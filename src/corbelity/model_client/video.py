"""Judging a video request against a service and a model, before anything is sent.

One function, `resolve_request()`, used by every path that sends a video request, so a
check a UI runs ahead of time (`resolve_video_request()` in the factory) is exactly the
check a real submission gets. It returns the settings to send, or raises.

Three layers, cheapest and most certain first:

  1. Structure -- types and shapes that are wrong for every provider (a duration that is
     not a whole number, an extend input that is both bytes and a handle).
  2. The client's capability, from the ProviderSpec -- whether this service takes video
     at all, and which forms of video input it can route.
  3. The model's stated capability, from the catalog's `video` block -- enforced only
     when stated. A model the catalog does not describe gets layers 1 and 2 and is then
     passed through for the provider to judge, so a model released after this package
     still works (DESIGN.md section 6, "Enforced when stated").

Nothing here substitutes. A value a model does not offer raises; it is never swapped for
a nearby one. The one thing filled in is a setting a constraint forces to a SINGLE value
and the caller left unset -- there is no other value it could be, so nothing is chosen on
the caller's behalf. A setting the caller gave a different value is refused.
"""
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from .errors import (
    TooManyImagesError,
    UnsupportedModalityError,
    UnsupportedVideoInputError,
    UnsupportedVideoSettingError,
)
from .media import (
    FIRST_FRAME,
    LAST_FRAME,
    REFERENCES,
    SUPPORTED_VIDEO_MIMES,
    VIDEO,
    ImageInput,
    VideoInput,
    VideoInputs,
    VideoOptions,
    parse_aspect_ratio,
    validate_images,
)

if TYPE_CHECKING:
    from .catalog import ModelInfo, VideoCapabilities, VideoConstraint
    from .registry import ProviderSpec

_RESOLUTION_PREFIX = "resolution:"


def resolve_request(spec: ProviderSpec, model: str, entry: ModelInfo | None,
                    inputs: VideoInputs, options: VideoOptions) -> VideoOptions:
    """Validate a video request and return the options to send.

    `entry` must come from the UNFILTERED catalog (`load_catalog()`, never
    `catalog_for()`): narrowing what a UI lists must never change what a call is allowed
    to do."""
    if VIDEO not in spec.modalities:
        raise UnsupportedModalityError(spec.name, VIDEO, spec.modalities)

    _check_structure(inputs, options)
    _check_input_forms(spec, model, inputs)

    caps = entry.video if entry is not None else None
    if caps is None:
        # Not described: structural rules only, then the provider decides. The one
        # structural rule about roles -- an end frame needs a start frame -- applies here
        # because interpolation is meaningless without both. A model the catalog DOES
        # describe states it as a constraint row instead, so its own table is the whole
        # truth about it.
        if inputs.last_frame is not None and inputs.first_frame is None:
            raise UnsupportedVideoInputError(
                spec.name, model,
                "last_frame needs first_frame: an end frame is interpolated towards from "
                "a start frame. Supply both, or use first_frame alone.",
            )
        return options

    return _check_capabilities(spec.name, model, caps, inputs, options)


# --------------------------------------------------------------------------- #
# Layer 1: structure
# --------------------------------------------------------------------------- #
def _check_structure(inputs: VideoInputs, options: VideoOptions) -> None:
    """Shapes that are wrong for every provider. Plain ValueError (DESIGN.md section 11):
    each is a caller mistake, not a capability."""
    for role, image in ((FIRST_FRAME, inputs.first_frame), (LAST_FRAME, inputs.last_frame)):
        if image is not None:
            if not isinstance(image, ImageInput):
                raise ValueError(f"{role} must be an ImageInput (got {type(image).__name__}).")
            validate_images([image])
    if inputs.references:
        validate_images(inputs.references)

    extend = inputs.extend
    if extend is not None:
        if not isinstance(extend, VideoInput):
            raise ValueError(f"extend must be a VideoInput (got {type(extend).__name__}).")
        has_bytes = extend.data is not None
        has_uri = extend.uri is not None
        if has_bytes == has_uri:
            raise ValueError(
                "extend must carry exactly one of data= or uri=: the video's bytes, or a "
                "provider's handle to a video it holds."
            )
        if has_uri and not (isinstance(extend.uri, str) and extend.uri.strip()):
            raise ValueError("extend.uri is empty.")
        if has_bytes:
            if not isinstance(extend.data, bytes) or not extend.data:
                raise ValueError("extend has no data.")
            if extend.mime_type not in SUPPORTED_VIDEO_MIMES:
                raise ValueError(
                    f"extend has unsupported type {extend.mime_type!r}; allowed: "
                    f"{', '.join(sorted(SUPPORTED_VIDEO_MIMES))}."
                )

    if options.aspect_ratio is not None and parse_aspect_ratio(options.aspect_ratio) is None:
        raise ValueError(f"aspect_ratio must be W:H, like '16:9' (got {options.aspect_ratio!r}).")
    if options.duration_seconds is not None and not _is_positive_int(options.duration_seconds):
        # bool is excluded explicitly: True is an int in Python and would read as 1 second.
        raise ValueError(
            f"duration_seconds must be a positive whole number (got {options.duration_seconds!r})."
        )
    if options.seed is not None and not _is_int(options.seed):
        raise ValueError(f"seed must be an int (got {options.seed!r}).")
    if options.generate_audio is not None and not isinstance(options.generate_audio, bool):
        raise ValueError(f"generate_audio must be True or False (got {options.generate_audio!r}).")
    for name in ("resolution", "negative_prompt", "person_generation"):
        value = getattr(options, name)
        if value is not None and not (isinstance(value, str) and value.strip()):
            raise ValueError(f"{name} must be a non-empty string (got {value!r}).")


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


# --------------------------------------------------------------------------- #
# Layer 2: the client's capability
# --------------------------------------------------------------------------- #
def _check_input_forms(spec: ProviderSpec, model: str, inputs: VideoInputs) -> None:
    extend = inputs.extend
    if extend is None or extend.form in spec.video_input_forms:
        return
    if extend.form == "bytes" and "uri" in spec.video_input_forms:
        raise UnsupportedVideoInputError(
            spec.name, model,
            "extend needs a handle to a video this service produced and still holds, not "
            "the video's bytes. Pass the earlier result (extend=result) or "
            "VideoInput(uri=...). Providers keep generated videos for a limited time "
            "(Veo: 2 days).",
        )
    raise UnsupportedVideoInputError(
        spec.name, model,
        f"this client cannot send a video input ({extend.form}) to {spec.name!r}."
        + (f" It accepts: {', '.join(spec.video_input_forms)}." if spec.video_input_forms
           else ""),
    )


# --------------------------------------------------------------------------- #
# Layer 3: the model's stated capability
# --------------------------------------------------------------------------- #
def _check_capabilities(service: str, model: str, caps: VideoCapabilities,
                        inputs: VideoInputs, options: VideoOptions) -> VideoOptions:
    roles = inputs.roles()

    if caps.inputs is not None:
        for role in roles:
            if role not in caps.inputs:
                takes = ", ".join(caps.inputs) if caps.inputs else "a text prompt only"
                raise UnsupportedVideoInputError(
                    service, model, f"does not take {role}. It takes: {takes}."
                )

    if caps.max_references is not None and len(inputs.references) > caps.max_references:
        raise TooManyImagesError(service, len(inputs.references), caps.max_references)

    # Role constraints first: they do not depend on settings, and a forbidden combination
    # is the more fundamental problem to report.
    for constraint in caps.constraints:
        if not _active(constraint, roles, options):
            continue
        if constraint.exclude_input:
            conflicts = [role for role in constraint.exclude_input if role in roles]
            if conflicts:
                raise UnsupportedVideoInputError(
                    service, model, _exclusion_message(constraint.when, conflicts)
                )
        if constraint.require_input and constraint.require_input not in roles:
            raise UnsupportedVideoInputError(
                service, model,
                f"{constraint.when} needs {constraint.require_input}. Supply both, or drop "
                f"{constraint.when}.",
            )

    resolved = _fill_forced(caps.constraints, roles, options)
    _check_required(service, model, caps.constraints, roles, options, resolved)
    _check_offered(service, model, caps, resolved)
    return resolved


def _active(constraint: VideoConstraint, roles: tuple[str, ...], options: VideoOptions) -> bool:
    if constraint.when.startswith(_RESOLUTION_PREFIX):
        return options.resolution == constraint.when[len(_RESOLUTION_PREFIX):]
    return constraint.when in roles


def _describe(when: str) -> str:
    if when.startswith(_RESOLUTION_PREFIX):
        return f"at resolution {when[len(_RESOLUTION_PREFIX):]}"
    return f"when {when} {'are' if when == REFERENCES else 'is'} supplied"


def _exclusion_message(when: str, conflicts: list[str]) -> str:
    """Names both sides of the conflict and both ways out. The provider's own rejection of
    these is often generic, so this is the only place a caller learns what was wrong."""
    others = " / ".join(conflicts)
    return (
        f"cannot combine {when} with {others}. Use {when} alone, or {others} without "
        f"{when}."
    )


def _fill_forced(constraints: tuple[VideoConstraint, ...], roles: tuple[str, ...],
                 options: VideoOptions) -> VideoOptions:
    """Fill each setting an active constraint forces, where the caller left it unset.

    Repeated until stable, because a filled setting can activate another row (filling
    resolution can switch on a "resolution:<v>" rule). Bounded by the number of rows:
    each pass that changes anything fills at least one more setting."""
    resolved = options
    for _ in range(len(constraints) + 1):
        changes: dict[str, Any] = {}
        for constraint in constraints:
            if constraint.require and _active(constraint, roles, resolved):
                for setting, value in constraint.require.items():
                    if getattr(resolved, setting) is None and setting not in changes:
                        changes[setting] = value
        if not changes:
            return resolved
        resolved = replace(resolved, **changes)
    return resolved


def _check_required(service: str, model: str, constraints: tuple[VideoConstraint, ...],
                    roles: tuple[str, ...], requested: VideoOptions,
                    resolved: VideoOptions) -> None:
    for constraint in constraints:
        if not (constraint.require and _active(constraint, roles, resolved)):
            continue
        for setting, value in constraint.require.items():
            actual = getattr(resolved, setting)
            if actual != value:
                asked = getattr(requested, setting)
                # Distinguish "you asked for something else" from "two rules disagree",
                # which happens when a filled value meets a second row wanting another.
                detail = (
                    f"{_describe(constraint.when)}, {setting} must be {value!r}."
                    if asked is not None
                    else f"{_describe(constraint.when)}, {setting} must be {value!r}, but "
                         f"another constraint already requires {actual!r}."
                )
                raise UnsupportedVideoSettingError(
                    service, model, setting, actual, reason=detail[0].upper() + detail[1:]
                )


def _check_offered(service: str, model: str, caps: VideoCapabilities,
                   options: VideoOptions) -> None:
    if options.aspect_ratio is not None and caps.aspect_ratios is not None:
        wanted = parse_aspect_ratio(options.aspect_ratio)
        if all(parse_aspect_ratio(ratio) != wanted for ratio in caps.aspect_ratios):
            raise UnsupportedVideoSettingError(
                service, model, "aspect_ratio", options.aspect_ratio, caps.aspect_ratios
            )
    if options.resolution is not None and caps.resolutions is not None \
            and options.resolution not in caps.resolutions:
        raise UnsupportedVideoSettingError(
            service, model, "resolution", options.resolution, caps.resolutions
        )
    if options.duration_seconds is not None and caps.durations is not None \
            and options.duration_seconds not in caps.durations:
        raise UnsupportedVideoSettingError(
            service, model, "duration_seconds", options.duration_seconds, caps.durations
        )
    if options.generate_audio is False and caps.audio == "always":
        raise UnsupportedVideoSettingError(
            service, model, "generate_audio", False,
            reason="This model always produces audio, so the setting cannot be honoured; "
                   "omit it rather than believe the result is silent.",
        )
    if options.generate_audio is True and caps.audio == "never":
        raise UnsupportedVideoSettingError(
            service, model, "generate_audio", True,
            reason="This model produces no audio.",
        )


__all__ = ["resolve_request"]
