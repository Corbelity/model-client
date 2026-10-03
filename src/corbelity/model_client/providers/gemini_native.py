"""Gemini, through Google's own SDK (`google-genai`) rather than the compatibility shim.

The `gemini` service reaches the same models through Google's OpenAI-compatibility
endpoint, and stays exactly as it was. This is a separate service, `gemini-native`, so
nothing written against the shim changes behaviour -- the same arrangement as
`ollama-local` and `ollama`. It is also where Gemini's video, image and speech routes will
land, since none of them exist on the compatibility endpoint.

Text and video (Veo). Requests are built from plain dicts, which the SDK accepts for every
type it defines; that keeps the SDK out of everything except `_build_client` and the one
type the operations API insists on, so the tests can fake this provider without
importing it.

Four things about the native text API that the shim used to hide, each handled below:

  * Reasoning comes back as parts marked `thought=True`. They are dropped from the text,
    rather than relying on the SDK's `.text` convenience, whose handling of non-text parts
    has changed between releases.
  * Thinking tokens are billed at the output rate but reported apart from the answer's
    tokens. `completion_tokens` adds them back so the flat figure is the billed figure;
    the split goes to `output_text_tokens` / `output_reasoning_tokens`.
  * Finish reasons are an enum whose `str()` is "FinishReason.STOP", not "STOP". The
    provider reports `.value`, or the base class's truncation check would never match.
  * A refused PROMPT yields no candidate at all, only `prompt_feedback.block_reason`. That
    is reported as finish_reason "prompt_blocked" with empty text, so the base class's
    empty-response warning fires with an honest reason instead of a bare None.

Thinking also spends from `max_output_tokens`. A low cap can be used up entirely by
reasoning, giving finish_reason MAX_TOKENS with no visible text; the empty-response warning
fires correctly, and raising max_tokens is the fix.

Video is a long-running operation: `_submit_video` starts it and returns its name,
`_poll_video` asks after it, `_fetch_video` downloads the result. The base class owns the
loop, the clock and the trace records (see jobs.py). Three things about Veo shape them:

  * A refusal on safety grounds is not an error. The operation finishes "done" with no
    video and a list of filter reasons, which is reported as the FILTERED state so the
    caller is told why rather than handed "no video".
  * The download goes to the configured endpoint, never to the host named in the
    response. The SDK takes only the file id from the video's URI and fetches it with
    this client's own base URL and key, so a response cannot steer the key elsewhere.
  * Generated videos are kept for two days. The URI is kept as `source_uri`, which is
    what extending the video later needs.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, ClassVar, cast

from ..catalog import load_catalog
from ..client import ModelClient, get_field, load_sdk
from ..errors import ModelClientError, VideoJobNotFoundError
from ..jobs import FAILED, FILTERED, RUNNING, SUCCEEDED, VideoPoll
from ..media import (
    ImageInput,
    LLMResult,
    MediaResult,
    VideoInputs,
    VideoOptions,
    sniff_video_mime,
)
from ..messages import Message
from ..registry import ProviderSpec, get_spec

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    # The SDK's TypedDicts, so that with the extra installed mypy checks every request key
    # against the SDK's own schema -- a renamed field fails type-checking in the
    # all-extras CI job rather than at call time. Without the extra they resolve to Any.
    from google.genai import Client
    from google.genai.types import (
        ContentDict,
        ContentUnionDict,
        GenerateContentConfigDict,
        GenerateVideosConfigDict,
        GenerateVideosSourceDict,
        ImageDict,
        PartDict,
        VideoGenerationReferenceImageDict,
    )

# Gemini names the assistant "model". The rest of this package says "assistant", and the
# history validator enforces that, so the translation happens here and nowhere else.
_ROLES = {"user": "user", "assistant": "model"}

# Reported in place of a finish reason when the prompt itself was refused. Lower-case and
# underscored so it reads as one of this package's values, not one of Gemini's.
PROMPT_BLOCKED = "prompt_blocked"


def _enum_value(value: Any) -> str | None:
    """The wire value of an SDK enum, or the string itself.

    `str()` on these enums gives the qualified name ("FinishReason.STOP"), which would
    match nothing a caller or INCOMPLETE_FINISH_REASONS compares against. `.value` is the
    API's own spelling. A plain string (from a dict-shaped response) passes through."""
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _answer_text(parts: Iterable[Any]) -> str:
    """The visible answer: every text part that is not reasoning, in order.

    Thought parts are only returned when thinking output is explicitly requested, which
    this client never does -- but a part flagged `thought` is reasoning whatever caused it
    to be sent, and answer text that silently includes it is the quiet kind of wrong."""
    pieces = []
    for part in parts:
        if get_field(part, "thought"):
            continue
        text = get_field(part, "text")
        if isinstance(text, str):
            pieces.append(text)
    return "".join(pieces)


def _input_split(details: Iterable[Any] | None) -> tuple[int | None, int | None]:
    """(text, image) input tokens from `prompt_tokens_details`, a list of per-modality
    counts. Each stays None unless the provider reported it -- "not counted" and "zero"
    are different statements."""
    text: int | None = None
    image: int | None = None
    for entry in details or ():
        count = get_field(entry, "token_count")
        if not isinstance(count, int):
            continue
        modality = _enum_value(get_field(entry, "modality"))
        if modality == "TEXT":
            text = (text or 0) + count
        elif modality == "IMAGE":
            image = (image or 0) + count
    return text, image


def _image_dict(image: ImageInput) -> ImageDict:
    return {"image_bytes": image.data, "mime_type": image.mime_type}


def _sum_present(*values: int | None) -> int | None:
    """Sum of the values that were reported, or None when none were."""
    present = [value for value in values if value is not None]
    return sum(present) if present else None


class GeminiNativeClient(ModelClient["Client"]):
    SPEC: ClassVar[ProviderSpec] = get_spec("gemini-native")

    def _build_client(self) -> Client:
        self._logger.info(
            "Initializing Gemini (native) client: %s (temperature=%s, max_tokens=%s)",
            self._model, self._temperature, self._max_tokens,
        )
        # Configuration is resolved BEFORE the SDK import, so a missing key raises the
        # same error whether or not the extra is installed (DESIGN.md §10).
        api_key = self._resolve_key()
        base_url = self._resolve_base_url()
        sdk = load_sdk("google.genai", self.SPEC.extra)
        return cast("Client", sdk.Client(
            # Passed explicitly rather than left to the SDK, which reads the environment
            # itself -- preferring GOOGLE_API_KEY over GEMINI_API_KEY, the opposite of the
            # order this spec declares.
            api_key=api_key,
            # Pinned for the same reason: the SDK switches to Vertex AI whenever
            # GOOGLE_GENAI_USE_VERTEXAI is set in the process environment. A library that
            # changed backend because of a variable some other tool set would be reaching
            # for the world rather than accepting it (DESIGN.md §8). Vertex, if it is ever
            # wanted, is a separate service name.
            vertexai=False,
            http_options={"base_url": base_url} if base_url else None,
        ))

    def _accepts_sampling_params(self) -> bool:
        """The catalog's `supports_sampling` flag, as the chat-completions dialect reads
        it. Unlisted models are assumed to accept temperature and top_p. Reads the full
        catalog, never a filtered one (DESIGN.md §6)."""
        entry = load_catalog(self._config.catalog_path).get(self._model)
        if entry is not None and entry.supports_sampling is not None:
            return entry.supports_sampling
        return True

    def _invoke(self, system: str, user: str, history: tuple[Message, ...],
                images: tuple[ImageInput, ...]) -> LLMResult:
        turns: list[ContentDict] = [
            {"role": _ROLES[turn["role"]], "parts": [{"text": turn["content"]}]}
            for turn in history
        ]
        # Images BEFORE the text: Google's prompting guidance is that a single image works
        # best placed ahead of the question about it -- the same order Anthropic
        # recommends, and the opposite of the OpenAI content-parts helper.
        parts: list[PartDict] = [
            {"inline_data": {"mime_type": image.mime_type, "data": image.data}}
            for image in images
        ]
        parts.append({"text": user})
        turns.append({"role": "user", "parts": parts})
        # Widened to the SDK's parameter type only here. `list` is invariant, so a
        # list[ContentDict] is not accepted where list[ContentUnionDict] is expected, even
        # though every element would be; the turns above stay checked as ContentDict.
        contents: list[ContentUnionDict] = [*turns]

        # Always `max_output_tokens`: the catalog's `max_tokens_param` exists for the
        # chat-completions dialect's spelling disagreement and does not apply here.
        config: GenerateContentConfigDict = {"max_output_tokens": self._max_tokens}
        # The system prompt is configuration, not a turn. Omitted when blank rather than
        # sent as an empty instruction, which the API has no use for.
        if system.strip():
            config["system_instruction"] = system
        if self._accepts_sampling_params():
            config["temperature"] = self._temperature
            if self._top_p is not None:
                config["top_p"] = self._top_p

        response = self._client.models.generate_content(
            model=self._model, contents=contents, config=config
        )
        return self._parse(response)

    def _parse(self, response: Any) -> LLMResult:
        usage = get_field(response, "usage_metadata")
        answer_tokens = get_field(usage, "candidates_token_count") if usage is not None else None
        thought_tokens = get_field(usage, "thoughts_token_count") if usage is not None else None
        input_text, input_image = _input_split(
            get_field(usage, "prompt_tokens_details") if usage is not None else None
        )
        tokens: dict[str, Any] = {
            "prompt_tokens": get_field(usage, "prompt_token_count") if usage is not None else None,
            # The billed output: the answer AND the thinking. Gemini reports them apart,
            # and candidates_token_count alone would price a reasoning model's call as
            # though it had not reasoned.
            "completion_tokens": _sum_present(answer_tokens, thought_tokens),
            "total_tokens": get_field(usage, "total_token_count") if usage is not None else None,
            "input_text_tokens": input_text,
            "input_image_tokens": input_image,
            "input_cached_tokens": (
                get_field(usage, "cached_content_token_count") if usage is not None else None
            ),
            "output_text_tokens": answer_tokens,
            "output_reasoning_tokens": thought_tokens,
        }

        candidates = get_field(response, "candidates") or []
        if not candidates:
            feedback = get_field(response, "prompt_feedback")
            reason = _enum_value(get_field(feedback, "block_reason")) if feedback else None
            if reason is None:
                # No candidate and no stated reason. Report it as it is: empty text, no
                # finish reason. The base class's empty-response warning covers it.
                return LLMResult(text="", finish_reason=None, **tokens)
            # The base class warns on finish=prompt_blocked; this adds the one detail it
            # cannot know, which is WHY -- the difference between "rephrase" and "this
            # will never be answered".
            self._logger.warning(
                "Gemini refused the prompt for %s (block_reason=%s%s).",
                self._model, reason,
                f": {get_field(feedback, 'block_reason_message')}"
                if get_field(feedback, "block_reason_message") else "",
            )
            return LLMResult(text="", finish_reason=PROMPT_BLOCKED, **tokens)

        candidate = candidates[0]
        content = get_field(candidate, "content")
        parts = get_field(content, "parts") if content is not None else None
        return LLMResult(
            text=_answer_text(parts or ()),
            finish_reason=_enum_value(get_field(candidate, "finish_reason")),
            **tokens,
        )

    # ----------------------------------------------------------------------- #
    # Video (Veo)
    # ----------------------------------------------------------------------- #
    def _sdk_types(self) -> Any:
        return load_sdk("google.genai.types", self.SPEC.extra)

    def _submit_video(self, prompt: str, inputs: VideoInputs,
                      options: VideoOptions) -> str:
        # Every role is mapped, including the two the built-in catalog does not yet
        # switch on for Veo (references, extend). A model the catalog does not describe
        # is passed through to the provider (DESIGN.md section 6); a role that reached
        # this point and was then left out of the request would be a silent drop.
        source: GenerateVideosSourceDict = {}
        # Omitted when blank rather than sent empty: an image-to-video request may carry
        # no prompt, and "no prompt" is the statement the API understands.
        if prompt.strip():
            source["prompt"] = prompt
        if inputs.first_frame is not None:
            source["image"] = _image_dict(inputs.first_frame)
        if inputs.extend is not None:
            source["video"] = {"uri": inputs.extend.uri}

        # Only what was requested (or forced by a catalog constraint): None is omitted so
        # the API's own default applies. The names match the SDK's one for one, and the
        # spec's video_settings has already refused any this API lacks.
        config: GenerateVideosConfigDict = cast(
            "GenerateVideosConfigDict", options.requested()
        )
        if inputs.last_frame is not None:
            config["last_frame"] = _image_dict(inputs.last_frame)
        if inputs.references:
            # ASSET is the reference type Google documents for Veo 3.1. The enum comes from
            # the SDK because the typed dict demands it.
            asset = self._sdk_types().VideoGenerationReferenceType.ASSET
            references: list[VideoGenerationReferenceImageDict] = [
                {"image": _image_dict(image), "reference_type": asset}
                for image in inputs.references
            ]
            config["reference_images"] = references

        operation = self._client.models.generate_videos(
            model=self._model, source=source, config=config or None,
        )
        name = get_field(operation, "name")
        if not isinstance(name, str) or not name:
            # Without the name nothing can find the job again, and it is running and
            # billed regardless. Say so, rather than return a job that can never finish.
            raise ModelClientError(
                f"Gemini accepted a video request for {self._model!r} but returned no "
                "operation name, so the job cannot be followed."
            )
        return name

    def _poll_video(self, operation: str) -> VideoPoll:
        # operations.get() wants an operation OBJECT: it reads `.name` and parses the
        # reply with the object's own class. A bare one built from the name is enough,
        # and is what lets a job be resumed in a process that never submitted it.
        handle = self._sdk_types().GenerateVideosOperation(name=operation)
        try:
            reply = self._client.operations.get(handle)
        except Exception as err:
            # 404 is the one failure with a different remedy (resubmit, do not retry), so
            # it gets its own error. Anything else propagates and the job stays pollable.
            if getattr(err, "code", None) == 404:
                raise VideoJobNotFoundError(self.SPEC.name, operation) from err
            raise
        return self._read_operation(operation, reply)

    def _read_operation(self, operation: str, reply: Any) -> VideoPoll:
        if not get_field(reply, "done"):
            # Veo reports no progress figure, so none is invented.
            return VideoPoll(state=RUNNING)

        error = get_field(reply, "error")
        if error:
            message = get_field(error, "message") or str(error)
            code = get_field(error, "code")
            return VideoPoll(
                state=FAILED, error=f"{message} (code {code})" if code else str(message)
            )

        response = get_field(reply, "response", "result")
        # get_field() reads None as "absent", so a missing response is simply no videos.
        generated = get_field(response, "generated_videos") or []
        videos = [
            video for video in (get_field(entry, "video") for entry in generated)
            if video is not None
        ]
        if videos:
            if len(videos) > 1:
                # Never asked for (number_of_videos is not sent), so worth a word.
                self._logger.warning(
                    "Video job %s returned %d videos; keeping the first.",
                    operation, len(videos),
                )
            return VideoPoll(state=SUCCEEDED, output=videos[0])

        reasons = tuple(
            str(reason)
            for reason in (get_field(response, "rai_media_filtered_reasons") or ())
        )
        count = get_field(response, "rai_media_filtered_count")
        if reasons or count:
            return VideoPoll(state=FILTERED, filtered_reasons=reasons)
        return VideoPoll(
            state=FAILED,
            error="the job finished without a video and gave no reason",
        )

    def _fetch_video(self, poll: VideoPoll) -> MediaResult:
        video = poll.output
        inline = get_field(video, "video_bytes")
        if isinstance(inline, bytes) and inline:
            data = inline
        else:
            # Fetched by file id through this client's endpoint (see the module note);
            # the URI's host is not contacted. A failure here propagates, the job stays
            # unfinalized, and the next poll tries the download again.
            data = self._client.files.download(file=video)
        declared = get_field(video, "mime_type")
        return MediaResult(
            data=data,
            # What the bytes ARE, falling back to what the provider says they are.
            mime_type=sniff_video_mime(
                data, default=declared if isinstance(declared, str) else "video/mp4"
            ),
            source_uri=get_field(video, "uri"),
        )
