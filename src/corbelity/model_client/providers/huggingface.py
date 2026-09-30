"""HuggingFace Inference: the only provider here that produces text, images and audio."""
from __future__ import annotations

import io
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, cast

from ..client import ModelClient, get_field, load_sdk
from ..media import (
    IMAGE,
    SOUND,
    TEXT,
    ImageInput,
    ImageOptions,
    LLMResult,
    MediaResult,
    sniff_audio_mime,
)
from ..messages import Message
from ..registry import ProviderSpec, get_spec
from . import openai_user_content

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from huggingface_hub import InferenceClient


def _is_model_not_supported(err: BaseException) -> bool:
    """Is this the router refusing a model the account's providers do not cover?

    Keyed on the status code plus the API's own machine-readable error code, never on its
    prose -- wording is not a contract, and a message match would break on a rephrasing.
    Read off the exception rather than imported from huggingface_hub.errors, because this
    module must not import the SDK at module scope (the extra is optional)."""
    response = getattr(err, "response", None)
    if getattr(response, "status_code", None) != 400:
        return False
    return "model_not_supported" in str(err)


class HuggingFaceClient(ModelClient["InferenceClient"]):
    SPEC: ClassVar[ProviderSpec] = get_spec("huggingface")

    # pipeline_tag per modality, used to build an actionable error when nothing serves a
    # model.
    _TASKS: ClassVar[dict[str, str]] = {
        TEXT: "text-generation",
        IMAGE: "text-to-image",
        SOUND: "text-to-speech",
    }

    def _build_client(self) -> InferenceClient:
        sdk = load_sdk("huggingface_hub", self.SPEC.extra)
        return cast("InferenceClient", sdk.InferenceClient(token=self._resolve_key()))

    def _unserved_message(self, modality: str) -> str:
        """Nobody serves this model for this task. The fix is a different model."""
        task = self._TASKS.get(modality, modality)
        return (
            f"No HuggingFace inference provider currently serves {self._model!r} for "
            f"{task}. Browse models that are served at "
            f"https://huggingface.co/models?pipeline_tag={task}&inference_provider=all"
        )

    def _not_enabled_message(self, modality: str) -> str:
        """Somebody serves it, but not a provider this account has turned on. The fix is
        account settings. Kept separate from _unserved_message on purpose: sending someone
        to look for another model when their account is what needs changing costs them the
        time it takes to rule out every model they try."""
        task = self._TASKS.get(modality, modality)
        return (
            f"No inference provider enabled on this HuggingFace account serves "
            f"{self._model!r} for {task}. Enable one at "
            f"https://hf.co/settings/inference-providers, or choose a model your enabled "
            f"providers already serve."
        )

    def _call(self, modality: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Translate the hub's two ways of declining to route a request into actionable
        errors. Both are bad-request conditions, so both raise ValueError and web layers
        that map ValueError to a 400 keep working -- the same reasoning as
        MissingCredentialsError's dual bases.

        Which path fires depends on the modality, and they are not interchangeable:

          * image and speech resolve a provider mapping client-side, and huggingface_hub
            calls next() on it. An empty mapping raises a BARE StopIteration carrying no
            message at all, which reaches a UI as "StopIteration: ".
          * text (task "conversational") short-circuits to a dedicated auto-router before
            any mapping is fetched -- see huggingface_hub/inference/_providers/__init__.py
            -- so StopIteration is unreachable for it. The router answers HTTP 400 with
            code "model_not_supported", which hf_raise_for_status turns into
            BadRequestError(HfHubHTTPError, ValueError).

        The router path arrived in huggingface-hub 2.0.0. Before it, text also raised
        StopIteration; the live text test is what caught the change."""
        try:
            return fn(*args, **kwargs)
        except StopIteration as err:
            raise ValueError(self._unserved_message(modality)) from err
        except ValueError as err:
            if _is_model_not_supported(err):
                raise ValueError(self._not_enabled_message(modality)) from err
            raise

    def _invoke(self, system: str, user: str, history: tuple[Message, ...],
                images: tuple[ImageInput, ...]) -> LLMResult:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                *history,
                {"role": "user", "content": openai_user_content(user, images)},
            ],
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
        if self._top_p is not None:
            payload["top_p"] = self._top_p

        response = self._call(TEXT, self._client.chat_completion, **payload)
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        # HuggingFace/TGI is OpenAI-shaped; coerce a None content to "" (see OpenRouter).
        content = choice.message.content
        return LLMResult(
            text="" if content is None else str(content),
            prompt_tokens=get_field(usage, "prompt_tokens") if usage is not None else None,
            completion_tokens=get_field(usage, "completion_tokens") if usage is not None else None,
            total_tokens=get_field(usage, "total_tokens") if usage is not None else None,
            finish_reason=getattr(choice, "finish_reason", None),
        )

    def _invoke_image(self, prompt: str, images: tuple[ImageInput, ...],
                      options: ImageOptions) -> MediaResult:
        # `images` and `options` are always empty here: the huggingface spec declares
        # neither image_input nor any size or quality, so generate_image() rejects all of
        # them before reaching a provider. The parameters are present because the seam
        # requires them, not because they are used.
        # text_to_image returns a PIL.Image; re-encode to PNG bytes so the caller never
        # needs Pillow to hand the result on.
        image = self._call(IMAGE, self._client.text_to_image, prompt, model=self._model)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return MediaResult(data=buffer.getvalue(), mime_type="image/png")

    def _invoke_speech(self, text: str) -> MediaResult:
        # Container varies by model and provider (Kokoro returns WAV, others FLAC or MP3),
        # so sniff it rather than assuming.
        audio = bytes(self._call(SOUND, self._client.text_to_speech, text, model=self._model))
        return MediaResult(data=audio, mime_type=sniff_audio_mime(audio))
