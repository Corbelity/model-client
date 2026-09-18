"""HuggingFace Inference: the only provider here that produces text, images and audio."""
from __future__ import annotations

import io
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, cast

from ..client import ModelClient, get_field, load_sdk
from ..media import IMAGE, SOUND, TEXT, ImageInput, LLMResult, MediaResult, sniff_audio_mime
from ..messages import Message
from ..registry import ProviderSpec, get_spec
from . import openai_user_content

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from huggingface_hub import InferenceClient


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

    def _call(self, modality: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """huggingface_hub raises a BARE StopIteration when no inference provider serves a
        model for a task -- it calls next() on an empty provider mapping, so the error has
        no message at all and reaches a UI as "StopIteration: ". Translate it into
        something the user can act on. (ValueError, so callers treat it as a bad request
        rather than a provider outage.)"""
        try:
            return fn(*args, **kwargs)
        except StopIteration as err:
            task = self._TASKS.get(modality, modality)
            raise ValueError(
                f"No HuggingFace inference provider currently serves {self._model!r} for "
                f"{task}. Browse models that are served at "
                f"https://huggingface.co/models?pipeline_tag={task}&inference_provider=all"
            ) from err

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

    def _invoke_image(self, prompt: str) -> MediaResult:
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
