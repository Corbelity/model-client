"""Provider implementations.

Each module here is cheap to import: none of them imports its SDK at module level (see
client.load_sdk), so this package can be imported with zero provider SDKs installed.

Shared request-shaping helpers live here because two of the four providers speak the
OpenAI content-parts dialect, and a helper duplicated across sibling modules drifts.
"""
from __future__ import annotations

from typing import Any

from ..media import ImageInput


def openai_user_content(user: str, images: tuple[ImageInput, ...]) -> str | list[dict[str, Any]]:
    """The OpenAI content-parts shape, shared by OpenRouter and HuggingFace -- both take
    `chat_completion`-style messages, so this exists once rather than twice.

    Returns the plain string when there are no images. That is what keeps a text-only
    request byte-identical to what this code sent before image input existed."""
    if not images:
        return user
    return [
        {"type": "text", "text": user},
        *(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{image.mime_type};base64,{image.base64()}"},
            }
            for image in images
        ),
    ]


def anthropic_user_content(user: str, images: tuple[ImageInput, ...]) -> str | list[dict[str, Any]]:
    """Anthropic's block shape. Images come FIRST: their guidance is that a question
    following the image produces better results than one preceding it."""
    if not images:
        return user
    return [
        *(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image.mime_type,
                    "data": image.base64(),
                },
            }
            for image in images
        ),
        {"type": "text", "text": user},
    ]


__all__ = ["anthropic_user_content", "openai_user_content"]
