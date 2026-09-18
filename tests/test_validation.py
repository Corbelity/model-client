"""History and image validation, and the MIME sniffers.

These are the parts most worth testing: they are pure, they encode provider disagreements
that are invisible until a request fails, and every one of them is a rule that a future
refactor could quietly relax.
"""
from __future__ import annotations

import pytest

from corbelity.model_client import (
    ImageInput,
    sniff_audio_mime,
    sniff_image_mime,
    validate_history,
    validate_images,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"rest"
JPEG = b"\xff\xd8\xff" + b"rest"
WEBP = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"rest"
WAV = b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE"


def turn(role: str, content: str = "x") -> dict[str, str]:
    return {"role": role, "content": content}


class TestValidateHistory:
    def test_none_and_empty_normalize_to_empty_tuple(self) -> None:
        assert validate_history(None) == ()
        assert validate_history([]) == ()

    def test_alternating_history_is_returned_as_a_tuple(self) -> None:
        history = [turn("user"), turn("assistant"), turn("user"), turn("assistant")]
        assert validate_history(history) == tuple(history)

    def test_must_start_with_user(self) -> None:
        with pytest.raises(ValueError, match="entry 0"):
            validate_history([turn("assistant"), turn("user"), turn("assistant")])

    def test_must_end_with_assistant(self) -> None:
        # The rule that makes `history + [current user turn]` still alternating.
        with pytest.raises(ValueError, match="ends with a 'user' turn"):
            validate_history([turn("user")])

    def test_consecutive_same_role_is_rejected_not_merged(self) -> None:
        with pytest.raises(ValueError, match="turns must alternate"):
            validate_history([turn("user"), turn("user"), turn("assistant")])

    def test_unknown_role_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="only 'user' and 'assistant'"):
            validate_history([turn("system"), turn("assistant")])

    def test_extra_keys_are_rejected(self) -> None:
        entry = {"role": "user", "content": "x", "name": "steve"}
        with pytest.raises(ValueError, match="unexpected key"):
            validate_history([entry, turn("assistant")])

    def test_missing_key_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="missing content"):
            validate_history([{"role": "user"}, turn("assistant")])

    def test_non_string_content_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-string content"):
            validate_history([{"role": "user", "content": 42}, turn("assistant")])


class TestSniffers:
    @pytest.mark.parametrize(
        ("data", "expected"),
        [(PNG, "image/png"), (JPEG, "image/jpeg"), (b"GIF89a...", "image/gif")],
    )
    def test_known_image_signatures(self, data: bytes, expected: str) -> None:
        assert sniff_image_mime(data) == expected

    def test_webp_tag_is_at_offset_eight(self) -> None:
        # The case a naive prefix check gets wrong: WEBP is RIFF + size + WEBP.
        assert sniff_image_mime(WEBP) == "image/webp"

    def test_riff_alone_is_not_an_image(self) -> None:
        # A bare RIFF prefix is just as likely to be a WAV file, so the image sniffer must
        # not claim it. Defaulting to a non-image type is what routes it to a readable
        # rejection in validate_images instead of an opaque provider 400.
        assert sniff_image_mime(WAV) == "application/octet-stream"

    def test_unknown_image_default_is_not_an_image_type(self) -> None:
        assert sniff_image_mime(b"nonsense") == "application/octet-stream"

    @pytest.mark.parametrize(
        ("data", "expected"),
        [
            (WAV, "audio/wav"),
            (b"fLaC...", "audio/flac"),
            (b"OggS...", "audio/ogg"),
            (b"ID3...", "audio/mpeg"),
            (b"\xff\xfb...", "audio/mpeg"),
        ],
    )
    def test_audio_signatures(self, data: bytes, expected: str) -> None:
        assert sniff_audio_mime(data) == expected

    def test_unknown_audio_falls_back_to_wav(self) -> None:
        assert sniff_audio_mime(b"nonsense") == "audio/wav"


class TestValidateImages:
    def test_none_normalizes_to_empty_tuple(self) -> None:
        assert validate_images(None) == ()

    def test_valid_image_passes_through(self) -> None:
        image = ImageInput(data=PNG, mime_type="image/png", name="shot.png")
        assert validate_images([image]) == (image,)

    def test_from_bytes_sniffs_the_type(self) -> None:
        assert ImageInput.from_bytes(JPEG).mime_type == "image/jpeg"

    def test_empty_data_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Image 0 has no data"):
            validate_images([ImageInput(data=b"", mime_type="image/png")])

    def test_unsupported_type_names_the_allowed_set(self) -> None:
        image = ImageInput(data=b"xx", mime_type="image/tiff")
        with pytest.raises(ValueError, match="allowed: image/gif"):
            validate_images([image])

    def test_wrong_type_entirely_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="not an ImageInput"):
            validate_images(["/path/to/file.png"])  # type: ignore[list-item]

    def test_base64_round_trips(self) -> None:
        import base64

        image = ImageInput(data=PNG, mime_type="image/png")
        assert base64.b64decode(image.base64()) == PNG
