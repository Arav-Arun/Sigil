from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from sigil.imaging import (
    SERPAPI_MAX_BYTES,
    ImageInputError,
    encode_for_serpapi,
    load_validated_image,
    save_normalized_image,
)


def create_image(path: Path, size: tuple[int, int] = (640, 480)) -> Path:
    Image.new("RGB", size, color=(80, 120, 160)).save(path, format="JPEG", quality=95)
    return path


def test_load_valid_image_and_hash(tmp_path: Path) -> None:
    source = create_image(tmp_path / "input.jpg")

    result = load_validated_image(source)

    assert result.image.mode == "RGB"
    assert result.image.size == (640, 480)
    assert len(result.sha256) == 64
    assert result.byte_count == source.stat().st_size


def test_rejects_corrupt_image(tmp_path: Path) -> None:
    source = tmp_path / "fake.jpg"
    source.write_bytes(b"this is not an image")

    with pytest.raises(ImageInputError, match="Invalid or unsafe image"):
        load_validated_image(source)


def test_rejects_tiny_image(tmp_path: Path) -> None:
    source = create_image(tmp_path / "tiny.jpg", (95, 200))

    with pytest.raises(ImageInputError, match="at least 96px"):
        load_validated_image(source)


def test_serpapi_encoding_respects_byte_limit(tmp_path: Path) -> None:
    source = create_image(tmp_path / "large.jpg", (2400, 1600))
    validated = load_validated_image(source)

    encoded = encode_for_serpapi(validated.image)

    assert len(encoded.data) <= SERPAPI_MAX_BYTES
    assert encoded.content_type == "image/jpeg"
    assert max(encoded.width, encoded.height) <= 1600


def test_normalized_copy_has_no_exif(tmp_path: Path) -> None:
    source = create_image(tmp_path / "source.jpg")
    validated = load_validated_image(source)
    destination = save_normalized_image(validated, tmp_path / "normalized.jpg")

    with Image.open(destination) as normalized:
        assert not normalized.getexif()
