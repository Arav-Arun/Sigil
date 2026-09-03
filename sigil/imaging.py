"""Safe, deterministic image ingestion shared by face and search stages."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_INPUT_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
MIN_IMAGE_SIDE = 96
SERPAPI_MAX_BYTES = 500_000
SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}


class ImageInputError(ValueError):
    """Raised when an image is unsafe, unsupported, or unusable."""


@dataclass(frozen=True, slots=True)
class ValidatedImage:
    path: Path
    image: Image.Image
    original_format: str
    sha256: str
    byte_count: int

    @property
    def width(self) -> int:
        return self.image.width

    @property
    def height(self) -> int:
        return self.image.height


@dataclass(frozen=True, slots=True)
class EncodedImage:
    data: bytes
    filename: str
    content_type: str
    width: int
    height: int


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_validated_image(path: str | Path) -> ValidatedImage:
    """Decode an image defensively, normalize orientation, and strip metadata in memory."""

    image_path = Path(path).expanduser().resolve()
    if not image_path.is_file():
        raise ImageInputError(f"Image file not found: {image_path}")

    byte_count = image_path.stat().st_size
    if byte_count == 0:
        raise ImageInputError("Image file is empty")
    if byte_count > MAX_INPUT_BYTES:
        raise ImageInputError(f"Image exceeds the {MAX_INPUT_BYTES // (1024 * 1024)} MB limit")

    data = image_path.read_bytes()
    previous_pixel_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    try:
        with Image.open(io.BytesIO(data)) as probe:
            original_format = (probe.format or "").upper()
            if original_format not in SUPPORTED_FORMATS:
                allowed = ", ".join(sorted(SUPPORTED_FORMATS))
                raise ImageInputError(f"Unsupported image format; expected one of: {allowed}")
            if probe.width * probe.height > MAX_IMAGE_PIXELS:
                raise ImageInputError(f"Image exceeds the {MAX_IMAGE_PIXELS:,}-pixel limit")
            probe.verify()

        with Image.open(io.BytesIO(data)) as decoded:
            normalized = ImageOps.exif_transpose(decoded).convert("RGB")
            normalized.load()
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as exc:
        raise ImageInputError(f"Invalid or unsafe image: {exc}") from exc
    finally:
        Image.MAX_IMAGE_PIXELS = previous_pixel_limit

    if min(normalized.size) < MIN_IMAGE_SIDE:
        raise ImageInputError(f"Image must be at least {MIN_IMAGE_SIDE}px on its shortest side")

    return ValidatedImage(
        path=image_path,
        image=normalized,
        original_format=original_format,
        sha256=sha256_bytes(data),
        byte_count=byte_count,
    )


def save_normalized_image(validated: ValidatedImage, destination: str | Path) -> Path:
    """Save stable sRGB pixels without carrying source EXIF metadata forward."""

    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    validated.image.save(destination_path, format="JPEG", quality=95, optimize=True)
    return destination_path


def encode_for_serpapi(
    image: Image.Image,
    *,
    max_bytes: int = SERPAPI_MAX_BYTES,
    max_side: int = 1600,
) -> EncodedImage:
    """Encode a query image beneath SerpApi's hard byte limit without geometric distortion."""

    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    if max_side < MIN_IMAGE_SIDE:
        raise ValueError(f"max_side must be at least {MIN_IMAGE_SIDE}")

    working = image.copy().convert("RGB")
    working.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

    for quality in (92, 88, 84, 80, 74, 68, 60, 52, 44):
        buffer = io.BytesIO()
        working.save(buffer, format="JPEG", quality=quality, optimize=True, progressive=True)
        payload = buffer.getvalue()
        if len(payload) <= max_bytes:
            return EncodedImage(
                data=payload,
                filename="query.jpg",
                content_type="image/jpeg",
                width=working.width,
                height=working.height,
            )

    # Highly textured images can remain large even at low JPEG quality. Shrink dimensions in
    # bounded steps instead of silently violating the API limit.
    while min(working.size) >= MIN_IMAGE_SIDE * 2:
        working = working.resize(
            (
                max(MIN_IMAGE_SIDE, working.width * 3 // 4),
                max(MIN_IMAGE_SIDE, working.height * 3 // 4),
            ),
            Image.Resampling.LANCZOS,
        )
        buffer = io.BytesIO()
        working.save(buffer, format="JPEG", quality=60, optimize=True, progressive=True)
        payload = buffer.getvalue()
        if len(payload) <= max_bytes:
            return EncodedImage(
                data=payload,
                filename="query.jpg",
                content_type="image/jpeg",
                width=working.width,
                height=working.height,
            )

    raise ImageInputError(f"Could not encode query image below {max_bytes} bytes")
