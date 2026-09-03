"""Face detection, alignment, quality gating, and embedding.

Public surface for the rest of the pipeline. Everything below this line works in RGB
uint8 numpy arrays; conversion from disk bytes happens once, in :mod:`sigil.imaging`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sigil.config import ensure_output_dir
from sigil.face.align import CROP_SIZE, align_face, portrait_crop
from sigil.face.engine import (
    EMBEDDING_DIM,
    DetectedFace,
    FaceEngine,
    FaceEngineError,
    cosine_distance,
    get_engine,
)
from sigil.face.quality import assess, pose_offsets
from sigil.imaging import ImageInputError, load_validated_image, save_normalized_image
from sigil.models import FaceObservation, PipelineErrorCode


class FacePipelineError(RuntimeError):
    """Raised when a face cannot be processed defensibly.

    Carries a stable :class:`PipelineErrorCode` so the CLI, tests, and JSON output all
    agree on *why* a run stopped.
    """

    def __init__(self, code: PipelineErrorCode, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def select_face(
    faces: list[DetectedFace],
    *,
    face_index: int | None = None,
    select_largest: bool = False,
) -> int:
    """Choose which detected face to use, never guessing silently on ambiguity."""

    if not faces:
        raise FacePipelineError(PipelineErrorCode.NO_FACE, "no face was detected in the input")

    if face_index is not None:
        if not 0 <= face_index < len(faces):
            raise FacePipelineError(
                PipelineErrorCode.MULTIPLE_FACES,
                f"--face-index {face_index} is outside 0..{len(faces) - 1}",
            )
        return face_index

    if len(faces) == 1:
        return 0

    if select_largest:
        # ``detect`` already returns faces sorted by area, largest first.
        return 0

    raise FacePipelineError(
        PipelineErrorCode.MULTIPLE_FACES,
        f"detected {len(faces)} faces; pass --face-index N or --largest to choose explicitly",
    )


def annotate(image: np.ndarray, faces: list[DetectedFace], selected: int) -> Image.Image:
    """Draw every detected face, highlighting the one that was used."""

    canvas = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    stroke = max(2, min(canvas.size) // 300)
    for index, face in enumerate(faces):
        chosen = index == selected
        color = (32, 220, 120) if chosen else (250, 176, 5)
        x1, y1, x2, y2 = face.bbox
        draw.rectangle((x1, y1, x2, y2), outline=color, width=stroke)
        label = f"#{index} {face.score:.2f}" + (" *" if chosen else "")
        draw.text((x1 + 2, max(0.0, y1 - 12)), label, fill=color)
        for point in face.landmarks:
            draw.ellipse(
                (point[0] - stroke, point[1] - stroke, point[0] + stroke, point[1] + stroke),
                fill=color,
            )
    return canvas


def detect_and_encode(
    image_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    engine: FaceEngine | None = None,
    face_index: int | None = None,
    select_largest: bool = False,
    minimum_quality: float = 0.30,
) -> FaceObservation:
    """Validate an image, detect and select one face, gate on quality, and embed it.

    The returned embedding is biometric data. It lives in memory and is excluded from
    ordinary serialization; it is never written to the evidence bundle or on-chain.
    """

    validated = load_validated_image(image_path)
    destination = Path(output_dir) if output_dir else ensure_output_dir()
    destination.mkdir(parents=True, exist_ok=True)
    # Saved for human review of exactly what the detector saw after EXIF/sRGB
    # normalization; the pipeline itself works from the in-memory array below.
    save_normalized_image(validated, destination / "normalized_input.jpg")

    rgb = np.asarray(validated.image, dtype=np.uint8)
    active = engine or get_engine()
    faces = active.detect(rgb)
    index = select_face(faces, face_index=face_index, select_largest=select_largest)
    face = faces[index]

    crop = align_face(rgb, face.landmarks)
    box, quality = assess(rgb, face, crop)

    crop_path = destination / "face_crop.jpg"
    Image.fromarray(crop).save(crop_path, format="JPEG", quality=95, optimize=True)

    # The search-facing crop. The 112x112 aligned chip above is what the recognition
    # model needs; it is the wrong input for a reverse-image engine, which indexes whole
    # pictures. See align.portrait_crop.
    portrait_path = destination / "portrait_crop.jpg"
    portrait_box = (box.x, box.y, box.x + box.width, box.y + box.height)
    Image.fromarray(portrait_crop(rgb, portrait_box)).save(
        portrait_path, format="JPEG", quality=92, optimize=True
    )
    annotate(rgb, faces, index).save(
        destination / "annotated_input.jpg", format="JPEG", quality=92, optimize=True
    )

    if quality.quality_score < minimum_quality:
        reasons = "; ".join(quality.warnings) or "insufficient face quality"
        raise FacePipelineError(
            PipelineErrorCode.LOW_QUALITY,
            f"quality {quality.quality_score:.3f} < {minimum_quality:.3f} ({reasons})",
        )

    embedding = active.embed_aligned(crop[None, ...])[0]

    return FaceObservation(
        source_image=str(validated.path),
        face_crop_path=str(crop_path),
        portrait_crop_path=str(portrait_path),
        detector_backend="scrfd_10g",
        model_name=active.model_id,
        detection_confidence=min(1.0, max(0.0, face.score)),
        bounding_box=box,
        quality=quality,
        embedding=[float(value) for value in embedding],
    )


def embed_all_faces(
    image: np.ndarray,
    *,
    engine: FaceEngine | None = None,
    multiscale: bool = True,
    min_face_px: int = 48,
) -> tuple[list[DetectedFace], np.ndarray, bool]:
    """Detect and embed *every* face in an image in one batched call.

    Candidate images routinely contain several people. Comparing only the largest face is
    the classic way to miss a true match in a group photo, so we embed them all.

    Returns ``(faces, embeddings, rescanned)``. ``rescanned`` records whether the small-face
    recovery path ran, so a decision that leaned on upscaling can be told apart from one
    that did not, in the UI and in the evidence.
    """

    active = engine or get_engine()
    source, faces = active.detect_for_recognition(
        image, min_face_px=min_face_px, multiscale=multiscale
    )
    if not faces:
        return [], np.zeros((0, EMBEDDING_DIM), dtype=np.float32), False
    # `source` may be an upscaled copy, and the coordinates belong to it.
    return faces, active.embed_faces(source, faces), source is not image


__all__ = [
    "CROP_SIZE",
    "EMBEDDING_DIM",
    "DetectedFace",
    "FaceEngine",
    "FaceEngineError",
    "FacePipelineError",
    "ImageInputError",
    "align_face",
    "annotate",
    "assess",
    "cosine_distance",
    "detect_and_encode",
    "embed_all_faces",
    "get_engine",
    "pose_offsets",
    "select_face",
]
