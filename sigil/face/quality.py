"""Explainable face-quality measurement.

Quality gating exists so the pipeline can say *why* it refused, rather than returning a
confident answer derived from four blurry pixels. Every signal here is cheap, and every
one maps to a human-readable warning that ends up in the evidence bundle.
"""

from __future__ import annotations

import numpy as np

from sigil.face.engine import DetectedFace
from sigil.models import BoundingBox, FaceQuality

MIN_USEFUL_FACE_PX = 112
BLUR_FLOOR = 40.0
BLUR_REFERENCE = 220.0


def _laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the Laplacian, the standard cheap sharpness proxy."""

    try:
        import cv2

        # cv2.Laplacian has no float32 -> float64 kernel; feed it float64 explicitly.
        return float(cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F).var())
    except ImportError:  # pragma: no cover - opencv is a declared dependency
        kernel = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
        padded = np.pad(gray.astype(np.float64), 1, mode="edge")
        response = sum(
            kernel[i, j] * padded[i : i + gray.shape[0], j : j + gray.shape[1]]
            for i in range(3)
            for j in range(3)
        )
        return float(np.var(response))


def pose_offsets(landmarks: np.ndarray) -> tuple[float, float]:
    """Return (roll_ratio, yaw_ratio) proxies derived from the five landmarks.

    ``roll_ratio`` is the vertical eye offset relative to inter-ocular distance.
    ``yaw_ratio`` is how far the nose sits from the eye midpoint, again normalized by
    inter-ocular distance. Both are 0 for a frontal, level face. These are proxies, not
    calibrated Euler angles, and they are labelled as such in the evidence.
    """

    left_eye, right_eye, nose = landmarks[0], landmarks[1], landmarks[2]
    interocular = float(np.linalg.norm(right_eye - left_eye))
    if interocular < 1e-6:
        return 1.0, 1.0
    roll = abs(float(right_eye[1] - left_eye[1])) / interocular
    eye_center = (left_eye + right_eye) / 2.0
    yaw = abs(float(nose[0] - eye_center[0])) / interocular
    return roll, yaw


def assess(
    image: np.ndarray,
    face: DetectedFace,
    aligned_crop: np.ndarray,
) -> tuple[BoundingBox, FaceQuality]:
    """Measure one detected face and produce a bounded, explainable quality score."""

    height, width = image.shape[:2]
    x1, y1, x2, y2 = face.bbox
    box = BoundingBox(
        x=max(0, round(x1)),
        y=max(0, round(y1)),
        width=max(1, round(x2 - x1)),
        height=max(1, round(y2 - y1)),
    )

    gray = aligned_crop.astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    blur_score = _laplacian_variance(gray)
    brightness = float(np.clip(gray.mean() / 255.0, 0.0, 1.0))
    roll, yaw = pose_offsets(face.landmarks)

    face_size = min(box.width, box.height)
    border_truncated = (
        box.x <= 1
        or box.y <= 1
        or box.x + box.width >= width - 1
        or box.y + box.height >= height - 1
    )

    warnings: list[str] = []
    if face_size < MIN_USEFUL_FACE_PX:
        warnings.append(
            f"face is {face_size}px; below the {MIN_USEFUL_FACE_PX}px comfort threshold"
        )
    if blur_score < BLUR_FLOOR:
        warnings.append(f"low sharpness (Laplacian variance {blur_score:.1f})")
    if brightness < 0.18:
        warnings.append("underexposed")
    elif brightness > 0.88:
        warnings.append("overexposed")
    if roll > 0.25:
        warnings.append(f"head roll is high (eye offset ratio {roll:.2f})")
    if yaw > 0.30:
        warnings.append(f"off-frontal pose (nose offset ratio {yaw:.2f})")
    if border_truncated:
        warnings.append("face touches the image border and may be cropped")
    if face.score < 0.7:
        warnings.append(f"low detector confidence ({face.score:.2f})")

    size_score = min(face_size / 160.0, 1.0)
    sharpness_score = min(blur_score / BLUR_REFERENCE, 1.0)
    exposure_score = max(0.0, 1.0 - abs(brightness - 0.5) * 2.0)
    pose_score = max(0.0, 1.0 - (roll / 0.35) * 0.5 - (yaw / 0.45) * 0.5)

    quality_score = (
        0.30 * size_score + 0.30 * sharpness_score + 0.20 * exposure_score + 0.20 * pose_score
    )
    if border_truncated:
        quality_score *= 0.85
    quality_score = float(np.clip(quality_score, 0.0, 1.0))

    return box, FaceQuality(
        face_size_px=face_size,
        blur_score=round(blur_score, 4),
        brightness=round(brightness, 4),
        border_truncated=border_truncated,
        quality_score=round(quality_score, 4),
        warnings=warnings,
    )


__all__ = ["assess", "pose_offsets"]
