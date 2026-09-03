"""Canonical ArcFace alignment.

Recognition accuracy depends far more on alignment than on the embedding model. Every
crop we embed is warped into the same canonical 112x112 frame using the detector's five
landmarks, so the model always sees eyes, nose, and mouth in the positions it was
trained on.
"""

from __future__ import annotations

import numpy as np

# The reference five-point template that ArcFace/InsightFace recognition models were
# trained against, expressed for a 112x112 crop as (x, y) for
# (left eye, right eye, nose, left mouth corner, right mouth corner).
ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

CROP_SIZE = 112


def umeyama_similarity(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (rotation, uniform scale, translation).

    Implements Umeyama (1991). Returns the 2x3 affine matrix mapping ``source`` onto
    ``destination``. A similarity transform is the right model here: it corrects roll and
    scale without the shear a full affine fit would introduce, which would distort facial
    geometry and degrade the embedding.
    """

    source = np.asarray(source, dtype=np.float64)
    destination = np.asarray(destination, dtype=np.float64)
    if source.shape != destination.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("source and destination must both be (N, 2) point arrays")

    num_points = source.shape[0]
    source_mean = source.mean(axis=0)
    destination_mean = destination.mean(axis=0)
    source_centered = source - source_mean
    destination_centered = destination - destination_mean

    covariance = destination_centered.T @ source_centered / num_points
    unitary, singular_values, vt = np.linalg.svd(covariance)

    # Guard against a reflection: a mirrored "fit" would align landmarks numerically while
    # flipping the face, which is exactly the failure a naive SVD fit produces.
    correction = np.eye(2)
    if np.linalg.det(unitary) * np.linalg.det(vt) < 0:
        correction[1, 1] = -1

    rotation = unitary @ correction @ vt
    source_variance = source_centered.var(axis=0).sum()
    if source_variance <= 0:
        raise ValueError("degenerate landmark set: all points are identical")

    scale = float((singular_values * np.diag(correction)).sum() / source_variance)
    translation = destination_mean - scale * rotation @ source_mean

    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:, :2] = scale * rotation
    matrix[:, 2] = translation
    return matrix


def align_face(
    image: np.ndarray,
    landmarks: np.ndarray,
    *,
    size: int = CROP_SIZE,
) -> np.ndarray:
    """Warp a face into the canonical ArcFace frame using its five landmarks."""

    import cv2

    landmarks = np.asarray(landmarks, dtype=np.float32)
    if landmarks.shape != (5, 2):
        raise ValueError(f"expected 5 landmark points, got {landmarks.shape}")

    template = ARCFACE_TEMPLATE
    if size != CROP_SIZE:
        template = template * (size / CROP_SIZE)

    matrix = umeyama_similarity(landmarks, template)
    return cv2.warpAffine(
        image,
        matrix.astype(np.float32),
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderValue=0.0,
    )


__all__ = ["ARCFACE_TEMPLATE", "CROP_SIZE", "align_face", "umeyama_similarity"]


# Reverse-image engines index whole pictures, not 112x112 face chips. A tight aligned crop
# throws away hair, ears, neck and background, which is exactly the material those engines
# use, and in practice it returns "some other young man" rather than this person. A
# head-and-shoulders crop keeps the identity signal, drops the clothing and props that
# make a full photo match shopping listings instead of people, and is large enough to be
# indexed. The multiplier is deliberately generous on the vertical: hair above the crown
# and the shoulder line below both matter more than side background.
PORTRAIT_SCALE_X = 1.9
PORTRAIT_SCALE_UP = 1.5
PORTRAIT_SCALE_DOWN = 2.1
PORTRAIT_MIN_EDGE = 512


def portrait_crop(image: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    """Head-and-shoulders crop around a detected face box, clamped to the image."""

    from PIL import Image

    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    face_w = max(1.0, x2 - x1)
    face_h = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2.0

    left = round(max(0.0, cx - face_w * PORTRAIT_SCALE_X / 2.0))
    right = round(min(float(width), cx + face_w * PORTRAIT_SCALE_X / 2.0))
    top = round(max(0.0, y1 - face_h * (PORTRAIT_SCALE_UP - 1.0)))
    bottom = round(min(float(height), y2 + face_h * (PORTRAIT_SCALE_DOWN - 1.0)))

    if right - left < 2 or bottom - top < 2:
        return image

    crop = image[top:bottom, left:right]
    edge = min(crop.shape[0], crop.shape[1])
    if edge < PORTRAIT_MIN_EDGE:
        # Upscaling does not add detail, but small images are indexed poorly and some
        # providers reject them outright.
        scale = PORTRAIT_MIN_EDGE / edge
        target = (round(crop.shape[1] * scale), round(crop.shape[0] * scale))
        resample = Image.Resampling.LANCZOS
        crop = np.asarray(Image.fromarray(crop).resize(target, resample), dtype=np.uint8)
    return crop
