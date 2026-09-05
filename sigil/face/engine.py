"""SCRFD detection and ArcFace embedding on ONNX Runtime.

Sigil runs the InsightFace ``buffalo_l`` ONNX models directly rather than through a
wrapper framework. That buys three things that matter for this project:

* it works, the DeepFace/TensorFlow path cannot even import against TF 2.21;
* speed, sessions are built once and reused, candidate faces are embedded in a single
  batched ``session.run``, and CoreML is used where it is available on Apple silicon;
* explainability, the detector post-processing and the alignment are ours, so every
  number in the evidence bundle can be traced to code in this repository.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from sigil.face.align import CROP_SIZE, align_face

logger = logging.getLogger(__name__)

DEFAULT_PACK = "buffalo_l"
DETECTOR_FILE = "det_10g.onnx"
EMBEDDER_FILE = "w600k_r50.onnx"

# SCRFD det_10g emits three feature-map strides, each with two anchors per location.
SCRFD_STRIDES = (8, 16, 32)
SCRFD_ANCHORS_PER_LOCATION = 2
SCRFD_INPUT_SIZE = 640

# Detection-cascade bounds. A relaxed pass will not go below RETRY_MIN_THRESHOLD, and
# the upscaled pass is capped so a pathological image cannot blow up latency.
RETRY_MIN_THRESHOLD = 0.25
RETRY_MAX_DET_SIZE = 1280

# Multi-scale recovery for small faces.
#
# The cascade above escalates only when detection finds *nothing*. A 28px face is found,
# so it never fires for the case that dominates real candidate media: a run over social
# thumbnails returned six of twelve candidates as INCONCLUSIVE, every one of them for
# "closest face is only 25-32px", undecidable on resolution rather than on identity.
#
# Upscaling adds no information. What it adds is *landmark precision*: at 28px SCRFD's
# five points are quantised to a few pixels, the similarity transform built from them is
# correspondingly wrong, and ArcFace is far more sensitive to a misaligned crop than to a
# soft one. Re-detecting on an upscaled copy buys sub-pixel landmarks, and the alignment
# warp then samples a cleanly interpolated source instead of upsampling 4x by itself.
#
# The recovery path is covered by the face-resolution tests and bounded below.
UPSCALE_TARGET_FACE_PX = 112  # the ArcFace input edge; below it the warp only upsamples
MAX_RECOGNITION_UPSCALE = 4.0
MAX_UPSCALED_PIXELS = 12_000_000  # refuse to inflate an already-large image

EMBEDDING_DIM = 512


@dataclass(frozen=True, slots=True)
class DetectedFace:
    """One detected face in original-image pixel coordinates."""

    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    score: float
    landmarks: np.ndarray  # (5, 2)

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)


class FaceEngineError(RuntimeError):
    """Raised when models are unavailable or inference fails."""


def model_pack_dir(pack: str = DEFAULT_PACK) -> Path:
    """Locate the ONNX model pack, downloading it once if necessary."""

    override = os.environ.get("SIGIL_MODEL_DIR")
    if override:
        path = Path(override).expanduser()
        if not (path / DETECTOR_FILE).is_file():
            raise FaceEngineError(f"SIGIL_MODEL_DIR={path} does not contain {DETECTOR_FILE}")
        return path

    path = Path.home() / ".insightface" / "models" / pack
    if (path / DETECTOR_FILE).is_file() and (path / EMBEDDER_FILE).is_file():
        return path

    try:
        from insightface.utils import storage
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise FaceEngineError("insightface is not installed; run `uv sync --all-extras`") from exc

    logger.info("Downloading the %s model pack (~280 MB, one time)", pack)
    storage.ensure_available("models", pack)
    if not (path / DETECTOR_FILE).is_file():
        raise FaceEngineError(f"Model pack {pack} did not provide {DETECTOR_FILE}")
    return path


def _session_providers(prefer_coreml: bool) -> list[str | tuple[str, dict[str, str]]]:
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    providers: list[str | tuple[str, dict[str, str]]] = []
    if prefer_coreml and "CoreMLExecutionProvider" in available:
        # MLProgram is the modern CoreML format; ALL lets the Neural Engine take work it
        # supports and silently leaves the rest on CPU.
        providers.append(
            (
                "CoreMLExecutionProvider",
                {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"},
            )
        )
    providers.append("CPUExecutionProvider")
    return providers


@contextlib.contextmanager
def _quiet_stderr(enabled: bool = True) -> Iterator[None]:
    """Silence CoreML's C-level chatter.

    The CoreML execution provider writes E5RT partitioning notices straight to file
    descriptor 2, below Python's logging. They are benign, the embedder's batch
    dimension is dynamic, so those subgraphs stay on CPU, and CoreML output was verified
    numerically identical to CPU, but they would flood a screen recording.
    """

    if not enabled:
        yield
        return
    sys.stderr.flush()
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


def _build_session(path: Path, *, prefer_coreml: bool, threads: int, quiet: bool = True) -> object:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    with _quiet_stderr(quiet):
        return ort.InferenceSession(
            str(path), sess_options=options, providers=_session_providers(prefer_coreml)
        )


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    """Greedy non-maximum suppression over (N, 4) boxes."""

    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        best = int(order[0])
        keep.append(best)
        if order.size == 1:
            break
        rest = order[1:]
        inter_x1 = np.maximum(x1[best], x1[rest])
        inter_y1 = np.maximum(y1[best], y1[rest])
        inter_x2 = np.minimum(x2[best], x2[rest])
        inter_y2 = np.minimum(y2[best], y2[rest])
        inter = np.maximum(0.0, inter_x2 - inter_x1) * np.maximum(0.0, inter_y2 - inter_y1)
        iou = inter / np.maximum(areas[best] + areas[rest] - inter, 1e-9)
        order = rest[iou <= threshold]
    return keep


@lru_cache(maxsize=8)
def _anchor_centers(height: int, width: int, stride: int) -> np.ndarray:
    """Anchor centre coordinates for one SCRFD feature map, in input-image pixels.

    Cached because the geometry only depends on the letterboxed input size, which is
    constant across a run.
    """

    ys, xs = np.mgrid[:height, :width]
    centers = np.stack([xs, ys], axis=-1).astype(np.float32) * stride
    centers = centers.reshape(-1, 2)
    if SCRFD_ANCHORS_PER_LOCATION > 1:
        centers = np.repeat(centers, SCRFD_ANCHORS_PER_LOCATION, axis=0)
    return centers


class FaceEngine:
    """Warmed, thread-safe SCRFD + ArcFace inference."""

    def __init__(
        self,
        *,
        pack: str = DEFAULT_PACK,
        det_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        det_size: int = SCRFD_INPUT_SIZE,
        prefer_coreml: bool = True,
        threads: int = 0,
        quiet: bool = True,
        flip_tta: bool = True,
    ) -> None:
        self.pack = pack
        self.det_threshold = det_threshold
        self.nms_threshold = nms_threshold
        self.det_size = det_size
        self.flip_tta = flip_tta
        self._prefer_coreml = prefer_coreml
        self._quiet = quiet
        self._threads = threads or max(1, (os.cpu_count() or 4) - 2)
        self._lock = threading.Lock()
        self._detector: object | None = None
        self._embedder: object | None = None
        self._det_input_name = ""
        self._emb_input_name = ""

    # -- session management ------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._detector is not None and self._embedder is not None:
            return
        with self._lock:
            if self._detector is not None and self._embedder is not None:
                return
            directory = model_pack_dir(self.pack)
            # Measured on an Apple M4: CoreML gives SCRFD nothing (47.7 ms CPU vs 48.0 ms
            # CoreML, its ops fall back anyway) but makes ArcFace 3.5x faster
            # (33.8 ms -> 9.7 ms). So the detector stays on CPU and only the embedder is
            # accelerated. Outputs were verified identical across providers.
            detector = _build_session(
                directory / DETECTOR_FILE,
                prefer_coreml=False,
                threads=self._threads,
                quiet=self._quiet,
            )
            embedder = _build_session(
                directory / EMBEDDER_FILE,
                prefer_coreml=self._prefer_coreml,
                threads=self._threads,
                quiet=self._quiet,
            )
            self._det_input_name = detector.get_inputs()[0].name  # type: ignore[attr-defined]
            self._emb_input_name = embedder.get_inputs()[0].name  # type: ignore[attr-defined]
            self._detector = detector
            self._embedder = embedder

    def warm_up(self) -> None:
        """Run one inference of each model so demo timings are not polluted by lazy init."""

        self._ensure_loaded()
        blank = np.zeros((self.det_size, self.det_size, 3), dtype=np.uint8)
        with _quiet_stderr(self._quiet):
            self.detect(blank)
            # Warm both the single and batched shapes so neither pays compilation cost
            # inside a measured stage.
            self.embed_aligned(np.zeros((1, CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8))
            self.embed_aligned(np.zeros((4, CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8))

    @property
    def providers(self) -> list[str]:
        """Providers actually in use, reported per model since they now differ."""

        self._ensure_loaded()
        detector = self._detector.get_providers()[0]  # type: ignore[union-attr]
        embedder = self._embedder.get_providers()[0]  # type: ignore[union-attr]
        return [f"detector:{detector}", f"embedder:{embedder}"]

    @property
    def model_id(self) -> str:
        """Stable identifier recorded in evidence so a result is tied to a model."""

        suffix = "+fliptta" if self.flip_tta else ""
        return f"insightface/{self.pack}:scrfd_10g+arcface_w600k_r50{suffix}"

    # -- detection ---------------------------------------------------------------

    def _letterbox(self, image: np.ndarray, det_size: int) -> tuple[np.ndarray, float]:
        """Resize preserving aspect ratio and pad to a square det_size canvas."""

        import cv2

        height, width = image.shape[:2]
        scale = min(det_size / max(height, 1), det_size / max(width, 1))
        new_width, new_height = round(width * scale), round(height * scale)
        interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
        canvas = np.zeros((det_size, det_size, 3), dtype=image.dtype)
        canvas[:new_height, :new_width] = resized
        return canvas, scale

    def detect(self, image: np.ndarray, *, retry: bool = True) -> list[DetectedFace]:
        """Detect every face, with a fallback cascade for hard images.

        The InsightFace reference runs a single pass at a fixed threshold and returns
        nothing when that pass fails. On a benchmark sweep that silently discarded 68 of
        ~9,100 LFW pairs, and in this pipeline a missed detection on a candidate image
        is a match the system will never even consider.

        So a failed pass escalates: first a lower score threshold, then a larger input
        canvas for images whose faces are small relative to the detector's receptive
        field. Escalation only runs when the cheap pass found nothing, so the common
        case pays nothing for it.
        """

        self._ensure_loaded()
        if image.ndim != 3 or image.shape[2] != 3:
            raise FaceEngineError("detect() expects an (H, W, 3) RGB array")

        faces = self._detect_once(image, self.det_threshold, self.det_size)
        if faces or not retry:
            return faces

        relaxed = max(RETRY_MIN_THRESHOLD, self.det_threshold * 0.5)
        faces = self._detect_once(image, relaxed, self.det_size)
        if faces:
            logger.debug("detection recovered at threshold %.2f", relaxed)
            return faces

        # A small or distant face can fall below the smallest anchor stride. Upscaling
        # the canvas gives it more pixels to land on.
        larger = self.det_size * 2
        if larger <= RETRY_MAX_DET_SIZE:
            faces = self._detect_once(image, relaxed, larger)
            if faces:
                logger.debug("detection recovered at det_size %d", larger)
        return faces

    def detect_for_recognition(
        self,
        image: np.ndarray,
        *,
        min_face_px: int = 48,
        multiscale: bool = True,
    ) -> tuple[np.ndarray, list[DetectedFace]]:
        """Detect, and when the best face is too small, re-detect on an upscaled copy.

        Returns ``(source_image, faces)``. The caller must align and crop from the
        *returned* image, not the one it passed in, because the coordinates belong to it.

        The image is returned rather than the scale factor on purpose: handing back a
        factor invites a caller to forget to apply it and silently crop the wrong region.
        """

        faces = self.detect(image)
        if not faces or not multiscale:
            return image, faces

        best_edge = max(min(face.width, face.height) for face in faces)
        if best_edge >= min_face_px:
            return image, faces

        scale = min(MAX_RECOGNITION_UPSCALE, UPSCALE_TARGET_FACE_PX / max(best_edge, 1.0))
        height, width = image.shape[:2]
        if scale <= 1.01 or height * width * scale * scale > MAX_UPSCALED_PIXELS:
            return image, faces

        import cv2

        enlarged = cv2.resize(
            image,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_LANCZOS4,
        )
        rescanned = self.detect(enlarged)
        if not rescanned:
            # Upscaling lost the detection entirely. The original result is still valid.
            return image, faces

        logger.debug(
            "multiscale: %.0fpx face rescanned at %.2fx, now %.0fpx",
            best_edge,
            scale,
            max(min(f.width, f.height) for f in rescanned),
        )
        return enlarged, rescanned

    def _detect_once(
        self, image: np.ndarray, threshold: float, det_size: int
    ) -> list[DetectedFace]:
        """One detector pass at a given threshold and input size."""

        canvas, scale = self._letterbox(image, det_size)
        # InsightFace feeds SCRFD through blobFromImage(..., swapRB=True) from a BGR
        # source, so the network actually sees RGB. We already work in RGB, so no swap.
        blob = canvas.astype(np.float32)
        blob = (blob - 127.5) / 128.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]

        outputs = self._detector.run(None, {self._det_input_name: blob})  # type: ignore[union-attr]
        # det_10g emits scores, bbox deltas, and kps deltas grouped by stride.
        group = len(SCRFD_STRIDES)
        scores_out = outputs[0:group]
        bbox_out = outputs[group : group * 2]
        kps_out = outputs[group * 2 : group * 3]

        boxes: list[np.ndarray] = []
        scores: list[np.ndarray] = []
        points: list[np.ndarray] = []

        for index, stride in enumerate(SCRFD_STRIDES):
            stride_scores = scores_out[index].reshape(-1)
            keep = np.nonzero(stride_scores >= threshold)[0]
            if keep.size == 0:
                continue

            side = det_size // stride
            centers = _anchor_centers(side, side, stride)[keep]
            stride_scores = stride_scores[keep]

            # SCRFD regresses distances from the anchor centre to each box edge, in
            # stride units. Multiply back into pixels, then convert to corners.
            deltas = bbox_out[index].reshape(-1, 4)[keep] * stride
            box = np.stack(
                [
                    centers[:, 0] - deltas[:, 0],
                    centers[:, 1] - deltas[:, 1],
                    centers[:, 0] + deltas[:, 2],
                    centers[:, 1] + deltas[:, 3],
                ],
                axis=-1,
            )

            kps_deltas = kps_out[index].reshape(-1, 5, 2)[keep] * stride
            kps = kps_deltas + centers[:, None, :]

            boxes.append(box)
            scores.append(stride_scores)
            points.append(kps)

        if not boxes:
            return []

        all_boxes = np.concatenate(boxes, axis=0)
        all_scores = np.concatenate(scores, axis=0)
        all_points = np.concatenate(points, axis=0)

        keep_indices = _nms(all_boxes, all_scores, self.nms_threshold)

        height, width = image.shape[:2]
        faces: list[DetectedFace] = []
        for i in keep_indices:
            box = all_boxes[i] / scale
            kps = all_points[i] / scale
            x1 = float(np.clip(box[0], 0, width))
            y1 = float(np.clip(box[1], 0, height))
            x2 = float(np.clip(box[2], 0, width))
            y2 = float(np.clip(box[3], 0, height))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            faces.append(
                DetectedFace(
                    bbox=(x1, y1, x2, y2),
                    score=float(all_scores[i]),
                    landmarks=kps.astype(np.float32),
                )
            )

        faces.sort(key=lambda face: face.area, reverse=True)
        return faces

    # -- embedding ---------------------------------------------------------------

    def _forward_embed(self, crops: np.ndarray) -> np.ndarray:
        """One embedder pass over a batch of aligned crops, without normalization."""

        # Same convention as the detector: ArcFace w600k expects RGB input.
        blob = crops.astype(np.float32)
        blob = (blob - 127.5) / 127.5
        blob = np.transpose(blob, (0, 3, 1, 2))

        with _quiet_stderr(self._quiet):
            raw = self._embedder.run(None, {self._emb_input_name: blob})[0]  # type: ignore[union-attr]
        embeddings = np.asarray(raw, dtype=np.float32).reshape(len(crops), -1)
        if embeddings.shape[1] != EMBEDDING_DIM:
            raise FaceEngineError(f"unexpected embedding width {embeddings.shape[1]}")
        return embeddings

    def embed_aligned(self, crops: np.ndarray, *, flip_tta: bool | None = None) -> np.ndarray:
        """Embed aligned 112x112 RGB crops into L2-normalized vectors.

        Batching matters: verifying ten candidates costs one ``session.run`` rather than
        ten, which is most of the difference between a 4-second and a 40-second demo.

        **Flip test-time augmentation.** The published ArcFace evaluation protocol sums
        the embedding of a crop with the embedding of its horizontal mirror before
        normalizing; the numbers quoted for these weights come from that protocol.
        InsightFace's own ``ArcFaceONNX.get_feat`` does *not* do it, it is a single
        forward pass, so following the paper here is a genuine improvement over the
        reference implementation, at the cost of a second batched pass.

        A face is close to symmetric, so the mirrored view is a real second observation
        of the same identity rather than noise; averaging the two suppresses
        pose- and lighting-specific components of the embedding.
        """

        self._ensure_loaded()
        crops = np.asarray(crops)
        if crops.ndim == 3:
            crops = crops[None, ...]
        if crops.ndim != 4 or crops.shape[1:] != (CROP_SIZE, CROP_SIZE, 3):
            raise FaceEngineError(f"expected (N, {CROP_SIZE}, {CROP_SIZE}, 3) crops")

        use_flip = self.flip_tta if flip_tta is None else flip_tta
        embeddings = self._forward_embed(crops)
        if use_flip:
            # Axis 2 is width: reversing it mirrors each crop left-to-right.
            embeddings = embeddings + self._forward_embed(crops[:, :, ::-1, :])

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized: np.ndarray = embeddings / np.maximum(norms, 1e-10)
        return normalized

    def embed_faces(self, image: np.ndarray, faces: list[DetectedFace]) -> np.ndarray:
        """Align and embed the given faces from one image in a single batch."""

        if not faces:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        crops = np.stack([align_face(image, face.landmarks) for face in faces])
        return self.embed_aligned(crops)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance in [0, 2] between two L2-normalized embeddings."""

    return float(1.0 - np.dot(np.asarray(a).ravel(), np.asarray(b).ravel()))


@lru_cache(maxsize=4)
def get_engine(
    pack: str = DEFAULT_PACK,
    det_threshold: float = 0.5,
    det_size: int = SCRFD_INPUT_SIZE,
    prefer_coreml: bool = True,
    flip_tta: bool = True,
) -> FaceEngine:
    """Process-wide cached engine so models load exactly once."""

    return FaceEngine(
        pack=pack,
        det_threshold=det_threshold,
        det_size=det_size,
        prefer_coreml=prefer_coreml,
        flip_tta=flip_tta,
    )


__all__ = [
    "EMBEDDING_DIM",
    "RETRY_MAX_DET_SIZE",
    "RETRY_MIN_THRESHOLD",
    "DetectedFace",
    "FaceEngine",
    "FaceEngineError",
    "cosine_distance",
    "get_engine",
    "model_pack_dir",
]
