"""Accuracy and latency measurement.

Every accuracy claim Sigil makes has to come from here. The protocol matters as much as
the numbers:

* **Thresholds are calibrated on the LFW train and 10-fold splits, and reported on the
  disjoint test split.** Tuning and reporting on the same pairs inflates the result;
  that is the single most common way a face-recognition benchmark lies.
* **Raw counts and denominators are always reported**, never a bare percentage.
* **The operating point is chosen for this task's cost profile.** Falsely identifying a
  stranger on social media is much worse than abstaining, so the threshold targets a low
  false-match rate rather than a balanced error rate.
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sigil.face import FaceEngine, cosine_distance, embed_all_faces, get_engine

logger = logging.getLogger(__name__)

# Operating point. Falsely naming a stranger on social media is the worst outcome this
# pipeline can produce, so the match threshold is calibrated at a strict false-match
# rate rather than at a balanced error rate.
TARGET_FMR = 0.001
REPORTING_FMR = 0.01
STRICT_FMR = 0.001

# Positive-quantile used to place the upper edge of the uncertainty band: above this,
# a pair is confidently *not* the same person.
REJECT_POSITIVE_QUANTILE = 0.99


@dataclass(slots=True)
class LatencyStats:
    samples: int = 0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    mean_ms: float = 0.0

    @classmethod
    def of(cls, values_ms: list[float]) -> LatencyStats:
        if not values_ms:
            return cls()
        ordered = sorted(values_ms)
        return cls(
            samples=len(ordered),
            p50_ms=round(statistics.median(ordered), 2),
            p95_ms=round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 2),
            mean_ms=round(statistics.fmean(ordered), 2),
        )


# Face sizes the resolution sweep simulates, as fractions of the LFW image. Downscaling
# the whole image and re-detecting is what actually happens to a search-provider
# thumbnail, so it measures the deployed failure mode rather than a synthetic blur.
RESOLUTION_SCALES = (0.30, 0.40, 0.55, 0.75, 1.00)

# Bands the measured size penalty is quantised into. Three is enough to capture the
# shape and few enough that each one is backed by hundreds of pairs rather than tens.
SIZE_BANDS = (64, 96)


@dataclass(slots=True)
class ResolutionBand:
    """How the distance distribution moves when the candidate face is small."""

    scale: float = 0.0
    face_px_median: int = 0
    pairs: int = 0
    detection_failures: int = 0
    genuine_p97: float = 0.0
    impostor_min: float = 0.0
    impostor_fmr_1e2: float = 0.0
    # How much closer impostors get at this size than at full resolution. Positive means
    # the boundary has to move down by this much to hold the same false-match rate.
    impostor_shift: float = 0.0


@dataclass(slots=True)
class SamePhotoCalibration:
    """Separation between a re-encoded copy of one photo and a different photo.

    The pipeline has to tell "the image you uploaded, served back as a JPEG thumbnail"
    apart from "a different photograph of the same person". The second is the useful
    discovery; only the first is a repost. Both are near-duplicates to a human, so the
    thresholds that separate them are worth measuring rather than guessing.
    """

    same_pairs: int = 0
    different_pairs: int = 0
    same_mean_error_p99: float = 0.0
    different_mean_error_min: float = 0.0
    same_dhash_p99: float = 0.0
    different_dhash_min: float = 0.0
    chosen_mean_error: float = 0.0
    chosen_dhash: float = 0.0


@dataclass(slots=True)
class BenchmarkReport:
    """Everything needed to reproduce and audit an accuracy claim."""

    model: str = ""
    providers: list[str] = field(default_factory=list)
    dataset: str = "LFW (sklearn fetch_lfw_pairs, funneled, colour)"
    calibration_pairs: int = 0
    test_pairs: int = 0
    test_positive: int = 0
    test_negative: int = 0
    detection_failures: int = 0
    # Coverage is reported because a detection failure is not a neutral event: the pair
    # silently leaves the evaluation, so a pipeline that detects less scores better. Any
    # metric below is conditional on coverage and must be read alongside it.
    coverage: float = 0.0
    tar_at_fmr_1e3_with_failures: float = 0.0
    roc_auc: float = 0.0
    eer: float = 0.0
    match_threshold: float = 0.0
    reject_threshold: float = 0.0
    tar_at_fmr_1e2: float = 0.0
    tar_at_fmr_1e3: float = 0.0
    false_matches: int = 0
    false_non_matches: int = 0
    inconclusive: int = 0
    positive_mean: float = 0.0
    negative_mean: float = 0.0
    detect_latency: LatencyStats = field(default_factory=LatencyStats)
    embed_latency: LatencyStats = field(default_factory=LatencyStats)
    resolution: list[ResolutionBand] = field(default_factory=list)
    # Measured penalty to subtract from the match threshold, keyed by the smallest face
    # size the band covers. Consumed by sigil.verify, and pinned there by a test.
    size_penalties: dict[str, float] = field(default_factory=dict)
    same_photo: SamePhotoCalibration = field(default_factory=SamePhotoCalibration)
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _as_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return (image * 255.0).clip(0, 255).astype(np.uint8)


def _pair_distances(
    pairs: np.ndarray,
    targets: np.ndarray,
    engine: FaceEngine,
    *,
    detect_ms: list[float],
    embed_ms: list[float],
    limit: int | None = None,
) -> tuple[list[float], list[float], int]:
    """Return (positive distances, negative distances, detection failures)."""

    positives: list[float] = []
    negatives: list[float] = []
    failures = 0
    total = len(pairs) if limit is None else min(limit, len(pairs))
    # Stride rather than truncate. LFW lists every genuine pair before every impostor
    # pair, so `pairs[:limit]` returned 398 genuine and 0 impostor, and the report then
    # printed ROC-AUC 0.00000 as though it had been measured.
    step = max(1, len(pairs) // total)
    indices = list(range(0, len(pairs), step))[:total]

    for index in indices:
        embeddings: list[np.ndarray] = []
        for side in (0, 1):
            image = _as_uint8(pairs[index, side])
            start = time.perf_counter()
            faces = engine.detect(image)
            detect_ms.append((time.perf_counter() - start) * 1000)
            if not faces:
                embeddings = []
                break
            start = time.perf_counter()
            embeddings.append(engine.embed_faces(image, faces[:1])[0])
            embed_ms.append((time.perf_counter() - start) * 1000)

        if len(embeddings) != 2:
            failures += 1
            continue

        distance = cosine_distance(embeddings[0], embeddings[1])
        (positives if targets[index] == 1 else negatives).append(distance)

        if (index + 1) % 250 == 0:
            logger.info("benchmark: %d/%d pairs", index + 1, total)

    return positives, negatives, failures


def threshold_at_fmr(negatives: list[float], fmr: float) -> float:
    """Largest distance accepting at most ``fmr`` of impostor pairs.

    Distances below the threshold are accepted, so the operating point is the ``fmr``
    quantile of the impostor distribution.
    """

    if not negatives:
        return 0.0
    ordered = sorted(negatives)
    index = max(0, int(fmr * len(ordered)) - 1)
    return float(ordered[index])


def _tar_at(positives: list[float], threshold: float) -> float:
    if not positives:
        return 0.0
    return sum(1 for value in positives if value <= threshold) / len(positives)


def _roc_auc(positives: list[float], negatives: list[float]) -> float:
    """Mann-Whitney U estimate; lower distance means more likely genuine.

    NaN, not 0.0, when a class is missing. Zero is a legitimate value for this statistic
    and printing it for an unmeasurable split is the exact failure this project refuses
    everywhere else: an absence of evidence rendered as a result.
    """

    if not positives or not negatives:
        return float("nan")
    scores = [(-value, 1) for value in positives] + [(-value, 0) for value in negatives]
    scores.sort()
    rank_sum = 0.0
    index = 0
    rank = 1
    while index < len(scores):
        stop = index
        while stop + 1 < len(scores) and scores[stop + 1][0] == scores[index][0]:
            stop += 1
        average_rank = (rank + (rank + (stop - index))) / 2
        for position in range(index, stop + 1):
            if scores[position][1] == 1:
                rank_sum += average_rank
        rank += stop - index + 1
        index = stop + 1
    n_pos, n_neg = len(positives), len(negatives)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _eer(positives: list[float], negatives: list[float]) -> float:
    """Equal error rate, found by sweeping every observed distance. NaN when unmeasurable."""

    if not positives or not negatives:
        return float("nan")
    best_gap, best_value = float("inf"), 0.0
    for threshold in sorted(set(positives + negatives)):
        fnmr = sum(1 for value in positives if value > threshold) / len(positives)
        fmr = sum(1 for value in negatives if value <= threshold) / len(negatives)
        gap = abs(fnmr - fmr)
        if gap < best_gap:
            best_gap, best_value = gap, (fnmr + fmr) / 2
    return best_value


def measure_resolution(
    pairs: np.ndarray,
    targets: np.ndarray,
    engine: FaceEngine,
    *,
    limit: int = 250,
) -> list[ResolutionBand]:
    """Measure how face size moves the genuine and impostor distance distributions.

    The whole image is downscaled and re-detected, which is what a search-provider
    thumbnail actually is, and the production ``embed_all_faces`` path runs so the
    small-face recovery is included in the measurement rather than around it.

    This exists because the operating point was chosen on full-resolution pairs and then
    applied to 112px thumbnails, and a real run matched a stranger at 0.6888 on one.
    """

    from PIL import Image

    step = max(1, len(pairs) // limit)
    indices = list(range(0, len(pairs), step))[:limit]
    bands: list[ResolutionBand] = []

    for scale in RESOLUTION_SCALES:
        positives: list[float] = []
        negatives: list[float] = []
        face_px: list[int] = []
        failures = 0

        for index in indices:
            embeddings: list[np.ndarray] = []
            for side in (0, 1):
                image = _as_uint8(pairs[index, side])
                if scale < 1.0:
                    height, width = image.shape[:2]
                    small = Image.fromarray(image).resize(
                        (max(1, int(width * scale)), max(1, int(height * scale))),
                        Image.Resampling.LANCZOS,
                    )
                    image = np.asarray(small, dtype=np.uint8)
                faces, vectors, _ = embed_all_faces(image, engine=engine, min_face_px=0)
                if not faces:
                    break
                best = int(np.argmax([f.width * f.height for f in faces]))
                face_px.append(int(min(faces[best].width, faces[best].height)))
                embeddings.append(vectors[best])

            if len(embeddings) != 2:
                failures += 1
                continue
            distance = cosine_distance(embeddings[0], embeddings[1])
            (positives if targets[index] == 1 else negatives).append(distance)

        band = ResolutionBand(
            scale=scale,
            face_px_median=int(statistics.median(face_px)) if face_px else 0,
            pairs=len(positives) + len(negatives),
            detection_failures=failures,
            genuine_p97=round(float(np.quantile(positives, 0.97)), 5) if positives else 0.0,
            impostor_min=round(min(negatives), 5) if negatives else 0.0,
            impostor_fmr_1e2=round(threshold_at_fmr(negatives, REPORTING_FMR), 5)
            if negatives
            else 0.0,
        )
        bands.append(band)
        logger.info(
            "resolution %.2f: %d px median face, %d pairs, impostor min %.4f",
            scale,
            band.face_px_median,
            band.pairs,
            band.impostor_min,
        )

    # Full resolution is the reference the shipped threshold was chosen against.
    reference = next((b for b in bands if b.scale == 1.0), None)
    if reference:
        for band in bands:
            band.impostor_shift = round(max(0.0, reference.impostor_min - band.impostor_min), 5)
    return bands


# Below this, a measured impostor shift is indistinguishable from sampling noise: each
# resolution band holds only ~125 impostor pairs, and the minimum of 125 samples is a
# high-variance statistic. Reporting a penalty smaller than this would dress noise up as
# a calibrated constant, which is the exact failure this module exists to prevent.
PENALTY_NOISE_FLOOR = 0.01


def size_penalties(bands: list[ResolutionBand]) -> dict[str, float]:
    """Quantise the measured impostor shift into the bands sigil.verify applies.

    A penalty is the amount the match threshold must come *down* by at that face size to
    keep the same distance to the nearest impostor that full resolution has. Bands are
    keyed by their lower edge in pixels; anything above the largest key is unpenalised.

    Shifts below PENALTY_NOISE_FLOOR are reported as zero rather than as small numbers.
    """

    penalties: dict[str, float] = {}
    for edge in SIZE_BANDS:
        covered = [b for b in bands if b.face_px_median and b.face_px_median < edge]
        # The worst band at or below this size sets the penalty: a threshold must be safe
        # across the whole band, not on its average.
        penalties[str(edge)] = round(max((b.impostor_shift for b in covered), default=0.0), 3)
    # Monotonic: a smaller face can never be given a gentler penalty than a larger one.
    running = 0.0
    for edge in sorted((int(k) for k in penalties), reverse=True):
        running = max(running, penalties[str(edge)])
        penalties[str(edge)] = running if running >= PENALTY_NOISE_FLOOR else 0.0
    return penalties


def measure_same_photo(pairs: np.ndarray, *, limit: int = 300) -> SamePhotoCalibration:
    """Calibrate the repost test: one photo re-encoded, versus a different photograph.

    Positives are the same pixels after the transforms a search index actually applies,
    JPEG recompression and downscaling. Negatives are LFW's own pairs, so they include
    *the same person photographed twice*, which is the case the test must not swallow.
    """

    import io

    from PIL import Image

    from sigil.verify import photo_difference

    step = max(1, len(pairs) // limit)
    indices = list(range(0, len(pairs), step))[:limit]

    same_mean: list[float] = []
    same_dhash: list[float] = []
    diff_mean: list[float] = []
    diff_dhash: list[float] = []

    for index in indices:
        source = Image.fromarray(_as_uint8(pairs[index, 0])).convert("RGB")
        other = Image.fromarray(_as_uint8(pairs[index, 1])).convert("RGB")

        for quality, factor in ((30, 1.0), (60, 0.5), (85, 0.25), (95, 1.0)):
            buffer = io.BytesIO()
            variant = source
            if factor < 1.0:
                variant = source.resize(
                    (max(1, int(source.width * factor)), max(1, int(source.height * factor))),
                    Image.Resampling.LANCZOS,
                )
            variant.save(buffer, format="JPEG", quality=quality)
            with Image.open(io.BytesIO(buffer.getvalue())) as decoded:
                mean_error, dhash_error = photo_difference(source, decoded.convert("RGB"))
            same_mean.append(mean_error)
            same_dhash.append(dhash_error)

        mean_error, dhash_error = photo_difference(source, other)
        diff_mean.append(mean_error)
        diff_dhash.append(dhash_error)

    def q99(values: list[float]) -> float:
        return round(float(np.quantile(values, 0.99)), 4) if values else 0.0

    calibration = SamePhotoCalibration(
        same_pairs=len(same_mean),
        different_pairs=len(diff_mean),
        same_mean_error_p99=q99(same_mean),
        different_mean_error_min=round(min(diff_mean), 4) if diff_mean else 0.0,
        same_dhash_p99=q99(same_dhash),
        different_dhash_min=round(min(diff_dhash), 4) if diff_dhash else 0.0,
    )
    # Sit above the same-photo tail and below the closest different photo. When the two
    # overlap, prefer the same-photo side: calling a repost "a distinct photo" costs a
    # duplicate in the grid, while the reverse hides a genuine second photograph.
    calibration.chosen_mean_error = round(
        min(
            calibration.same_mean_error_p99 * 1.2,
            max(calibration.same_mean_error_p99, calibration.different_mean_error_min * 0.5),
        ),
        3,
    )
    calibration.chosen_dhash = round(
        min(
            calibration.same_dhash_p99 * 1.2,
            max(calibration.same_dhash_p99, calibration.different_dhash_min * 0.5),
        ),
        3,
    )
    return calibration


def run_quality_calibration(
    *, prefer_coreml: bool = True, output: Path | None = None
) -> dict[str, Any]:
    """Measure only the resolution sweep and the repost test, and merge them in.

    The accuracy figures in `docs/benchmark.json` come from a full LFW run that takes
    tens of minutes. These two sections do not, and re-running everything to refresh
    them would invite exactly the failure the full run guards against: a partial
    measurement replacing a complete one. So this merges into the committed report and
    refuses to invent the sections it did not measure.
    """

    try:
        from sklearn.datasets import fetch_lfw_pairs
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError("benchmarking needs scikit-learn: uv sync --extra bench") from exc

    engine = get_engine(prefer_coreml=prefer_coreml)
    engine.warm_up()
    split = fetch_lfw_pairs(subset="test", color=True, resize=1.0, funneled=True, slice_=None)

    bands = measure_resolution(split.pairs, split.target, engine)
    payload = {
        "resolution": [asdict(band) for band in bands],
        "size_penalties": size_penalties(bands),
        "same_photo": asdict(measure_same_photo(split.pairs)),
    }

    if output:
        if not output.is_file():
            raise RuntimeError(
                f"{output} does not exist; run a full `sigil benchmark` before calibrating"
            )
        existing = json.loads(output.read_text(encoding="utf-8"))
        existing.update(payload)
        output.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        logger.info("quality calibration merged into %s", output)
    return payload


def run_benchmark(
    *,
    limit: int | None = None,
    prefer_coreml: bool = True,
    output: Path | None = None,
) -> BenchmarkReport:
    """Calibrate on LFW train + 10_folds, then report on the held-out test split."""

    try:
        from sklearn.datasets import fetch_lfw_pairs
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError("benchmarking needs scikit-learn: uv sync --extra bench") from exc

    engine = get_engine(prefer_coreml=prefer_coreml)
    engine.warm_up()
    report = BenchmarkReport(model=engine.model_id, providers=engine.providers)

    detect_ms: list[float] = []
    embed_ms: list[float] = []

    # Calibration uses the train split plus the 10-fold set. More impostor pairs means
    # the low-FMR operating point is estimated from real data rather than extrapolated:
    # a threshold at FMR 1e-3 is meaningless with only a thousand negatives.
    cal_pos: list[float] = []
    cal_neg: list[float] = []
    cal_fail = 0
    for subset in ("train", "10_folds"):
        logger.info("loading LFW calibration split: %s", subset)
        try:
            split = fetch_lfw_pairs(
                subset=subset, color=True, resize=1.0, funneled=True, slice_=None
            )
        except Exception as exc:
            logger.warning("calibration subset %s unavailable: %s", subset, exc)
            continue
        pos, neg, fail = _pair_distances(
            split.pairs, split.target, engine, detect_ms=detect_ms, embed_ms=embed_ms, limit=limit
        )
        cal_pos.extend(pos)
        cal_neg.extend(neg)
        cal_fail += fail

    report.calibration_pairs = len(cal_pos) + len(cal_neg)
    if not cal_neg:
        raise RuntimeError("no calibration impostor pairs; cannot choose a threshold")

    # Thresholds come only from calibration data, never from the reported split.
    report.match_threshold = round(threshold_at_fmr(cal_neg, TARGET_FMR), 4)
    # The band's upper edge sits above almost every genuine pair, so anything beyond it
    # is confidently a different person.
    positive_edge = float(np.quantile(cal_pos, REJECT_POSITIVE_QUANTILE)) if cal_pos else 0.0
    report.reject_threshold = round(max(report.match_threshold + 0.05, positive_edge), 4)

    logger.info("loading LFW held-out test split")
    test = fetch_lfw_pairs(subset="test", color=True, resize=1.0, funneled=True, slice_=None)
    pos, neg, fail = _pair_distances(
        test.pairs, test.target, engine, detect_ms=detect_ms, embed_ms=embed_ms, limit=limit
    )

    report.test_pairs = len(pos) + len(neg)
    report.test_positive = len(pos)
    report.test_negative = len(neg)
    report.detection_failures = cal_fail + fail
    report.roc_auc = round(_roc_auc(pos, neg), 5)
    report.eer = round(_eer(pos, neg), 5)
    report.tar_at_fmr_1e2 = round(_tar_at(pos, threshold_at_fmr(neg, REPORTING_FMR)), 5)
    report.tar_at_fmr_1e3 = round(_tar_at(pos, threshold_at_fmr(neg, STRICT_FMR)), 5)
    report.positive_mean = round(float(np.mean(pos)) if pos else 0.0, 5)
    report.negative_mean = round(float(np.mean(neg)) if neg else 0.0, 5)

    # Apply the calibrated three-state rule to the held-out split.
    report.false_matches = sum(1 for value in neg if value <= report.match_threshold)
    report.false_non_matches = sum(1 for value in pos if value >= report.reject_threshold)
    report.inconclusive = sum(
        1 for value in pos + neg if report.match_threshold < value < report.reject_threshold
    )

    attempted = report.test_pairs + fail
    report.coverage = round(report.test_pairs / attempted, 5) if attempted else 0.0
    # Counting a detection failure as a failure to verify, which is what it is in
    # production: a face the pipeline never found is a match it can never make.
    report.tar_at_fmr_1e3_with_failures = (
        round(report.tar_at_fmr_1e3 * report.coverage, 5) if attempted else 0.0
    )

    report.detect_latency = LatencyStats.of(detect_ms)
    report.embed_latency = LatencyStats.of(embed_ms)

    logger.info("measuring the resolution sweep")
    report.resolution = measure_resolution(test.pairs, test.target, engine)
    report.size_penalties = size_penalties(report.resolution)
    logger.info("calibrating the repost test")
    report.same_photo = measure_same_photo(test.pairs)
    report.notes = [
        f"Thresholds calibrated on {report.calibration_pairs} pairs "
        f"(LFW train + 10_folds) at FMR<={TARGET_FMR}.",
        f"Metrics reported on {report.test_pairs} disjoint held-out test pairs.",
        f"False matches: {report.false_matches}/{report.test_negative} impostor pairs.",
        f"False non-matches: {report.false_non_matches}/{report.test_positive} genuine pairs.",
        f"Coverage {report.coverage:.4f}: {fail} of {attempted} test pairs had a face the "
        "detector could not find. Metrics above are conditional on the pairs it did find.",
        "The detection cascade lowers headline AUC because it stops silently discarding "
        "the hardest images. That is the metric becoming more honest, not the model "
        "getting worse; coverage rises and the pipeline answers more queries.",
        "Flip TTA follows the published ArcFace evaluation protocol, which averages a "
        "crop with its mirror. InsightFace's own get_feat() does not do this.",
        "LFW is a frontal, celebrity-photo benchmark; social-media media is harder.",
        "Latency is measured under whatever load the machine had at the time. Accuracy is "
        "not: a full re-run on a busy machine reproduced every accuracy figure in "
        "docs/benchmark.json exactly and moved only p50/p95. Quote latency from an idle run.",
    ]

    if output:
        # A truncated run must never replace the committed report. `--limit 300` produced
        # a calibration on 300 pairs, wrote it to docs/benchmark.json, and the shipped
        # thresholds silently stopped matching the measurement they claim to come from.
        # The pinning test caught it, which is what that test is for, but the write should
        # not have been possible: a partial measurement is not the shipped one.
        if limit is not None:
            report.notes.append(
                f"Report NOT written to {output}: this was a truncated run (--limit "
                f"{limit}). Only a full run may replace the committed benchmark."
            )
            logger.warning("limited run; %s left untouched", output)
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report.to_json(), indent=2), encoding="utf-8")
            logger.info("benchmark report written to %s", output)

    return report


__all__ = [
    "PENALTY_NOISE_FLOOR",
    "BenchmarkReport",
    "LatencyStats",
    "ResolutionBand",
    "SamePhotoCalibration",
    "measure_resolution",
    "measure_same_photo",
    "run_benchmark",
    "run_quality_calibration",
    "size_penalties",
    "threshold_at_fmr",
]
