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

from sigil.face import FaceEngine, cosine_distance, get_engine

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


__all__ = ["BenchmarkReport", "LatencyStats", "run_benchmark", "threshold_at_fmr"]
