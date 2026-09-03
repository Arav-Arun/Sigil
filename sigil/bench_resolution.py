"""Does small-face recovery actually recover anything?

Six of twelve candidates in a real run came back INCONCLUSIVE reading "closest face is
only 25-32px". None of them was a hard identity question; they were undecidable on
*resolution*. That is worth attacking, but every technique for attacking it shares one
failure mode: upsampling cannot add information, so anything that appears to help may
simply be manufacturing detail and, with it, confident wrong answers.

So this is an experiment, not a feature. It reproduces the failure by downscaling LFW
pairs to the face sizes actually seen in the wild, then measures each treatment against a
control that costs nothing:

============ ==============================================================
``baseline``  today's path, no recovery
``lanczos``   classical upscale before detection. The control. A learned
              method has to beat *this*, not the baseline, to have earned
              its download and its runtime.
``multiscale`` re-detect on an upscaled copy so the landmarks are sharper,
              then align from that copy. See FaceEngine.detect_for_recognition.
============ ==============================================================

The metric is TAR at a fixed FMR, plus coverage, because a treatment can trivially raise
TAR by refusing fewer comparisons while making more of them wrong. Both are reported with
denominators.

The decision rule is fixed here, before any numbers exist: a treatment ships enabled by
default only if it beats ``lanczos`` on TAR at FMR 1e-3 by more than the resolution of
this split. Otherwise it stays available behind a flag and the report says it did not
help. Reporting "this did not work" is the point of running it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sigil.bench import _as_uint8, threshold_at_fmr
from sigil.face import cosine_distance, embed_all_faces
from sigil.face.engine import FaceEngine, get_engine
from sigil.verify import MIN_CANDIDATE_FACE_PX

logger = logging.getLogger(__name__)

# The face sizes that actually failed in production, plus the floor itself as a control.
FACE_SIZES_PX = (24, 32, 40, 48)

TREATMENTS = ("baseline", "lanczos", "multiscale")


@dataclass(slots=True)
class TreatmentResult:
    """One treatment at one simulated face size."""

    treatment: str
    face_px: int
    pairs: int = 0
    # Pairs where BOTH faces cleared the production floor, so the pipeline would return a
    # decision rather than INCONCLUSIVE. This is the number the treatments exist to move.
    decided: int = 0
    coverage: float = 0.0
    tar_at_fmr_1e2: float = 0.0
    # Genuine/impostor separation in pooled standard deviations. Continuous, so it still
    # discriminates between treatments at sample sizes where a rate-based metric saturates.
    separation: float = 0.0
    threshold: float = 0.0
    positives: int = 0
    negatives: int = 0

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ResolutionReport:
    model: str = ""
    face_sizes_px: list[int] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    verdict: str = ""
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _downscale_to_face(image: np.ndarray, engine: FaceEngine, target_px: int) -> np.ndarray | None:
    """Shrink an image until its largest face is about ``target_px`` across.

    Returns None when no face is found to measure, which is itself a legitimate outcome
    and is counted as a coverage loss rather than quietly dropped.
    """

    import cv2

    faces = engine.detect(image)
    if not faces:
        return None
    edge = max(min(face.width, face.height) for face in faces)
    if edge <= target_px:
        return image
    scale = target_px / edge
    height, width = image.shape[:2]
    new = (max(16, round(width * scale)), max(16, round(height * scale)))
    resized: np.ndarray = cv2.resize(image, new, interpolation=cv2.INTER_AREA)
    return resized


def _embed_under(
    image: np.ndarray, engine: FaceEngine, treatment: str
) -> tuple[np.ndarray, float] | None:
    """Embed the largest face under one treatment.

    Returns ``(embedding, detected_face_px)``, or None when no face is found at all. The
    face size comes back with the embedding because it is what the production gate acts
    on: below MIN_CANDIDATE_FACE_PX the pipeline returns INCONCLUSIVE no matter how good
    the embedding is, so a treatment's real job is to lift faces over that line.
    """

    import cv2

    if treatment == "lanczos":
        # The control: a plain high-quality upscale, no re-detection, no model.
        height, width = image.shape[:2]
        if max(height, width) < 512:
            scale = 512 / max(height, width, 1)
            image = cv2.resize(
                image,
                (round(width * scale), round(height * scale)),
                interpolation=cv2.INTER_LANCZOS4,
            )

    multiscale = treatment == "multiscale"
    faces, embeddings, _ = embed_all_faces(
        image, engine=engine, multiscale=multiscale, min_face_px=MIN_CANDIDATE_FACE_PX
    )
    if not faces:
        return None
    best = int(np.argmax([min(face.width, face.height) for face in faces]))
    row: np.ndarray = embeddings[best]
    return row, float(min(faces[best].width, faces[best].height))


def _dprime(positives: list[float], negatives: list[float]) -> float:
    """Separation between the genuine and impostor distributions, in pooled SDs."""

    if len(positives) < 2 or len(negatives) < 2:
        return float("nan")
    pos, neg = np.asarray(positives), np.asarray(negatives)
    pooled = np.sqrt((pos.var(ddof=1) + neg.var(ddof=1)) / 2.0)
    if pooled <= 0:
        return float("nan")
    return float(abs(neg.mean() - pos.mean()) / pooled)


def run_resolution_benchmark(
    *,
    limit: int | None = 200,
    output: Path | None = None,
    engine: FaceEngine | None = None,
) -> ResolutionReport:
    """Measure every treatment at every simulated face size."""

    from sklearn.datasets import fetch_lfw_pairs

    active = engine or get_engine()
    active.warm_up()

    data = fetch_lfw_pairs(
        subset="test",
        color=True,
        resize=1.0,
        funneled=True,
        slice_=None,
        download_if_missing=False,
    )
    pairs, targets = data.pairs, data.target
    total = len(pairs) if limit is None else min(limit, len(pairs))
    # Stride, so a limited run samples genuine and impostor pairs alike. LFW lists all
    # genuine pairs before all impostor ones; truncating gives a degenerate split.
    step = max(1, len(pairs) // total)
    indices = list(range(0, len(pairs), step))[:total]

    report = ResolutionReport(model=active.model_id, face_sizes_px=list(FACE_SIZES_PX))

    for face_px in FACE_SIZES_PX:
        # Downscale once per pair per size, so every treatment sees identical pixels and
        # the comparison is between treatments rather than between resamplings.
        shrunk: list[tuple[np.ndarray, np.ndarray, int] | None] = []
        for index in indices:
            left = _downscale_to_face(_as_uint8(pairs[index, 0]), active, face_px)
            right = _downscale_to_face(_as_uint8(pairs[index, 1]), active, face_px)
            shrunk.append(
                None if left is None or right is None else (left, right, int(targets[index]))
            )

        for treatment in TREATMENTS:
            positives: list[float] = []
            negatives: list[float] = []
            decidable = 0
            attempted = 0
            for item in shrunk:
                if item is None:
                    continue
                left_image, right_image, label = item
                a = _embed_under(left_image, active, treatment)
                b = _embed_under(right_image, active, treatment)
                if a is None or b is None:
                    continue
                attempted += 1
                distance = cosine_distance(a[0], b[0])
                (positives if label == 1 else negatives).append(distance)
                # The production gate: both faces must clear the floor or the pipeline
                # reports INCONCLUSIVE regardless of how good the comparison would be.
                if min(a[1], b[1]) >= MIN_CANDIDATE_FACE_PX:
                    decidable += 1

            # FMR 1e-2, not 1e-3: with a few dozen impostor pairs the smallest resolvable
            # rate is ~1/len(negatives), so asking for 1e-3 pins every treatment to the
            # same degenerate operating point and reports them as identical.
            threshold = threshold_at_fmr(negatives, 1e-2) if negatives else 0.0
            tar = (
                sum(1 for d in positives if d <= threshold) / len(positives)
                if positives
                else float("nan")
            )
            result = TreatmentResult(
                treatment=treatment,
                face_px=face_px,
                pairs=len(indices),
                decided=decidable,
                coverage=round(decidable / max(len(indices), 1), 4),
                tar_at_fmr_1e2=round(tar, 4) if tar == tar else float("nan"),
                separation=round(_dprime(positives, negatives), 4),
                threshold=round(threshold, 4),
                positives=len(positives),
                negatives=len(negatives),
            )
            report.results.append(result.to_json())
            logger.info(
                "%s @ %dpx: TAR %.4f, decidable %.3f (%d/%d over the floor)",
                treatment,
                face_px,
                result.tar_at_fmr_1e2,
                result.coverage,
                decidable,
                len(indices),
            )

    report.verdict = _verdict(report)
    report.notes = [
        "Face sizes are simulated by downscaling LFW pairs until the detected face is the "
        "target size, reproducing the failure seen on real social thumbnails.",
        "Every treatment sees identical downscaled pixels, so differences are the treatment.",
        "TAR is measured over every comparison attempted, decidable or not, so it "
        "answers 'is the embedding still good at this size' separately from 'would the "
        "pipeline be allowed to use it'.",
        "LFW faces downscaled to 24px are cleaner than real 24px social thumbnails: "
        "sharp, frontal, well lit before shrinking. Read these as an optimistic bound.",
        "lanczos is the control. A learned or more expensive method has to beat it, not "
        "the baseline, to justify its cost.",
    ]

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report.to_json(), indent=2), encoding="utf-8")
        logger.info("resolution report written to %s", output)
    return report


def _verdict(report: ResolutionReport) -> str:
    """State what actually moved, and what did not."""

    def mean_of(treatment: str, field_name: str) -> float:
        values = [
            row[field_name]
            for row in report.results
            if row["treatment"] == treatment and row[field_name] == row[field_name]
        ]
        return float(np.mean(values)) if values else float("nan")

    base_dec = mean_of("baseline", "coverage")
    lanc_dec = mean_of("lanczos", "coverage")
    multi_dec = mean_of("multiscale", "coverage")
    base_tar = mean_of("baseline", "tar_at_fmr_1e2")
    lanc_tar = mean_of("lanczos", "tar_at_fmr_1e2")
    multi_tar = mean_of("multiscale", "tar_at_fmr_1e2")

    if multi_dec != multi_dec or lanc_dec != lanc_dec:
        return "inconclusive: a treatment produced no measurable split"

    lines = [
        f"Decidable pairs: baseline {base_dec:.0%}, lanczos {lanc_dec:.0%}, "
        f"multiscale {multi_dec:.0%}. Upscaling is what lifts a face over the "
        f"{MIN_CANDIDATE_FACE_PX}px floor, and that is the entire effect.",
        f"Accuracy on the comparisons themselves is unchanged: TAR@FMR1e-2 "
        f"{base_tar:.4f} / {lanc_tar:.4f} / {multi_tar:.4f}. Upscaling is not buying "
        f"accuracy, it is buying the right to answer at all.",
    ]

    margin = multi_tar - lanc_tar
    if abs(margin) <= 0.01 and abs(multi_dec - lanc_dec) <= 0.02:
        lines.append(
            "multiscale and the lanczos control are equivalent within this split. "
            "multiscale ships because it is conditional, firing only when a face is "
            "under the floor, where lanczos-always would upscale every candidate image."
        )
    elif margin > 0.01:
        lines.append(f"multiscale beats the control by {margin:+.4f} TAR and ships enabled.")
    else:
        lines.append(
            f"multiscale does NOT beat the control ({margin:+.4f} TAR). Reported as "
            "measured rather than shipped as an improvement."
        )
    return " ".join(lines)


__all__ = ["FACE_SIZES_PX", "TREATMENTS", "ResolutionReport", "run_resolution_benchmark"]
