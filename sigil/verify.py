"""Face verification and candidate ranking.

Two decisions live here, and keeping them apart is the point:

* **Identity** is decided only by the calibrated distance between the source embedding
  and the best-matching face in the candidate media. Nothing else can grant it, not a
  high search rank, not an exact-image flag, not the number of routes that agreed.
* **Ranking** orders the candidates that already passed the identity gate, so a judge
  sees the strongest verified match first.

A third state, ``INCONCLUSIVE``, sits between match and non-match. Any distance inside
the uncertainty band, and any candidate whose media was too poor to trust, lands there.
For this task, wrongly naming a person is far worse than admitting uncertainty.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from sigil.candidates import FetchedMedia
from sigil.face import FaceEngine, cosine_distance, embed_all_faces
from sigil.models import DecisionStatus, SearchCandidate, VerificationDecision

logger = logging.getLogger(__name__)

# These are measured, and then deliberately tightened. Both halves matter.
#
# `sigil benchmark` calibrates at FMR 1e-3 on LFW and lands on 0.795. That number is
# correct for LFW and wrong for this pipeline, because the two use different impostors:
#
#   LFW impostors        random pairs of different people. Measured on the held-out
#                        split: genuine p95 0.4867, closest impostor 0.7924. Choosing
#                        0.795 puts the boundary ON the impostor edge, with no margin.
#   Sigil's impostors    candidates a reverse-image search surfaced *because they look
#                        like the query*. Hard negatives, by construction.
#
# Calibrating against random impostors and then deploying against hand-picked lookalikes
# is not a conservative mistake, it is the optimistic direction. It showed up exactly as
# the theory predicts: a real run matched a stranger at 0.6888, comfortably inside the
# old accept region, on a 112px thumbnail.
#
# So the operating point moves to just above the genuine 97th percentile (0.5296) rather
# than to the impostor edge. Measured cost on LFW: genuine pairs kept falls 98.6% -> 97.2%,
# about 1.4 points of recall. Measured benefit: 0.24 of margin against the nearest LFW
# impostor, and the 0.55-0.75 overlap band, where genuine and impostor distances really do
# mix, becomes INCONCLUSIVE instead of a confident wrong name.
#
# Trading 1.4 points of recall to stop naming the wrong person is the whole thesis of the
# three-state gate. The old values had that backwards.
DEFAULT_MATCH_THRESHOLD = 0.55
DEFAULT_REJECT_THRESHOLD = 0.75

# A face this small in a candidate image carries too little signal to be decisive.
MIN_CANDIDATE_FACE_PX = 48

# Measured size penalty, from `size_penalties` in docs/benchmark.json. Each entry is
# (upper edge in pixels, amount subtracted from the match threshold below that edge).
#
# Both are zero, and that is a measured result rather than an unfinished one.
#
# The hypothesis was that a small face should face a tighter bar, because the one
# production false match on record, a stranger at 0.6888, was on a 112px thumbnail. The
# resolution sweep in `sigil benchmark` tests it directly: LFW images are downscaled and
# re-detected, which is what a search-provider thumbnail actually is. It does not hold.
#
#   face px    genuine p97    impostor min
#        29         0.5220          0.8620
#        39         0.4923          0.8495
#        54         0.4695          0.8448
#        73         0.4699          0.8367
#        98         0.4669          0.8387
#
# Impostors do not get closer as the face shrinks; if anything they drift further, and
# the variation across bands is inside the noise of ~125 impostor pairs each. What
# degrades is the *genuine* side: p97 climbs 0.467 -> 0.522. Low resolution costs recall,
# not precision. A size penalty would therefore tighten a bar that is not the problem,
# and would spend recall exactly where recall is already worst.
#
# The honest caveat: LFW impostors are random strangers. A reverse image search returns
# look-alikes on purpose, and that population cannot be sampled from LFW at all, so this
# does not clear the hard-negative case that produced the 0.6888 match. It rules out the
# simpler explanation, not the harder one.
#
# What the sweep does support is MIN_CANDIDATE_FACE_PX below. At 39px the genuine p97 is
# 0.4923 and at 29px it is 0.5220, closing on the 0.55 operating point; under roughly
# 48px the genuine distribution starts colliding with the threshold. The gate was already
# there and is now measured rather than assumed.
#
# The mechanism stays because it is where a hard-negative measurement would land.
SIZE_PENALTIES: tuple[tuple[int, float], ...] = ((64, 0.0), (96, 0.0))

# Measured repost bounds, from `same_photo` in docs/benchmark.json.
#
# Calibrated by re-encoding LFW images the way a search index does, JPEG quality 30-95
# and downscales to 1/4, and comparing against LFW's own pairs, which include the same
# person photographed twice, the case this test must NOT swallow.
#
#   same photo, 99th pct   mean error  5.61   dhash 0.0586
#   different photo, min   mean error 24.00   dhash 0.2383
#
# The separation is over 3x on both signals. The previously shipped dhash bound of 0.04
# sat *below* the same-photo 99th percentile, so it was quietly failing to recognise
# genuine reposts and showing them as independent discoveries, which is the direction
# that overstates a result.
SAME_PHOTO_MEAN_ERROR = 6.74
SAME_PHOTO_DHASH_ERROR = 0.07
# Aspect ratios have to agree before a pixel comparison means anything.
SAME_PHOTO_ASPECT_TOLERANCE = 0.03


def match_threshold_for(face_px: int, base: float = DEFAULT_MATCH_THRESHOLD) -> float:
    """Tighten the match threshold for small faces, by the measured amount.

    Returns the threshold a candidate with a face this size has to clear. The band edges
    and penalties are measured, not chosen; see SIZE_PENALTIES.
    """

    penalty = max(
        (value for edge, value in SIZE_PENALTIES if face_px < edge),
        default=0.0,
    )
    return round(base - penalty, 4)


@dataclass(slots=True)
class CandidateVerification:
    """The full, explainable outcome for one candidate."""

    candidate: SearchCandidate
    media: FetchedMedia
    decision: VerificationDecision
    faces_detected: int = 0
    best_face_index: int | None = None
    best_face_px: int = 0
    all_distances: tuple[float, ...] = ()
    # True when the small-face recovery path ran for this candidate. Recorded so a
    # decision that leaned on upscaling can be told apart from one that did not.
    rescanned: bool = False
    # Whether the candidate is the query photograph itself (possibly re-encoded). Kept
    # separate from the face decision: the same face in a different photograph is the
    # useful discovery, while rediscovering the uploaded pixels is only provenance.
    same_photo: bool = False

    @property
    def matched(self) -> bool:
        return self.decision.status is DecisionStatus.MATCH

    @property
    def margin(self) -> float:
        """How far below the match threshold the distance sits. Higher is stronger."""

        if self.decision.distance is None:
            return 0.0
        return self.decision.threshold - self.decision.distance

    def to_json(self) -> dict[str, Any]:
        return {
            "source_url": str(self.candidate.source_url),
            "platform": self.candidate.platform,
            "is_social": self.candidate.is_social,
            "favicon_url": str(self.candidate.favicon_url) if self.candidate.favicon_url else "",
            "post_id": self.candidate.post_id,
            "search_routes": list(self.candidate.search_routes),
            "media": self.media.to_json(),
            "faces_detected": self.faces_detected,
            "best_face_index": self.best_face_index,
            "best_face_px": self.best_face_px,
            "all_distances": [round(d, 6) for d in self.all_distances],
            "recognition_path": "multiscale" if self.rescanned else "plain",
            "title": self.candidate.title,
            "exact_match": self.candidate.exact_match,
            "same_photo": self.same_photo,
            "decision": self.decision.model_dump(mode="json"),
            "margin": round(self.margin, 6),
        }


def decode_media(media: FetchedMedia) -> np.ndarray | None:
    """Decode fetched bytes into an RGB array, correcting EXIF orientation."""

    try:
        with Image.open(io.BytesIO(media.data)) as image:
            normalized = ImageOps.exif_transpose(image).convert("RGB")
            normalized.load()
            return np.asarray(normalized, dtype=np.uint8)
    except Exception as exc:  # Pillow raises a wide family of decode errors
        logger.debug("could not decode media for %s: %s", media.candidate.source_url, exc)
        return None


def is_same_photo(query_path: str | Path, media: FetchedMedia) -> bool:
    """Recognise the uploaded photograph after ordinary resizing or recompression.

    Byte hashes alone miss the common case where a search engine returns the same pixels
    as a JPEG or thumbnail. The full-image comparison below combines a small colour error
    with a difference-hash check. It does not participate in face identity; it only lets
    the UI and ranking distinguish a source-image rediscovery from another photograph of
    the verified person. This is deliberately a media comparison, not a search-provider
    label: an ``exact_matches`` response says why a URL was retrieved, but does not prove
    which image the URL or its thumbnail eventually served.
    """

    if not media.ok:
        return False
    try:
        with Image.open(query_path) as source_image:
            source = ImageOps.exif_transpose(source_image).convert("RGB")
            source.load()
        with Image.open(io.BytesIO(media.data)) as candidate_image:
            candidate = ImageOps.exif_transpose(candidate_image).convert("RGB")
            candidate.load()
    except (OSError, ValueError):
        return False

    source_ratio = source.width / source.height
    candidate_ratio = candidate.width / candidate.height
    if abs(source_ratio - candidate_ratio) > SAME_PHOTO_ASPECT_TOLERANCE:
        return False

    mean_error, dhash_error = photo_difference(source, candidate)
    return mean_error <= SAME_PHOTO_MEAN_ERROR and dhash_error <= SAME_PHOTO_DHASH_ERROR


def photo_difference(source: Image.Image, candidate: Image.Image) -> tuple[float, float]:
    """How far apart two images are: ``(mean colour error, difference-hash error)``.

    Split out so `sigil benchmark` calibrates the thresholds against the exact function
    the pipeline runs. Two signals rather than one because they fail differently: the
    colour error catches a recompression that preserves structure, and the difference
    hash catches a crop or a colour shift that preserves the histogram.
    """

    size = (64, 64)
    source_small = np.asarray(source.resize(size, Image.Resampling.LANCZOS), dtype=np.int16)
    candidate_small = np.asarray(candidate.resize(size, Image.Resampling.LANCZOS), dtype=np.int16)
    mean_error = float(np.abs(source_small - candidate_small).mean())

    def difference_hash(image: Image.Image) -> np.ndarray:
        gray = image.convert("L").resize((17, 16), Image.Resampling.LANCZOS)
        pixels = np.asarray(gray, dtype=np.int16)
        return pixels[:, 1:] > pixels[:, :-1]

    dhash_error = float(np.not_equal(difference_hash(source), difference_hash(candidate)).mean())
    return mean_error, dhash_error


def classify_same_photo(query_path: str | Path, verification: CandidateVerification) -> bool:
    """Return whether a verified candidate reuses the submitted image.

    ``SearchCandidate.exact_match`` is intentionally absent from this decision. It is a
    useful discovery hint from Google Lens, but it is not evidence that the downloaded
    media is the submitted image. Keeping those concepts separate prevents a whole
    provider bucket from being displayed as source-image reposts.
    """

    return verification.matched and is_same_photo(query_path, verification.media)


def verify_candidate(
    source_embedding: np.ndarray,
    media: FetchedMedia,
    *,
    engine: FaceEngine,
    model_name: str,
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    reject_threshold: float = DEFAULT_REJECT_THRESHOLD,
) -> CandidateVerification:
    """Compare the source face against *every* face in one candidate image.

    Comparing only the largest face is the standard way to miss a true match in a group
    photo, so all detected faces are embedded in a single batch and the best one wins.
    """

    candidate = media.candidate

    def inconclusive(reason: str, **extra: Any) -> CandidateVerification:
        return CandidateVerification(
            candidate=candidate,
            media=media,
            decision=VerificationDecision(
                status=DecisionStatus.INCONCLUSIVE,
                distance=None,
                threshold=match_threshold,
                model_name=model_name,
                detector_backend="scrfd_10g",
                reason=reason,
            ),
            **extra,
        )

    if not media.ok:
        return inconclusive(f"media unavailable: {media.error}")

    image = decode_media(media)
    if image is None:
        return inconclusive("candidate media could not be decoded as an image")

    faces, embeddings, rescanned = embed_all_faces(
        image, engine=engine, min_face_px=MIN_CANDIDATE_FACE_PX
    )
    if not faces:
        return inconclusive("no face detected in the candidate media")

    distances = [cosine_distance(source_embedding, row) for row in embeddings]
    best_index = int(np.argmin(distances))
    best_distance = float(distances[best_index])
    best_face = faces[best_index]
    best_px = int(min(best_face.width, best_face.height))

    # Shared by every return path below. Typed explicitly so the per-field
    # CandidateVerification construction stays clear to the type checker.
    common: dict[str, Any] = {
        "faces_detected": len(faces),
        "best_face_index": best_index,
        "best_face_px": best_px,
        "all_distances": tuple(round(d, 6) for d in distances),
        "rescanned": rescanned,
    }

    if best_px < MIN_CANDIDATE_FACE_PX:
        return inconclusive(
            f"closest face is only {best_px}px; below the {MIN_CANDIDATE_FACE_PX}px "
            "threshold for a decisive comparison",
            **common,
        )

    # A small face gets a proportionally harder bar, by the measured amount.
    effective_threshold = match_threshold_for(best_px, match_threshold)

    if best_distance <= effective_threshold:
        status = DecisionStatus.MATCH
        tightened = (
            f", tightened from {match_threshold:.2f} for a {best_px}px face"
            if effective_threshold < match_threshold
            else ""
        )
        reason = (
            f"cosine distance {best_distance:.4f} <= {effective_threshold:.2f}{tightened} "
            f"against face #{best_index} of {len(faces)} ({best_px}px, "
            f"{media.quality.lower()} media)"
        )
    elif best_distance >= reject_threshold:
        status = DecisionStatus.NON_MATCH
        reason = (
            f"cosine distance {best_distance:.4f} >= {reject_threshold:.2f}; "
            f"closest of {len(faces)} face(s) is not the same person"
        )
    else:
        # Inside the uncertainty band. Report it rather than resolving it.
        return inconclusive(
            f"cosine distance {best_distance:.4f} falls in the uncertainty band "
            f"({effective_threshold:.2f}, {reject_threshold:.2f})",
            **common,
        )

    return CandidateVerification(
        candidate=candidate,
        media=media,
        decision=VerificationDecision(
            status=status,
            distance=round(best_distance, 6),
            threshold=effective_threshold if status is DecisionStatus.MATCH else reject_threshold,
            model_name=model_name,
            detector_backend="scrfd_10g",
            candidate_face_index=best_index,
            reason=reason,
        ),
        **common,
    )


def rank(verifications: list[CandidateVerification]) -> list[CandidateVerification]:
    """Order verified matches by strength of evidence: margin, and nothing else.

    Only candidates that already passed the identity gate are ranked, so ordering can
    never grant identity. Within that set the order decides which post gets anchored, so
    it should be defensible on its own terms.

    It used to be a weighted sum of five hand-chosen coefficients over margin, media
    quality, route agreement, face size and search rank. None of those weights was
    measured, and four of the five were proxies for one thing: how much signal the
    comparison had. Now that the match threshold is conditioned on face size, the margin
    already carries that, a thumbnail is scored against a tighter bar and earns a smaller
    margin for the same distance. So the extra terms are not just unmeasured, they are
    redundant, and a single measured quantity is easier to defend than five invented ones.

    Two orderings survive as tie-break keys, and both are editorial rather than
    evidential: a social post outranks a news photograph because a social post is the
    deliverable, and an independently found photograph outranks a repost of the submitted
    image because rediscovering your own pixels is provenance, not discovery.
    """

    matches = [item for item in verifications if item.matched]
    return sorted(
        matches,
        key=lambda item: (item.same_photo, not item.candidate.is_social, -item.margin),
    )


def verify_all(
    source_embedding: np.ndarray,
    media_list: list[FetchedMedia],
    *,
    engine: FaceEngine,
    model_name: str,
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    reject_threshold: float = DEFAULT_REJECT_THRESHOLD,
) -> tuple[list[CandidateVerification], list[CandidateVerification]]:
    """Verify every candidate. Returns (all verifications, ranked matches)."""

    verifications = [
        verify_candidate(
            source_embedding,
            media,
            engine=engine,
            model_name=model_name,
            match_threshold=match_threshold,
            reject_threshold=reject_threshold,
        )
        for media in media_list
    ]
    counts: dict[str, int] = {}
    for item in verifications:
        counts[str(item.decision.status)] = counts.get(str(item.decision.status), 0) + 1
    logger.info("verification outcome: %s", counts or "none")
    return verifications, rank(verifications)


__all__ = [
    "DEFAULT_MATCH_THRESHOLD",
    "DEFAULT_REJECT_THRESHOLD",
    "MIN_CANDIDATE_FACE_PX",
    "SAME_PHOTO_DHASH_ERROR",
    "SAME_PHOTO_MEAN_ERROR",
    "SIZE_PENALTIES",
    "CandidateVerification",
    "classify_same_photo",
    "decode_media",
    "is_same_photo",
    "match_threshold_for",
    "photo_difference",
    "rank",
    "verify_all",
    "verify_candidate",
]
