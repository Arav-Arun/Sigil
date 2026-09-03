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
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from sigil.candidates import FetchedMedia, MediaQuality
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

    # Shared by every return path below. Typed explicitly so the ** expansion into
    # CandidateVerification keeps its per-field types.
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

    if best_distance <= match_threshold:
        status = DecisionStatus.MATCH
        reason = (
            f"cosine distance {best_distance:.4f} <= {match_threshold:.2f} "
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
            f"({match_threshold:.2f}, {reject_threshold:.2f})",
            **common,
        )

    return CandidateVerification(
        candidate=candidate,
        media=media,
        decision=VerificationDecision(
            status=status,
            distance=round(best_distance, 6),
            threshold=match_threshold if status is DecisionStatus.MATCH else reject_threshold,
            model_name=model_name,
            detector_backend="scrfd_10g",
            candidate_face_index=best_index,
            reason=reason,
        ),
        **common,
    )


def _quality_weight(quality: MediaQuality) -> float:
    return {
        MediaQuality.ORIGINAL: 1.0,
        MediaQuality.OPENGRAPH: 0.9,
        MediaQuality.OEMBED: 0.85,
        MediaQuality.THUMBNAIL: 0.7,
    }[quality]


def rank(verifications: list[CandidateVerification]) -> list[CandidateVerification]:
    """Order verified matches by strength of evidence.

    Only candidates that already passed the identity gate are ranked. Search rank and the
    exact-image flag act as tie-breakers among genuine matches; they can never promote a
    candidate that failed verification.
    """

    matches = [item for item in verifications if item.matched]

    def score(item: CandidateVerification) -> float:
        return (
            # Distance margin dominates: it is the only identity evidence.
            4.0 * item.margin
            + 0.8 * _quality_weight(item.media.quality)
            + 0.4 * (1.0 if item.candidate.exact_match else 0.0)
            + 0.3 * min(len(item.candidate.search_routes), 3) / 3.0
            + 0.2 * min(item.best_face_px, 400) / 400.0
            - 0.02 * min(item.candidate.search_rank, 25)
        )

    # Social posts sort ahead of everything else *within* the verified set, because the
    # deliverable is a social-media post and a news photograph, however sharp, is not one.
    # This orders matches; it cannot create one. A candidate the face gate rejected is not
    # in this list at all, so no amount of platform preference can promote it.
    return sorted(matches, key=lambda item: (not item.candidate.is_social, -score(item)))


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
    "CandidateVerification",
    "decode_media",
    "rank",
    "verify_all",
    "verify_candidate",
]
