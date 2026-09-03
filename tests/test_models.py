from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from sigil.models import (
    BoundingBox,
    DecisionStatus,
    FaceObservation,
    FaceQuality,
    VerificationDecision,
)


def test_embedding_is_excluded_from_serialization() -> None:
    observation = FaceObservation(
        source_image="input.jpg",
        face_crop_path="crop.jpg",
        detector_backend="retinaface",
        model_name="Buffalo_L",
        detection_confidence=0.99,
        bounding_box=BoundingBox(x=1, y=2, width=100, height=100),
        quality=FaceQuality(
            face_size_px=100,
            blur_score=250.0,
            brightness=0.5,
            quality_score=0.9,
        ),
        embedding=[0.1, 0.2, 0.3],
    )

    assert "embedding" not in observation.model_dump()
    assert "0.1" not in observation.model_dump_json()


def test_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        BoundingBox(x=0, y=0, width=10, height=10, typo=True)


def test_verification_supports_explicit_inconclusive_state() -> None:
    decision = VerificationDecision(
        status=DecisionStatus.INCONCLUSIVE,
        distance=None,
        threshold=0.42,
        model_name="Buffalo_L",
        detector_backend="retinaface",
        reason="Candidate image has no sufficiently large face",
    )

    assert decision.status is DecisionStatus.INCONCLUSIVE
    assert decision.distance is None


def test_datetime_serializes_as_iso_8601() -> None:
    now = datetime(2026, 9, 2, 12, 0, tzinfo=UTC)
    decision = VerificationDecision(
        status=DecisionStatus.MATCH,
        distance=0.2,
        threshold=0.42,
        model_name="Buffalo_L",
        detector_backend="retinaface",
        reason="distance passed the calibrated threshold",
    )

    assert decision.model_dump(mode="json")["status"] == "MATCH"
    assert now.isoformat() == "2026-09-02T12:00:00+00:00"


def test_match_cannot_exceed_threshold() -> None:
    with pytest.raises(ValidationError, match="MATCH distance"):
        VerificationDecision(
            status=DecisionStatus.MATCH,
            distance=0.7,
            threshold=0.5,
            model_name="Buffalo_L",
            detector_backend="retinaface",
            reason="invalid threshold relationship",
        )
