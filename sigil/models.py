"""Typed data contracts shared by every Sigil pipeline stage."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

SCHEMA_VERSION = 2

Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]


class StrictModel(BaseModel):
    """Base model that rejects misspelled/unknown persisted fields."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class DecisionStatus(StrEnum):
    """A three-state identity decision; uncertainty is never coerced to success."""

    MATCH = "MATCH"
    NON_MATCH = "NON_MATCH"
    INCONCLUSIVE = "INCONCLUSIVE"


class PipelineErrorCode(StrEnum):
    """Stable error codes for CLI, tests, and machine-readable output."""

    NO_FACE = "NO_FACE"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    LOW_QUALITY = "LOW_QUALITY"
    SEARCH_EMPTY = "SEARCH_EMPTY"
    SEARCH_UNAVAILABLE = "SEARCH_UNAVAILABLE"
    CANDIDATE_UNFETCHABLE = "CANDIDATE_UNFETCHABLE"
    NO_VERIFIED_MATCH = "NO_VERIFIED_MATCH"
    CHAIN_PENDING = "CHAIN_PENDING"
    CHAIN_MISMATCH = "CHAIN_MISMATCH"
    INVALID_CONFIGURATION = "INVALID_CONFIGURATION"
    INVALID_INPUT = "INVALID_INPUT"


class BoundingBox(StrictModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class FaceQuality(StrictModel):
    """Normalized input-quality measurements used to explain gating."""

    face_size_px: int = Field(gt=0)
    blur_score: NonNegativeFloat
    brightness: Confidence
    border_truncated: bool = False
    quality_score: Confidence
    warnings: list[str] = Field(default_factory=list)


class FaceObservation(StrictModel):
    schema_version: int = Field(default=SCHEMA_VERSION, ge=1)
    source_image: str
    face_crop_path: str
    # Head-and-shoulders crop, used only as a search query. Never fed to the recogniser.
    portrait_crop_path: str = ""
    detector_backend: str
    model_name: str
    detection_confidence: Confidence
    bounding_box: BoundingBox
    quality: FaceQuality
    # Embeddings are sensitive biometric data. They are usable in memory but excluded from
    # ordinary serialization, logs, evidence manifests, and on-chain records.
    embedding: list[float] = Field(min_length=1, exclude=True, repr=False)


class SearchCandidate(StrictModel):
    schema_version: int = Field(default=SCHEMA_VERSION, ge=1)
    source_url: HttpUrl
    platform: str
    # Whether the candidate sits on an allowlisted social platform. Used for ordering and
    # for the report; never for deciding identity, which only the face gate does.
    is_social: bool = False
    post_id: str = ""
    title: str = ""
    image_url: HttpUrl | None = None
    thumbnail_url: HttpUrl | None = None
    # The source site's own icon, as reported by the search provider.
    favicon_url: HttpUrl | None = None
    search_rank: int = Field(ge=1)
    # A discovery hint: the search provider put this URL in its exact-image result
    # bucket. It is never proof that the media we downloaded is the submitted image;
    # that stricter conclusion is recorded as CandidateVerification.same_photo.
    exact_match: bool = False
    discovered_at: datetime
    search_routes: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict, exclude=True, repr=False)


class VerificationDecision(StrictModel):
    schema_version: int = Field(default=SCHEMA_VERSION, ge=1)
    status: DecisionStatus
    distance: NonNegativeFloat | None = None
    threshold: NonNegativeFloat
    model_name: str
    detector_backend: str
    candidate_face_index: int | None = Field(default=None, ge=0)
    reason: str

    @model_validator(mode="after")
    def validate_distance_against_decision(self) -> Self:
        if self.status is DecisionStatus.INCONCLUSIVE:
            return self
        if self.distance is None:
            raise ValueError(f"{self.status} requires a measured distance")
        if self.status is DecisionStatus.MATCH and self.distance > self.threshold:
            raise ValueError("MATCH distance must be at or below the threshold")
        if self.status is DecisionStatus.NON_MATCH and self.distance <= self.threshold:
            raise ValueError("NON_MATCH distance must be above the threshold")
        return self


class ChainReceipt(StrictModel):
    schema_version: int = Field(default=SCHEMA_VERSION, ge=1)
    chain_id: int = Field(gt=0)
    contract_address: str
    evidence_root: str = Field(pattern=r"^0x[0-9a-fA-F]{64}$")
    transaction_hash: str = Field(pattern=r"^0x[0-9a-fA-F]{64}$")
    block_number: int = Field(ge=0)
    submitter: str
    anchored_at: datetime
    gas_used: int = Field(default=0, ge=0)
    explorer_url: str = ""
