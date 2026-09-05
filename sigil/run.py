"""End-to-end orchestration: face → search → verify → evidence → anchor.

The stage boundaries here are deliberate. Discovery and evidence construction complete
before the chain is touched at all, so a chain outage costs the anchoring step and
nothing else: the bundle still exists, still has a root, and ``sigil anchor`` can finish
the job later without re-running a single search.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sigil import __version__
from sigil.candidates import fetch_all_sync
from sigil.chain import SEPOLIA_CHAIN_ID, ChainClient, ChainError
from sigil.config import DATA_DIR, ConfigurationError, Settings, get_settings
from sigil.evidence.bundle import attach_receipt, build_manifest, write_bundle, write_pending
from sigil.evidence.canonical import canonicalize
from sigil.face import FacePipelineError, detect_and_encode, get_engine
from sigil.imaging import sha256_bytes, sha256_file
from sigil.models import ChainReceipt, DecisionStatus, FaceObservation, PipelineErrorCode
from sigil.search.providers.serpapi import WebSearchError, client_from_settings
from sigil.search.routes import discover
from sigil.verify import (
    DEFAULT_MATCH_THRESHOLD,
    DEFAULT_REJECT_THRESHOLD,
    CandidateVerification,
    classify_same_photo,
    rank,
    verify_all,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RunResult:
    """Everything a run produced, successful or not."""

    run_id: str
    started_at: datetime
    ok: bool = False
    error_code: PipelineErrorCode | None = None
    error_message: str = ""
    bundle_dir: Path | None = None
    evidence_root: str = ""
    # The face that was detected and encoded. Kept so callers can report on the
    # identification step itself. FaceObservation.embedding is excluded from
    # serialization, so this never carries biometric data out of the process.
    face: FaceObservation | None = None
    selected: CandidateVerification | None = None
    verifications: list[CandidateVerification] = field(default_factory=list)
    receipt: ChainReceipt | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    search_records: list[dict[str, Any]] = field(default_factory=list)
    budget: dict[str, int] = field(default_factory=dict)
    # Which discovery sources ran, what each returned, and how long each took. Recorded
    # because "the search found nothing" and "several sources errored" look
    # identical from the outside, and only one of them is an answer about the world.
    sources: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "ok": self.ok,
            "error_code": str(self.error_code) if self.error_code else None,
            "error_message": self.error_message,
            "bundle_dir": str(self.bundle_dir) if self.bundle_dir else None,
            "evidence_root": self.evidence_root,
            "selected": self.selected.to_json() if self.selected else None,
            # The index is the candidate's position in the examined list, which is what
            # the thumbnail filenames are keyed on. Filtering in the UI must not lose it.
            "candidates": [
                {**item.to_json(), "index": i} for i, item in enumerate(self.verifications)
            ],
            "receipt": self.receipt.model_dump(mode="json") if self.receipt else None,
            "timings_ms": {k: round(v, 2) for k, v in self.timings_ms.items()},
            "search_records": self.search_records,
            "search_budget": self.budget,
            "sources": self.sources,
        }


@contextmanager
def _stage(result: RunResult, name: str, on_stage: Any = None) -> Generator[None, None, None]:
    if on_stage:
        on_stage(name)
    start = time.perf_counter()
    try:
        yield
    finally:
        result.timings_ms[name] = (time.perf_counter() - start) * 1000


def run_pipeline(
    image_path: str | Path,
    *,
    settings: Settings | None = None,
    output_root: Path | None = None,
    face_index: int | None = None,
    select_largest: bool = False,
    max_candidates: int | None = None,
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    reject_threshold: float = DEFAULT_REJECT_THRESHOLD,
    no_cache: bool = False,
    use_cache: bool = True,
    search_budget: int = 4,
    name_hint: str = "",
    skip_chain: bool = False,
    expected_chain_id: int | None = SEPOLIA_CHAIN_ID,
    run_id: str | None = None,
    started_at: datetime | None = None,
    on_stage: Any = None,
) -> RunResult:
    """Run the full pipeline. Never raises for expected failures, check ``result.ok``."""

    resolved = settings or get_settings()
    started = started_at or datetime.now(UTC)
    run_id = run_id or f"{started:%Y%m%dT%H%M%S%fZ}-{sha256_file(image_path)[:8]}"
    max_candidates = max_candidates if max_candidates is not None else resolved.max_candidates
    result = RunResult(run_id=run_id, started_at=started)

    base = Path(output_root) if output_root else DATA_DIR / "bundles"
    bundle_dir = base / run_id
    bundle_dir.mkdir(parents=True, exist_ok=True)

    engine = get_engine()

    # -- 1. face ---------------------------------------------------------------
    try:
        with _stage(result, "warm_up", on_stage):
            engine.warm_up()
        with _stage(result, "face", on_stage):
            face = detect_and_encode(
                image_path,
                bundle_dir / "media",
                engine=engine,
                face_index=face_index,
                select_largest=select_largest,
            )
    except FacePipelineError as exc:
        result.error_code, result.error_message = exc.code, exc.message
        return result

    result.face = face

    import numpy as np

    source_embedding = np.asarray(face.embedding, dtype=np.float32)

    # -- 2. search -------------------------------------------------------------
    # The client is built here rather than inside discover() so that a failed search
    # still reports what it spent. It used to be created inside, which meant a run that
    # burned three live searches and found nothing reported "0 live, 0 cached": the one
    # number the operator needs in order to trust their remaining quota was the number
    # that disappeared exactly when it mattered.
    search_client = client_from_settings(resolved, use_cache=use_cache, budget_limit=search_budget)
    try:
        with _stage(result, "search", on_stage):
            discovery = discover(
                face.source_image,
                face.portrait_crop_path or face.face_crop_path,
                face.face_crop_path,
                settings=resolved,
                client=search_client,
                max_results=max_candidates,
                no_cache=no_cache,
                use_cache=use_cache,
                budget_limit=search_budget,
                name_hint=name_hint,
            )
    except WebSearchError as exc:
        result.error_code, result.error_message = exc.code, exc.message
        result.search_records = list(search_client.search_records)
        result.budget = search_client.budget.summary()
        return result

    result.search_records = discovery["search_records"]
    result.budget = discovery["budget"]
    result.sources = discovery.get("sources", [])

    # -- 3. acquire ------------------------------------------------------------
    with _stage(result, "acquire", on_stage):
        media = fetch_all_sync(
            discovery["candidates"],
            concurrency=resolved.download_concurrency,
            timeout=resolved.http_timeout_seconds,
        )

    # -- 4. verify -------------------------------------------------------------
    with _stage(result, "verify", on_stage):
        verifications, _ = verify_all(
            source_embedding,
            media,
            engine=engine,
            model_name=engine.model_id,
            match_threshold=match_threshold,
            reject_threshold=reject_threshold,
        )
    for item in verifications:
        item.same_photo = classify_same_photo(face.source_image, item)
    ranked = rank(verifications)
    result.verifications = verifications

    if not ranked:
        inconclusive = sum(
            1 for v in verifications if v.decision.status is DecisionStatus.INCONCLUSIVE
        )
        result.error_code = PipelineErrorCode.NO_VERIFIED_MATCH
        result.error_message = (
            f"{len(verifications)} candidate(s) examined, none passed the identity gate "
            f"({inconclusive} inconclusive). Reporting no match rather than guessing."
        )
        return result

    selected = ranked[0]
    result.selected = selected

    # -- 5. evidence -----------------------------------------------------------
    with _stage(result, "evidence", on_stage):
        search_payload = canonicalize(discovery["raw_responses"])
        configuration = {
            "match_threshold": match_threshold,
            "reject_threshold": reject_threshold,
            "max_candidates": max_candidates,
            "detector": "scrfd_10g",
            "det_size": engine.det_size,
            "det_threshold": engine.det_threshold,
            "distance_metric": "cosine",
        }
        # The corroborating set, not just the winner. Reposts of the submitted image are
        # counted separately because rediscovering your own pixels is provenance, and
        # counting it as independent support would inflate exactly the number that is
        # supposed to say how much independent support there is.
        distinct = [item for item in ranked if not item.same_photo]
        distances = sorted(
            item.decision.distance for item in ranked if item.decision.distance is not None
        )
        corroboration = {
            "verified_matches": len(ranked),
            "distinct_photos": len(distinct),
            "source_image_reposts": len(ranked) - len(distinct),
            "best_distance": distances[0] if distances else None,
            # The gap to the next-best verified photo. A lone match has none, and saying
            # so is more honest than reporting a number that does not exist.
            "runner_up_distance": distances[1] if len(distances) > 1 else None,
            # Sorted so the set is order-independent, and digests rather than URLs so the
            # commitment is to the media actually compared.
            "media_sha256": sorted(item.media.sha256 for item in ranked),
        }
        manifest = build_manifest(
            input_sha256=sha256_file(face.source_image),
            crop_sha256=sha256_file(face.face_crop_path),
            candidate_sha256=selected.media.sha256,
            search_response_sha256=sha256_bytes(search_payload),
            canonical_post_url=str(selected.candidate.source_url),
            platform=selected.candidate.platform,
            post_id=selected.candidate.post_id,
            media_quality=str(selected.media.quality),
            media_url=selected.media.final_url,
            discovered_at=selected.candidate.discovered_at.isoformat(),
            decision={
                "status": str(selected.decision.status),
                "distance": selected.decision.distance,
                "threshold": selected.decision.threshold,
                "candidate_face_index": selected.decision.candidate_face_index,
                "faces_detected": selected.faces_detected,
            },
            corroboration=corroboration,
            configuration=configuration,
            model_id=engine.model_id,
            pipeline_version=__version__,
            search_routes=list(selected.candidate.search_routes),
        )
        _, tree = write_bundle(
            bundle_dir,
            manifest=manifest,
            artifacts={
                "input.jpg": Path(face.source_image).read_bytes(),
                "aligned_crop.jpg": Path(face.face_crop_path).read_bytes(),
                "candidate.bin": selected.media.data,
            },
            search_responses=discovery["raw_responses"],
            context={
                "run_id": run_id,
                "routes_run": discovery["routes_run"],
                "entities_inferred": discovery["entities_inferred"],
                "search_records": discovery["search_records"],
                "sources": discovery.get("sources", []),
                "budget": discovery["budget"],
                "candidates": [item.to_json() for item in verifications],
                "face_quality": face.quality.model_dump(mode="json"),
                "providers": engine.providers,
            },
        )

    result.bundle_dir = bundle_dir
    result.evidence_root = tree.root_hex

    with _stage(result, "report", on_stage):
        from sigil.report import html_report, write_thumbnails

        # Presentation only, and deliberately outside the Merkle root: these are for a
        # human scanning the grid, not part of the claim. The candidate bytes that the
        # decision was actually computed on are digested in the manifest.
        write_thumbnails(verifications, bundle_dir / "media" / "thumbs")

        html_report(
            bundle_dir,
            manifest=manifest,
            root=tree.root_hex,
            verifications=verifications,
            timings=result.timings_ms,
            search_records=discovery["search_records"],
        )

    # -- 6. anchor -------------------------------------------------------------
    if skip_chain:
        result.ok = True
        return result

    try:
        with _stage(result, "anchor", on_stage):
            resolved.require("chain-write")
            client = ChainClient(
                resolved.sepolia_rpc_url.get_secret_value(),
                resolved.contract_address,
                expected_chain_id=expected_chain_id,
            )
            receipt = client.anchor(
                tree.root,
                manifest["schema_version"],
                resolved.private_key.get_secret_value(),
            )
        attach_receipt(bundle_dir, receipt)
        result.receipt = receipt
        result.ok = True

        # Regenerate the report now that the receipt exists.
        from sigil.report import html_report

        html_report(
            bundle_dir,
            manifest=manifest,
            root=tree.root_hex,
            verifications=verifications,
            receipt=receipt,
            timings=result.timings_ms,
            search_records=discovery["search_records"],
        )
    except (ChainError, ValueError) as exc:
        # Discovery and evidence succeeded; only anchoring failed. The bundle is intact
        # and `sigil anchor --bundle` can complete it without repeating any search.
        default_code = (
            PipelineErrorCode.INVALID_CONFIGURATION
            if isinstance(exc, ConfigurationError)
            else PipelineErrorCode.CHAIN_PENDING
        )
        code = getattr(exc, "code", default_code)
        result.error_code = code
        result.error_message = f"{exc} (evidence bundle is intact at {bundle_dir})"
        # A transaction that was submitted but did not confirm is recorded, so
        # `sigil anchor --bundle` resumes that exact hash rather than signing again.
        pending = getattr(exc, "transaction_hash", "")
        if pending:
            write_pending(bundle_dir, pending, expected_chain_id or SEPOLIA_CHAIN_ID)
            result.error_message = (
                f"{exc} (evidence bundle is intact at {bundle_dir}; "
                f"resume with `sigil anchor --bundle {bundle_dir}`)"
            )

    return result


__all__ = ["RunResult", "run_pipeline"]
