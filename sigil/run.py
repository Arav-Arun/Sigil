"""End-to-end orchestration: face → search → verify → evidence → anchor.

The stage boundaries here are deliberate. Discovery and evidence construction complete
before the chain is touched at all, so a chain outage costs the anchoring step and
nothing else: the bundle still exists, still has a root, and ``sigil anchor`` can finish
the job later without re-running a single search.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from sigil import __version__
from sigil.candidates import fetch_all_sync
from sigil.chain import ChainClient, ChainError
from sigil.config import DATA_DIR, ConfigurationError, Settings, get_settings
from sigil.evidence.bundle import attach_receipt, build_manifest, write_bundle
from sigil.evidence.canonical import canonicalize
from sigil.face import FacePipelineError, detect_and_encode, get_engine
from sigil.imaging import sha256_bytes, sha256_file
from sigil.models import ChainReceipt, DecisionStatus, FaceObservation, PipelineErrorCode
from sigil.search.routes import discover
from sigil.search.serpapi import WebSearchError, client_from_settings
from sigil.verify import (
    DEFAULT_MATCH_THRESHOLD,
    DEFAULT_REJECT_THRESHOLD,
    CandidateVerification,
    is_same_photo,
    rank,
    verify_all,
)

logger = logging.getLogger(__name__)

# Capitalised runs that are not people. Without these, headline furniture like "The New
# York" outvotes the actual subject.
_ENTITY_NOISE = frozenset(
    {
        "the new york",
        "new york times",
        "the washington post",
        "bbc news",
        "the guardian",
        "associated press",
        "first african",
        "african american",
        "united states",
        "secretary of",
        "of state",
        "the court",
        "book talk",
        "the joint",
    }
)


class PipelineError(RuntimeError):
    def __init__(self, code: PipelineErrorCode, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


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
    # because "the search found nothing" and "three of four sources errored" look
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


def _harvest_entity(verifications: list[CandidateVerification]) -> str:
    """Find the name that the *confirmed* pages agree on.

    This is the one place a name can be trusted, and only because of the order of events:
    the face gate passed first, on those exact pages, so a name recurring across them is
    attached to a person the pipeline has already identified rather than to a guess. The
    name is still never evidence of identity; it is only a better query.

    A title-derived name has to appear on at least two independently-found matches. When
    only the query photograph was rediscovered, a name-like media filename may be used as
    a weaker search hint. Neither path decides identity; every result from the hint still
    has to pass the original face embedding gate.
    """

    from collections import Counter

    counts: Counter[str] = Counter()
    for item in verifications:
        if not item.matched:
            continue
        seen: set[str] = set()
        words = re.findall(r"[A-Z][a-z]+", item.candidate.title or "")
        for size in (3, 2):
            for start in range(len(words) - size + 1):
                phrase = " ".join(words[start : start + size])
                if phrase.lower() in _ENTITY_NOISE or phrase in seen:
                    continue
                seen.add(phrase)
                counts[phrase] += 1

    for phrase, count in counts.most_common():
        if count >= 2:
            return phrase

    filename_noise = {
        "avatar",
        "face",
        "headshot",
        "image",
        "img",
        "photo",
        "picture",
        "profile",
        "query",
        "selfie",
        "thumbnail",
    }
    for item in verifications:
        if not item.matched or not item.same_photo:
            continue
        media_urls = [item.candidate.image_url, item.media.final_url]
        for media_url in media_urls:
            if not media_url:
                continue
            stem = Path(unquote(urlsplit(str(media_url)).path)).stem
            words = [word for word in re.split(r"[-_.]+", stem) if word.isalpha()]
            if not 1 <= len(words) <= 3:
                continue
            if any(word.lower() in filename_noise for word in words):
                continue
            if all(3 <= len(word) <= 30 for word in words):
                return " ".join(word.capitalize() for word in words)
    return ""


def _person_name_from_label(label: str) -> str:
    """Accept a compact person-like image label as a search hint, never as evidence."""

    normalized = " ".join(label.split()).strip()
    words = normalized.split()
    if not 2 <= len(words) <= 5:
        return ""
    if normalized.lower() in _ENTITY_NOISE | {"go for gold"}:
        return ""
    if not all(re.fullmatch(r"[A-Z][A-Za-z'\-]{1,29}", word) for word in words):
        return ""
    return normalized


def _expand_matched_pages(
    verifications: list[CandidateVerification],
    source_embedding: Any,
    *,
    resolved: Settings,
    engine: Any,
    discovered_at: datetime,
    match_threshold: float,
    reject_threshold: float,
    on_stage: Any,
    result: RunResult,
    raw_responses: dict[str, Any],
) -> tuple[list[CandidateVerification], str]:
    """Check every photograph on the page that already passed face verification."""

    from sigil.search.providers.pages import PageHarvestProvider

    pages = list(
        dict.fromkeys(
            str(item.candidate.source_url)
            for item in verifications
            if item.matched and not item.candidate.is_social
        )
    )
    if not pages:
        return [], ""

    try:
        with _stage(result, "expand", on_stage):
            outcome = PageHarvestProvider(timeout=resolved.http_timeout_seconds).harvest(
                pages, discovered_at=discovered_at, max_pages=2
            )
            outcome.name = "page-harvest:verified-match"
            raw_responses[outcome.name] = outcome.raw

            # Link the already verified image back to its label on the source page. For
            # example, the search result title may be merely "Go For Gold", while the
            # page's exact image element says alt="Gaurish Baliga". This label only seeds
            # another search; it never changes a face decision.
            verified_by_image = {
                str(url): item
                for item in verifications
                if item.matched and item.same_photo
                for url in (item.candidate.image_url, item.media.final_url)
                if url
            }
            entity_hint = ""
            for candidate in outcome.candidates:
                item = verified_by_image.get(str(candidate.image_url or ""))
                if item is None:
                    continue
                label_entity = _person_name_from_label(candidate.title)
                if label_entity:
                    item.candidate.title = candidate.title
                    entity_hint = entity_hint or label_entity
            if entity_hint:
                outcome.entities = [entity_hint]
            result.sources.append(outcome.summary())

            known_media = {
                str(url)
                for item in verifications
                for url in (item.candidate.image_url, item.media.final_url)
                if url
            }
            fresh = [
                candidate
                for candidate in outcome.candidates
                if str(candidate.image_url or "") not in known_media
            ]
            if not fresh:
                return [], entity_hint
            media = fetch_all_sync(
                fresh,
                concurrency=resolved.download_concurrency,
                timeout=resolved.http_timeout_seconds,
            )
            checked, _ = verify_all(
                source_embedding,
                media,
                engine=engine,
                model_name=engine.model_id,
                match_threshold=match_threshold,
                reject_threshold=reject_threshold,
            )
    except Exception as exc:
        logger.warning("verified-page expansion failed: %s: %s", type(exc).__name__, exc)
        return [], ""

    logger.info(
        "verified-page expansion added %d candidate(s), %d verified",
        len(checked),
        sum(1 for item in checked if item.matched),
    )
    return checked, entity_hint


def _expand_from_match(
    verifications: list[CandidateVerification],
    source_embedding: Any,
    *,
    entity_hint: str,
    search_client: Any,
    resolved: Settings,
    engine: Any,
    discovered_at: datetime,
    match_threshold: float,
    reject_threshold: float,
    known_urls: set[str],
    on_stage: Any,
    result: RunResult,
    raw_responses: dict[str, Any],
) -> list[CandidateVerification]:
    """Search again using a name harvested from pages the face gate already confirmed.

    Every other route guesses from the query image and is fixed at discovery time. This
    one runs *after* verification, so it can use something no earlier stage had: a name
    that appears on several pages which were independently confirmed to show this face.

    Exa's findSimilar was tried here first and is not used: asked for pages like a
    YouTube watch URL it returned a property listing, a git repository and a video site,
    because a watch page carries almost no crawlable text to be similar to. Searching the
    confirmed name returns pages about the person instead.

    Failure is never fatal. Expansion is a bonus round, and a run that already has a
    verified match must not be lost because a second provider was slow or unconfigured.
    """

    from sigil.search.providers.base import ProviderResult
    from sigil.search.providers.exa import ExaProvider
    from sigil.search.routes import _site_query, merge_candidates, parse_candidates

    entity = entity_hint or _harvest_entity(verifications)
    if not entity:
        logger.info("expansion skipped: the confirmed pages agree on no name")
        return []

    try:
        with _stage(result, "expand", on_stage):
            outcomes: list[ProviderResult] = []

            route = f"R5:serp-confirmed-name:{entity}"
            started = time.perf_counter()
            try:
                payload = search_client.web(
                    _site_query(entity),
                    route=route,
                    country=resolved.search_country,
                    language=resolved.search_language,
                )
                outcomes.append(
                    ProviderResult(
                        name=route,
                        candidates=parse_candidates(
                            payload, route=route, discovered_at=discovered_at
                        ),
                        entities=[entity],
                        raw=payload,
                        elapsed_ms=(time.perf_counter() - started) * 1000,
                    )
                )
            except Exception as exc:
                logger.warning(
                    "confirmed-name SerpApi search failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )
                outcomes.append(
                    ProviderResult(
                        name=route,
                        entities=[entity],
                        elapsed_ms=(time.perf_counter() - started) * 1000,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )

            exa = ExaProvider(resolved.exa_api_key.get_secret_value())
            if exa.configured():
                started = time.perf_counter()
                try:
                    exa_outcome = exa.search([entity], discovered_at=discovered_at)
                    exa_outcome.name = f"exa:confirmed-name:{entity}"
                    exa_outcome.entities = [entity]
                    exa_outcome.elapsed_ms = (time.perf_counter() - started) * 1000
                    outcomes.append(exa_outcome)
                except Exception as exc:
                    logger.warning(
                        "confirmed-name Exa search failed: %s: %s", type(exc).__name__, exc
                    )
                    outcomes.append(
                        ProviderResult(
                            name=f"exa:confirmed-name:{entity}",
                            entities=[entity],
                            elapsed_ms=(time.perf_counter() - started) * 1000,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )

            candidates = []
            for outcome in outcomes:
                result.sources.append(outcome.summary())
                if outcome.raw:
                    raw_responses[outcome.name] = outcome.raw
                candidates.extend(outcome.candidates)

            fresh = [
                candidate
                for candidate in merge_candidates(candidates)
                if str(candidate.source_url) not in known_urls
            ]
            logger.info(
                "expansion on confirmed name %r: %d result(s), %d new",
                entity,
                len(candidates),
                len(fresh),
            )
            if not fresh:
                return []

            # Recorded on every expansion candidate, so the audit trail shows which
            # results were found *because of* an earlier match rather than independently.
            for candidate in fresh:
                candidate.search_routes = [*candidate.search_routes, f"confirmed-name:{entity}"]

            media = fetch_all_sync(
                fresh,
                concurrency=resolved.download_concurrency,
                timeout=resolved.http_timeout_seconds,
            )
            checked, _ = verify_all(
                source_embedding,
                media,
                engine=engine,
                model_name=engine.model_id,
                match_threshold=match_threshold,
                reject_threshold=reject_threshold,
            )
    except Exception as exc:  # a bonus round must not sink a successful run
        logger.warning("expansion failed: %s: %s", type(exc).__name__, exc)
        return []

    matched = sum(1 for item in checked if item.matched)
    logger.info("expansion added %d candidate(s), %d verified", len(checked), matched)
    return checked


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
    expected_chain_id: int | None = None,
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
        if item.matched:
            item.same_photo = item.candidate.exact_match or is_same_photo(
                face.source_image, item.media
            )
    ranked = rank(verifications)
    result.verifications = verifications

    # -- 4b. expand from what was confirmed ------------------------------------
    # Every other route guesses from the query image. This one starts from a page that
    # already passed the face gate and asks for more like it, so recall compounds after
    # the first hit instead of being fixed at discovery time.
    #
    # Bounded to a single round on purpose: feeding expansion results back in again would
    # drift away from the original face by way of "similar page" hops, and each hop is a
    # weaker link to the person than the one before it.
    if ranked:
        page_expanded, page_entity = _expand_matched_pages(
            verifications,
            source_embedding,
            resolved=resolved,
            engine=engine,
            discovered_at=datetime.now(UTC),
            match_threshold=match_threshold,
            reject_threshold=reject_threshold,
            on_stage=on_stage,
            result=result,
            raw_responses=discovery["raw_responses"],
        )
        if page_entity and page_entity not in discovery["entities_inferred"]:
            discovery["entities_inferred"].append(page_entity)
        for item in page_expanded:
            if item.matched:
                item.same_photo = item.candidate.exact_match or is_same_photo(
                    face.source_image, item.media
                )
        if page_expanded:
            verifications = verifications + page_expanded
            result.verifications = verifications
            ranked = rank(verifications)

        expanded = _expand_from_match(
            verifications,
            source_embedding,
            entity_hint=page_entity,
            search_client=search_client,
            resolved=resolved,
            engine=engine,
            discovered_at=datetime.now(UTC),
            match_threshold=match_threshold,
            reject_threshold=reject_threshold,
            known_urls={str(v.candidate.source_url) for v in verifications},
            on_stage=on_stage,
            result=result,
            raw_responses=discovery["raw_responses"],
        )
        if expanded:
            for item in expanded:
                if item.matched:
                    item.same_photo = item.candidate.exact_match or is_same_photo(
                        face.source_image, item.media
                    )
            verifications = verifications + expanded
            result.verifications = verifications
            ranked = rank(verifications)

        # Post-verification routes use the same quota-aware client and belong in the same
        # audit trail as discovery. Refresh these snapshots after expansion rather than
        # leaving the UI and evidence bundle with the pre-expansion totals.
        discovery["search_records"] = list(search_client.search_records)
        discovery["budget"] = search_client.budget.summary()
        for route in discovery["raw_responses"]:
            if route not in discovery["routes_run"]:
                discovery["routes_run"].append(route)
        result.search_records = discovery["search_records"]
        result.budget = discovery["budget"]

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

    return result


__all__ = ["PipelineError", "RunResult", "run_pipeline"]
