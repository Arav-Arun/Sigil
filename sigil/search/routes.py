"""Multi-route discovery with progressive escalation.

===== ==================================================================== ========
Route  What it asks                                                         Cost
===== ==================================================================== ========
R1     Lens ``type=all`` on the full image: visual matches, pages *about*   1 search
       the image, and the entity Lens inferred, in one response
R2     Entity pivot: that inferred name, restricted to social domains       1 search
R3     Lens ``type=all`` on a head-and-shoulders crop, identity-led         1 search
===== ==================================================================== ========

R1 runs alone first because it is usually sufficient and because R2 depends on the entity
it returns. If R1 comes back short, R2 and R3 run **concurrently**, since neither needs
the other. A well-indexed subject therefore costs one search; a hard one costs three.

``type=all`` replaced a pair of narrower calls (``exact_matches`` then ``visual_matches``).
It returns strictly more, including the ``organic_results`` block of pages that discuss
the image, for half the quota.

Two things worth knowing about the provider, because they shape everything above:

*Google Lens does not do face recognition.* It is suppressed, deliberately, for privacy.
Given a whole photo it matches objects, so a portrait of someone in distinctive glasses
returns eyewear retailers. Given a face crop it returns visually similar strangers. Lens
finds *this person* when the photo, or the person, is already published somewhere Google
has indexed and captioned. For a subject with no public photo presence it returns nothing,
and Sigil reports that rather than inventing a match. That ceiling is the provider's, not
this code's, and the README states it plainly.

*Nothing here decides identity.* Routes decide what gets looked at. Every candidate still
has to pass the face gate in :mod:`sigil.verify`, which is why this module is permissive:
throwing a result away before the face check is the one mistake that cannot be recovered.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from sigil.config import Settings, get_settings
from sigil.models import PipelineErrorCode, SearchCandidate
from sigil.search.normalize import (
    canonicalize_url,
    detect_platform,
    extract_post_id,
    is_candidate_url,
    is_social_url,
)
from sigil.search.providers.base import ProviderResult, run_providers
from sigil.search.providers.exa import ExaProvider
from sigil.search.providers.pages import PageHarvestProvider
from sigil.search.providers.wikidata import WikidataProvider
from sigil.search.serpapi import SerpApiClient, WebSearchError, client_from_settings

logger = logging.getLogger(__name__)

# Lens groups results under different keys depending on the requested type.
LENS_RESULT_KEYS = ("visual_matches", "exact_matches", "image_results", "organic_results")

# Entity pivot needs a plausible person name, not a stock-photo caption.
_ENTITY_STOPWORDS = frozenset(
    {"stock", "photo", "image", "getty", "shutterstock", "alamy", "wallpaper", "png", "jpg"}
)


def _iter_results(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Collect result items across every shape Lens uses, keeping the block they came from.

    The block matters: with ``type=all`` a single response carries exact matches, visual
    matches and page results together, and only the first of those means "this is the same
    photograph". Deriving that from the route name instead would mark all of them exact.
    """

    items: list[tuple[str, dict[str, Any]]] = []
    for key in LENS_RESULT_KEYS:
        block = payload.get(key)
        if isinstance(block, list):
            items.extend((key, item) for item in block if isinstance(item, dict))
    return items


def parse_candidates(
    payload: dict[str, Any],
    *,
    route: str,
    discovered_at: datetime,
) -> list[SearchCandidate]:
    """Turn one provider response into candidates, tolerating missing fields.

    A candidate needs a fetchable image, because a page whose picture cannot be retrieved
    cannot be face-checked and would only ever appear as an unverifiable claim.
    """

    candidates: list[SearchCandidate] = []
    for fallback_rank, (block, item) in enumerate(_iter_results(payload), start=1):
        link = item.get("link") or item.get("source")
        if not isinstance(link, str) or not is_candidate_url(link):
            continue
        image_url = item.get("original") or item.get("image") or None
        thumbnail_url = item.get("thumbnail") or None
        if not image_url and not thumbnail_url:
            continue
        try:
            canonical = canonicalize_url(link)
            candidates.append(
                SearchCandidate(
                    source_url=canonical,
                    platform=detect_platform(canonical),
                    is_social=is_social_url(canonical),
                    post_id=extract_post_id(canonical),
                    title=str(item.get("title") or item.get("snippet") or "")[:500],
                    image_url=image_url,
                    thumbnail_url=thumbnail_url,
                    search_rank=int(item.get("position") or fallback_rank),
                    exact_match=block == "exact_matches",
                    discovered_at=discovered_at,
                    search_routes=[route],
                    # Providers hand back the site's own favicon. Keeping it means the
                    # results grid can show where a candidate came from without asking a
                    # third-party favicon service, which would leak the whole result list
                    # to a company that had nothing to do with the search.
                    favicon_url=item.get("favicon") or item.get("source_icon") or None,
                    raw=item,
                )
            )
        except (TypeError, ValueError, ValidationError) as exc:
            logger.debug("skipping malformed candidate from %s: %s", route, exc)
    return candidates


def infer_entities(payload: dict[str, Any], limit: int = 3) -> list[str]:
    """Extract plausible identity strings that Lens associated with the image.

    This is a *recall* mechanism only. A name never establishes identity in Sigil, it
    just tells the next search where to look, and every post it finds still has to pass
    the face-verification gate.
    """

    entities: list[str] = []

    knowledge = payload.get("knowledge_graph")
    if isinstance(knowledge, list):
        for entry in knowledge:
            if isinstance(entry, dict) and isinstance(entry.get("title"), str):
                entities.append(entry["title"])
    elif isinstance(knowledge, dict) and isinstance(knowledge.get("title"), str):
        entities.append(knowledge["title"])

    # `related_content` is where Lens actually puts the identity it inferred, under
    # `query`. Reading only `text` meant no entity was ever found, so the entity-pivot
    # route, the one route that specifically targets social platforms, almost never ran.
    for key in ("related_content", "text_results"):
        block = payload.get(key)
        if isinstance(block, list):
            for entry in block:
                if not isinstance(entry, dict):
                    continue
                for field in ("query", "text", "title"):
                    value = entry.get(field)
                    if isinstance(value, str) and value:
                        entities.append(value)
                        break

    cleaned: list[str] = []
    for raw in entities:
        name = " ".join(raw.split())[:80]
        words = name.split()
        if not 2 <= len(words) <= 5:
            continue
        if any(word.lower() in _ENTITY_STOPWORDS for word in words):
            continue
        if not all(word[0].isupper() for word in words if word[:1].isalpha()):
            continue
        if name not in cleaned:
            cleaned.append(name)
        if len(cleaned) >= limit:
            break
    return cleaned


# Google degrades badly on long site: disjunctions: an eleven-way OR chain reliably
# returned nothing at all for a name that plainly has social coverage. Four platforms is
# the point where it still behaves, and they are the four that carry personal posts.
PIVOT_DOMAINS = ("instagram.com", "x.com", "facebook.com", "linkedin.com")


def _site_query(entity: str, domains: tuple[str, ...] = PIVOT_DOMAINS) -> str:
    sites = " OR ".join(f"site:{domain}" for domain in domains)
    return f'"{entity}" ({sites})'


def merge_candidates(candidates: list[SearchCandidate]) -> list[SearchCandidate]:
    """Deduplicate by canonical URL, keeping every route that found each candidate."""

    merged: dict[tuple[str, str], SearchCandidate] = {}
    for candidate in candidates:
        # (url, post_id), not url alone. A harvested page contributes one candidate per
        # photograph on it, all sharing the page URL, and a URL-only key threw away every
        # one but the first. Ordinary posts have a stable post_id, so cross-route merging
        # of the same post is unaffected.
        key = (str(candidate.source_url), candidate.post_id)
        existing = merged.get(key)
        if existing is None:
            merged[key] = candidate
            continue
        preferred = candidate if candidate.search_rank < existing.search_rank else existing
        merged[key] = preferred.model_copy(
            update={
                "exact_match": existing.exact_match or candidate.exact_match,
                "search_routes": sorted(set(existing.search_routes + candidate.search_routes)),
                "image_url": existing.image_url or candidate.image_url,
                "thumbnail_url": existing.thumbnail_url or candidate.thumbnail_url,
                "favicon_url": existing.favicon_url or candidate.favicon_url,
                "title": existing.title or candidate.title,
            }
        )

    return sorted(
        merged.values(),
        # Social posts first, because the task asks for a social post and a slot spent on
        # a news page is a social post not fetched. Then multi-route agreement, which is
        # a stronger signal than any single route's rank. None of this decides identity.
        key=lambda item: (
            not item.is_social,
            -len(item.search_routes),
            not item.exact_match,
            item.search_rank,
            str(item.source_url),
        ),
    )


def _wrap(fn: Any, name: str, *args: Any) -> ProviderResult:
    """Adapt a SerpApi route helper to the provider contract."""

    outcome = fn(name, *args)
    return ProviderResult(name=outcome["name"], candidates=outcome["found"], raw=outcome["payload"])


def discover(
    image_path: str | Path,
    crop_path: str | Path | None = None,
    *,
    settings: Settings | None = None,
    client: SerpApiClient | None = None,
    max_results: int = 15,
    enough: int = 8,
    no_cache: bool = False,
    use_cache: bool = True,
    budget_limit: int = 4,
    provider_timeout: float = 25.0,
    name_hint: str = "",
) -> dict[str, Any]:
    """Run the discovery routes and return candidates plus a full audit trail.

    Raises :class:`WebSearchError` with ``SEARCH_EMPTY`` when the search genuinely found
    nothing, and with ``SEARCH_UNAVAILABLE`` when the provider failed. Those two are kept
    distinct on purpose: only the first is an honest answer about the world.
    """

    resolved = settings or get_settings()
    active = client or client_from_settings(
        resolved, use_cache=use_cache, budget_limit=budget_limit
    )
    exa = ExaProvider(resolved.exa_api_key.get_secret_value())
    wikidata = WikidataProvider()
    discovered_at = datetime.now(UTC)

    country = resolved.search_country
    language = resolved.search_language

    candidates: list[SearchCandidate] = []
    raw_responses: dict[str, Any] = {}
    routes_run: list[str] = []
    entities: list[str] = []

    def lens_route(name: str, path: str | Path, search_type: str = "all") -> dict[str, Any]:
        uploaded = active.upload_image(path)
        payload = active.lens(
            uploaded,
            route=name,
            search_type=search_type,
            country=country,
            language=language,
            no_cache=no_cache,
        )
        found = parse_candidates(payload, route=name, discovered_at=discovered_at)
        logger.info("route %s: %d candidate(s)", name, len(found))
        return {"name": name, "payload": payload, "found": found}

    def web_route(name: str, entity: str) -> dict[str, Any]:
        payload = active.web(_site_query(entity), route=name, country=country, language=language)
        found = parse_candidates(payload, route=name, discovered_at=discovered_at)
        logger.info("route %s (%s): %d candidate(s)", name, entity, len(found))
        return {"name": name, "payload": payload, "found": found}

    def collect(outcome: dict[str, Any]) -> dict[str, Any]:
        raw_responses[outcome["name"]] = outcome["payload"]
        routes_run.append(outcome["name"])
        candidates.extend(outcome["found"])
        payload: dict[str, Any] = outcome["payload"]
        return payload

    source_reports: list[dict[str, Any]] = []

    try:
        # Wave one is the only stage that can run on the image alone, and it is usually
        # sufficient on its own. It also produces the entity that every other source needs,
        # which is why it cannot simply join the fan-out below.
        first = lens_route("R1:lens-all-full", image_path, "all")
        entities = infer_entities(collect(first))

        # A caller-supplied name goes to the front. Lens infers a name only for people it
        # already recognises, which excludes exactly the subjects that are hardest to
        # find, so without this the pivot never fires for anyone who is not a celebrity.
        # The name decides where to look and nothing else: every page it reaches is still
        # fetched, face-checked and just as capable of being rejected.
        if name_hint:
            entities = [name_hint, *(e for e in entities if e != name_hint)]

        social = sum(1 for candidate in candidates if candidate.is_social)
        # A caller-supplied name always escalates. Wave one returned sixty visually similar
        # strangers and therefore looked "sufficient", so the escalation never fired and the
        # name was never used, which is the one case the caller bothered to supply it for.
        if name_hint or len(candidates) < enough or not social:
            # Wave two. Four sources that fail in different ways, run at once: wall clock
            # is the slowest of them rather than the sum, and losing any one of them costs
            # coverage instead of taking the run down.
            jobs: list[tuple[str, Any]] = []
            budget_left = active.budget.remaining

            if crop_path and budget_left:
                jobs.append(
                    (
                        "R2:lens-all-portrait",
                        lambda: _wrap(lens_route, "R2:lens-all-portrait", crop_path, "all"),
                    )
                )
                budget_left -= 1
            if entities and budget_left:
                jobs.append(
                    (
                        "R3:serp-entity-pivot",
                        lambda: _wrap(web_route, "R3:serp-entity-pivot", entities[0]),
                    )
                )
                budget_left -= 1
            if entities and exa is not None and exa.configured():
                jobs.append(("exa", lambda: exa.search(entities, discovered_at=discovered_at)))
            if entities and wikidata is not None:
                jobs.append(
                    ("wikidata", lambda: wikidata.search(entities, discovered_at=discovered_at))
                )

            for report in run_providers(jobs, timeout=provider_timeout):
                source_reports.append(report.summary())
                if report.ok and report.candidates:
                    routes_run.append(report.name)
                    candidates.extend(report.candidates)
                    if report.raw:
                        raw_responses[report.name] = report.raw

        # Harvest. Non-social pages, the personal sites and writeups, carry several photos
        # of a person while the search returned at most one of them. A subject's own
        # portfolio held four pictures of them and the provider surfaced none.
        harvest_pages = [
            str(c.source_url)
            for c in merge_candidates(candidates)
            if not c.is_social and not str(c.source_url).startswith("data:")
        ]
        if harvest_pages:
            harvested = PageHarvestProvider().harvest(harvest_pages, discovered_at=discovered_at)
            source_reports.append(harvested.summary())
            if harvested.candidates:
                routes_run.append(harvested.name)
                candidates.extend(harvested.candidates)
    except WebSearchError:
        if not candidates:
            raise
        logger.warning("a search route failed; continuing with %d candidate(s)", len(candidates))

    ranked = merge_candidates(candidates)[:max_results]
    if not ranked:
        raise WebSearchError(
            PipelineErrorCode.SEARCH_EMPTY,
            f"no web result carrying a checkable image across {len(routes_run)} route(s): "
            f"{', '.join(routes_run)}",
        )

    return {
        "discovered_at": discovered_at.isoformat(),
        "routes_run": routes_run,
        "entities_inferred": entities,
        "candidates": ranked,
        "search_records": list(active.search_records),
        "raw_responses": raw_responses,
        "budget": active.budget.summary(),
        "cache": active.cache.stats,
        "sources": source_reports,
    }


__all__ = [
    "LENS_RESULT_KEYS",
    "discover",
    "infer_entities",
    "merge_candidates",
    "parse_candidates",
]
