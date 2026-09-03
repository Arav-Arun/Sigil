"""Exa as a neural discovery source.

Exa retrieves by meaning rather than by keyword or by pixel, which makes it complementary
to the two failure modes that dominate here. Google Lens finds the *photo* and misses the
person; a keyword search finds the *name* and drowns in namesakes. Exa is asked for pages
that are about this person on a social platform, and it returns pages that read that way.

Two capabilities are used:

``/search``  neural retrieval, optionally restricted to the social allowlist.

``/findSimilar`` was tried here and removed. Asked for pages like a YouTube watch URL it
returned a property listing, a git repository and a video site, because a watch page
carries almost no crawlable text to be similar *to*. The expansion round in
:mod:`sigil.run` searches a face-confirmed name instead, which works.

Exa also returns page images alongside results, so candidates arrive with media already
attached instead of needing a second fetch to discover an ``og:image``. That is a latency
win, not just a convenience.

Unconfigured is a normal state: without ``EXA_API_KEY`` the source reports itself
unavailable and the fan-out proceeds without it.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import requests

from sigil.config import SOCIAL_DOMAINS
from sigil.models import SearchCandidate
from sigil.search.normalize import (
    canonicalize_url,
    detect_platform,
    extract_post_id,
    is_candidate_url,
    is_social_url,
)
from sigil.search.providers.base import ProviderResult

logger = logging.getLogger(__name__)

EXA_SEARCH = "https://api.exa.ai/search"
EXA_SIMILAR = "https://api.exa.ai/findSimilar"

DEFAULT_RESULTS = 15


class ExaProvider:
    """Neural web retrieval over the social allowlist."""

    name = "exa"

    def __init__(
        self,
        api_key: str = "",
        *,
        session: requests.Session | None = None,
        timeout: float = 15.0,
    ) -> None:
        self._api_key = api_key
        self._session = session or requests.Session()
        self._timeout = timeout

    def configured(self) -> bool:
        return bool(self._api_key)

    def _post(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        response = self._session.post(
            url,
            json=body,
            headers={"x-api-key": self._api_key, "content-type": "application/json"},
            timeout=self._timeout,
        )
        if response.status_code == 401:
            raise RuntimeError("EXA_API_KEY was rejected")
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        return payload

    def _to_candidates(
        self, payload: dict[str, Any], *, route: str, discovered_at: datetime
    ) -> list[SearchCandidate]:
        candidates: list[SearchCandidate] = []
        for rank, item in enumerate(payload.get("results") or [], start=1):
            if not isinstance(item, dict):
                continue
            link = item.get("url")
            if not isinstance(link, str) or not is_candidate_url(link):
                continue
            # `.get("imageLinks", [None])[0]` looks safe and is not: when the key exists
            # but holds an empty list, the default never applies and the subscript raises.
            # Exa returns exactly that for pages it found no image on, which is most of
            # them, so the whole expansion round died on its first result.
            links = (item.get("extras") or {}).get("imageLinks") or []
            image = item.get("image") or (links[0] if links else None)
            if not image:
                # Nothing to face-check. Exa found a page, but a page is not evidence.
                continue
            try:
                canonical = canonicalize_url(link)
            except (TypeError, ValueError):
                continue
            candidates.append(
                SearchCandidate(
                    source_url=canonical,
                    platform=detect_platform(canonical),
                    is_social=is_social_url(canonical),
                    post_id=extract_post_id(canonical),
                    title=str(item.get("title") or "")[:500],
                    image_url=image,
                    thumbnail_url=None,
                    search_rank=rank,
                    exact_match=False,
                    discovered_at=discovered_at,
                    search_routes=[route],
                    raw={"score": item.get("score"), "published": item.get("publishedDate")},
                )
            )
        return candidates

    def search(
        self,
        entities: list[str],
        *,
        discovered_at: datetime,
        limit: int = DEFAULT_RESULTS,
    ) -> ProviderResult:
        result = ProviderResult(name=self.name)
        if not entities:
            return result

        entity = entities[0]
        body = {
            "query": f"{entity} social media profile or post photo",
            "type": "auto",
            "numResults": limit,
            "includeDomains": list(SOCIAL_DOMAINS),
            "contents": {"text": False, "extras": {"imageLinks": 1}},
        }
        payload = self._post(EXA_SEARCH, body)
        result.raw["search"] = {
            "query": body["query"],
            "results": len(payload.get("results") or []),
        }
        result.candidates = self._to_candidates(
            payload, route=f"exa:search:{entity}", discovered_at=discovered_at
        )
        return result


__all__ = ["ExaProvider"]
