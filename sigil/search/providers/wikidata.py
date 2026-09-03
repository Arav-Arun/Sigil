"""Wikidata and Wikimedia Commons as a structured discovery source.

Different in kind from a crawler: Wikidata is curated, so the portrait it returns for a
person is the *right* person by construction, and it is served at full resolution rather
than as a 100px avatar. That matters because most reverse-image results arrive as
thumbnails too small for a decisive face comparison.

It needs no API key and no account, which is why it runs even when nothing else is
configured. Its limitation is obvious and worth stating: it only knows people who have a
Wikidata entry, so it corroborates public figures and says nothing about anyone else.

A Wikidata portrait is not a social-media post and never satisfies the task on its own.
It is ranked below social results and, like every other candidate, is only ever admitted
by the face gate.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from urllib.parse import quote

import requests

from sigil.models import SearchCandidate
from sigil.search.providers.base import ProviderResult

logger = logging.getLogger(__name__)

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"

# Wikidata's own guidance: identify the tool and a contact, or expect to be throttled.
USER_AGENT = "Sigil/0.3 (face-evidence research tool; https://github.com/Arav-Arun/HHgoa-FaceID)"

# P18 is "image". P4862 and friends exist but P18 is the canonical portrait.
IMAGE_PROPERTY = "P18"

# Wide enough that a face inside it clears the recogniser's decisive-comparison floor.
THUMB_WIDTH = 800


class WikidataProvider:
    """Resolve an inferred name to a curated portrait."""

    name = "wikidata"

    def __init__(self, *, session: requests.Session | None = None, timeout: float = 8.0) -> None:
        self._session = session or requests.Session()
        # Not setdefault: requests.Session ships with its own User-Agent already set,
        # so setdefault is a no-op and Wikimedia answers 403 to the default agent.
        self._session.headers["User-Agent"] = USER_AGENT
        self._timeout = timeout

    def configured(self) -> bool:
        return True  # no credential, so it is always available

    def _get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        response = self._session.get(url, params=params, timeout=self._timeout)
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        return payload

    def _search_entity(self, name: str) -> str:
        payload = self._get(
            WIKIDATA_API,
            {
                "action": "wbsearchentities",
                "search": name,
                "language": "en",
                "uselang": "en",
                "type": "item",
                "limit": 1,
                "format": "json",
            },
        )
        hits = payload.get("search") or []
        return str(hits[0]["id"]) if hits and isinstance(hits[0], dict) else ""

    def _entity_image(self, entity_id: str) -> tuple[str, str]:
        """Return (commons file name, sitelink URL) for an entity, or empty strings."""

        payload = self._get(
            WIKIDATA_API,
            {
                "action": "wbgetentities",
                "ids": entity_id,
                "props": "claims|sitelinks/urls|labels",
                "languages": "en",
                "format": "json",
            },
        )
        entity = (payload.get("entities") or {}).get(entity_id) or {}
        claims = (entity.get("claims") or {}).get(IMAGE_PROPERTY) or []
        file_name = ""
        for claim in claims:
            value = ((claim.get("mainsnak") or {}).get("datavalue") or {}).get("value")
            if isinstance(value, str) and value:
                file_name = value
                break

        sitelinks = entity.get("sitelinks") or {}
        article = (sitelinks.get("enwiki") or {}).get("url") or ""
        return file_name, str(article)

    def _commons_thumbnail(self, file_name: str) -> str:
        payload = self._get(
            COMMONS_API,
            {
                "action": "query",
                "titles": f"File:{file_name}",
                "prop": "imageinfo",
                "iiprop": "url",
                "iiurlwidth": THUMB_WIDTH,
                "format": "json",
            },
        )
        pages = (payload.get("query") or {}).get("pages") or {}
        for page in pages.values():
            info = (page.get("imageinfo") or [{}])[0]
            url = info.get("thumburl") or info.get("url")
            if isinstance(url, str) and url:
                return url
        return ""

    def search(self, entities: list[str], *, discovered_at: datetime) -> ProviderResult:
        result = ProviderResult(name=self.name)
        for rank, entity in enumerate(entities[:2], start=1):
            entity_id = self._search_entity(entity)
            if not entity_id:
                continue
            file_name, article = self._entity_image(entity_id)
            if not file_name:
                continue
            image_url = self._commons_thumbnail(file_name)
            if not image_url:
                continue

            page = article or f"https://www.wikidata.org/wiki/{entity_id}"
            result.raw[entity] = {"entity_id": entity_id, "file": file_name, "page": page}
            result.candidates.append(
                SearchCandidate(
                    source_url=page,
                    platform="wikipedia" if article else "wikidata",
                    is_social=False,
                    post_id=entity_id,
                    title=f"{entity} ({entity_id})",
                    image_url=image_url,
                    thumbnail_url=None,
                    search_rank=rank,
                    exact_match=False,
                    discovered_at=discovered_at,
                    search_routes=[f"wikidata:{entity_id}"],
                    raw={"commons_file": file_name, "url": quote(page, safe=":/")},
                )
            )
        return result


__all__ = ["WikidataProvider"]
