"""Content-addressed cache for search responses.

SerpApi's free tier allows 100 searches per month. Without a cache, every debugging run,
every test, and every rehearsal permanently consumes quota, and quota exhaustion two
hours before a deadline is an unrecoverable failure.

The cache key is derived from the *query image bytes* plus the route and parameters, so
re-running the same input is free while genuinely new inputs still go live. The recorded
demo passes ``--no-cache`` so the search shown on camera is unambiguously live.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 14 * 24 * 3600


def cache_dir() -> Path:
    override = os.environ.get("SIGIL_CACHE_DIR")
    base = Path(override).expanduser() if override else Path.home() / ".cache" / "sigil" / "search"
    base.mkdir(parents=True, exist_ok=True)
    return base


def cache_key(image_sha256: str, route: str, params: dict[str, Any]) -> str:
    """Stable key over the query image and every parameter that changes the result."""

    material = json.dumps(
        {"image": image_sha256, "route": route, "params": params},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class SearchCache:
    """A small on-disk JSON cache with a TTL.

    Deliberately not an LRU: entries are keyed by content, so a stale entry is only ever
    stale with respect to the live web, which the TTL handles.
    """

    enabled: bool = True
    ttl_seconds: int = DEFAULT_TTL_SECONDS
    directory: Path | None = None
    hits: int = 0
    misses: int = 0

    def _path(self, key: str) -> Path:
        base = self.directory or cache_dir()
        return base / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        path = self._path(key)
        if not path.is_file():
            self.misses += 1
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.misses += 1
            return None

        age = time.time() - float(record.get("cached_at", 0))
        if age > self.ttl_seconds:
            logger.debug("cache entry %s expired (%.0fs old)", key[:12], age)
            self.misses += 1
            return None

        self.hits += 1
        payload: dict[str, Any] = record["payload"]
        return payload

    def put(self, key: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        path = self._path(key)
        temporary = path.with_suffix(".tmp")
        try:
            temporary.write_text(
                json.dumps({"cached_at": time.time(), "payload": payload}),
                encoding="utf-8",
            )
            temporary.replace(path)
        except OSError as exc:  # pragma: no cover - disk problems are environmental
            logger.warning("could not write search cache entry: %s", exc)

    @property
    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}


__all__ = ["DEFAULT_TTL_SECONDS", "SearchCache", "cache_dir", "cache_key"]
