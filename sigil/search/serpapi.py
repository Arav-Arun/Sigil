"""SerpApi client: image upload, Google Lens, and site-restricted web search.

Every call here costs real quota, so the client is built around not wasting it:

* one image upload per run, reused across every Lens route (``image_id`` is valid for
  ten minutes);
* a content-addressed cache consulted before any live call;
* a hard per-run budget that raises instead of over-spending;
* error classification that distinguishes "no results" (a legitimate answer) from
  "provider unavailable" (a failure), because conflating them is how a pipeline ends up
  reporting a fabricated success.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter

from sigil.config import Settings, get_settings
from sigil.imaging import encode_for_serpapi, load_validated_image, sha256_bytes
from sigil.models import PipelineErrorCode
from sigil.search.cache import SearchCache, cache_key
from sigil.search.quota import ACCOUNT_ENDPOINT, AccountStatus, QuotaError, SearchBudget

logger = logging.getLogger(__name__)

IMAGE_UPLOAD_ENDPOINT = "https://serpapi.com/image"
SEARCH_ENDPOINT = "https://serpapi.com/search.json"
USER_AGENT = "Sigil/0.3 (+https://github.com/Arav-Arun/HHgoa-FaceID)"

# SerpApi invalidates uploaded images after ten minutes. Refresh a little early so a slow
# run never fails on a boundary.
IMAGE_ID_TTL_SECONDS = 8 * 60

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


# Parameters that must not enter the cache key.
#
# ``image_id`` is the killer: SerpApi mints a new one for every upload, so including it
# meant the key changed on every run and the cache could never hit. The image *content*
# is already the key material, which is the thing that actually determines the result.
# ``no_cache`` is a hint to SerpApi's own cache, and ``api_key`` is a credential that has
# no business being hashed into a filename.
VOLATILE_CACHE_PARAMS = frozenset({"api_key", "image_id", "no_cache"})


class WebSearchError(RuntimeError):
    """A search failed. Carries a stable code so callers can react appropriately."""

    def __init__(self, code: PipelineErrorCode, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def build_http_session() -> requests.Session:
    """A plain session. Retries are handled explicitly so each one can be logged."""

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    adapter = HTTPAdapter(pool_connections=4, pool_maxsize=8, max_retries=0)
    session.mount("https://", adapter)
    return session


@dataclass(frozen=True, slots=True)
class UploadedImage:
    """A SerpApi upload identifier paired with the digest of the bytes it holds.

    The digest travels with the identifier deliberately. The cache key is derived from
    image *content*, and reading that content digest off mutable client state broke as
    soon as two routes uploaded two different images at the same time: one route could
    key its response by the other route's image.
    """

    image_id: str
    sha256: str


class SerpApiClient:
    """Quota-aware SerpApi client. Safe to call from several threads at once."""

    def __init__(
        self,
        api_key: str,
        *,
        timeout: float = 20.0,
        retries: int = 3,
        session: requests.Session | None = None,
        cache: SearchCache | None = None,
        budget: SearchBudget | None = None,
    ) -> None:
        if not api_key:
            raise WebSearchError(
                PipelineErrorCode.INVALID_CONFIGURATION, "SERPAPI_KEY is not configured"
            )
        self._api_key = api_key
        self._timeout = timeout
        self._retries = retries
        self._session = session or build_http_session()
        self.cache = cache or SearchCache()
        self.budget = budget or SearchBudget()
        self._image_id: str | None = None
        self._image_id_at: float = 0.0
        self._image_sha256: str = ""
        self._lock = threading.Lock()
        self.search_records: list[dict[str, Any]] = []

    # -- low-level request handling ---------------------------------------------

    def _request(
        self,
        method: str,
        url: str,
        *,
        what: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Issue one request with capped exponential backoff and honest error mapping."""

        last_error = ""
        for attempt in range(self._retries + 1):
            try:
                response = self._session.request(method, url, timeout=self._timeout, **kwargs)
            except requests.Timeout as exc:
                last_error = f"timeout after {self._timeout}s: {exc}"
            except requests.RequestException as exc:
                last_error = f"transport error: {exc}"
            else:
                if response.status_code in RETRYABLE_STATUS:
                    last_error = f"HTTP {response.status_code}"
                    if response.status_code == 429:
                        # Respect the server's pacing when it tells us one.
                        retry_after = response.headers.get("Retry-After")
                        if retry_after and retry_after.isdigit():
                            self._sleep(float(retry_after), attempt, forced=True)
                            continue
                elif response.status_code == 401:
                    raise WebSearchError(
                        PipelineErrorCode.INVALID_CONFIGURATION,
                        "SerpApi rejected the API key (HTTP 401)",
                    )
                elif response.status_code in (402, 403):
                    raise WebSearchError(
                        PipelineErrorCode.SEARCH_UNAVAILABLE,
                        f"SerpApi account cannot serve this request (HTTP {response.status_code}) "
                        "- usually an exhausted plan",
                    )
                else:
                    try:
                        payload: dict[str, Any] = response.json()
                    except ValueError as exc:
                        raise WebSearchError(
                            PipelineErrorCode.SEARCH_UNAVAILABLE,
                            f"{what} returned non-JSON content: {exc}",
                        ) from exc

                    error = payload.get("error")
                    if error:
                        text = str(error)
                        # SerpApi reports "no results" as an error string. That is a
                        # legitimate empty answer, not a provider failure.
                        if "hasn't returned any results" in text or "no results" in text.lower():
                            return {"__empty__": True, "error": text}
                        raise WebSearchError(PipelineErrorCode.SEARCH_UNAVAILABLE, text)
                    response.raise_for_status()
                    return payload

            if attempt < self._retries:
                self._sleep(0.75 * (2**attempt), attempt)

        raise WebSearchError(
            PipelineErrorCode.SEARCH_UNAVAILABLE,
            f"{what} failed after {self._retries + 1} attempts ({last_error})",
        )

    @staticmethod
    def _sleep(base: float, attempt: int, *, forced: bool = False) -> None:
        # Full jitter: avoids synchronised retries when several routes back off together.
        delay = base if forced else random.uniform(0, base)
        logger.debug("backing off %.2fs (attempt %d)", delay, attempt + 1)
        time.sleep(min(delay, 20.0))

    # -- account ------------------------------------------------------------------

    def account(self) -> AccountStatus:
        """Read plan health. This endpoint does not consume search quota."""

        payload = self._request(
            "GET", ACCOUNT_ENDPOINT, what="account lookup", params={"api_key": self._api_key}
        )
        return AccountStatus.from_payload(payload)

    # -- image upload -------------------------------------------------------------

    def upload_image(self, image_path: str | Path, *, force: bool = False) -> UploadedImage:
        """Upload one query image and return its identifier alongside its digest."""

        validated = load_validated_image(image_path)
        encoded = encode_for_serpapi(validated.image)
        digest = sha256_bytes(encoded.data)

        fresh = (
            self._image_id is not None
            and self._image_sha256 == digest
            and (time.monotonic() - self._image_id_at) < IMAGE_ID_TTL_SECONDS
        )
        if fresh and not force:
            logger.debug("reusing SerpApi image_id for %s", digest[:12])
            return UploadedImage(image_id=str(self._image_id), sha256=digest)

        payload = self._request(
            "POST",
            IMAGE_UPLOAD_ENDPOINT,
            what="image upload",
            data={"api_key": self._api_key},
            files={"image": (encoded.filename, encoded.data, encoded.content_type)},
        )
        image_id = payload.get("image_id")
        if not isinstance(image_id, str) or not image_id:
            raise WebSearchError(
                PipelineErrorCode.SEARCH_UNAVAILABLE, "SerpApi upload returned no image_id"
            )

        self._image_id = image_id
        self._image_id_at = time.monotonic()
        self._image_sha256 = digest
        return UploadedImage(image_id=image_id, sha256=digest)

    # -- searches -----------------------------------------------------------------

    def _cached_search(
        self, route: str, params: dict[str, Any], key_material: str
    ) -> dict[str, Any]:
        """Run one search, consulting the cache and the budget first."""

        cacheable = {k: v for k, v in params.items() if k not in VOLATILE_CACHE_PARAMS}
        key = cache_key(key_material, route, cacheable)

        hit = self.cache.get(key)
        if hit is not None:
            with self._lock:
                self.budget.record_cache_hit()
            logger.info("search route %s served from cache", route)
            self._record(route, hit, from_cache=True)
            return hit

        with self._lock:
            self.budget.spend(route)
        payload = self._request("GET", SEARCH_ENDPOINT, what=f"search route {route}", params=params)
        if payload.get("__empty__"):
            payload = {"search_metadata": {"status": "Empty"}, "__empty__": True}
        self.cache.put(key, payload)
        self._record(route, payload, from_cache=False)
        return payload

    def _record(self, route: str, payload: dict[str, Any], *, from_cache: bool) -> None:
        """Keep an auditable trace of every search that contributed to a result."""

        metadata = payload.get("search_metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        self.search_records.append(
            {
                "route": route,
                "from_cache": from_cache,
                "search_id": metadata.get("id", ""),
                "status": metadata.get("status", ""),
                "processed_at": metadata.get("processed_at", ""),
                "total_time_taken": metadata.get("total_time_taken", 0),
                "observed_at": datetime.now(UTC).isoformat(),
            }
        )

    def lens(
        self,
        image: UploadedImage,
        *,
        route: str,
        search_type: str,
        country: str,
        language: str,
        no_cache: bool,
    ) -> dict[str, Any]:
        """Google Lens search over a previously uploaded image."""

        params = {
            "engine": "google_lens",
            "image_id": image.image_id,
            "type": search_type,
            "country": country,
            "hl": language,
            "safe": "active",
            "no_cache": str(no_cache).lower(),
            "api_key": self._api_key,
        }
        return self._cached_search(route, params, image.sha256)

    def web(
        self,
        query: str,
        *,
        route: str,
        country: str,
        language: str,
        num: int = 20,
    ) -> dict[str, Any]:
        """Site-restricted Google web search, used by the entity-pivot route."""

        params = {
            "engine": "google",
            "q": query,
            "gl": country,
            "hl": language,
            "num": num,
            "safe": "active",
            "api_key": self._api_key,
        }
        return self._cached_search(route, params, query)


def client_from_settings(
    settings: Settings | None = None,
    *,
    use_cache: bool = True,
    budget_limit: int = 4,
) -> SerpApiClient:
    resolved = settings or get_settings()
    resolved.require("search")
    return SerpApiClient(
        resolved.serpapi_key.get_secret_value(),
        timeout=resolved.http_timeout_seconds,
        retries=resolved.http_retries + 1,
        cache=SearchCache(enabled=use_cache),
        budget=SearchBudget(limit=budget_limit),
    )


__all__ = [
    "IMAGE_UPLOAD_ENDPOINT",
    "SEARCH_ENDPOINT",
    "QuotaError",
    "SerpApiClient",
    "WebSearchError",
    "build_http_session",
    "client_from_settings",
]
