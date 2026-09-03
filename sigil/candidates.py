"""Fetch the media behind each discovered post.

A search result is a URL, not evidence. Before a candidate can be face-verified, its
media has to be retrieved under controlled conditions, bounded size, bounded redirects,
verified content type, and a digest captured *before* decoding so the bytes that were
hashed are exactly the bytes that were evaluated.

Social platforms are hostile to naive fetching: many post URLs are login-walled HTML
pages rather than images. The resolution ladder below tries the direct media first, then
OpenGraph/oEmbed metadata from the post page, then the thumbnail the search provider
returned, labelling each result with its evidence quality so a thumbnail-derived match
is never presented as if it came from the full-resolution original.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import httpx
from PIL import Image

from sigil.models import SearchCandidate

logger = logging.getLogger(__name__)

MAX_MEDIA_BYTES = 12 * 1024 * 1024
MAX_HTML_BYTES = 2 * 1024 * 1024
MAX_REDIRECTS = 5
IMAGE_CONTENT_TYPES = ("image/jpeg", "image/png", "image/webp", "image/gif", "image/avif")

BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 Sigil/0.3"
)

OG_IMAGE = re.compile(
    r'<meta[^>]+(?:property|name)=["\']og:image(?::secure_url)?["\'][^>]+content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
OG_IMAGE_REVERSED = re.compile(
    r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:property|name)=["\']og:image(?::secure_url)?["\']',
    re.IGNORECASE,
)
OEMBED_ENDPOINTS = {
    "youtube": "https://www.youtube.com/oembed?format=json&url={url}",
}


class MediaQuality(StrEnum):
    """How the media was obtained. Recorded in evidence; never silently upgraded."""

    ORIGINAL = "ORIGINAL"
    OPENGRAPH = "OPENGRAPH"
    OEMBED = "OEMBED"
    THUMBNAIL = "THUMBNAIL"


@dataclass(slots=True)
class FetchedMedia:
    """Retrieved bytes plus everything needed to audit the retrieval."""

    candidate: SearchCandidate
    data: bytes = b""
    sha256: str = ""
    content_type: str = ""
    final_url: str = ""
    status_code: int = 0
    byte_count: int = 0
    quality: MediaQuality = MediaQuality.THUMBNAIL
    retrieved_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.data) and not self.error

    def to_json(self) -> dict[str, Any]:
        return {
            "source_url": str(self.candidate.source_url),
            "media_url": self.final_url,
            "sha256": self.sha256,
            "content_type": self.content_type,
            "byte_count": self.byte_count,
            "http_status": self.status_code,
            "quality": str(self.quality),
            "retrieved_at": self.retrieved_at.isoformat(),
            "error": self.error,
        }


def _is_image(content_type: str) -> bool:
    return content_type.split(";")[0].strip().lower() in IMAGE_CONTENT_TYPES


async def _get(
    client: httpx.AsyncClient, url: str, *, limit: int, accept: str
) -> tuple[httpx.Response, bytes]:
    """Stream a response, aborting as soon as it exceeds the byte limit."""

    chunks: list[bytes] = []
    total = 0
    async with client.stream("GET", url, headers={"Accept": accept}) as response:
        declared = response.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > limit:
            raise ValueError(f"declared size {declared} exceeds the {limit}-byte limit")
        async for chunk in response.aiter_bytes():
            total += len(chunk)
            if total > limit:
                raise ValueError(f"response exceeded the {limit}-byte limit")
            chunks.append(chunk)
    return response, b"".join(chunks)


async def _resolve_via_page(client: httpx.AsyncClient, post_url: str, platform: str) -> str | None:
    """Find a media URL from the post page's OpenGraph tag or oEmbed endpoint."""

    endpoint = OEMBED_ENDPOINTS.get(platform)
    if endpoint:
        try:
            response = await client.get(endpoint.format(url=post_url))
            if response.status_code == 200:
                thumbnail = response.json().get("thumbnail_url")
                if isinstance(thumbnail, str) and thumbnail:
                    return thumbnail
        except (httpx.HTTPError, ValueError) as exc:
            logger.debug("oEmbed lookup failed for %s: %s", post_url, exc)

    try:
        response, body = await _get(client, post_url, limit=MAX_HTML_BYTES, accept="text/html")
    except (httpx.HTTPError, ValueError) as exc:
        logger.debug("page fetch failed for %s: %s", post_url, exc)
        return None

    if response.status_code != 200:
        return None

    html = body.decode("utf-8", errors="replace")
    for pattern in (OG_IMAGE, OG_IMAGE_REVERSED):
        match = pattern.search(html)
        if match:
            return match.group(1).replace("&amp;", "&")
    return None


# Below this, a fetched image is worth keeping but not worth stopping the ladder for: a
# face inside it lands under the recogniser's decisive-comparison floor, so the candidate
# would come back INCONCLUSIVE purely because of resolution.
DECISIVE_EDGE_PX = 256


def _smaller_edge(data: bytes) -> int:
    """Shorter edge of an encoded image, or 0 when it cannot be read."""

    try:
        with Image.open(io.BytesIO(data)) as image:
            return int(min(image.size))
    except Exception:  # Pillow raises a wide family of decode errors
        return 0


async def _fetch_one(
    client: httpx.AsyncClient,
    candidate: SearchCandidate,
    semaphore: asyncio.Semaphore,
) -> FetchedMedia:
    """Walk the resolution ladder for one candidate."""

    result = FetchedMedia(candidate=candidate)
    best: FetchedMedia | None = None
    best_edge = 0

    # Order matters, and it is not "largest first".
    #
    # The provider's own image and thumbnail depict *the thing that was matched*, so they
    # go first. A page's og:image is whatever that site chose to advertise, and whether
    # that is the same picture depends entirely on what kind of page it is:
    #
    #   instagram.com/p/DF5ifsizco2   a specific post; og:image IS the post's image
    #   in.pinterest.com/user/board   a board; og:image is the cover, unrelated
    #
    # Asking the second kind is how a search for a person returned a photograph of a lion.
    # `post_id` is empty for exactly the pages where the question is meaningless, so the
    # page rung is offered only when the URL names a specific post.
    #
    # It still matters: providers serve ~200px thumbnails, and half of one run's candidates
    # came back INCONCLUSIVE at 25-32px, undecidable on resolution alone, while the post's
    # own og:image was full size.
    provider_rungs: list[tuple[str, MediaQuality]] = []
    if candidate.image_url:
        provider_rungs.append((str(candidate.image_url), MediaQuality.ORIGINAL))
    if candidate.thumbnail_url:
        provider_rungs.append((str(candidate.thumbnail_url), MediaQuality.THUMBNAIL))
    ladder = list(provider_rungs)
    if candidate.post_id:
        ladder.append(("__page__", MediaQuality.OPENGRAPH))

    async with semaphore:
        for url, quality in ladder:
            target = url
            if url == "__page__":
                resolved = await _resolve_via_page(
                    client, str(candidate.source_url), candidate.platform
                )
                if not resolved:
                    continue
                target = resolved

            try:
                response, data = await _get(client, target, limit=MAX_MEDIA_BYTES, accept="image/*")
            except (httpx.HTTPError, ValueError) as exc:
                result.error = f"{type(exc).__name__}: {exc}"
                continue

            content_type = response.headers.get("content-type", "")
            if response.status_code != 200:
                result.error = f"HTTP {response.status_code} for {target}"
                continue
            if not _is_image(content_type):
                result.error = f"unexpected content-type {content_type!r} for {target}"
                continue
            if len(data) < 1024:
                result.error = f"media too small ({len(data)} bytes)"
                continue

            # Digest the exact bytes that were received, before any decoding.
            candidate_result = FetchedMedia(
                candidate=candidate,
                data=data,
                sha256=hashlib.sha256(data).hexdigest(),
                content_type=content_type.split(";")[0].strip(),
                final_url=str(response.url),
                status_code=response.status_code,
                byte_count=len(data),
                quality=quality,
            )

            edge = _smaller_edge(data)
            if edge >= DECISIVE_EDGE_PX:
                return candidate_result

            # Big enough to fetch, too small to decide on. The ladder used to stop at the
            # first success, so a 100px avatar ended the search and the candidate came back
            # INCONCLUSIVE ("closest face is only 41px") while a larger copy of the very
            # same photo sat one rung further down. Keep the best seen and keep climbing.
            if edge > best_edge:
                best_edge, best = edge, candidate_result
            logger.debug(
                "%s: %s copy is only %dpx, trying the next rung",
                candidate.source_url,
                quality,
                edge,
            )

    if best is not None:
        return best
    if not result.error:
        result.error = "no retrievable media for this candidate"
    return result


async def fetch_all(
    candidates: list[SearchCandidate],
    *,
    concurrency: int = 5,
    timeout: float = 15.0,
) -> list[FetchedMedia]:
    """Fetch every candidate's media concurrently, preserving input order."""

    if not candidates:
        return []

    semaphore = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(
        follow_redirects=True,
        max_redirects=MAX_REDIRECTS,
        timeout=httpx.Timeout(timeout, connect=min(timeout, 10.0)),
        limits=limits,
        headers={"User-Agent": BROWSER_UA},
        # No cookie jar: Sigil never authenticates to a platform or scrapes behind a login.
        cookies=None,
    ) as client:
        results = await asyncio.gather(
            *(_fetch_one(client, candidate, semaphore) for candidate in candidates)
        )
    return list(results)


def fetch_all_sync(
    candidates: list[SearchCandidate], *, concurrency: int = 5, timeout: float = 15.0
) -> list[FetchedMedia]:
    """Blocking wrapper for the synchronous pipeline."""

    return asyncio.run(fetch_all(candidates, concurrency=concurrency, timeout=timeout))


__all__ = [
    "MAX_MEDIA_BYTES",
    "FetchedMedia",
    "MediaQuality",
    "fetch_all",
    "fetch_all_sync",
]
