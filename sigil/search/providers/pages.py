"""Harvest every image on a page, not just the one the search engine chose.

A reverse-image result gives one picture per page: the thumbnail the provider decided was
the match. That is the right picture for a social post, which is about one image. It is
the wrong picture for a personal site, a portfolio, a team page or a conference writeup,
where the person appears in several photos and the provider picked at most one of them,
often the logo.

The gap this closes was found the hard way. A subject's own portfolio carried four
photographs of them; Google Lens returned none of them, and the pipeline reported nothing
found. Fetching that page and checking every image on it produced matches at distances of
0.24 to 0.52, well inside the gate. The recognition was never the weak link. Reaching the
page was.

Two rules keep this a search rather than a lookup:

* Pages come from a *search*, never from a hand-supplied URL. A name only decides where to
  look, exactly as the existing entity pivot does, and the face gate still decides who is
  in the picture.
* Every harvested image is an ordinary candidate. It is fetched, face-checked and ranked
  like any other, and is just as capable of being rejected.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from urllib.parse import urljoin, urlsplit

import requests

from sigil.models import SearchCandidate
from sigil.search.normalize import (
    canonicalize_url,
    detect_platform,
    is_public_http_url,
    is_social_url,
)
from sigil.search.providers.base import ProviderResult

logger = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (compatible; Sigil/0.3; +https://github.com/Arav-Arun/HHgoa-FaceID)"

MAX_PAGE_BYTES = 3_000_000
MAX_IMAGES_PER_PAGE = 12
MAX_PAGES = 4

# Images that are never a person. Filtering here is a latency decision, not an identity
# one: the face gate would reject them anyway, but each one costs a fetch and an inference.
_SKIP_PATTERN = re.compile(
    r"(logo|icon|favicon|sprite|banner|badge|avatar-default|placeholder|"
    r"qr[-_]|/tech/|\.svg($|\?))",
    re.I,
)

_IMG_SRC = re.compile(r"<img[^>]+?src=[\"']([^\"']+)", re.I)
_IMG_SRCSET = re.compile(r"<img[^>]+?srcset=[\"']([^\"']+)", re.I)
_META_IMAGE = re.compile(r"(?:og:image|twitter:image)[\"'][^>]*content=[\"']([^\"']+)", re.I)
# Modern front ends often keep image paths in JSON payloads rather than in <img> tags, so
# the portfolio that started this would have yielded almost nothing from markup alone.
_JSON_PATH = re.compile(r"[\"'](/[^\"']*?\.(?:jpe?g|png|webp))[\"']", re.I)


def extract_image_urls(html: str, base_url: str, limit: int = MAX_IMAGES_PER_PAGE) -> list[str]:
    """Pull plausible photo URLs out of a page, in a stable order."""

    found: list[str] = []
    seen: set[str] = set()

    def add(raw: str) -> None:
        candidate = raw.strip()
        if not candidate or candidate.startswith("data:"):
            return
        absolute = urljoin(base_url, candidate)
        if absolute in seen or _SKIP_PATTERN.search(absolute):
            return
        if not is_public_http_url(absolute):
            return
        seen.add(absolute)
        found.append(absolute)

    for match in _META_IMAGE.finditer(html):
        add(match.group(1))
    for match in _IMG_SRC.finditer(html):
        add(match.group(1))
    for match in _IMG_SRCSET.finditer(html):
        # "a.jpg 1x, b.jpg 2x" -> take the largest, which is the last entry.
        parts = [piece.strip().split(" ")[0] for piece in match.group(1).split(",")]
        if parts:
            add(parts[-1])
    for match in _JSON_PATH.finditer(html):
        add(match.group(1))

    return found[:limit]


class PageHarvestProvider:
    """Fetch pages a search returned and turn every photo on them into a candidate."""

    name = "page-harvest"

    def __init__(self, *, session: requests.Session | None = None, timeout: float = 12.0) -> None:
        self._session = session or requests.Session()
        self._session.headers["User-Agent"] = USER_AGENT
        self._timeout = timeout

    def configured(self) -> bool:
        return True  # no credential; it only reads pages another source already found

    def _fetch(self, url: str) -> str:
        target = url
        for _ in range(6):
            if not is_public_http_url(target):
                raise ValueError("refusing a non-public page URL")
            response = self._session.get(
                target, timeout=self._timeout, stream=True, allow_redirects=False
            )
            try:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if not location:
                        raise ValueError("redirect response has no Location header")
                    target = urljoin(response.url, location)
                    continue
                response.raise_for_status()
                if "html" not in response.headers.get("content-type", "").lower():
                    return ""
                body = response.raw.read(MAX_PAGE_BYTES + 1, decode_content=True) or b""
                if len(body) > MAX_PAGE_BYTES:
                    raise ValueError(f"page exceeded the {MAX_PAGE_BYTES}-byte limit")
                return body.decode(response.encoding or "utf-8", "replace")
            finally:
                response.close()
        raise ValueError("too many redirects")

    def harvest(
        self,
        page_urls: list[str],
        *,
        discovered_at: datetime,
        max_pages: int = MAX_PAGES,
    ) -> ProviderResult:
        """Turn the images on each page into candidates."""

        result = ProviderResult(name=self.name)
        for rank, page in enumerate(page_urls[:max_pages], start=1):
            try:
                html = self._fetch(page)
            except Exception as exc:
                logger.debug("page harvest skipped %s: %s", page, exc)
                continue
            if not html:
                continue

            images = extract_image_urls(html, page)
            result.raw[page] = {"images": len(images)}
            logger.info("page harvest %s: %d image(s)", page, len(images))

            for index, image_url in enumerate(images):
                try:
                    canonical = canonicalize_url(page)
                    result.candidates.append(
                        SearchCandidate(
                            source_url=canonical,
                            platform=detect_platform(canonical),
                            is_social=is_social_url(canonical),
                            # The page is one URL but yields many images, so the image
                            # itself has to distinguish them or dedupe collapses the lot.
                            post_id=f"img{index}",
                            title=f"image {index + 1} on {urlsplit(page).netloc}",
                            image_url=image_url,
                            thumbnail_url=None,
                            search_rank=rank * 100 + index,
                            exact_match=False,
                            discovered_at=discovered_at,
                            search_routes=[f"page-harvest:{urlsplit(page).netloc}"],
                            raw={"page": page, "image": image_url},
                        )
                    )
                except Exception as exc:
                    logger.debug("skipping harvested image %s: %s", image_url, exc)
        return result


__all__ = ["MAX_IMAGES_PER_PAGE", "MAX_PAGES", "PageHarvestProvider", "extract_image_urls"]
