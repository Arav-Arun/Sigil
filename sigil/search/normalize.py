"""URL canonicalization and social-platform identification.

Substring matching on hostnames is a real vulnerability, not a style problem:
``notinstagram.com`` and ``instagram.com.evil.net`` both contain ``instagram.com``.
Everything here parses the URL and compares normalized hostnames.
"""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from sigil.config import SOCIAL_DOMAINS

TRACKING_PARAMETERS = frozenset(
    {"fbclid", "gclid", "igshid", "ref_src", "ref_url", "si", "s", "t", "share_id", "mibextid"}
)

PLATFORM_ALIASES = {
    "twitter.com": "x",
    "x.com": "x",
    "youtu.be": "youtube",
    "youtube.com": "youtube",
    "threads.net": "threads",
}

# Post-identifier extraction per platform. Used for deduplication and for binding a
# stable identity into the evidence manifest.
POST_ID_PATTERNS = {
    "instagram": re.compile(r"^/(?:p|reel|tv)/([A-Za-z0-9_-]+)"),
    "x": re.compile(r"^/[^/]+/status/(\d+)"),
    "facebook": re.compile(r"/(?:posts|videos|photos)/(?:[^/]+/)?(\d+)"),
    "tiktok": re.compile(r"^/@[^/]+/video/(\d+)"),
    "reddit": re.compile(r"^/r/[^/]+/comments/([a-z0-9]+)"),
    "youtube": re.compile(r"^/(?:watch|shorts/|embed/)?"),
    "linkedin": re.compile(r"^/(?:posts|feed/update)/([A-Za-z0-9_:%-]+)"),
    "threads": re.compile(r"^/@[^/]+/post/([A-Za-z0-9_-]+)"),
}


def normalized_hostname(url: str) -> str:
    """Lowercased hostname with a trailing dot and a leading ``www.`` removed."""

    try:
        hostname = (urlsplit(url).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""
    return hostname.removeprefix("www.")


def is_social_url(url: str, domains: tuple[str, ...] = SOCIAL_DOMAINS) -> bool:
    """True when the parsed hostname is, or is a subdomain of, an allowlisted domain."""

    hostname = normalized_hostname(url)
    if not hostname:
        return False
    return any(hostname == domain or hostname.endswith(f".{domain}") for domain in domains)


def detect_platform(url: str) -> str:
    """A short label for where a candidate lives.

    Social platforms get their canonical name. Everything else gets its hostname rather
    than the word "unknown": a result on ``news.bbc.co.uk`` is not unknown, and telling a
    reviewer which site a candidate came from is the whole point of the column.
    """

    hostname = normalized_hostname(url)
    for domain in SOCIAL_DOMAINS:
        if hostname == domain or hostname.endswith(f".{domain}"):
            return PLATFORM_ALIASES.get(domain, domain.split(".")[0])
    return hostname or "unknown"


# Hosts that serve images but never a page about a person. Excluding them is a *latency*
# decision, not an identity one: the face gate would reject them anyway, and every slot
# they occupy is a real candidate that never gets fetched.
NON_PROFILE_HOSTS = frozenset(
    {
        # Lens returns its own redirector for video results. It is not a page.
        "google.com",
        "gettyimages.com",
        "shutterstock.com",
        "alamy.com",
        "istockphoto.com",
        "dreamstime.com",
        "123rf.com",
        "depositphotos.com",
        "stock.adobe.com",
    }
)


def is_candidate_url(url: str) -> bool:
    """True when a result is a web page worth fetching and face-checking.

    Deliberately permissive. An earlier version accepted *only* the social allowlist,
    which discarded every result before the face check had a chance to look at it: a
    genuine match on a news site was thrown away while a face-free product listing on an
    allowlisted domain would have been kept. Identity is decided by the face gate, so the
    URL filter's job is only to drop things that cannot be a page about a person.
    """

    if not is_public_http_url(url):
        return False
    hostname = normalized_hostname(url)
    return not any(hostname == host or hostname.endswith(f".{host}") for host in NON_PROFILE_HOSTS)


def is_public_http_url(url: str) -> bool:
    """Accept a normal public HTTP(S) URL, never an obvious local network target.

    Search responses are untrusted input. Rejecting loopback, link-local and private
    literal addresses keeps them from turning the media fetcher into a probe of the
    machine or cloud metadata service. Hostname resolution is deliberately left to the
    HTTP client because requiring DNS here would make offline parsing and mocked tests
    brittle; redirects are checked again by the fetchers.
    """

    try:
        split = urlsplit(url.strip())
    except ValueError:
        return False
    if split.scheme.lower() not in {"http", "https"} or split.username or split.password:
        return False
    hostname = normalized_hostname(url)
    if not hostname or hostname == "localhost" or hostname.endswith(".localhost"):
        return False
    try:
        return ipaddress.ip_address(hostname).is_global
    except ValueError:
        # Domain names need a dot so bare local host aliases cannot pass.
        return "." in hostname


def extract_post_id(url: str) -> str:
    """Best-effort stable post identifier; empty string when the shape is unknown."""

    split = urlsplit(url)
    platform = detect_platform(url)
    if platform == "youtube":
        host = normalized_hostname(url)
        if host == "youtu.be":
            return split.path.lstrip("/")
        query = dict(parse_qsl(split.query))
        if "v" in query:
            return query["v"]
        for prefix in ("/shorts/", "/embed/"):
            if split.path.startswith(prefix):
                return split.path[len(prefix) :].split("/")[0]
        return ""
    pattern = POST_ID_PATTERNS.get(platform)
    if pattern is None:
        return ""
    match = pattern.search(split.path)
    return match.group(1) if match and match.groups() else ""


def canonicalize_url(url: str) -> str:
    """Normalize a URL without discarding parameters that identify the post.

    ``v`` on YouTube is the post identity, so a blanket parameter strip would be wrong;
    only known tracking parameters and ``utm_*`` are removed.
    """

    split = urlsplit(url.strip())
    if split.scheme.lower() not in {"http", "https"} or not split.hostname:
        raise ValueError("expected an absolute HTTP(S) URL")

    hostname = split.hostname.lower().rstrip(".").removeprefix("www.")
    if hostname == "twitter.com" or hostname.endswith(".twitter.com"):
        hostname = "x.com"
    if hostname.startswith("m.") and hostname.count(".") >= 2:
        hostname = hostname[2:]

    scheme = split.scheme.lower()
    port = split.port
    default_port = (scheme == "https" and port == 443) or (scheme == "http" and port == 80)
    netloc = f"{hostname}:{port}" if port is not None and not default_port else hostname

    path = split.path.rstrip("/") or "/"
    query = urlencode(
        sorted(
            (key, value)
            for key, value in parse_qsl(split.query, keep_blank_values=True)
            if key.lower() not in TRACKING_PARAMETERS and not key.lower().startswith("utm_")
        ),
        doseq=True,
    )
    return urlunsplit((scheme, netloc, path, query, ""))


__all__ = [
    "NON_PROFILE_HOSTS",
    "PLATFORM_ALIASES",
    "TRACKING_PARAMETERS",
    "canonicalize_url",
    "detect_platform",
    "extract_post_id",
    "is_candidate_url",
    "is_public_http_url",
    "is_social_url",
    "normalized_hostname",
]
