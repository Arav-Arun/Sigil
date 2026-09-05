"""Genuine live discovery of social-media posts matching a face."""

from sigil.search.cache import SearchCache, cache_key
from sigil.search.normalize import (
    canonicalize_url,
    detect_platform,
    extract_post_id,
    is_social_url,
    normalized_hostname,
)
from sigil.search.providers.serpapi import SerpApiClient, WebSearchError, client_from_settings
from sigil.search.quota import AccountStatus, QuotaError, SearchBudget, check_reserve
from sigil.search.routes import (
    discover,
    infer_entities,
    merge_candidates,
    parse_candidates,
)

__all__ = [
    "AccountStatus",
    "QuotaError",
    "SearchBudget",
    "SearchCache",
    "SerpApiClient",
    "WebSearchError",
    "cache_key",
    "canonicalize_url",
    "check_reserve",
    "client_from_settings",
    "detect_platform",
    "discover",
    "extract_post_id",
    "infer_entities",
    "is_social_url",
    "merge_candidates",
    "normalized_hostname",
    "parse_candidates",
]
