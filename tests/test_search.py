"""Search normalization, result parsing, quota control, and caching.

No test here touches the network. The SerpApi client is exercised through a fake
transport so the retry, error-classification, and budget paths are covered without
consuming a single search from the free tier.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from sigil.models import PipelineErrorCode
from sigil.search.cache import SearchCache, cache_key
from sigil.search.normalize import (
    canonicalize_url,
    detect_platform,
    extract_post_id,
    is_candidate_url,
    is_public_http_url,
    is_social_url,
    normalized_hostname,
)
from sigil.search.quota import (
    DEFAULT_RESERVE,
    AccountStatus,
    QuotaError,
    SearchBudget,
    check_reserve,
)
from sigil.search.routes import (
    infer_entities,
    merge_candidates,
    parse_candidates,
    select_candidates,
)
from sigil.search.serpapi import UploadedImage

NOW = datetime(2026, 9, 2, 12, 0, tzinfo=UTC)


class TestSocialAllowlist:
    @pytest.mark.parametrize(
        "url",
        [
            "https://www.instagram.com/p/CxYz/",
            "https://instagram.com/p/CxYz/",
            "https://m.facebook.com/user/posts/123",
            "https://twitter.com/jack/status/20",
            "https://x.com/jack/status/20",
            "https://in.linkedin.com/posts/abc",
            "https://youtu.be/dQw4w9WgXcQ",
        ],
    )
    def test_accepts_real_social_urls(self, url):
        assert is_social_url(url)

    @pytest.mark.parametrize(
        "url",
        [
            # These are the attacks a substring match would fall for.
            "https://notinstagram.com/p/CxYz/",
            "https://instagram.com.evil.net/p/CxYz/",
            "https://evil.com/?u=instagram.com",
            "https://fakex.com/jack/status/20",
            "https://example.com/photo.jpg",
            "not-a-url",
            "",
        ],
    )
    def test_rejects_lookalikes_and_junk(self, url):
        assert not is_social_url(url)

    def test_hostname_normalization_strips_www_and_trailing_dot(self):
        assert normalized_hostname("https://www.instagram.com./p/A") == "instagram.com"


class TestCanonicalizeUrl:
    def test_maps_twitter_to_x(self):
        assert canonicalize_url("https://twitter.com/jack/status/20") == (
            "https://x.com/jack/status/20"
        )

    def test_strips_tracking_parameters(self):
        assert canonicalize_url("https://x.com/a/status/1?s=20&utm_source=news&t=abc") == (
            "https://x.com/a/status/1"
        )

    def test_keeps_identity_bearing_parameters(self):
        # Dropping ?v= would destroy the identity of a YouTube post.
        assert "v=dQw4w9WgXcQ" in canonicalize_url(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&si=xyz"
        )

    def test_removes_trailing_slash_and_lowercases_host(self):
        assert canonicalize_url("https://INSTAGRAM.com/p/CxYz/") == "https://instagram.com/p/CxYz"

    def test_drops_default_ports_but_keeps_others(self):
        assert canonicalize_url("https://x.com:443/a") == "https://x.com/a"
        assert canonicalize_url("https://x.com:8443/a") == "https://x.com:8443/a"

    def test_is_idempotent(self):
        once = canonicalize_url("https://twitter.com/a/status/1?s=1")
        assert canonicalize_url(once) == once

    def test_rejects_a_relative_url(self):
        with pytest.raises(ValueError):
            canonicalize_url("/p/CxYz")

    def test_rejects_a_non_http_scheme(self):
        with pytest.raises(ValueError):
            canonicalize_url("ftp://instagram.com/p/A")


class TestPublicFetchUrls:
    @pytest.mark.parametrize("url", ["https://example.com/a", "https://cdn.example.org/image.jpg"])
    def test_accepts_public_domains(self, url):
        assert is_public_http_url(url)

    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1/private",
            "http://169.254.169.254/latest/meta-data",
            "http://10.0.0.5/admin",
            "http://[::1]/private",
            "http://localhost:8420/",
            "https://user:pass@example.com/",
        ],
    )
    def test_rejects_private_or_credentialed_fetch_targets(self, url):
        assert not is_public_http_url(url)
        assert not is_candidate_url(url)


class TestPostId:
    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("https://instagram.com/p/CxYz123", "CxYz123"),
            ("https://instagram.com/reel/AbC_9", "AbC_9"),
            ("https://x.com/jack/status/20", "20"),
            ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtube.com/shorts/abc123", "abc123"),
            ("https://reddit.com/r/pics/comments/abc123/title", "abc123"),
            ("https://tiktok.com/@user/video/7123456", "7123456"),
        ],
    )
    def test_extracts_known_shapes(self, url, expected):
        assert extract_post_id(url) == expected

    def test_returns_empty_for_an_unknown_shape(self):
        assert extract_post_id("https://instagram.com/someuser") == ""


class TestPlatform:
    @pytest.mark.parametrize(
        ("url", "platform"),
        [
            ("https://twitter.com/a", "x"),
            ("https://x.com/a", "x"),
            ("https://youtu.be/a", "youtube"),
            ("https://instagram.com/p/a", "instagram"),
            # Not a shrug: a reviewer needs to know which site a candidate came from,
            # and "unknown" is less informative than the hostname it already had.
            ("https://example.com/a", "example.com"),
        ],
    )
    def test_identifies_the_platform(self, url, platform):
        assert detect_platform(url) == platform


# Every candidate needs a fetchable image; this stands in for one.
IMG = "https://cdn.example.com/t.jpg"


class TestParseCandidates:
    def test_reads_the_visual_matches_shape(self):
        payload = {
            "visual_matches": [
                {
                    "position": 1,
                    "title": "A post",
                    "link": "https://www.instagram.com/p/CxYz/",
                    "thumbnail": "https://cdn.example.com/t.jpg",
                }
            ]
        }
        found = parse_candidates(payload, route="R2:lens-visual-full", discovered_at=NOW)
        assert len(found) == 1
        assert str(found[0].source_url) == "https://instagram.com/p/CxYz"
        assert found[0].post_id == "CxYz"

    def test_reads_the_exact_matches_shape(self):
        # The original scaffold only read `visual_matches`, so this route silently
        # produced zero candidates.
        payload = {
            "exact_matches": [{"link": "https://x.com/a/status/1", "position": 2, "thumbnail": IMG}]
        }
        found = parse_candidates(payload, route="R1:lens-exact-full", discovered_at=NOW)
        assert len(found) == 1
        assert found[0].exact_match is True

    def test_reads_the_organic_results_shape(self):
        payload = {
            "organic_results": [
                {"link": "https://x.com/a/status/9", "position": 1, "thumbnail": IMG}
            ]
        }
        assert len(parse_candidates(payload, route="R4:entity-pivot", discovered_at=NOW)) == 1

    def test_keeps_non_social_results(self):
        # These used to be discarded before the face check ever saw them, which threw
        # away genuine matches on news sites while keeping face-free product listings
        # that happened to sit on an allowlisted domain. Identity is the face gate's
        # decision, so the URL filter only drops things that cannot be a page at all.
        payload = {
            "visual_matches": [
                {"link": "https://news.example.com/a", "position": 1, "thumbnail": IMG}
            ]
        }
        found = parse_candidates(payload, route="R2", discovered_at=NOW)
        assert len(found) == 1
        assert found[0].is_social is False
        assert found[0].platform == "news.example.com"

    def test_marks_social_results(self):
        payload = {
            "visual_matches": [
                {"link": "https://instagram.com/p/A", "position": 1, "thumbnail": IMG}
            ]
        }
        assert parse_candidates(payload, route="R2", discovered_at=NOW)[0].is_social is True

    def test_drops_results_with_no_image_to_check(self):
        # A page whose picture cannot be fetched cannot be face-checked, so keeping it
        # would only ever produce an unverifiable claim.
        payload = {"visual_matches": [{"link": "https://instagram.com/p/A", "position": 1}]}
        assert parse_candidates(payload, route="R2", discovered_at=NOW) == []

    def test_drops_stock_photo_agencies(self):
        payload = {
            "visual_matches": [
                {"link": "https://www.gettyimages.com/detail/1", "position": 1, "thumbnail": IMG}
            ]
        }
        assert parse_candidates(payload, route="R2", discovered_at=NOW) == []

    def test_survives_malformed_entries(self):
        payload = {
            "visual_matches": [
                None,
                "garbage",
                {},
                {"link": 42},
                {"link": "https://x.com/a/status/1", "thumbnail": IMG},
            ]
        }
        assert len(parse_candidates(payload, route="R2", discovered_at=NOW)) == 1

    def test_handles_a_response_with_no_results(self):
        assert parse_candidates({}, route="R2", discovered_at=NOW) == []


class TestMergeCandidates:
    def _candidate(self, url, rank, route, exact=False):
        return parse_candidates(
            {"visual_matches": [{"link": url, "position": rank, "thumbnail": IMG}]},
            route=route,
            discovered_at=NOW,
        )[0].model_copy(update={"exact_match": exact})

    def test_deduplicates_by_canonical_url(self):
        merged = merge_candidates(
            [
                self._candidate("https://twitter.com/a/status/1", 3, "R1"),
                self._candidate("https://x.com/a/status/1?s=20", 1, "R2"),
            ]
        )
        assert len(merged) == 1
        assert merged[0].search_routes == ["R1", "R2"]

    def test_keeps_the_best_rank_when_merging(self):
        merged = merge_candidates(
            [
                self._candidate("https://x.com/a/status/1", 7, "R1"),
                self._candidate("https://x.com/a/status/1", 2, "R2"),
            ]
        )
        assert merged[0].search_rank == 2

    def test_exact_match_flag_is_sticky(self):
        merged = merge_candidates(
            [
                self._candidate("https://x.com/a/status/1", 5, "R1", exact=True),
                self._candidate("https://x.com/a/status/1", 1, "R2", exact=False),
            ]
        )
        assert merged[0].exact_match is True

    def test_several_images_from_one_page_survive_deduplication(self):
        # A harvested page contributes one candidate per photograph, all sharing the page
        # URL. Keying deduplication on the URL alone kept only the first and threw away
        # the rest, which on a personal site is most of the evidence.
        from sigil.models import SearchCandidate

        page = "https://example.com/about"
        candidates = [
            SearchCandidate(
                source_url=page,
                platform="example.com",
                post_id=f"img{i}",
                image_url=f"https://example.com/{i}.jpg",
                search_rank=i + 1,
                discovered_at=NOW,
                search_routes=["page-harvest:example.com"],
            )
            for i in range(3)
        ]
        assert len(merge_candidates(candidates)) == 3

    def test_multi_route_agreement_outranks_a_single_route(self):
        merged = merge_candidates(
            [
                self._candidate("https://x.com/a/status/1", 9, "R1"),
                self._candidate("https://x.com/a/status/1", 9, "R2"),
                self._candidate("https://x.com/b/status/2", 1, "R1"),
            ]
        )
        assert str(merged[0].source_url) == "https://x.com/a/status/1"


class TestSelectCandidates:
    def _candidate(self, number: int, route: str, *, exact: bool = False):
        from sigil.models import SearchCandidate

        return SearchCandidate(
            source_url=f"https://x.com/person/status/{number}",
            platform="x",
            is_social=True,
            post_id=str(number),
            search_rank=number,
            exact_match=exact,
            discovered_at=NOW,
            search_routes=[route],
        )

    def test_a_noisy_full_image_route_cannot_crowd_out_face_results(self):
        candidates = [self._candidate(i, "R1:lens-all-full") for i in range(1, 21)]
        candidates += [self._candidate(100 + i, "R3:lens-all-face") for i in range(1, 5)]

        selected = select_candidates(candidates, 6)

        routes = [route for candidate in selected for route in candidate.search_routes]
        assert routes.count("R1:lens-all-full") == 3
        assert routes.count("R3:lens-all-face") == 3

    def test_an_exact_result_is_retained_but_does_not_take_every_slot(self):
        candidates = [
            self._candidate(1, "R1:lens-all-full", exact=True),
            *[self._candidate(i, "R1:lens-all-full") for i in range(2, 10)],
            self._candidate(100, "R3:lens-all-face"),
        ]

        selected = select_candidates(candidates, 3)

        assert selected[0].exact_match is True
        assert any("R3:lens-all-face" in candidate.search_routes for candidate in selected)


def test_discover_always_runs_portrait_and_face_routes_when_full_image_is_noisy(monkeypatch):
    from sigil.config import Settings
    from sigil.search.quota import SearchBudget
    from sigil.search.routes import discover

    class FakeCache:
        def __init__(self):
            self.stats = {}

    class FakeClient:
        def __init__(self):
            self.budget = SearchBudget(limit=4)
            self.search_records = []
            self.cache = FakeCache()
            self.routes = []

        def upload_image(self, path):
            return UploadedImage(str(path), str(path))

        def lens(self, uploaded, *, route, **kwargs):
            self.budget.spend(route)
            self.routes.append(route)
            offset = {"R1:lens-all-full": 0, "R2:lens-all-portrait": 100, "R3:lens-all-face": 200}[
                route
            ]
            count = 12 if route == "R1:lens-all-full" else 2
            return {
                "visual_matches": [
                    {
                        "position": i + 1,
                        "title": f"result {offset + i}",
                        "link": f"https://x.com/person/status/{offset + i}",
                        "thumbnail": f"https://img.example.com/{offset + i}.jpg",
                    }
                    for i in range(count)
                ]
            }

    client = FakeClient()
    result = discover(
        "whole.jpg",
        "portrait.jpg",
        "face.jpg",
        settings=Settings(_env_file=None),
        client=client,
        max_results=6,
    )

    assert set(client.routes) == {
        "R1:lens-all-full",
        "R2:lens-all-portrait",
        "R3:lens-all-face",
    }
    selected_routes = {route for item in result["candidates"] for route in item.search_routes}
    assert selected_routes == set(client.routes)


class TestInferEntities:
    def test_reads_a_knowledge_graph_title(self):
        assert infer_entities({"knowledge_graph": [{"title": "Ada Lovelace"}]}) == ["Ada Lovelace"]

    def test_accepts_a_dict_knowledge_graph(self):
        assert infer_entities({"knowledge_graph": {"title": "Ada Lovelace"}}) == ["Ada Lovelace"]

    def test_rejects_stock_photo_captions(self):
        assert infer_entities({"knowledge_graph": [{"title": "Getty Images Stock"}]}) == []

    def test_rejects_single_words_and_long_phrases(self):
        payload = {"knowledge_graph": [{"title": "Ada"}, {"title": " ".join(["Word"] * 8)}]}
        assert infer_entities(payload) == []

    def test_requires_capitalised_words(self):
        assert infer_entities({"knowledge_graph": [{"title": "a person somewhere"}]}) == []

    def test_deduplicates_and_limits(self):
        payload = {"knowledge_graph": [{"title": "Ada Lovelace"}] * 5}
        assert infer_entities(payload, limit=2) == ["Ada Lovelace"]


class TestSearchBudget:
    def test_allows_spending_up_to_the_limit(self):
        budget = SearchBudget(limit=2)
        budget.spend("R1")
        budget.spend("R2")
        assert budget.spent == 2
        assert budget.remaining == 0

    def test_raises_rather_than_silently_overspending(self):
        budget = SearchBudget(limit=1)
        budget.spend("R1")
        with pytest.raises(QuotaError, match="budget exhausted"):
            budget.spend("R2")

    def test_cache_hits_do_not_consume_budget(self):
        budget = SearchBudget(limit=1)
        budget.record_cache_hit()
        budget.record_cache_hit()
        assert budget.spent == 0
        assert budget.summary()["cached_searches"] == 2


class TestQuotaReserve:
    def test_allows_a_run_that_stays_above_the_reserve(self):
        status = AccountStatus("Free", 100, 100, 0)
        check_reserve(status, needed=4, reserve=DEFAULT_RESERVE)

    def test_blocks_a_run_that_would_eat_the_demo_reserve(self):
        status = AccountStatus("Free", 16, 16, 84)
        with pytest.raises(QuotaError, match="reserved for the final demo"):
            check_reserve(status, needed=4, reserve=DEFAULT_RESERVE)

    def test_parses_a_partial_account_payload(self):
        status = AccountStatus.from_payload({"plan_searches_left": "7"})
        assert status.searches_left == 7
        assert status.healthy

    def test_treats_a_missing_field_as_zero(self):
        assert AccountStatus.from_payload({}).searches_left == 0


class TestSearchCache:
    def test_round_trips_a_payload(self, tmp_path):
        cache = SearchCache(directory=tmp_path)
        cache.put("k", {"a": 1})
        assert cache.get("k") == {"a": 1}
        assert cache.hits == 1

    def test_misses_an_absent_key(self, tmp_path):
        cache = SearchCache(directory=tmp_path)
        assert cache.get("absent") is None
        assert cache.misses == 1

    def test_disabled_cache_never_serves_or_stores(self, tmp_path):
        cache = SearchCache(enabled=False, directory=tmp_path)
        cache.put("k", {"a": 1})
        assert cache.get("k") is None
        assert not list(tmp_path.iterdir())

    def test_expires_entries_past_the_ttl(self, tmp_path):
        cache = SearchCache(directory=tmp_path, ttl_seconds=0)
        cache.put("k", {"a": 1})
        assert cache.get("k") is None

    def test_survives_a_corrupt_entry(self, tmp_path):
        cache = SearchCache(directory=tmp_path)
        (tmp_path / "bad.json").write_text("{not json")
        assert cache.get("bad") is None

    def test_key_depends_on_image_route_and_params(self):
        base = cache_key("sha", "R1", {"a": 1})
        assert base != cache_key("other", "R1", {"a": 1})
        assert base != cache_key("sha", "R2", {"a": 1})
        assert base != cache_key("sha", "R1", {"a": 2})

    def test_key_is_stable_across_param_ordering(self):
        assert cache_key("s", "R1", {"a": 1, "b": 2}) == cache_key("s", "R1", {"b": 2, "a": 1})


class TestSerpApiClient:
    """Error classification and retry behaviour, driven by a fake transport."""

    def _client(self, responses, **kwargs):
        import requests

        from sigil.search.serpapi import SerpApiClient

        class FakeSession(requests.Session):
            def __init__(self):
                super().__init__()
                self.calls = []

            def request(self, method, url, **rq):  # type: ignore[override]
                self.calls.append((method, url))
                return responses.pop(0)

        session = FakeSession()
        return SerpApiClient("key", session=session, **kwargs), session

    def _response(self, status=200, payload=None, headers=None):
        import requests

        response = requests.Response()
        response.status_code = status
        response.headers.update(headers or {})
        response._content = __import__("json").dumps(payload or {}).encode()
        return response

    def test_rejects_an_empty_api_key(self):
        from sigil.search.serpapi import SerpApiClient, WebSearchError

        with pytest.raises(WebSearchError) as info:
            SerpApiClient("")
        assert info.value.code is PipelineErrorCode.INVALID_CONFIGURATION

    def test_maps_401_to_a_configuration_error(self):
        from sigil.search.serpapi import WebSearchError

        client, _ = self._client([self._response(401)])
        with pytest.raises(WebSearchError) as info:
            client.account()
        assert info.value.code is PipelineErrorCode.INVALID_CONFIGURATION

    def test_maps_402_to_search_unavailable(self):
        from sigil.search.serpapi import WebSearchError

        client, _ = self._client([self._response(402)])
        with pytest.raises(WebSearchError) as info:
            client.account()
        assert info.value.code is PipelineErrorCode.SEARCH_UNAVAILABLE

    def test_cache_hits_across_runs_despite_a_fresh_image_id(self, tmp_path):
        # Regression. SerpApi mints a new image_id on every upload, so a cache key built
        # over the raw parameters changed on every run and the cache never hit: each
        # rehearsal silently burned live quota. The image bytes decide the result, so the
        # upload identifier must not decide the key.

        payload = {"visual_matches": [], "search_metadata": {"id": "x", "status": "Success"}}
        client, session = self._client(
            [self._response(200, payload)], cache=SearchCache(directory=tmp_path)
        )

        def lens(image_id: str):
            return client.lens(
                UploadedImage(image_id, "deadbeef"),
                route="R1:lens-exact-full",
                search_type="exact_matches",
                country="us",
                language="en",
                no_cache=False,
            )

        first = lens("upload-id-one")
        second = lens("upload-id-two")

        assert first == second
        assert len(session.calls) == 1, "the second search must not reach the network"
        assert client.budget.spent == 1
        assert client.budget.cached == 1

    def test_no_cache_flag_does_not_fragment_the_cache(self, tmp_path):
        # --no-cache asks SerpApi to bypass *its* cache. Letting it change our key would
        # mean a live rehearsal's result could never serve the run that follows it.

        payload = {"visual_matches": [], "search_metadata": {"id": "x", "status": "Success"}}
        client, session = self._client(
            [self._response(200, payload)], cache=SearchCache(directory=tmp_path)
        )

        for flag in (True, False):
            client.lens(
                UploadedImage("same-id", "deadbeef"),
                route="R2:lens-visual-full",
                search_type="visual_matches",
                country="us",
                language="en",
                no_cache=flag,
            )
        assert len(session.calls) == 1
        assert client.budget.cached == 1

    def test_treats_no_results_as_empty_not_as_failure(self):
        # SerpApi reports "no results" via the error field. Conflating that with an
        # outage is how a pipeline ends up claiming a fabricated success.
        client, _ = self._client(
            [self._response(200, {"error": "Google hasn't returned any results"})]
        )
        assert client.account().searches_left == 0

    def test_retries_a_500_then_succeeds(self):
        client, session = self._client(
            [self._response(500), self._response(200, {"plan_searches_left": 42})], retries=2
        )
        assert client.account().searches_left == 42
        assert len(session.calls) == 2

    def test_gives_up_after_exhausting_retries(self):
        from sigil.search.serpapi import WebSearchError

        client, _ = self._client([self._response(503)] * 3, retries=2)
        with pytest.raises(WebSearchError) as info:
            client.account()
        assert info.value.code is PipelineErrorCode.SEARCH_UNAVAILABLE

    def test_a_cached_route_does_not_spend_budget(self, tmp_path):
        from sigil.search.cache import SearchCache

        client, session = self._client(
            [self._response(200, {"visual_matches": [], "search_metadata": {"id": "x"}})],
            cache=SearchCache(directory=tmp_path),
            budget=SearchBudget(limit=5),
        )
        client.lens(
            UploadedImage("img", "deadbeef"),
            route="R2",
            search_type="visual_matches",
            country="in",
            language="en",
            no_cache=False,
        )
        client.lens(
            UploadedImage("img", "deadbeef"),
            route="R2",
            search_type="visual_matches",
            country="in",
            language="en",
            no_cache=False,
        )
        assert len(session.calls) == 1
        assert client.budget.spent == 1
        assert client.budget.cached == 1

    def test_records_an_audit_trail_for_every_route(self, tmp_path):
        from sigil.search.cache import SearchCache

        client, _ = self._client(
            [self._response(200, {"search_metadata": {"id": "abc123", "status": "Success"}})],
            cache=SearchCache(enabled=False, directory=tmp_path),
        )
        client.lens(
            UploadedImage("img", "deadbeef"),
            route="R1",
            search_type="exact_matches",
            country="in",
            language="en",
            no_cache=True,
        )
        assert client.search_records[0]["search_id"] == "abc123"
        assert client.search_records[0]["from_cache"] is False
