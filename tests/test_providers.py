"""Tests for the multi-source discovery layer.

The property that matters here is not that any one source works. It is that the fan-out
degrades: an unconfigured source is skipped, a failing source is recorded rather than
raising, and a slow source cannot decide the latency of the whole search.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

import pytest

from sigil.search.providers.base import ProviderResult, run_providers
from sigil.search.providers.exa import ExaProvider
from sigil.search.providers.wikidata import WikidataProvider

NOW = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)


class TestFanOut:
    def test_runs_every_job_and_collects_results(self):
        jobs = [
            ("a", lambda: ProviderResult(name="a")),
            ("b", lambda: ProviderResult(name="b")),
        ]
        assert {r.name for r in run_providers(jobs)} == {"a", "b"}

    def test_a_failing_source_is_recorded_not_raised(self):
        # One broken source must cost coverage, never the whole run.
        def boom() -> ProviderResult:
            raise RuntimeError("provider exploded")

        results = {
            r.name: r
            for r in run_providers([("ok", lambda: ProviderResult(name="ok")), ("bad", boom)])
        }
        assert results["ok"].ok
        assert not results["bad"].ok
        assert "provider exploded" in results["bad"].error

    def test_sources_run_concurrently_not_serially(self):
        def slow(name: str):
            def run() -> ProviderResult:
                time.sleep(0.4)
                return ProviderResult(name=name)

            return run

        started = time.perf_counter()
        results = run_providers([(n, slow(n)) for n in ("a", "b", "c")])
        elapsed = time.perf_counter() - started
        assert len(results) == 3
        assert elapsed < 0.9, f"three 0.4s sources took {elapsed:.2f}s, so they ran serially"

    def test_a_hanging_source_is_dropped_at_the_timeout(self):
        def hang() -> ProviderResult:
            time.sleep(5)
            return ProviderResult(name="slow")

        results = {
            r.name: r
            for r in run_providers(
                [("quick", lambda: ProviderResult(name="quick")), ("slow", hang)], timeout=0.5
            )
        }
        assert results["quick"].ok
        assert "timed out" in results["slow"].error

    def test_no_jobs_is_not_an_error(self):
        assert run_providers([]) == []

    def test_results_carry_timings(self):
        result = run_providers([("a", lambda: ProviderResult(name="a"))])[0]
        assert result.elapsed_ms >= 0
        assert result.summary()["source"] == "a"


class TestConfiguration:
    def test_exa_is_unavailable_without_a_key(self):
        assert not ExaProvider("").configured()

    def test_exa_is_available_with_a_key(self):
        assert ExaProvider("k").configured()

    def test_wikidata_needs_no_credential(self):
        # The point of keeping it: it still works when nothing else is configured.
        assert WikidataProvider().configured()


class TestExaParsing:
    def _provider(self) -> ExaProvider:
        return ExaProvider("k")

    def test_keeps_results_that_carry_an_image(self):
        payload = {
            "results": [
                {"url": "https://instagram.com/p/A", "title": "post", "image": "https://c/i.jpg"}
            ]
        }
        found = self._provider()._to_candidates(payload, route="exa", discovered_at=NOW)
        assert len(found) == 1
        assert found[0].is_social is True
        assert found[0].platform == "instagram"

    def test_drops_results_with_no_image(self):
        # Exa found a page, but a page with no picture cannot be face-checked.
        payload = {"results": [{"url": "https://instagram.com/p/A", "title": "post"}]}
        assert self._provider()._to_candidates(payload, route="exa", discovered_at=NOW) == []

    def test_reads_the_extras_image_links_shape(self):
        payload = {
            "results": [
                {
                    "url": "https://x.com/a/status/1",
                    "extras": {"imageLinks": ["https://c/i.jpg"]},
                }
            ]
        }
        assert len(self._provider()._to_candidates(payload, route="exa", discovered_at=NOW)) == 1

    def test_survives_an_empty_image_links_list(self):
        # Regression: `.get("imageLinks", [None])[0]` raises IndexError when the key is
        # present but empty, which is what Exa returns for a page with no image. It took
        # down the entire expansion round on its first result.
        payload = {"results": [{"url": "https://x.com/a/status/1", "extras": {"imageLinks": []}}]}
        assert self._provider()._to_candidates(payload, route="exa", discovered_at=NOW) == []

    def test_survives_a_malformed_response(self):
        for payload in ({}, {"results": None}, {"results": ["junk", None, {}]}):
            assert self._provider()._to_candidates(payload, route="exa", discovered_at=NOW) == []

    def test_rejects_a_non_page_url(self):
        payload = {"results": [{"url": "ftp://x/y", "image": "https://c/i.jpg"}]}
        assert self._provider()._to_candidates(payload, route="exa", discovered_at=NOW) == []


class TestWikidataParsing:
    def test_returns_nothing_without_an_entity(self):
        assert WikidataProvider().search([], discovered_at=NOW).candidates == []

    def test_sets_a_wikimedia_user_agent(self):
        # requests.Session ships a default User-Agent, so setdefault silently kept it and
        # Wikimedia answered 403. The header must be assigned, not defaulted.
        provider = WikidataProvider()
        assert "Sigil" in provider._session.headers["User-Agent"]


@pytest.mark.skipif(
    "not config.getoption('--run-live', default=False)",
    reason="needs the network; enable with --run-live",
)
class TestWikidataLive:
    def test_resolves_a_public_figure_to_a_portrait(self):
        result = WikidataProvider().search(["Colin Powell"], discovered_at=NOW)
        assert result.candidates
        assert str(result.candidates[0].image_url).startswith("https://")


class TestPageHarvest:
    """A page is not one image, and a search engine only ever hands back one."""

    def _urls(self, html: str, base: str = "https://example.com/page"):
        from sigil.search.providers.pages import extract_image_urls

        return extract_image_urls(html, base)

    def test_reads_img_tags_and_resolves_relative_paths(self):
        html = '<img src="/assets/one.jpg"><img src="https://cdn.example.com/two.png">'
        assert self._urls(html) == [
            "https://example.com/assets/one.jpg",
            "https://cdn.example.com/two.png",
        ]

    def test_reads_paths_out_of_json_payloads(self):
        # The portfolio that motivated this keeps its photos in a JSON blob, not in <img>
        # tags, so markup-only extraction found almost nothing on it.
        html = '<script>{"gallery":["/assets/glimpses/talk-520.webp","/assets/pfp.webp"]}</script>'
        assert self._urls(html) == [
            "https://example.com/assets/glimpses/talk-520.webp",
            "https://example.com/assets/pfp.webp",
        ]

    def test_prefers_the_largest_srcset_entry(self):
        html = '<img srcset="/small.jpg 1x, /large.jpg 2x">'
        assert self._urls(html) == ["https://example.com/large.jpg"]

    def test_skips_furniture_that_is_never_a_person(self):
        html = (
            '<img src="/logo.png"><img src="/favicon.ico"><img src="/tech/java.svg">'
            '<img src="/team/photo.jpg">'
        )
        assert self._urls(html) == ["https://example.com/team/photo.jpg"]

    def test_deduplicates_and_ignores_data_uris(self):
        html = '<img src="/a.jpg"><img src="/a.jpg"><img src="data:image/png;base64,AAA">'
        assert self._urls(html) == ["https://example.com/a.jpg"]

    def test_each_image_becomes_its_own_candidate(self):
        from datetime import UTC, datetime

        from sigil.search.providers.pages import PageHarvestProvider

        provider = PageHarvestProvider()
        provider._fetch = lambda url: (  # type: ignore[method-assign]
            '<img src="/a.jpg" alt="Gaurish Baliga"><img src="/b.jpg">'
        )
        result = provider.harvest(["https://example.com/about"], discovered_at=datetime.now(UTC))
        assert len(result.candidates) == 2
        # Same page URL, so the post_id is what keeps them apart through deduplication.
        assert {c.post_id for c in result.candidates} == {"img0", "img1"}
        assert len({str(c.image_url) for c in result.candidates}) == 2
        assert result.candidates[0].title == "Gaurish Baliga"
        assert result.raw["https://example.com/about"]["labels"] == {
            "https://example.com/a.jpg": "Gaurish Baliga"
        }
