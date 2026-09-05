"""Verification decisions and ranking.

The behaviour that matters most here is negative: a candidate that fails the face gate
must never be promoted by a good search rank, and an uncertain distance must surface as
INCONCLUSIVE rather than being rounded to a decision.
"""

from __future__ import annotations

import io
from datetime import UTC, datetime

import numpy as np
import pytest
from PIL import Image

from sigil.candidates import FetchedMedia, MediaQuality
from sigil.models import DecisionStatus, SearchCandidate
from sigil.verify import (
    DEFAULT_MATCH_THRESHOLD,
    DEFAULT_REJECT_THRESHOLD,
    CandidateVerification,
    is_same_photo,
    rank,
    verify_candidate,
)

NOW = datetime(2026, 9, 2, 12, 0, tzinfo=UTC)


def make_candidate(url="https://x.com/a/status/1", rank_=1, exact=False, routes=("R1",)):
    return SearchCandidate(
        source_url=url,
        platform="x",
        post_id="1",
        search_rank=rank_,
        exact_match=exact,
        discovered_at=NOW,
        search_routes=list(routes),
    )


def make_media(candidate=None, *, ok=True, quality=MediaQuality.ORIGINAL, error=""):
    return FetchedMedia(
        candidate=candidate or make_candidate(),
        data=b"x" * 4096 if ok else b"",
        sha256="ab" * 32 if ok else "",
        content_type="image/jpeg" if ok else "",
        byte_count=4096 if ok else 0,
        status_code=200 if ok else 0,
        quality=quality,
        error=error,
    )


def make_verification(
    distance, *, status=None, quality=MediaQuality.ORIGINAL, same_photo=False, **candidate_kwargs
):
    from sigil.models import VerificationDecision

    if status is None:
        status = (
            DecisionStatus.MATCH
            if distance <= DEFAULT_MATCH_THRESHOLD
            else DecisionStatus.NON_MATCH
        )
    candidate = make_candidate(**candidate_kwargs)
    threshold = (
        DEFAULT_MATCH_THRESHOLD if status is DecisionStatus.MATCH else DEFAULT_REJECT_THRESHOLD
    )
    return CandidateVerification(
        candidate=candidate,
        media=make_media(candidate, quality=quality),
        decision=VerificationDecision(
            status=status,
            distance=distance,
            threshold=threshold,
            model_name="test",
            detector_backend="scrfd_10g",
            candidate_face_index=0,
            reason="test",
        ),
        faces_detected=1,
        best_face_index=0,
        best_face_px=200,
        same_photo=same_photo,
    )


class TestUnavailableMedia:
    def test_unfetchable_media_is_inconclusive_not_a_non_match(self):
        # "We could not look" is different from "we looked and it is not them".
        result = verify_candidate(
            np.zeros(512, dtype=np.float32),
            make_media(ok=False, error="HTTP 403"),
            engine=None,  # never reached
            model_name="test",
        )
        assert result.decision.status is DecisionStatus.INCONCLUSIVE
        assert "403" in result.decision.reason

    def test_undecodable_media_is_inconclusive(self):
        media = make_media()
        media.data = b"not an image at all" * 200
        result = verify_candidate(
            np.zeros(512, dtype=np.float32), media, engine=None, model_name="test"
        )
        assert result.decision.status is DecisionStatus.INCONCLUSIVE
        assert "decode" in result.decision.reason


class TestThreeStateGate:
    """The band between the thresholds must never collapse into a decision."""

    @pytest.fixture
    def source(self, face_images, face_engine):
        import tempfile

        from sigil.face import detect_and_encode

        with tempfile.TemporaryDirectory() as tmp:
            observation = detect_and_encode(
                face_images["anchor"], tmp, engine=face_engine, select_largest=True
            )
        return np.asarray(observation.embedding, dtype=np.float32)

    def _media_from(self, path):
        candidate = make_candidate()
        data = path.read_bytes()
        import hashlib

        return FetchedMedia(
            candidate=candidate,
            data=data,
            sha256=hashlib.sha256(data).hexdigest(),
            content_type="image/jpeg",
            byte_count=len(data),
            status_code=200,
            quality=MediaQuality.ORIGINAL,
        )

    def test_same_person_matches(self, source, face_images, face_engine):
        result = verify_candidate(
            source,
            self._media_from(face_images["same_person"]),
            engine=face_engine,
            model_name="test",
        )
        assert result.decision.status is DecisionStatus.MATCH
        assert result.decision.distance <= DEFAULT_MATCH_THRESHOLD

    def test_different_person_does_not_match(self, source, face_images, face_engine):
        result = verify_candidate(
            source,
            self._media_from(face_images["other_person"]),
            engine=face_engine,
            model_name="test",
        )
        assert result.decision.status is not DecisionStatus.MATCH

    def test_an_impossible_threshold_forces_inconclusive(self, source, face_images, face_engine):
        # Squeeze the band around the observed distance: the answer must become
        # INCONCLUSIVE rather than flipping to a confident verdict.
        result = verify_candidate(
            source,
            self._media_from(face_images["same_person"]),
            engine=face_engine,
            model_name="test",
            match_threshold=0.0,
            reject_threshold=2.0,
        )
        assert result.decision.status is DecisionStatus.INCONCLUSIVE
        assert "uncertainty band" in result.decision.reason

    def test_reports_every_detected_face(self, source, face_images, face_engine):
        result = verify_candidate(
            source,
            self._media_from(face_images["same_person"]),
            engine=face_engine,
            model_name="test",
        )
        assert result.faces_detected >= 1
        assert len(result.all_distances) == result.faces_detected


class TestRanking:
    def test_only_matches_are_ranked(self):
        items = [
            make_verification(0.95, url="https://x.com/a/status/1"),
            make_verification(0.20, url="https://x.com/b/status/2"),
        ]
        ranked = rank(items)
        assert len(ranked) == 1
        assert str(ranked[0].candidate.source_url) == "https://x.com/b/status/2"

    def test_search_rank_cannot_promote_a_failed_candidate(self):
        # The critical invariant: being result #1 does not make you a match.
        items = [
            make_verification(0.99, url="https://x.com/top/status/1", rank_=1),
            make_verification(0.25, url="https://x.com/low/status/2", rank_=25),
        ]
        ranked = rank(items)
        assert [str(r.candidate.source_url) for r in ranked] == ["https://x.com/low/status/2"]

    def test_a_larger_margin_wins(self):
        items = [
            make_verification(0.55, url="https://x.com/a/status/1"),
            make_verification(0.15, url="https://x.com/b/status/2"),
        ]
        assert str(rank(items)[0].candidate.source_url) == "https://x.com/b/status/2"

    def test_media_quality_breaks_a_near_tie(self):
        items = [
            make_verification(0.30, url="https://x.com/a/status/1", quality=MediaQuality.THUMBNAIL),
            make_verification(0.30, url="https://x.com/b/status/2", quality=MediaQuality.ORIGINAL),
        ]
        assert str(rank(items)[0].candidate.source_url) == "https://x.com/b/status/2"

    def test_an_alternate_photo_outranks_the_uploaded_photo(self):
        items = [
            make_verification(0.0, url="https://x.com/exact/status/1", rank_=1, same_photo=True),
            make_verification(0.35, url="https://x.com/alternate/status/2", rank_=20),
        ]

        assert str(rank(items)[0].candidate.source_url) == "https://x.com/alternate/status/2"

    def test_empty_input_ranks_to_empty(self):
        assert rank([]) == []

    def test_margin_is_positive_for_a_match(self):
        assert make_verification(0.30).margin == pytest.approx(DEFAULT_MATCH_THRESHOLD - 0.30)


class TestSamePhoto:
    def _media(self, image: Image.Image) -> FetchedMedia:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=82)
        candidate = make_candidate()
        return FetchedMedia(
            candidate=candidate,
            data=buffer.getvalue(),
            sha256="ab" * 32,
            content_type="image/jpeg",
            byte_count=buffer.tell(),
            status_code=200,
            quality=MediaQuality.ORIGINAL,
        )

    def test_recognises_the_same_pixels_after_reencoding(self, tmp_path):
        pixels = np.zeros((120, 160, 3), dtype=np.uint8)
        pixels[:, :80] = (230, 40, 40)
        pixels[:, 80:] = (20, 80, 220)
        source = Image.fromarray(pixels)
        path = tmp_path / "query.png"
        source.save(path)

        assert is_same_photo(path, self._media(source.resize((800, 600))))

    def test_rejects_a_different_photo(self, tmp_path):
        source = Image.new("RGB", (160, 120), (230, 40, 40))
        path = tmp_path / "query.png"
        source.save(path)
        different = Image.new("RGB", (160, 120), (20, 80, 220))

        assert not is_same_photo(path, self._media(different))

    def test_serialization_exposes_duplicate_status_and_candidate_context(self):
        item = make_verification(0.0, exact=True, same_photo=True)
        payload = item.to_json()

        assert payload["exact_match"] is True
        assert payload["same_photo"] is True
        assert "title" in payload


class TestCalibrationIsPinned:
    """The shipped thresholds must be the ones the benchmark actually measured.

    Without this, someone can tune a threshold by hand and the published accuracy table
    silently stops describing the code that runs.
    """

    def test_the_shipped_threshold_is_tighter_than_the_calibrated_one(self):
        """The benchmark says what LFW allows; we ship something stricter, on purpose.

        The calibration answers "what threshold gives FMR 1e-3 against random impostor
        pairs". This pipeline's impostors are not random: a reverse-image search returns
        candidates *because* they resemble the query, so the negatives it must survive are
        the hard tail of that distribution rather than a sample from all of it.

        Shipping the calibrated value put the boundary on the impostor edge and a real run
        then matched a stranger at 0.6888. So the shipped value must stay at or below the
        calibrated one. Equality is what this test used to assert, and equality is the bug.
        """

        import json
        from pathlib import Path

        report_path = Path(__file__).resolve().parent.parent / "docs" / "benchmark.json"
        if not report_path.is_file():
            pytest.skip("no committed benchmark report")
        report = json.loads(report_path.read_text())

        assert report["match_threshold"] >= DEFAULT_MATCH_THRESHOLD, (
            f"shipped match threshold {DEFAULT_MATCH_THRESHOLD} is MORE permissive than "
            f"the calibrated {report['match_threshold']}. The shipped value may be "
            "stricter than the benchmark, never looser."
        )
        assert report["reject_threshold"] >= DEFAULT_REJECT_THRESHOLD

    def test_the_match_threshold_leaves_margin_against_the_nearest_impostor(self):
        """0.7924 is the closest impostor pair measured on the held-out LFW split.

        A threshold at or above it has no margin at all, which is how the old 0.795 let a
        lookalike through. Anything a similarity search surfaces is harder than a random
        LFW pair, so the margin has to be real rather than nominal.
        """

        nearest_impostor = 0.7924
        assert nearest_impostor - 0.15 > DEFAULT_MATCH_THRESHOLD, (
            f"match threshold {DEFAULT_MATCH_THRESHOLD} leaves only "
            f"{nearest_impostor - DEFAULT_MATCH_THRESHOLD:.3f} against the nearest "
            "measured impostor; the search surfaces harder negatives than that"
        )

    def test_the_uncertainty_band_is_non_empty(self):
        assert DEFAULT_REJECT_THRESHOLD > DEFAULT_MATCH_THRESHOLD

    def test_the_false_match_rate_stays_within_the_reporting_bound(self):
        """The held-out false-match rate must not exceed the 1e-2 reporting bound.

        Asserting *zero* false matches would be asserting luck: with ~500 impostor pairs
        the smallest observable non-zero rate is 1/500 = 2e-3, so a run at a true FMR of
        1e-3 will show 0 or 1 more or less at random. The operating point is chosen on
        the ~4000 calibration negatives, where 1e-3 is actually resolvable; this test
        guards the far weaker property that the test split has not blown past 1%.
        """

        import json
        from pathlib import Path

        report_path = Path(__file__).resolve().parent.parent / "docs" / "benchmark.json"
        if not report_path.is_file():
            pytest.skip("no committed benchmark report")
        report = json.loads(report_path.read_text())

        negatives = report["test_negative"]
        assert negatives > 0
        observed = report["false_matches"] / negatives
        assert observed <= 0.01, (
            f"{report['false_matches']} false matches in {negatives} impostor pairs "
            f"(FMR {observed:.4f}), falsely identifying someone is the worst outcome "
            "this pipeline can produce, so this is a hard ceiling"
        )

    def test_detection_coverage_is_reported_and_high(self):
        """Coverage must be tracked, because a detection failure is a silent exclusion."""

        import json
        from pathlib import Path

        report_path = Path(__file__).resolve().parent.parent / "docs" / "benchmark.json"
        if not report_path.is_file():
            pytest.skip("no committed benchmark report")
        report = json.loads(report_path.read_text())
        assert report.get("coverage", 0) >= 0.99, (
            f"coverage {report.get('coverage')}, too many pairs are dropping out of the "
            "evaluation, which silently flatters every other metric"
        )


class TestBrowserVerifier:
    """verify.html reimplements the proof independently, so it can drift independently."""

    def _source(self) -> str:
        from pathlib import Path

        path = Path(__file__).resolve().parent.parent / "verify.html"
        return path.read_text(encoding="utf-8")

    def test_selector_matches_the_contract_abi(self):
        """A wrong selector reverts instead of failing, which reads as a chain problem.

        The page precomputes keccak256("verify(bytes32)")[0:4] so it needs no hashing
        library. That constant once disagreed with the contract, and the only symptom was
        "execution reverted" from the node, which looks like an RPC fault rather than a
        bug in the page. Recomputing it here turns a silent demo failure into a red test.
        """

        import re

        from eth_utils import keccak

        match = re.search(r'const VERIFY_SELECTOR = "(0x[0-9a-f]{8})"', self._source())
        assert match, "verify.html no longer declares VERIFY_SELECTOR"
        expected = "0x" + keccak(text="verify(bytes32)")[:4].hex()
        assert match.group(1) == expected, (
            f"verify.html calls {match.group(1)} but SigilRegistry.verify(bytes32) is "
            f"{expected}; the browser verifier would revert against the deployed contract"
        )

    def test_reads_the_contract_address_from_the_receipt(self):
        """The page must not require the operator to type an address during a demo."""

        source = self._source()
        assert 'byName["receipt.json"]' in source
        assert "contract_address" in source

    def test_links_an_anchored_receipt_to_sepolia_etherscan(self):
        source = self._source()
        assert "https://sepolia.etherscan.io/tx/" in source
        assert "View transaction on Etherscan" in source
