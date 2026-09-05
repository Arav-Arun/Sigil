"""Tests for the local face index.

The index is the half of a face-search engine that is genuinely just code, so the things
worth pinning are the ones that would quietly make it useless: a distance in the wrong
units, a query that returns itself as a match, or a saved index that loses its labels.
"""

from __future__ import annotations

import numpy as np
import pytest

from sigil.face import EMBEDDING_DIM
from sigil.index import FaceIndex, IndexStats, SearchHit, evaluate_index, index_path


def _unit(seed: int, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """A deterministic unit vector; the engine only ever emits normalised embeddings."""

    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(dim).astype(np.float32)
    return vector / np.linalg.norm(vector)


def _index(count: int = 6) -> FaceIndex:
    vectors = np.vstack([_unit(i) for i in range(count)])
    return FaceIndex(
        embeddings=vectors,
        labels=[f"person{i // 2}" for i in range(count)],
        paths=[f"/corpus/{i}.jpg" for i in range(count)],
        stats=IndexStats(vectors=count, identities=count // 2),
    )


class TestSearch:
    def test_an_empty_index_returns_nothing_rather_than_raising(self):
        assert FaceIndex().search(_unit(0)) == []

    def test_a_vector_in_the_index_finds_itself_first_at_distance_zero(self):
        index = _index()
        hit = index.search(index.embeddings[3], top=1)[0]
        assert hit.path == "/corpus/3.jpg"
        assert hit.distance == pytest.approx(0.0, abs=1e-6)

    def test_results_come_back_in_ascending_distance(self):
        hits = _index().search(_unit(0), top=6)
        distances = [hit.distance for hit in hits]
        assert distances == sorted(distances)

    def test_distance_is_cosine_so_it_shares_units_with_the_face_gate(self):
        # The gate compares against DEFAULT_MATCH_THRESHOLD. If the index returned
        # similarity instead of distance, every neighbour would read as a perfect match.
        index = _index()
        a, b = index.embeddings[0], index.embeddings[1]
        expected = 1.0 - float(a @ b)
        found = next(h for h in index.search(a, top=6) if h.path == "/corpus/1.jpg")
        assert found.distance == pytest.approx(expected, abs=1e-6)

    def test_top_is_clamped_to_the_index_size(self):
        assert len(_index(3).search(_unit(0), top=50)) == 3

    def test_an_unnormalised_query_is_normalised_before_scoring(self):
        index = _index()
        scaled = index.embeddings[2] * 7.5
        assert index.search(scaled, top=1)[0].distance == pytest.approx(0.0, abs=1e-6)


class TestPersistence:
    def test_round_trips_through_disk(self, tmp_path):
        original = _index()
        path = tmp_path / "corpus.npz"
        original.save(path)
        loaded = FaceIndex.load(path)

        assert len(loaded) == len(original)
        assert loaded.labels == original.labels
        assert loaded.paths == original.paths
        assert np.allclose(loaded.embeddings, original.embeddings)

    def test_saving_records_the_size_on_disk(self, tmp_path):
        index = _index()
        path = tmp_path / "corpus.npz"
        index.save(path)
        assert index.stats.bytes_on_disk == path.stat().st_size > 0
        loaded = FaceIndex.load(path)
        assert loaded.stats.bytes_on_disk == path.stat().st_size > 0


class TestEvaluation:
    def test_a_query_never_counts_its_own_row_as_the_answer(self):
        """Leave-one-out, or recall@1 is 100% and means nothing.

        Every query vector is already in the index, so without excluding it the nearest
        neighbour is always the query itself and the metric measures nothing at all.
        """

        # Two identities, two images each, with the pairs deliberately far apart, so the
        # only way to score is by finding a *different* image of the same person.
        embeddings = np.vstack([_unit(1), _unit(2), _unit(3), _unit(4)])
        index = FaceIndex(
            embeddings=embeddings,
            labels=["a", "a", "b", "b"],
            paths=["/1.jpg", "/2.jpg", "/3.jpg", "/4.jpg"],
        )
        report = evaluate_index(index, queries=4)
        assert report["queries"] == 4
        # Random vectors are not really the same person, so recall should be near chance
        # rather than the perfect score self-matching would produce.
        assert report["recall_at_1"] < 1.0

    def test_identities_with_one_image_are_skipped(self):
        index = FaceIndex(
            embeddings=np.vstack([_unit(1), _unit(2)]),
            labels=["only-one", "also-one"],
            paths=["/1.jpg", "/2.jpg"],
        )
        assert evaluate_index(index)["queries"] == 0

    def test_an_empty_index_evaluates_to_nothing(self):
        assert evaluate_index(FaceIndex())["queries"] == 0


class TestPaths:
    def test_index_path_is_under_the_gitignored_data_directory(self):
        # Embeddings are biometric. The index must never land somewhere committable.
        path = index_path("lfw")
        assert path.parent.name == "index"
        assert path.name == "lfw.npz"


class TestSearchHit:
    def test_serializes_without_the_embedding(self):
        payload = SearchHit(label="p", path="/p.jpg", distance=0.1234567).to_json()
        assert payload == {"label": "p", "path": "/p.jpg", "distance": 0.123457}
