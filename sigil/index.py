"""A local face index: the part of "be like lenso.ai" that is genuinely just code.

It is worth being precise about what a face-search engine is, because the parts have very
different costs and lumping them together is how the question gets waved away:

===========================  ==========================================================
 embed a face                 done; ArcFace, 512-d, already in this repo
 store N embeddings           this module
 nearest-neighbour search     this module. Embeddings are L2-normalised, so cosine
                              similarity is a dot product and querying the whole index
                              is one matrix multiply
 measure recall and latency   this module, with denominators
 -------------------------    ----------------------------------------------------------
 **crawl social media at**    not code. Scraping Instagram, Facebook and LinkedIn at
 **scale to fill it**         scale breaks their terms, needs authentication bypass and
                              rotating infrastructure, and builds a permanent biometric
                              record of millions of people who were never asked
===========================  ==========================================================

Only the last row is the hard part, and it is hard for reasons that more engineering does
not fix. So this module builds the whole machine and leaves the choice of what to put in
it to whoever runs it, with two corpora that are defensible by default:

``lfw``       a public face-recognition research dataset, already on disk for the
              benchmark. Used here to *measure* the index honestly: recall@1 against
              5,749 identities is a real number, not a demo.
``runs``      faces this tool already fetched during its own searches. Nothing new is
              collected; it indexes what it has, so a repeat query costs no API call.

An index of people who did not consent is a decision, not a feature, and it is the one
that would raise recall on private individuals the most. This code does not make it.

Embeddings are biometric data. The index is local, is written under ``data/index/`` which
is gitignored, and is never anchored, uploaded or committed.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sigil.config import DATA_DIR
from sigil.face import EMBEDDING_DIM, embed_all_faces
from sigil.face.engine import FaceEngine, get_engine

logger = logging.getLogger(__name__)

INDEX_DIR = DATA_DIR / "index"

# Beyond roughly this many vectors a brute-force matmul stops being the right answer and
# an approximate structure (HNSW, IVF-PQ) earns its complexity. Below it, exact search is
# faster than building a graph and has no recall loss to apologise for.
EXACT_SEARCH_CEILING = 1_000_000


@dataclass(slots=True)
class SearchHit:
    """One neighbour, with the distance in the same units the face gate uses."""

    label: str
    path: str
    distance: float

    def to_json(self) -> dict[str, Any]:
        return {"label": self.label, "path": self.path, "distance": round(self.distance, 6)}


@dataclass(slots=True)
class IndexStats:
    vectors: int = 0
    identities: int = 0
    dimensions: int = EMBEDDING_DIM
    bytes_on_disk: int = 0
    build_seconds: float = 0.0
    images_seen: int = 0
    images_without_a_face: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "vectors": self.vectors,
            "identities": self.identities,
            "dimensions": self.dimensions,
            "bytes_on_disk": self.bytes_on_disk,
            "megabytes_on_disk": round(self.bytes_on_disk / 1e6, 2),
            "build_seconds": round(self.build_seconds, 1),
            "images_seen": self.images_seen,
            "images_without_a_face": self.images_without_a_face,
            "coverage": round(
                (self.images_seen - self.images_without_a_face) / max(self.images_seen, 1), 4
            ),
        }


@dataclass(slots=True)
class FaceIndex:
    """Exact nearest-neighbour search over L2-normalised face embeddings."""

    embeddings: np.ndarray = field(
        default_factory=lambda: np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    )
    labels: list[str] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)
    stats: IndexStats = field(default_factory=IndexStats)

    def __len__(self) -> int:
        return int(self.embeddings.shape[0])

    def search(self, query: np.ndarray, top: int = 10) -> list[SearchHit]:
        """Return the ``top`` nearest faces.

        Embeddings are unit vectors, so cosine distance is ``1 - dot`` and the whole index
        is scored by one matrix multiply. argpartition then does the selection in linear
        time rather than sorting everything.
        """

        if len(self) == 0:
            return []
        vector = np.asarray(query, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm

        similarities = self.embeddings @ vector
        count = min(top, len(self))
        top_indices = np.argpartition(-similarities, count - 1)[:count]
        top_indices = top_indices[np.argsort(-similarities[top_indices])]
        return [
            SearchHit(
                label=self.labels[i],
                path=self.paths[i],
                distance=float(1.0 - similarities[i]),
            )
            for i in top_indices
        ]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            embeddings=self.embeddings,
            labels=np.array(self.labels, dtype=object),
            paths=np.array(self.paths, dtype=object),
            stats=json.dumps(self.stats.to_json()),
        )
        self.stats.bytes_on_disk = path.stat().st_size
        logger.info("index written to %s (%d vectors)", path, len(self))

    @classmethod
    def load(cls, path: Path) -> FaceIndex:
        with np.load(path, allow_pickle=True) as data:
            stats = IndexStats(
                **{
                    k: v
                    for k, v in json.loads(str(data["stats"])).items()
                    if k in IndexStats.__slots__
                }
            )
            return cls(
                embeddings=data["embeddings"].astype(np.float32),
                labels=list(data["labels"]),
                paths=list(data["paths"]),
                stats=stats,
            )


def _iter_lfw(limit: int | None) -> list[tuple[str, Path]]:
    """(identity, path) for the LFW corpus scikit-learn already downloaded."""

    root = Path.home() / "scikit_learn_data" / "lfw_home" / "lfw_funneled"
    if not root.is_dir():
        raise FileNotFoundError(
            f"LFW is not on disk at {root}. Run `sigil benchmark` once to fetch it."
        )
    items: list[tuple[str, Path]] = []
    for person in sorted(root.iterdir()):
        if not person.is_dir():
            continue
        for image in sorted(person.glob("*.jpg")):
            items.append((person.name, image))
            if limit and len(items) >= limit:
                return items
    return items


def _iter_runs(limit: int | None) -> list[tuple[str, Path]]:
    """(source page, path) for candidate media this tool already fetched."""

    items: list[tuple[str, Path]] = []
    for bundle in sorted((DATA_DIR / "bundles").glob("*")):
        thumbs = bundle / "media" / "thumbs"
        if not thumbs.is_dir():
            continue
        for image in sorted(thumbs.glob("*.jpg")):
            items.append((bundle.name, image))
            if limit and len(items) >= limit:
                return items
    return items


CORPORA = {"lfw": _iter_lfw, "runs": _iter_runs}


def build_index(
    corpus: str = "lfw",
    *,
    limit: int | None = None,
    engine: FaceEngine | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> FaceIndex:
    """Detect, embed and index every face in a corpus.

    ``on_progress(done, total)`` is called as work completes. Building over the full LFW
    corpus takes twenty minutes or more, and the first version reported nothing at all
    until it finished: the progress went to ``logger.info``, which the CLI does not enable,
    so there was no way to tell a slow run from a hung one. A long job has to say where it
    is.
    """

    if corpus not in CORPORA:
        raise ValueError(f"unknown corpus {corpus!r}; expected one of {sorted(CORPORA)}")

    active = engine or get_engine()
    active.warm_up()
    items = CORPORA[corpus](limit)

    started = time.perf_counter()
    vectors: list[np.ndarray] = []
    labels: list[str] = []
    paths: list[str] = []
    missing = 0

    from PIL import Image

    for position, (label, path) in enumerate(items, start=1):
        try:
            with Image.open(path) as handle:
                image = np.asarray(handle.convert("RGB"), dtype=np.uint8)
        except Exception:
            missing += 1
            continue
        faces, embeddings, _ = embed_all_faces(image, engine=active)
        if not faces:
            missing += 1
            continue
        # One vector per image: the largest face. Indexing every face in a group photo
        # would bloat the index with bystanders nobody searched for.
        best = int(np.argmax([min(f.width, f.height) for f in faces]))
        vectors.append(embeddings[best])
        labels.append(label)
        paths.append(str(path))
        if on_progress is not None:
            on_progress(position, len(items))

    matrix = (
        np.vstack(vectors).astype(np.float32)
        if vectors
        else np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    )
    stats = IndexStats(
        vectors=len(vectors),
        identities=len(set(labels)),
        build_seconds=time.perf_counter() - started,
        images_seen=len(items),
        images_without_a_face=missing,
    )
    return FaceIndex(embeddings=matrix, labels=labels, paths=paths, stats=stats)


def evaluate_index(index: FaceIndex, *, queries: int = 300) -> dict[str, Any]:
    """Measure retrieval quality and speed, leave-one-out.

    Each query is a vector already in the index, and the index's own copy is excluded from
    its own result. Recall@1 is then "does the nearest *other* face belong to the same
    person", which is the question a face-search engine is actually answering. Identities
    with a single image are skipped: they have no correct answer to find.
    """

    if len(index) == 0:
        return {"queries": 0}

    counts: dict[str, int] = {}
    for label in index.labels:
        counts[label] = counts.get(label, 0) + 1
    eligible = [i for i, label in enumerate(index.labels) if counts[label] > 1]
    if not eligible:
        return {"queries": 0, "note": "no identity has more than one image"}

    rng = np.random.default_rng(0)
    chosen = rng.choice(eligible, size=min(queries, len(eligible)), replace=False)

    hits_at_1 = hits_at_10 = 0
    latencies: list[float] = []
    for position in chosen:
        query = index.embeddings[position]
        started = time.perf_counter()
        results = index.search(query, top=11)
        latencies.append((time.perf_counter() - started) * 1000)
        # Drop the query's own row; it is trivially its own nearest neighbour.
        others = [hit for hit in results if hit.path != index.paths[position]][:10]
        truth = index.labels[position]
        if others and others[0].label == truth:
            hits_at_1 += 1
        if any(hit.label == truth for hit in others):
            hits_at_10 += 1

    latency = np.array(latencies)
    return {
        "queries": len(chosen),
        "identities_searched": len(set(index.labels)),
        "vectors": len(index),
        "recall_at_1": round(hits_at_1 / len(chosen), 4),
        "recall_at_10": round(hits_at_10 / len(chosen), 4),
        "query_p50_ms": round(float(np.percentile(latency, 50)), 3),
        "query_p95_ms": round(float(np.percentile(latency, 95)), 3),
        "exact_search": True,
        "note": (
            f"Exact search: every query scores all {len(index)} vectors, so recall is not "
            "approximated away. Beyond about "
            f"{EXACT_SEARCH_CEILING:,} vectors an HNSW or IVF-PQ structure would be worth "
            "its complexity; below it a matrix multiply wins."
        ),
    }


def index_path(corpus: str) -> Path:
    return INDEX_DIR / f"{corpus}.npz"


__all__ = [
    "CORPORA",
    "EXACT_SEARCH_CEILING",
    "FaceIndex",
    "IndexStats",
    "SearchHit",
    "build_index",
    "evaluate_index",
    "index_path",
]
