"""Shared fixtures.

Test images come from LFW via scikit-learn rather than being committed to the repo: no
licensing question, no biometric data in git history, and the same source the benchmark
uses. When scikit-learn or its cache is unavailable, face tests skip rather than fail.
"""

from __future__ import annotations

import io
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture(scope="session")
def lfw_pairs():
    sklearn_datasets = pytest.importorskip("sklearn.datasets")
    try:
        data = sklearn_datasets.fetch_lfw_pairs(
            subset="test",
            color=True,
            resize=1.0,
            funneled=True,
            slice_=None,
            download_if_missing=False,
        )
    except Exception as exc:  # dataset not cached locally
        pytest.skip(f"LFW not available offline: {exc}")
    return data


@pytest.fixture(scope="session")
def face_engine():
    pytest.importorskip("onnxruntime")
    from sigil.face.engine import FaceEngineError, get_engine

    try:
        engine = get_engine()
        engine.warm_up()
    except FaceEngineError as exc:
        pytest.skip(f"face models unavailable: {exc}")
    return engine


def _to_uint8(array: np.ndarray) -> np.ndarray:
    return array if array.dtype == np.uint8 else (array * 255).clip(0, 255).astype(np.uint8)


@pytest.fixture(scope="session")
def face_images(lfw_pairs, tmp_path_factory) -> dict[str, Path]:
    """Two images of the same person and one of a different person, written to disk."""

    pairs, target = lfw_pairs.pairs, lfw_pairs.target
    positive = int(np.argmax(target == 1))
    negative = int(np.argmax(target == 0))

    directory = tmp_path_factory.mktemp("faces")
    written: dict[str, Path] = {}
    for name, array in (
        ("anchor", pairs[positive, 0]),
        ("same_person", pairs[positive, 1]),
        ("other_person", pairs[negative, 1]),
    ):
        path = directory / f"{name}.jpg"
        Image.fromarray(_to_uint8(array)).save(path, format="JPEG", quality=95)
        written[name] = path
    return written


@pytest.fixture
def jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (256, 256), (120, 130, 140)).save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 9, 2, 12, 0, 0, tzinfo=UTC)


def pytest_addoption(parser):
    """`--run-live` opts in to the few tests that touch the network."""

    parser.addoption("--run-live", action="store_true", default=False, help="run network tests")
