from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from sigil.candidates import MediaQuality
from sigil.config import ConfigurationError, Settings
from sigil.models import FaceObservation, PipelineErrorCode
from sigil.run import run_pipeline


def test_run_pipeline_no_face(tmp_path: Path) -> None:
    blank = tmp_path / "blank.jpg"
    Image.new("RGB", (200, 200), color=(128, 128, 128)).save(blank)

    settings = Settings(_env_file=None)
    result = run_pipeline(blank, settings=settings, output_root=tmp_path / "bundles")

    assert not result.ok
    assert result.error_code == PipelineErrorCode.NO_FACE


def test_run_pipeline_chain_configuration_error_preserves_bundle(
    tmp_path: Path, monkeypatch: object
) -> None:
    # Test that when anchoring raises ConfigurationError (e.g. missing chain credentials),
    # run_pipeline gracefully returns INVALID_CONFIGURATION and preserves the evidence bundle.
    test_img = tmp_path / "query.jpg"
    Image.new("RGB", (200, 200), color=(200, 200, 200)).save(test_img)

    fake_crop = tmp_path / "crop.jpg"
    Image.new("RGB", (100, 100), color=(200, 200, 200)).save(fake_crop)

    mock_face = FaceObservation(
        source_image=str(test_img),
        face_crop_path=str(fake_crop),
        portrait_crop_path="",
        detector_backend="scrfd_10g",
        model_name="arcface_w600k_r50",
        detection_confidence=0.99,
        bounding_box={"x": 10, "y": 10, "width": 80, "height": 80},
        quality={
            "face_size_px": 80,
            "blur_score": 100.0,
            "brightness": 0.5,
            "border_truncated": False,
            "quality_score": 0.95,
        },
        embedding=[0.1] * 512,
    )

    settings = Settings(
        _env_file=None,
        SERPAPI_KEY="mock-key",
        SEPOLIA_RPC_URL="https://sepolia.mock",
        CONTRACT_ADDRESS="0x" + "1" * 40,
    )

    def mock_require(*stages: str) -> None:
        if "chain-write" in stages:
            raise ConfigurationError("Missing PRIVATE_KEY")

    with (
        patch("sigil.run.detect_and_encode", return_value=mock_face),
        patch(
            "sigil.run.discover",
            return_value={
                "candidates": [MagicMock(source_url="https://x.com/post/1")],
                "search_records": [],
                "budget": {},
                "sources": [],
                "raw_responses": {},
                "routes_run": [],
                "entities_inferred": [],
            },
        ),
        patch("sigil.run.fetch_all_sync", return_value=[MagicMock()]),
        patch(
            "sigil.run.verify_all",
            return_value=(
                [
                    MagicMock(
                        matched=True,
                        media=MagicMock(
                            sha256="aa" * 32,
                            quality=MediaQuality.ORIGINAL,
                            final_url="https://x.com/img.jpg",
                            data=b"fake",
                        ),
                        candidate=MagicMock(
                            source_url="https://x.com/post/1",
                            platform="x",
                            post_id="1",
                            title="",
                            discovered_at=mock_face.quality.face_size_px
                            and MagicMock(isoformat=lambda: "2026-09-01T00:00:00Z"),
                            search_routes=["R1"],
                            search_rank=1,
                            exact_match=False,
                            is_social=True,
                        ),
                        decision=MagicMock(
                            status="MATCH", distance=0.1, threshold=0.36, candidate_face_index=0
                        ),
                        faces_detected=1,
                        best_face_px=100,
                        margin=0.26,
                        same_photo=False,
                    )
                ],
                [
                    MagicMock(
                        matched=True,
                        media=MagicMock(
                            sha256="aa" * 32,
                            quality=MediaQuality.ORIGINAL,
                            final_url="https://x.com/img.jpg",
                            data=b"fake",
                        ),
                        candidate=MagicMock(
                            source_url="https://x.com/post/1",
                            platform="x",
                            post_id="1",
                            discovered_at=MagicMock(isoformat=lambda: "2026-09-01T00:00:00Z"),
                            search_routes=["R1"],
                            search_rank=1,
                            exact_match=False,
                            is_social=True,
                        ),
                        decision=MagicMock(
                            status="MATCH", distance=0.1, threshold=0.36, candidate_face_index=0
                        ),
                        faces_detected=1,
                        best_face_px=100,
                        margin=0.26,
                        same_photo=False,
                    )
                ],
            ),
        ),
        patch.object(Settings, "require", side_effect=mock_require),
    ):
        result = run_pipeline(test_img, settings=settings, output_root=tmp_path / "bundles")
        assert not result.ok
        assert result.error_code == PipelineErrorCode.INVALID_CONFIGURATION
        assert result.bundle_dir is not None
        assert result.bundle_dir.is_dir()
        assert "evidence bundle is intact" in result.error_message
