"""Face engine correctness: alignment maths, detection, and the quality/selection gates.

The alignment and selection tests run everywhere. The tests that need real weights are
marked ``models`` and skip cleanly when the ONNX pack is not present.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from sigil.face import FacePipelineError, detect_and_encode, select_face
from sigil.face.align import ARCFACE_TEMPLATE, align_face, umeyama_similarity
from sigil.face.engine import DetectedFace, _nms, cosine_distance
from sigil.face.quality import pose_offsets
from sigil.imaging import ImageInputError
from sigil.models import PipelineErrorCode

pytestmark = pytest.mark.filterwarnings("ignore")


def make_face(x1: float, y1: float, x2: float, y2: float, score: float = 0.9) -> DetectedFace:
    width, height = x2 - x1, y2 - y1
    landmarks = np.array(
        [
            [x1 + width * 0.35, y1 + height * 0.40],
            [x1 + width * 0.65, y1 + height * 0.40],
            [x1 + width * 0.50, y1 + height * 0.58],
            [x1 + width * 0.38, y1 + height * 0.76],
            [x1 + width * 0.62, y1 + height * 0.76],
        ],
        dtype=np.float32,
    )
    return DetectedFace(bbox=(x1, y1, x2, y2), score=score, landmarks=landmarks)


class TestUmeyama:
    def test_recovers_an_exact_similarity_transform(self):
        source = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]])
        angle, scale, shift = np.pi / 6, 2.5, np.array([3.0, -1.0])
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        destination = (scale * source @ rotation.T) + shift

        matrix = umeyama_similarity(source, destination)
        recovered = source @ matrix[:, :2].T + matrix[:, 2]
        assert np.allclose(recovered, destination, atol=1e-9)

    def test_recovered_scale_matches(self):
        source = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0], [1.0, 1.0]])
        matrix = umeyama_similarity(source, source * 3.0)
        assert np.isclose(np.linalg.norm(matrix[:, 0]), 3.0)

    def test_does_not_produce_a_reflection(self):
        # A mirrored fit would align the landmark numbers while flipping the face.
        source = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [0.0, 1.0], [1.0, 1.0]])
        destination = source[:, ::-1].copy()
        matrix = umeyama_similarity(source, destination)
        assert np.linalg.det(matrix[:, :2]) > 0

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError):
            umeyama_similarity(np.zeros((5, 2)), np.zeros((4, 2)))

    def test_rejects_degenerate_points(self):
        with pytest.raises(ValueError):
            umeyama_similarity(np.zeros((5, 2)), ARCFACE_TEMPLATE)


class TestAlign:
    def test_output_is_the_canonical_crop_size(self):
        image = np.random.default_rng(0).integers(0, 255, (300, 300, 3), dtype=np.uint8)
        landmarks = ARCFACE_TEMPLATE * 2.0 + 50.0
        assert align_face(image, landmarks).shape == (112, 112, 3)

    def test_template_landmarks_map_to_the_template(self):
        image = np.zeros((112, 112, 3), dtype=np.uint8)
        matrix = umeyama_similarity(ARCFACE_TEMPLATE, ARCFACE_TEMPLATE)
        mapped = ARCFACE_TEMPLATE @ matrix[:, :2].T + matrix[:, 2]
        assert np.allclose(mapped, ARCFACE_TEMPLATE, atol=1e-6)
        assert align_face(image, ARCFACE_TEMPLATE).shape == (112, 112, 3)

    def test_rejects_the_wrong_landmark_count(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            align_face(image, np.zeros((3, 2)))


class TestNMS:
    def test_suppresses_a_heavy_overlap(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=float)
        assert _nms(boxes, np.array([0.9, 0.8]), 0.4) == [0]

    def test_keeps_disjoint_boxes(self):
        boxes = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=float)
        assert sorted(_nms(boxes, np.array([0.9, 0.8]), 0.4)) == [0, 1]

    def test_keeps_the_highest_scoring_box(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=float)
        assert _nms(boxes, np.array([0.5, 0.95]), 0.4) == [1]

    def test_handles_an_empty_input(self):
        assert _nms(np.zeros((0, 4)), np.zeros(0), 0.4) == []


class TestSelectFace:
    def test_no_faces_raises_no_face(self):
        with pytest.raises(FacePipelineError) as info:
            select_face([])
        assert info.value.code is PipelineErrorCode.NO_FACE

    def test_a_single_face_is_used_automatically(self):
        assert select_face([make_face(0, 0, 50, 50)]) == 0

    def test_multiple_faces_refuse_to_guess(self):
        faces = [make_face(0, 0, 50, 50), make_face(60, 0, 110, 50)]
        with pytest.raises(FacePipelineError) as info:
            select_face(faces)
        assert info.value.code is PipelineErrorCode.MULTIPLE_FACES

    def test_largest_policy_selects_index_zero(self):
        # detect() returns faces sorted by area, so index 0 is the largest.
        faces = [make_face(0, 0, 100, 100), make_face(0, 0, 20, 20)]
        assert select_face(faces, select_largest=True) == 0

    def test_explicit_index_is_honoured(self):
        faces = [make_face(0, 0, 50, 50), make_face(60, 0, 110, 50)]
        assert select_face(faces, face_index=1) == 1

    def test_out_of_range_index_raises(self):
        with pytest.raises(FacePipelineError) as info:
            select_face([make_face(0, 0, 50, 50)], face_index=5)
        assert info.value.code is PipelineErrorCode.MULTIPLE_FACES


class TestPoseOffsets:
    def test_frontal_level_face_is_near_zero(self):
        landmarks = np.array([[30.0, 40.0], [70.0, 40.0], [50.0, 58.0], [35.0, 76.0], [65.0, 76.0]])
        roll, yaw = pose_offsets(landmarks)
        assert roll < 0.01
        assert yaw < 0.01

    def test_tilted_eyes_raise_roll(self):
        landmarks = np.array([[30.0, 30.0], [70.0, 50.0], [50.0, 58.0], [35.0, 76.0], [65.0, 76.0]])
        roll, _ = pose_offsets(landmarks)
        assert roll > 0.4

    def test_offset_nose_raises_yaw(self):
        landmarks = np.array([[30.0, 40.0], [70.0, 40.0], [66.0, 58.0], [35.0, 76.0], [65.0, 76.0]])
        _, yaw = pose_offsets(landmarks)
        assert yaw > 0.3

    def test_degenerate_landmarks_are_clamped(self):
        assert pose_offsets(np.zeros((5, 2))) == (1.0, 1.0)


class TestCosineDistance:
    def test_identical_vectors_are_zero(self):
        vector = np.array([0.6, 0.8], dtype=np.float32)
        assert cosine_distance(vector, vector) == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_vectors_are_one(self):
        assert cosine_distance(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)

    def test_opposite_vectors_are_two(self):
        assert cosine_distance(np.array([1.0, 0.0]), np.array([-1.0, 0.0])) == pytest.approx(2.0)


class TestDetectAndEncode:
    def test_rejects_a_missing_file(self, tmp_path):
        with pytest.raises(ImageInputError):
            detect_and_encode(tmp_path / "missing.jpg", tmp_path)

    def test_reports_no_face_on_a_blank_image(self, tmp_path, face_engine):
        path = tmp_path / "blank.jpg"
        Image.new("RGB", (400, 400), (128, 128, 128)).save(path)
        with pytest.raises(FacePipelineError) as info:
            detect_and_encode(path, tmp_path / "out", engine=face_engine)
        assert info.value.code is PipelineErrorCode.NO_FACE


class TestRealFaces:
    """End-to-end behaviour on genuine photographs."""

    def test_encodes_a_real_face(self, face_images, face_engine, tmp_path):
        observation = detect_and_encode(face_images["anchor"], tmp_path / "out", engine=face_engine)
        assert len(observation.embedding) == 512
        assert observation.detection_confidence > 0.5
        assert observation.quality.quality_score > 0.0
        assert np.isclose(np.linalg.norm(observation.embedding), 1.0, atol=1e-5)

    def test_writes_review_artifacts(self, face_images, face_engine, tmp_path):
        out = tmp_path / "out"
        detect_and_encode(face_images["anchor"], out, engine=face_engine)
        for name in ("face_crop.jpg", "annotated_input.jpg", "normalized_input.jpg"):
            assert (out / name).is_file(), name

    def test_same_person_is_closer_than_a_different_person(
        self, face_images, face_engine, tmp_path
    ):
        def embed(key: str) -> np.ndarray:
            observation = detect_and_encode(
                face_images[key], tmp_path / key, engine=face_engine, select_largest=True
            )
            return np.asarray(observation.embedding, dtype=np.float32)

        anchor = embed("anchor")
        same = cosine_distance(anchor, embed("same_person"))
        other = cosine_distance(anchor, embed("other_person"))
        assert same < other, f"same={same:.4f} not closer than other={other:.4f}"
        assert same < 0.7
        assert other > 0.7

    def test_detection_is_deterministic(self, face_images, face_engine):
        image = np.asarray(Image.open(face_images["anchor"]).convert("RGB"), dtype=np.uint8)
        first, second = face_engine.detect(image), face_engine.detect(image)
        assert len(first) == len(second)
        assert first[0].bbox == second[0].bbox
