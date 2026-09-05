"""Canonical JSON, Merkle proofs, tamper localization, and bundle round-trips."""

from __future__ import annotations

import copy
import json

import pytest

from sigil.evidence.bundle import (
    BundleError,
    build_manifest,
    load_manifest,
    verify_bundle,
    write_bundle,
)
from sigil.evidence.canonical import canonicalize, loads_canonical
from sigil.evidence.merkle import (
    MerkleTree,
    build,
    flatten,
    leaf_hash,
    locate_tampering,
    verify_proof,
)
from sigil.imaging import sha256_bytes


class TestCanonical:
    def test_orders_keys_and_strips_whitespace(self):
        assert canonicalize({"b": 1, "a": 2}) == b'{"a":2,"b":1}'

    def test_matches_rfc8785_number_formatting(self):
        assert canonicalize([1.0, 0.5, -0.0, 100, 1e21]) == b"[1,0.5,0,100,1e+21]"

    def test_uses_fixed_notation_at_the_jcs_lower_boundary(self):
        assert canonicalize([1e-6, -2e-6, 1e-7]) == b"[0.000001,-0.000002,1e-7]"

    def test_nested_structures_round_trip(self):
        value = {"z": [1, {"y": None}], "a": {"b": True, "c": False}}
        assert loads_canonical(canonicalize(value)) == value

    def test_escapes_control_characters(self):
        assert canonicalize({"k": "a\nb\tc"}) == b'{"k":"a\\nb\\tc"}'

    def test_unicode_is_emitted_literally(self):
        assert canonicalize({"k": "café"}) == '{"k":"café"}'.encode()

    def test_rejects_non_finite_numbers(self):
        with pytest.raises(ValueError):
            canonicalize({"k": float("nan")})
        with pytest.raises(ValueError):
            canonicalize({"k": float("inf")})

    def test_is_stable_across_dict_insertion_order(self):
        first = {"alpha": 1, "beta": {"x": 1, "y": 2}}
        second = {"beta": {"y": 2, "x": 1}, "alpha": 1}
        assert canonicalize(first) == canonicalize(second)

    def test_golden_bytes_do_not_drift(self):
        # A pinned expectation: if canonicalization ever changes, every previously
        # anchored root silently becomes unverifiable. This test is the tripwire.
        manifest = {"schema_version": 1, "post": {"url": "https://x.com/a/status/1"}}
        assert canonicalize(manifest) == (
            b'{"post":{"url":"https://x.com/a/status/1"},"schema_version":1}'
        )


class TestFlatten:
    def test_produces_dotted_paths(self):
        assert flatten({"a": {"b": 1}}) == {"a.b": 1}

    def test_indexes_lists(self):
        assert flatten({"a": [10, 20]}) == {"a[0]": 10, "a[1]": 20}

    def test_keeps_empty_containers_as_leaves(self):
        assert flatten({"a": {}, "b": []}) == {"a": {}, "b": []}


class TestMerkle:
    @pytest.fixture
    def manifest(self):
        return {
            "post": {"url": "https://x.com/a/status/1", "platform": "x"},
            "decision": {"status": "MATCH", "distance": 0.28},
            "digests": {"input": "ab" * 32, "candidate": "cd" * 32},
        }

    def test_root_is_deterministic(self, manifest):
        assert build(manifest).root_hex == build(copy.deepcopy(manifest)).root_hex

    def test_root_changes_when_any_value_changes(self, manifest):
        other = copy.deepcopy(manifest)
        other["decision"]["distance"] = 0.29
        assert build(manifest).root_hex != build(other).root_hex

    def test_leaf_binds_the_field_name(self):
        # Moving a value to a different field must invalidate its proof.
        assert leaf_hash("a", "value") != leaf_hash("b", "value")

    def test_domain_separation_between_leaves_and_nodes(self, manifest):
        tree = build(manifest)
        # An internal node must never be presentable as a leaf.
        assert tree.root not in tree.leaves

    def test_every_field_has_a_valid_inclusion_proof(self, manifest):
        tree = build(manifest)
        flat = flatten(manifest)
        for field, proof in tree.all_proofs().items():
            assert verify_proof(proof, flat[field], tree.root), field

    def test_proof_rejects_a_substituted_value(self, manifest):
        tree = build(manifest)
        proof = tree.proof("post.url")
        assert not verify_proof(proof, "https://x.com/a/status/999", tree.root)

    def test_odd_leaf_counts_build_correctly(self):
        for count in (1, 2, 3, 5, 7, 9, 33):
            tree = MerkleTree({f"f{i}": i for i in range(count)})
            flat = {f"f{i}": i for i in range(count)}
            for field, proof in tree.all_proofs().items():
                assert verify_proof(proof, flat[field], tree.root), (count, field)

    def test_rejects_an_empty_field_set(self):
        with pytest.raises(ValueError):
            MerkleTree({})


class TestTamperLocalization:
    @pytest.fixture
    def setup(self):
        manifest = {
            "post": {"url": "https://x.com/a/status/1", "platform": "x"},
            "decision": {"status": "MATCH", "distance": 0.28},
        }
        tree = build(manifest)
        stored = {field: proof.leaf for field, proof in tree.all_proofs().items()}
        return manifest, tree, stored

    def test_intact_manifest_passes(self, setup):
        manifest, tree, stored = setup
        assert locate_tampering(manifest, tree.root, stored).ok

    def test_names_the_single_modified_field(self, setup):
        manifest, tree, stored = setup
        tampered = copy.deepcopy(manifest)
        tampered["post"]["url"] = "https://x.com/impostor/status/2"
        report = locate_tampering(tampered, tree.root, stored)
        assert not report.ok
        assert report.modified == ("post.url",)
        assert "post.url" in report.summary()

    def test_names_every_modified_field(self, setup):
        manifest, tree, stored = setup
        tampered = copy.deepcopy(manifest)
        tampered["post"]["url"] = "https://x.com/impostor/status/2"
        tampered["decision"]["distance"] = 0.01
        report = locate_tampering(tampered, tree.root, stored)
        assert report.modified == ("decision.distance", "post.url")

    def test_detects_an_added_field(self, setup):
        manifest, tree, stored = setup
        tampered = copy.deepcopy(manifest)
        tampered["decision"]["confidence"] = 0.99
        report = locate_tampering(tampered, tree.root, stored)
        assert report.added == ("decision.confidence",)

    def test_detects_a_removed_field(self, setup):
        manifest, tree, stored = setup
        tampered = copy.deepcopy(manifest)
        del tampered["post"]["platform"]
        report = locate_tampering(tampered, tree.root, stored)
        assert report.removed == ("post.platform",)

    def test_a_one_character_edit_is_caught(self, setup):
        manifest, tree, stored = setup
        tampered = copy.deepcopy(manifest)
        tampered["post"]["url"] = "https://x.com/a/status/2"  # 1 -> 2
        assert locate_tampering(tampered, tree.root, stored).modified == ("post.url",)


class TestBundle:
    @pytest.fixture
    def manifest(self):
        return build_manifest(
            input_sha256="ab" * 32,
            crop_sha256="cd" * 32,
            candidate_sha256="ef" * 32,
            search_response_sha256=sha256_bytes(canonicalize({})),
            canonical_post_url="https://x.com/a/status/1",
            platform="x",
            post_id="1",
            media_quality="ORIGINAL",
            media_url="https://pbs.twimg.com/media/x.jpg",
            discovered_at="2026-09-02T12:00:00+00:00",
            decision={"status": "MATCH", "distance": 0.28, "threshold": 0.6},
            corroboration={"verified_matches": 2, "distinct_photos": 2, "media_sha256": []},
            configuration={"match_threshold": 0.6},
            model_id="insightface/buffalo_l",
            pipeline_version="0.3.0",
            search_routes=["R1:lens-exact-full"],
        )

    def test_manifest_excludes_wall_clock_time(self, manifest):
        # Hashing "now" would make the root irreproducible, which is exactly the bug
        # the original scaffold shipped.
        assert "hashed_at" not in json.dumps(manifest)

    def test_write_and_verify_round_trip(self, tmp_path, manifest):
        path, tree = write_bundle(
            tmp_path / "bundle",
            manifest=manifest,
            artifacts={"input.jpg": b"x" * 2048},
            search_responses={"R1:lens-exact-full": {"search_metadata": {"id": "abc"}}},
            context={"run_id": "test"},
        )
        outcome = verify_bundle(path, check_artifacts=False)
        assert outcome.ok
        assert outcome.root == tree.root_hex

    def test_manifest_on_disk_hashes_to_the_root(self, tmp_path, manifest):
        path, tree = write_bundle(
            tmp_path / "bundle",
            manifest=manifest,
            artifacts={},
            search_responses={},
            context={},
        )
        assert build(load_manifest(path)).root_hex == tree.root_hex

    def test_editing_the_manifest_fails_verification_and_names_the_field(self, tmp_path, manifest):
        path, _ = write_bundle(
            tmp_path / "bundle",
            manifest=manifest,
            artifacts={},
            search_responses={},
            context={},
        )
        edited = load_manifest(path)
        edited["post"]["canonical_url"] = "https://x.com/impostor/status/9"
        (path / "manifest.json").write_bytes(canonicalize(edited))

        outcome = verify_bundle(path, check_artifacts=False)
        assert not outcome.ok
        assert outcome.tamper.modified == ("post.canonical_url",)
        assert outcome.summary().startswith("FAIL")

    def test_corrupting_a_stored_artifact_is_detected(self, tmp_path, manifest):
        payload = b"y" * 4096
        import hashlib

        digest = hashlib.sha256(payload).hexdigest()
        manifest["digests"]["input"] = digest
        manifest["digests"]["aligned_crop"] = digest
        manifest["digests"]["candidate_media"] = digest
        path, _ = write_bundle(
            tmp_path / "bundle",
            manifest=manifest,
            artifacts={
                "input.jpg": payload,
                "aligned_crop.jpg": payload,
                "candidate.bin": payload,
            },
            search_responses={},
            context={},
        )
        assert verify_bundle(path).ok

        (path / "media" / "input.jpg").write_bytes(payload[:-1] + b"z")
        outcome = verify_bundle(path)
        assert not outcome.ok
        assert outcome.artifact_failures

    def test_deleting_a_stored_artifact_is_detected(self, tmp_path, manifest):
        payload = b"z" * 2048
        import hashlib

        digest = hashlib.sha256(payload).hexdigest()
        manifest["digests"].update(
            {"input": digest, "aligned_crop": digest, "candidate_media": digest}
        )
        path, _ = write_bundle(
            tmp_path / "bundle",
            manifest=manifest,
            artifacts={
                "input.jpg": payload,
                "aligned_crop.jpg": payload,
                "candidate.bin": payload,
            },
            search_responses={},
            context={},
        )
        (path / "media" / "candidate.bin").unlink()

        outcome = verify_bundle(path)
        assert not outcome.ok
        assert "candidate.bin (missing)" in outcome.artifact_failures

    def test_editing_stored_search_response_is_detected(self, tmp_path, manifest):
        response = {"R1:lens-exact-full": {"search_metadata": {"id": "abc"}}}
        manifest["digests"]["search_response"] = sha256_bytes(canonicalize(response))
        payload = b"x"
        digest = sha256_bytes(payload)
        manifest["digests"].update(
            {"input": digest, "aligned_crop": digest, "candidate_media": digest}
        )
        path, _ = write_bundle(
            tmp_path / "bundle",
            manifest=manifest,
            artifacts={
                "input.jpg": payload,
                "aligned_crop.jpg": payload,
                "candidate.bin": payload,
            },
            search_responses=response,
            context={},
        )
        (path / "search" / "responses.json").write_text('{"forged":true}', encoding="utf-8")

        outcome = verify_bundle(path)
        assert not outcome.ok
        assert any("search responses" in item for item in outcome.artifact_failures)

    def test_missing_bundle_raises(self, tmp_path):
        with pytest.raises(BundleError):
            verify_bundle(tmp_path / "nope")
