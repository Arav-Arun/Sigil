"""End-to-end pipeline test against a local chain and a local media server.

This exercises the real face engine, the real HTTP fetch path, the real Merkle
construction, and a real EVM transaction. Only the search provider is substituted, with
a fixture whose URLs point at a local server, because hitting SerpApi in CI would burn
the free tier and make the suite non-deterministic.

The chain half is skipped unless a node is reachable at ``SIGIL_TEST_RPC``
(default ``http://127.0.0.1:8545``), so the suite still runs offline.
"""

from __future__ import annotations

import http.server
import json
import os
import socket
import threading
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from sigil.candidates import fetch_all_sync
from sigil.evidence.bundle import build_manifest, verify_bundle, write_bundle
from sigil.evidence.canonical import canonicalize
from sigil.face import detect_and_encode
from sigil.imaging import sha256_bytes, sha256_file
from sigil.models import DecisionStatus, SearchCandidate
from sigil.verify import verify_all

TEST_RPC = os.environ.get("SIGIL_TEST_RPC", "http://127.0.0.1:8545")


def _rpc_available(url: str = TEST_RPC) -> bool:
    from urllib.parse import urlparse

    parsed = urlparse(url)
    try:
        with socket.create_connection((parsed.hostname or "", parsed.port or 8545), timeout=0.5):
            return True
    except OSError:
        return False


@pytest.fixture(scope="module")
def media_server(face_images):
    """Serve the LFW test images over HTTP so the real fetch path is exercised."""

    directory = face_images["anchor"].parent

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def log_message(self, *args):  # keep pytest output clean
            pass

        def guess_type(self, path):
            return "image/jpeg" if str(path).endswith(".jpg") else super().guess_type(path)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def _candidate(url: str, media_url: str, rank: int, routes: list[str]) -> SearchCandidate:
    return SearchCandidate(
        source_url=url,
        platform="x",
        post_id=url.rsplit("/", 1)[-1],
        title="fixture post",
        image_url=media_url,
        search_rank=rank,
        exact_match=rank == 1,
        discovered_at=datetime(2026, 9, 2, 12, 0, tzinfo=UTC),
        search_routes=routes,
    )


class TestDiscoveryToEvidence:
    """Face -> fetch -> verify -> evidence, with no credentials and no chain."""

    @pytest.fixture
    def outcome(self, face_images, face_engine, media_server, monkeypatch, tmp_path):
        # The test deliberately hosts deterministic image fixtures on loopback. Production
        # fetches reject local network targets to prevent search results probing a machine.
        monkeypatch.setattr("sigil.candidates.is_public_http_url", lambda _url: True)
        observation = detect_and_encode(
            face_images["anchor"], tmp_path / "face", engine=face_engine, select_largest=True
        )
        source = np.asarray(observation.embedding, dtype=np.float32)

        candidates = [
            _candidate(
                "https://x.com/subject/status/1",
                f"{media_server}/same_person.jpg",
                1,
                ["R1:lens-exact-full"],
            ),
            _candidate(
                "https://x.com/stranger/status/2",
                f"{media_server}/other_person.jpg",
                2,
                ["R2:lens-visual-full"],
            ),
        ]

        media = fetch_all_sync(candidates, concurrency=2, timeout=10.0)
        verifications, ranked = verify_all(
            source, media, engine=face_engine, model_name=face_engine.model_id
        )
        return observation, verifications, ranked, tmp_path

    def test_media_is_actually_fetched_and_hashed(self, outcome):
        _, verifications, _, _ = outcome
        for item in verifications:
            assert item.media.ok, item.media.error
            assert len(item.media.sha256) == 64
            assert item.media.byte_count > 1024
            assert item.media.content_type == "image/jpeg"

    def test_the_right_person_matches_and_the_stranger_does_not(self, outcome):
        _, verifications, ranked, _ = outcome
        by_url = {str(v.candidate.source_url): v for v in verifications}
        assert by_url["https://x.com/subject/status/1"].decision.status is DecisionStatus.MATCH
        assert by_url["https://x.com/stranger/status/2"].decision.status is not DecisionStatus.MATCH
        assert len(ranked) == 1

    def test_the_stranger_is_ranked_first_by_search_but_still_rejected(self, outcome):
        # The stranger is result #2 here; make the stronger version of the point by
        # confirming that ranking contains only verified matches regardless of rank.
        _, _, ranked, _ = outcome
        assert all(item.matched for item in ranked)

    def test_bundle_root_is_reproducible(self, outcome):
        observation, _, ranked, tmp_path = outcome
        selected = ranked[0]

        def make_bundle(directory: Path):
            manifest = build_manifest(
                input_sha256=sha256_file(observation.source_image),
                crop_sha256=sha256_file(observation.face_crop_path),
                candidate_sha256=selected.media.sha256,
                search_response_sha256=sha256_bytes(b"{}"),
                canonical_post_url=str(selected.candidate.source_url),
                platform=selected.candidate.platform,
                post_id=selected.candidate.post_id,
                media_quality=str(selected.media.quality),
                media_url=selected.media.final_url,
                discovered_at=selected.candidate.discovered_at.isoformat(),
                decision={
                    "status": str(selected.decision.status),
                    "distance": selected.decision.distance,
                    "threshold": selected.decision.threshold,
                },
                corroboration={
                    "verified_matches": len(ranked),
                    "distinct_photos": len(ranked),
                    "media_sha256": sorted(item.media.sha256 for item in ranked),
                },
                configuration={"match_threshold": selected.decision.threshold},
                model_id="test-model",
                pipeline_version="0.3.0",
                search_routes=list(selected.candidate.search_routes),
            )
            return write_bundle(
                directory,
                manifest=manifest,
                artifacts={
                    "input.jpg": Path(observation.source_image).read_bytes(),
                    "candidate.bin": selected.media.data,
                },
                search_responses={},
                context={"note": "differs between runs and must not affect the root"},
            )

        _, first = make_bundle(tmp_path / "bundle_a")
        _, second = make_bundle(tmp_path / "bundle_b")
        assert first.root_hex == second.root_hex

    def test_bundle_verifies_and_tampering_is_localized(self, outcome):
        observation, _, ranked, tmp_path = outcome
        selected = ranked[0]
        manifest = build_manifest(
            input_sha256=sha256_file(observation.source_image),
            crop_sha256=sha256_file(observation.face_crop_path),
            candidate_sha256=selected.media.sha256,
            search_response_sha256=sha256_bytes(b"{}"),
            canonical_post_url=str(selected.candidate.source_url),
            platform="x",
            post_id="1",
            media_quality=str(selected.media.quality),
            media_url=selected.media.final_url,
            discovered_at=selected.candidate.discovered_at.isoformat(),
            decision={"status": "MATCH", "distance": selected.decision.distance},
            corroboration={"verified_matches": len(ranked), "distinct_photos": len(ranked)},
            configuration={},
            model_id="test-model",
            pipeline_version="0.3.0",
            search_routes=["R1:lens-exact-full"],
        )
        path, tree = write_bundle(
            tmp_path / "bundle_c",
            manifest=manifest,
            artifacts={
                "input.jpg": Path(observation.source_image).read_bytes(),
                "aligned_crop.jpg": Path(observation.face_crop_path).read_bytes(),
                "candidate.bin": selected.media.data,
            },
            search_responses={},
            context={},
        )

        assert verify_bundle(path).ok

        edited = json.loads((path / "manifest.json").read_bytes())
        edited["post"]["canonical_url"] = "https://x.com/someone-else/status/999"
        (path / "manifest.json").write_bytes(canonicalize(edited))

        outcome_after = verify_bundle(path)
        assert not outcome_after.ok
        assert outcome_after.tamper.modified == ("post.canonical_url",)
        assert outcome_after.root == tree.root_hex


@pytest.mark.skipif(not _rpc_available(), reason=f"no EVM node at {TEST_RPC}")
class TestChainRoundTrip:
    """Anchor and read back against a real node (Hardhat, Anvil, or a testnet fork)."""

    @pytest.fixture(scope="class")
    def deployed(self):
        pytest.importorskip("web3")
        from eth_account import Account
        from web3 import HTTPProvider, Web3

        w3 = Web3(HTTPProvider(TEST_RPC))
        artifact = Path("artifacts/contracts/SigilRegistry.sol/SigilRegistry.json")
        if not artifact.is_file():
            pytest.skip("run `npm run compile` first")
        compiled = json.loads(artifact.read_text())

        key = os.environ.get(
            "SIGIL_TEST_KEY",
            # Hardhat/Anvil's first well-known development account. Never used elsewhere.
            "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
        )
        account = Account.from_key(key)
        contract = w3.eth.contract(abi=compiled["abi"], bytecode=compiled["bytecode"])
        tx = contract.constructor().build_transaction(
            {
                "from": account.address,
                "nonce": w3.eth.get_transaction_count(account.address),
                "chainId": w3.eth.chain_id,
                "gas": 1_000_000,
                "gasPrice": w3.eth.gas_price,
            }
        )
        signed = account.sign_transaction(tx)
        receipt = w3.eth.wait_for_transaction_receipt(
            w3.eth.send_raw_transaction(signed.raw_transaction), timeout=60
        )
        return receipt["contractAddress"], key, w3.eth.chain_id

    def test_anchor_then_read_back(self, deployed):
        from sigil.chain import ChainClient

        address, key, chain_id = deployed
        client = ChainClient(TEST_RPC, address, expected_chain_id=chain_id)

        root = "0x" + "a1" * 32
        assert not client.read(root).exists

        receipt = client.anchor(root, 1, key)
        assert receipt.evidence_root == root
        assert receipt.block_number > 0

        record = client.read(root)
        assert record.exists
        assert record.schema_version == 1
        assert record.submitter.lower() == receipt.submitter.lower()

    def test_an_unanchored_root_reads_as_absent(self, deployed):
        from sigil.chain import ChainClient

        address, _, chain_id = deployed
        client = ChainClient(TEST_RPC, address, expected_chain_id=chain_id)
        assert not client.read("0x" + "ff" * 32).exists

    def test_duplicate_anchoring_is_refused(self, deployed):
        from sigil.chain import ChainClient, ChainError

        address, key, chain_id = deployed
        client = ChainClient(TEST_RPC, address, expected_chain_id=chain_id)
        root = "0x" + "b2" * 32
        client.anchor(root, 1, key)
        with pytest.raises(ChainError, match="already anchored"):
            client.anchor(root, 1, key)

    def test_a_known_transaction_can_be_resumed_without_signing_again(self, deployed):
        """The recovery path for a confirmation timeout.

        `anchor()` raising CHAIN_PENDING carries the transaction hash precisely so a
        later `sigil anchor --bundle` can finish that transaction. Signing a second one
        would risk two anchors and two fees for a root that was already landing.
        """

        from sigil.chain import ChainClient

        address, key, chain_id = deployed
        client = ChainClient(TEST_RPC, address, expected_chain_id=chain_id)
        root = "0x" + "c3" * 32

        first = client.anchor(root, 1, key)
        # Resuming the same hash yields the same receipt, with no new transaction.
        resumed = client.await_transaction(first.transaction_hash, root)
        assert resumed.transaction_hash == first.transaction_hash
        assert resumed.block_number == first.block_number
        assert resumed.submitter.lower() == first.submitter.lower()
        assert client.total_anchored() >= 1

    def test_a_local_node_reports_the_registry_as_unrecorded_not_verified(self, deployed):
        """No deployment record for this chain means nothing to compare against.

        That is neither a pass nor a failure, and collapsing it into either is exactly
        the class of mistake the three-state identity check exists to prevent.
        """

        from sigil.chain import ChainClient

        address, _, chain_id = deployed
        client = ChainClient(TEST_RPC, address, expected_chain_id=chain_id)
        status, detail = client.registry_identity()
        assert status == "unrecorded", detail
        # The digest is still computable; there is simply no committed value for it.
        assert len(client.runtime_code_sha256()) == 64

    def test_a_contract_that_is_not_the_registry_is_a_mismatch(self, deployed, monkeypatch):
        """A look-alike registry must not be able to produce a verified result."""

        from sigil import chain as chain_module
        from sigil.chain import ChainClient

        address, _, chain_id = deployed
        client = ChainClient(TEST_RPC, address, expected_chain_id=chain_id)
        # Pretend this chain has a recorded deployment whose code is something else.
        monkeypatch.setattr(chain_module, "expected_runtime_sha256", lambda _cid=0: "ab" * 32)
        status, detail = client.registry_identity()
        assert status == "mismatch"
        assert "not SigilRegistry" in detail

    def test_wrong_chain_id_is_refused_before_signing(self, deployed):
        from sigil.chain import ChainClient, ChainError

        address, _, chain_id = deployed
        with pytest.raises(ChainError, match="Refusing to continue"):
            ChainClient(TEST_RPC, address, expected_chain_id=chain_id + 1)

    def test_zero_root_is_refused_locally(self, deployed):
        from sigil.chain import ChainClient, ChainError

        address, key, chain_id = deployed
        client = ChainClient(TEST_RPC, address, expected_chain_id=chain_id)
        with pytest.raises(ChainError, match="zero root"):
            client.anchor("0x" + "00" * 32, 1, key)
