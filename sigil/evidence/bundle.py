"""Evidence bundle construction and verification.

A bundle is a self-contained directory that a stranger can check without running the
pipeline, without any credential, and (until the final on-chain step) without a network.

The critical design rule is the **hashed/unhashed split**. Only deterministic facts go
inside the Merkle tree:

* what was searched (input digest, model and configuration versions);
* what was found (canonical post URL, platform, post id, media digest, search response
  digest);
* what was concluded (decision, distance, thresholds).

Everything that varies between runs of the same evidence, local paths, wall-clock
timings, the transaction receipt that can only exist *after* anchoring, stays outside
the root. Hashing ``datetime.now()`` into the root, as the original scaffold did, makes
the record impossible to reproduce and therefore worthless.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sigil.evidence.canonical import canonicalize
from sigil.evidence.merkle import (
    MerkleProof,
    MerkleTree,
    TamperReport,
    build,
    flatten,
    locate_tampering,
    verify_proof,
)
from sigil.imaging import sha256_bytes
from sigil.models import SCHEMA_VERSION, ChainReceipt

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
PROOFS_NAME = "merkle-proofs.json"
RECEIPT_NAME = "receipt.json"
PENDING_NAME = "pending.json"
CONTEXT_NAME = "context.json"
SEARCH_DIR = "search"
RESPONSES_NAME = "responses.json"
MEDIA_DIR = "media"


class BundleError(RuntimeError):
    """Raised when a bundle is malformed or missing required parts."""


@dataclass(frozen=True, slots=True)
class VerificationOutcome:
    """Result of checking a bundle. ``ok`` is only true when *everything* passed."""

    ok: bool
    root: str
    tamper: TamperReport
    artifact_failures: tuple[str, ...] = ()

    def summary(self) -> str:
        """Describe the offline result. The on-chain half is reported by the caller."""

        if self.ok:
            return "PASS, evidence intact"
        parts: list[str] = []
        if not self.tamper.ok:
            parts.append(self.tamper.summary())
        if self.artifact_failures:
            parts.append("artifact digest mismatch: " + ", ".join(self.artifact_failures))
        return "FAIL, " + "; ".join(parts)


def build_manifest(
    *,
    input_sha256: str,
    crop_sha256: str,
    candidate_sha256: str,
    search_response_sha256: str,
    canonical_post_url: str,
    platform: str,
    post_id: str,
    media_quality: str,
    media_url: str,
    discovered_at: str,
    decision: dict[str, Any],
    corroboration: dict[str, Any],
    configuration: dict[str, Any],
    model_id: str,
    pipeline_version: str,
    search_routes: list[str],
) -> dict[str, Any]:
    """Assemble the deterministic section that gets hashed into the root.

    ``corroboration`` commits to the *set* of verified matches, not just the one that
    ranked first. A single candidate that scraped past the threshold and three
    independent photographs that cleared it comfortably are very different claims, and a
    manifest that records only the winner cannot tell them apart. Anchoring the set means
    the strength of the finding is part of what was sealed, so it cannot be quietly
    restated afterwards as stronger than it was.
    """

    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": pipeline_version,
        "model": model_id,
        "configuration_sha256": sha256_bytes(canonicalize(configuration)),
        "digests": {
            "input": input_sha256,
            "aligned_crop": crop_sha256,
            "candidate_media": candidate_sha256,
            "search_response": search_response_sha256,
        },
        "post": {
            "canonical_url": canonical_post_url,
            "platform": platform,
            "post_id": post_id,
            "media_url": media_url,
            "media_quality": media_quality,
            "discovered_at": discovered_at,
            "search_routes": sorted(search_routes),
        },
        "decision": decision,
        "corroboration": corroboration,
    }


def write_bundle(
    directory: str | Path,
    *,
    manifest: dict[str, Any],
    artifacts: dict[str, bytes],
    search_responses: dict[str, Any],
    context: dict[str, Any],
) -> tuple[Path, MerkleTree]:
    """Write a complete bundle and return its path and Merkle tree."""

    path = Path(directory)
    (path / SEARCH_DIR).mkdir(parents=True, exist_ok=True)
    (path / MEDIA_DIR).mkdir(parents=True, exist_ok=True)

    tree = build(manifest)

    # The manifest is stored in canonical form so the file on disk hashes to the root
    # byte-for-byte, with no re-serialization step in between.
    (path / MANIFEST_NAME).write_bytes(canonicalize(manifest))

    proofs = {
        "root": tree.root_hex,
        "schema_version": manifest["schema_version"],
        "leaf_count": len(tree.order),
        "proofs": {field: proof.to_json() for field, proof in tree.all_proofs().items()},
    }
    (path / PROOFS_NAME).write_text(json.dumps(proofs, indent=2), encoding="utf-8")

    for name, payload in artifacts.items():
        (path / MEDIA_DIR / name).write_bytes(payload)

    for route, payload in search_responses.items():
        safe = route.replace(":", "_").replace("/", "_")
        (path / SEARCH_DIR / f"{safe}.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
    # This canonical aggregate is the exact byte sequence whose digest is committed
    # in the manifest. Per-route files remain for people inspecting the bundle.
    (path / SEARCH_DIR / RESPONSES_NAME).write_bytes(canonicalize(search_responses))

    # Context is explicitly outside the root: useful, but not part of the claim.
    (path / CONTEXT_NAME).write_text(
        json.dumps(
            {**context, "written_at": datetime.now(UTC).isoformat(), "root": tree.root_hex},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    logger.info("evidence bundle written to %s (root %s)", path, tree.root_hex)
    return path, tree


def attach_receipt(directory: str | Path, receipt: ChainReceipt) -> Path:
    """Store the anchoring receipt *after* the root exists. Never inside the root."""

    path = Path(directory) / RECEIPT_NAME
    path.write_text(receipt.model_dump_json(indent=2), encoding="utf-8")
    return path


def write_pending(directory: str | Path, transaction_hash: str, chain_id: int) -> Path:
    """Record a submitted-but-unconfirmed anchor transaction.

    Written when the chain accepted the transaction but it did not confirm in time. It
    exists so the resume path waits for *that* transaction instead of signing a second
    one for the same root, which is how a timeout turns into two anchors and a wasted
    fee. Removed as soon as a receipt lands. Outside the Merkle root: operational
    state, not part of the claim.
    """

    path = Path(directory) / PENDING_NAME
    path.write_text(
        json.dumps({"transaction_hash": transaction_hash, "chain_id": chain_id}, indent=2),
        encoding="utf-8",
    )
    return path


def load_pending(directory: str | Path) -> str:
    """The pending transaction hash for this bundle, or empty if there is none."""

    path = Path(directory) / PENDING_NAME
    if not path.is_file():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(data.get("transaction_hash") or "")


def clear_pending(directory: str | Path) -> None:
    """Drop the pending marker once the anchor is confirmed."""

    (Path(directory) / PENDING_NAME).unlink(missing_ok=True)


def load_manifest(directory: str | Path) -> dict[str, Any]:
    path = Path(directory) / MANIFEST_NAME
    if not path.is_file():
        raise BundleError(f"no {MANIFEST_NAME} in {directory}")
    try:
        data: dict[str, Any] = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise BundleError(f"could not read {path}: {exc}") from exc
    return data


def load_proofs(directory: str | Path) -> dict[str, Any]:
    path = Path(directory) / PROOFS_NAME
    if not path.is_file():
        raise BundleError(f"no {PROOFS_NAME} in {directory}")
    try:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError(f"could not read {path}: {exc}") from exc
    return data


def verify_bundle(
    directory: str | Path,
    *,
    expected_root: str | None = None,
    check_artifacts: bool = True,
) -> VerificationOutcome:
    """Verify a bundle offline: rebuild the root and re-digest every stored artifact."""

    path = Path(directory)
    manifest = load_manifest(path)
    proofs = load_proofs(path)

    root = expected_root or str(proofs.get("root", ""))
    if not root:
        raise BundleError("no expected root available to verify against")

    try:
        proof_entries = proofs["proofs"]
        if not isinstance(proof_entries, dict):
            raise TypeError("proofs must be an object")
        stored_leaves = {field: str(entry["leaf"]) for field, entry in proof_entries.items()}
    except (KeyError, TypeError) as exc:
        raise BundleError(f"malformed {PROOFS_NAME}: {exc}") from exc
    tamper = locate_tampering(manifest, root, stored_leaves)

    artifact_failures: list[str] = []
    if check_artifacts:
        digests = manifest.get("digests", {})
        media_dir = path / MEDIA_DIR
        # Compact, manifest-only examples intentionally omit the media directory. Once a
        # bundle contains that directory, however, silently accepting a deleted file
        # makes the completeness check meaningless.
        if media_dir.is_dir():
            for name, expected in (
                ("input.jpg", digests.get("input")),
                ("aligned_crop.jpg", digests.get("aligned_crop")),
                ("candidate.bin", digests.get("candidate_media")),
            ):
                artifact = media_dir / name
                if not expected:
                    artifact_failures.append(f"{name} (digest missing from manifest)")
                elif not artifact.is_file():
                    artifact_failures.append(f"{name} (missing)")
                else:
                    actual = sha256_bytes(artifact.read_bytes())
                    if actual != expected:
                        artifact_failures.append(
                            f"{name} (expected {expected[:12]}…, got {actual[:12]}…)"
                        )

        # search/responses.json is the canonical aggregate whose digest the manifest
        # commits to. The per-route files beside it are for humans and are not hashed.
        # Manifest-only examples ship no search directory at all; once one exists,
        # a missing aggregate is a deletion, not an omission.
        search_dir = path / SEARCH_DIR
        aggregate = search_dir / RESPONSES_NAME
        if search_dir.is_dir() and not aggregate.is_file():
            artifact_failures.append(f"search responses ({RESPONSES_NAME} missing)")
        elif aggregate.is_file():
            expected_search = digests.get("search_response")
            try:
                actual_search = sha256_bytes(
                    canonicalize(json.loads(aggregate.read_text(encoding="utf-8")))
                )
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
                artifact_failures.append(f"search responses (unreadable: {exc})")
            else:
                if not expected_search:
                    artifact_failures.append("search responses (digest missing from manifest)")
                elif actual_search != expected_search:
                    artifact_failures.append(
                        "search responses "
                        f"(expected {expected_search[:12]}…, got {actual_search[:12]}…)"
                    )

        # A valid root is not enough if the stored per-field inclusion proofs were
        # corrupted or replaced. Verify their coverage, identity and path to the root.
        current_fields = flatten(manifest)
        if set(proof_entries) != set(current_fields):
            artifact_failures.append("merkle proofs (field coverage mismatch)")
        if proofs.get("leaf_count") != len(current_fields):
            artifact_failures.append("merkle proofs (leaf count mismatch)")
        for field, value in current_fields.items():
            entry = proof_entries.get(field)
            if entry is None:
                continue
            try:
                proof = MerkleProof.from_json(entry)
                valid = proof.field == field and verify_proof(proof, value, root)
            except (KeyError, TypeError, ValueError):
                valid = False
            if not valid:
                artifact_failures.append(f"merkle proof ({field})")

    return VerificationOutcome(
        ok=tamper.ok and not artifact_failures,
        root=root,
        tamper=tamper,
        artifact_failures=tuple(artifact_failures),
    )


__all__ = [
    "CONTEXT_NAME",
    "MANIFEST_NAME",
    "MEDIA_DIR",
    "PENDING_NAME",
    "PROOFS_NAME",
    "RECEIPT_NAME",
    "RESPONSES_NAME",
    "SEARCH_DIR",
    "BundleError",
    "VerificationOutcome",
    "attach_receipt",
    "build_manifest",
    "clear_pending",
    "load_manifest",
    "load_pending",
    "load_proofs",
    "verify_bundle",
    "write_bundle",
    "write_pending",
]
