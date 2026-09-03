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
from sigil.evidence.merkle import MerkleTree, TamperReport, build, locate_tampering
from sigil.imaging import sha256_bytes
from sigil.models import SCHEMA_VERSION, ChainReceipt

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
PROOFS_NAME = "merkle-proofs.json"
RECEIPT_NAME = "receipt.json"
CONTEXT_NAME = "context.json"
SEARCH_DIR = "search"
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
    chain_checked: bool = False
    chain_ok: bool = False
    chain_detail: str = ""

    def summary(self) -> str:
        if self.ok:
            return "PASS, evidence intact and anchored"
        parts: list[str] = []
        if not self.tamper.ok:
            parts.append(self.tamper.summary())
        if self.artifact_failures:
            parts.append("artifact digest mismatch: " + ", ".join(self.artifact_failures))
        if self.chain_checked and not self.chain_ok:
            parts.append(self.chain_detail or "on-chain record missing")
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
    configuration: dict[str, Any],
    model_id: str,
    pipeline_version: str,
    search_routes: list[str],
) -> dict[str, Any]:
    """Assemble the deterministic section that gets hashed into the root."""

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

    stored_leaves = {field: str(entry["leaf"]) for field, entry in proofs.get("proofs", {}).items()}
    tamper = locate_tampering(manifest, root, stored_leaves)

    artifact_failures: list[str] = []
    if check_artifacts:
        digests = manifest.get("digests", {})
        for name, expected in (
            ("input.jpg", digests.get("input")),
            ("aligned_crop.jpg", digests.get("aligned_crop")),
            ("candidate.bin", digests.get("candidate_media")),
        ):
            artifact = path / MEDIA_DIR / name
            if not expected or not artifact.is_file():
                continue
            actual = sha256_bytes(artifact.read_bytes())
            if actual != expected:
                artifact_failures.append(f"{name} (expected {expected[:12]}…, got {actual[:12]}…)")

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
    "PROOFS_NAME",
    "RECEIPT_NAME",
    "SEARCH_DIR",
    "BundleError",
    "VerificationOutcome",
    "attach_receipt",
    "build_manifest",
    "load_manifest",
    "load_proofs",
    "verify_bundle",
    "write_bundle",
]
