"""Deterministic evidence: canonical JSON, Merkle proofs, and bundles."""

from sigil.evidence.canonical import canonicalize, loads_canonical
from sigil.evidence.merkle import (
    MerkleProof,
    MerkleTree,
    TamperReport,
    build,
    flatten,
    leaf_hash,
    locate_tampering,
    verify_proof,
)

__all__ = [
    "MerkleProof",
    "MerkleTree",
    "TamperReport",
    "build",
    "canonicalize",
    "flatten",
    "leaf_hash",
    "loads_canonical",
    "locate_tampering",
    "verify_proof",
]
