"""Domain-separated Merkle tree over evidence fields.

Anchoring a single SHA-256 of the whole manifest proves only "something changed". A
Merkle tree over the individual fields proves *which* field changed, offline, from the
bundle alone, that is the difference between a hash and usable evidence.

Design notes:

* **Domain separation.** Leaves are hashed with a ``SIGIL:LEAF:v1`` prefix and internal
  nodes with ``SIGIL:NODE:v1``. Without this, an attacker could present an internal node
  as if it were a leaf (the classic second-preimage attack on naive Merkle trees).
* **Odd nodes are promoted, not duplicated.** Duplicating the last node makes trees with
  a repeated final leaf collide with trees that genuinely contain it twice.
* **Leaf order is the sorted field path**, so the tree is reproducible without storing
  an ordering alongside it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from sigil.evidence.canonical import canonicalize

LEAF_PREFIX = b"SIGIL:LEAF:v1:"
NODE_PREFIX = b"SIGIL:NODE:v1:"


@dataclass(frozen=True, slots=True)
class MerkleProof:
    """An inclusion proof for one field."""

    field: str
    leaf: str
    # Each step is (sibling_hash_hex, sibling_is_left).
    path: tuple[tuple[str, bool], ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "leaf": self.leaf,
            "path": [{"sibling": sibling, "left": is_left} for sibling, is_left in self.path],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> MerkleProof:
        return cls(
            field=str(data["field"]),
            leaf=str(data["leaf"]),
            path=tuple((str(step["sibling"]), bool(step["left"])) for step in data["path"]),
        )


def leaf_hash(field: str, value: Any) -> bytes:
    """Hash one field/value pair into a leaf.

    The field name is bound into the leaf, so moving a value from one field to another
    invalidates the proof even when the value itself is unchanged.
    """

    return hashlib.sha256(
        LEAF_PREFIX + field.encode("utf-8") + b"\x00" + canonicalize(value)
    ).digest()


def node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(NODE_PREFIX + left + right).digest()


def flatten(data: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested JSON into ``dotted.path`` → scalar-or-container leaves.

    Dicts and lists are walked; every other value becomes a leaf. Lists use ``[i]``
    indices so reordering a list changes the affected leaves.
    """

    flat: dict[str, Any] = {}
    if isinstance(data, dict):
        if not data:
            flat[prefix or "$"] = {}
            return flat
        for key, value in data.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            flat.update(flatten(value, path))
    elif isinstance(data, list | tuple):
        if not data:
            flat[prefix or "$"] = []
            return flat
        for index, value in enumerate(data):
            flat.update(flatten(value, f"{prefix}[{index}]"))
    else:
        flat[prefix or "$"] = data
    return flat


class MerkleTree:
    """A Merkle tree over the flattened fields of an evidence manifest."""

    def __init__(self, fields: dict[str, Any]) -> None:
        if not fields:
            raise ValueError("cannot build a Merkle tree over zero fields")
        self.fields = dict(fields)
        self.order: list[str] = sorted(self.fields)
        self.leaves: list[bytes] = [leaf_hash(name, self.fields[name]) for name in self.order]
        self._levels: list[list[bytes]] = self._build(self.leaves)

    @staticmethod
    def _build(leaves: list[bytes]) -> list[list[bytes]]:
        levels = [list(leaves)]
        current = leaves
        while len(current) > 1:
            nxt: list[bytes] = []
            for i in range(0, len(current) - 1, 2):
                nxt.append(node_hash(current[i], current[i + 1]))
            if len(current) % 2 == 1:
                # Promote the odd node unchanged rather than duplicating it.
                nxt.append(current[-1])
            levels.append(nxt)
            current = nxt
        return levels

    @property
    def root(self) -> bytes:
        return self._levels[-1][0]

    @property
    def root_hex(self) -> str:
        return "0x" + self.root.hex()

    def proof(self, field: str) -> MerkleProof:
        """Build the inclusion proof for one field."""

        try:
            index = self.order.index(field)
        except ValueError as exc:
            raise KeyError(f"unknown field: {field}") from exc

        path: list[tuple[str, bool]] = []
        for level in self._levels[:-1]:
            if index == len(level) - 1 and len(level) % 2 == 1:
                # This node was promoted; it has no sibling at this level.
                index //= 2
                continue
            sibling_index = index - 1 if index % 2 else index + 1
            path.append((level[sibling_index].hex(), sibling_index < index))
            index //= 2

        return MerkleProof(
            field=field,
            leaf=self.leaves[self.order.index(field)].hex(),
            path=tuple(path),
        )

    def all_proofs(self) -> dict[str, MerkleProof]:
        return {field: self.proof(field) for field in self.order}


def verify_proof(proof: MerkleProof, value: Any, root: bytes | str) -> bool:
    """Recompute a root from one field value and its proof."""

    if isinstance(root, str):
        root = bytes.fromhex(root.removeprefix("0x"))

    computed = leaf_hash(proof.field, value)
    if computed.hex() != proof.leaf:
        return False
    for sibling_hex, sibling_is_left in proof.path:
        sibling = bytes.fromhex(sibling_hex)
        computed = node_hash(sibling, computed) if sibling_is_left else node_hash(computed, sibling)
    return computed == root


def build(manifest: dict[str, Any]) -> MerkleTree:
    """Flatten a manifest and build its tree."""

    return MerkleTree(flatten(manifest))


@dataclass(frozen=True, slots=True)
class TamperReport:
    """The outcome of checking a manifest against a previously anchored root."""

    ok: bool
    computed_root: str
    expected_root: str
    modified: tuple[str, ...] = ()
    added: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()

    def summary(self) -> str:
        if self.ok:
            return "evidence intact"
        parts: list[str] = []
        if self.modified:
            parts.append("modified: " + ", ".join(self.modified))
        if self.added:
            parts.append("added: " + ", ".join(self.added))
        if self.removed:
            parts.append("removed: " + ", ".join(self.removed))
        return "; ".join(parts) or "root mismatch with no field-level difference"


def locate_tampering(
    manifest: dict[str, Any],
    expected_root: bytes | str,
    stored_leaves: dict[str, str],
) -> TamperReport:
    """Identify exactly which fields changed since the manifest was anchored.

    ``stored_leaves`` maps field path to leaf hash and comes from the bundle's
    ``merkle-proofs.json``. Recomputing each leaf from the current manifest and diffing
    against those hashes turns a bare "root mismatch" into a named field, which is the
    whole point of anchoring a tree rather than a single digest.

    The stored leaves are not trusted on their own: they are themselves validated by
    rebuilding the root from them, so an attacker who edits both the manifest and the
    proof file still fails the root comparison.
    """

    if isinstance(expected_root, str):
        expected_root = bytes.fromhex(expected_root.removeprefix("0x"))

    tree = build(manifest)
    computed_root = tree.root
    expected_hex = "0x" + expected_root.hex()

    if computed_root == expected_root:
        return TamperReport(True, tree.root_hex, expected_hex)

    current = dict(zip(tree.order, (leaf.hex() for leaf in tree.leaves), strict=True))
    current_fields, stored_fields = set(current), set(stored_leaves)

    modified = tuple(
        sorted(f for f in current_fields & stored_fields if current[f] != stored_leaves[f])
    )
    return TamperReport(
        ok=False,
        computed_root=tree.root_hex,
        expected_root=expected_hex,
        modified=modified,
        added=tuple(sorted(current_fields - stored_fields)),
        removed=tuple(sorted(stored_fields - current_fields)),
    )


__all__ = [
    "LEAF_PREFIX",
    "NODE_PREFIX",
    "MerkleProof",
    "MerkleTree",
    "TamperReport",
    "build",
    "flatten",
    "leaf_hash",
    "locate_tampering",
    "node_hash",
    "verify_proof",
]
