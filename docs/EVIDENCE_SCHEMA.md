# Evidence schema v1

What Sigil hashes, what it deliberately does not, and how to verify a bundle by hand.

---

## The hashed / unhashed split

This is the most important decision in the format.

**Inside the Merkle root**, only facts that are identical on every rebuild of the same
evidence:

```
schema_version              1
pipeline_version            "0.3.0"
model                       "insightface/buffalo_l:scrfd_10g+arcface_w600k_r50"
configuration_sha256        digest of the canonicalized settings that produced the result
digests.input               sha256 of the input image bytes
digests.aligned_crop        sha256 of the 112x112 aligned crop
digests.candidate_media     sha256 of the fetched candidate bytes, taken BEFORE decoding
digests.search_response     sha256 of the canonicalized provider responses
post.canonical_url          normalized post URL
post.platform               "x", "instagram", …
post.post_id                extracted stable identifier
post.media_url              the URL the media actually came from after redirects
post.media_quality          ORIGINAL | OPENGRAPH | OEMBED | THUMBNAIL
post.discovered_at          when the search returned this candidate
post.search_routes          which routes surfaced it
decision.status             MATCH | NON_MATCH | INCONCLUSIVE
decision.distance           cosine distance
decision.threshold          the calibrated threshold applied
decision.candidate_face_index
decision.faces_detected
```

**Outside the root**, real, useful, and excluded on purpose:

| Field | Why it is excluded |
|---|---|
| stage timings | vary run to run; hashing them makes the root irreproducible |
| local filesystem paths | differ per machine |
| `receipt.json` | cannot exist before anchoring, including it would be circular |
| `context.json` | rejected candidates, entities inferred, provider search IDs |
| `report.html`, `contact_sheet.png` | presentation |

> The original scaffold hashed `datetime.now()` into its content hash. That single line
> made every record impossible to reproduce, and therefore worthless as evidence. The
> split above exists to make that class of mistake structural rather than a matter of
> care.

**Never anywhere in the bundle:** face embeddings, raw biometric vectors, or any personal
data beyond the public post URL. Embeddings are `exclude=True` on the pydantic model, so
they cannot reach serialization by accident.

---

## Bundle layout

```
data/bundles/<run-id>/
├── manifest.json          canonical bytes, hashes directly to the root
├── merkle-proofs.json     root, leaf count, and a per-field inclusion proof
├── context.json           audit context, deliberately outside the root
├── receipt.json           written only after a successful anchor
├── report.html            self-contained; images inlined as data URIs
├── contact_sheet.png      every candidate with its decision and distance
├── media/
│   ├── input.jpg
│   ├── aligned_crop.jpg
│   ├── annotated_input.jpg
│   └── candidate.bin
└── search/
    └── R1_lens-exact-full.json …   sanitized provider responses
```

`manifest.json` is stored in canonical form, so the bytes on disk hash to the root with
no re-serialization step between reading and verifying.

---

## Canonicalization

**RFC 8785 (JSON Canonicalization Scheme).** `json.dumps(sort_keys=True)` is not enough:
it leaves number formatting unspecified and escapes differently from the spec. Two
implementations that disagree on either produce different roots for the same evidence,
which defeats the purpose.

The three rules that matter:

| Rule | Effect |
|---|---|
| Keys sorted by **UTF-16 code unit** | Python's default code-point order differs above the BMP |
| ECMAScript number formatting | `1.0` → `1`, `-0.0` → `0`, `1e21` → `1e+21` |
| Minimal escaping, literal UTF-8 | `"café"` stays `"café"`, not `"café"` |

Golden-byte tests pin this. If canonicalization ever changes, every previously anchored
root becomes unverifiable, so the tests are a tripwire, not a formality.

---

## The Merkle tree

Leaves are the **flattened** manifest, `post.canonical_url`, `digests.input`, and so on, sorted by field path, so the tree is reproducible without storing an ordering.

```
leaf   = SHA256("SIGIL:LEAF:v1:" ‖ field_path ‖ 0x00 ‖ canonical_json(value))
node   = SHA256("SIGIL:NODE:v1:" ‖ left ‖ right)
```

Three deliberate choices:

- **Domain separation.** Different prefixes for leaves and nodes. Without this, an
  internal node can be presented as if it were a leaf, the classic second-preimage
  attack on naive Merkle trees.
- **The field path is bound into the leaf.** Moving a value from one field to another
  invalidates its proof even though the value itself is unchanged.
- **Odd nodes are promoted, not duplicated.** Duplicating the last node makes a tree with
  a repeated final leaf collide with one that genuinely contains it twice.

Only the 32-byte root and the schema version go on-chain.

---

## Verifying by hand

The bundle is designed so you never have to trust `sigil verify`.

```python
import json
from pathlib import Path
from sigil.evidence.canonical import canonicalize
from sigil.evidence.merkle import build, verify_proof, flatten

bundle = Path("data/bundles/<run-id>")
manifest = json.loads((bundle / "manifest.json").read_bytes())
proofs = json.loads((bundle / "merkle-proofs.json").read_text())

# 1. The stored manifest bytes are canonical.
assert (bundle / "manifest.json").read_bytes() == canonicalize(manifest)

# 2. The root rebuilds from the manifest.
tree = build(manifest)
assert tree.root_hex == proofs["root"]

# 3. Each field's inclusion proof checks out on its own.
from sigil.evidence.merkle import MerkleProof

flat = flatten(manifest)
for field, entry in proofs["proofs"].items():
    assert verify_proof(MerkleProof.from_json(entry), flat[field], tree.root)

# 4. The stored artifacts hash to the digests the manifest claims.
import hashlib

for name, key in [("input.jpg", "input"), ("candidate.bin", "candidate_media")]:
    data = (bundle / "media" / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == manifest["digests"][key]
```

Step 4 is the one people skip. Without it, a manifest can be internally consistent while
describing media that is no longer there.

Then read the chain, no key required:

```python
from sigil.chain import ChainClient

record = ChainClient(RPC_URL, CONTRACT_ADDRESS, expected_chain_id=None).read(proofs["root"])
assert record.exists
print(record.submitter, record.anchored_at, record.schema_version)
```

---

## Tamper localization

When the root does not match, `locate_tampering()` recomputes every leaf from the current
manifest and diffs against the leaves recorded in `merkle-proofs.json`, reporting
`modified`, `added`, and `removed` field paths.

The stored leaves are not trusted on their own: the root is rebuilt from the manifest
independently, so editing both the manifest *and* the proof file still fails the root
comparison. The leaves only make the failure **legible**, they cannot make a failure
pass.

---

## Versioning

`schema_version` is anchored alongside the root. A future format change increments it and
changes the domain-separation prefixes, so v1 roots stay verifiable under v1 rules and
cannot be confused with v2 roots.
