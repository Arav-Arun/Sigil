# Architecture

The pipeline diagram lives in the [README](../README.md). This document covers the
decisions behind it, what was chosen, what was rejected, and why.

---

## 1. Running ONNX weights directly instead of DeepFace

**Forced by a blocker.** DeepFace does not import in this environment: `retinaface` calls
`validate_for_keras3()`, which raises against TensorFlow 2.21 because `tf-keras` is
absent. The face stack had never executed once.

The obvious fix, `pip install tf-keras`, buys a working import and inherits everything
else: ~600 MB of TensorFlow, a model reload on every call, and alignment behaviour buried
inside a wrapper. DeepFace's strongest models are InsightFace ONNX weights anyway, so
Sigil loads those directly.

| | DeepFace path | Sigil |
|---|---|---|
| Imports | no | yes |
| Install | ~600 MB | ~90 MB |
| Detector | RetinaFace (TF) | SCRFD `det_10g` (ONNX) |
| Embedder | via wrapper | ArcFace `w600k_r50` (ONNX) |
| Per-call cost | model reload | warmed sessions |
| Embedding latency |, | 9.7 ms (CoreML) vs 33.8 ms (CPU) |

**The cost:** ~200 lines of SCRFD anchor decoding and NMS written here. That was validated
bit-exact against the InsightFace reference implementation, bbox delta 0.000 px, embedding
cosine 1.000000, and that validation immediately paid for itself by catching a real bug:
InsightFace feeds its models through `cv2.dnn.blobFromImage(..., swapRB=True)` from a BGR
source, so the networks actually see **RGB**. Sigil works in RGB natively and was swapping
to BGR, which dropped embedding agreement to cosine 0.94. A wrapper would have hidden that.

**Provider split.** Measured on an Apple M4:

| Model | CPU | CoreML | Shipped |
|---|---|---|---|
| SCRFD detect | 47.7 ms | 48.0 ms | **CPU** |
| ArcFace embed | 33.8 ms | 9.7 ms | **CoreML** |

CoreML cannot partition SCRFD's graph, so it falls back to CPU and only emits noise. The
embedder gains 3.5×. Outputs across providers agree to 2.4e-07, so the split costs nothing
in correctness.

---

## 2. Four search routes with progressive escalation

Reverse image search on a bare face crop rarely surfaces social posts. Four complementary
routes run cheapest-and-most-precise first, stopping as soon as enough candidates exist:

| Route | Question it asks | When it wins |
|---|---|---|
| R1 | Lens exact-match, full image | the exact photo was reposted |
| R2 | Lens visual-match, full image | same scene or person, different crop |
| R3 | Lens visual-match, aligned crop | background context is misleading |
| R4 | entity pivot → `site:`-restricted web search | the person is *named* but the photo is not indexed on social |

**R4 deserves scrutiny.** It takes the identity Lens infers and searches social domains
for that name. It is a *recall* mechanism only, a name never establishes identity in
Sigil. Every post it surfaces still has to pass the same face gate as any other candidate.
Without that separation, R4 would be a way to launder a guess into a result.

**Progressive escalation** exists because of the free tier: a run that finds enough
candidates in R1 spends one search, not four.

---

## 3. Identity and ranking are separate

The single most important invariant in the codebase:

> A high search rank, an exact-image flag, or agreement across routes can order candidates
> that already passed the face gate. **None of them can grant identity.**

`sigil/verify.py` enforces it structurally: `rank()` filters to `matched` candidates
before scoring anything. There is no code path where a ranking signal can promote a
`NON_MATCH`.

This is tested with the case that would actually occur in the wild, a stranger returned
as search result #1 with the exact-image flag set. It scores 0.9247, is rejected, and the
real subject at rank #2 (0.2111) wins.

**Best-face-wins, not largest-face.** Candidate images routinely contain several people.
All detected faces are embedded in a single batched `session.run` and the closest one
decides. Comparing only the largest face is the standard way to miss a true match in a
group photo.

**Three states, not two.** `INCONCLUSIVE` covers a distance inside the uncertainty band,
media that could not be fetched, media that could not be decoded, and a face too small to
be decisive. "We could not look" and "we looked and it is not them" are different claims,
and collapsing them is how a pipeline ends up asserting things it did not check.

---

## 4. Thresholds are measured, not chosen

`sigil benchmark` calibrates on the LFW train and 10-fold splits and reports on the
disjoint test split. Tuning and reporting on the same pairs is the most common way a
face-recognition benchmark misleads.

The operating point targets **FMR 1e-3**, not a balanced error rate, because falsely
naming a stranger on social media is far worse than abstaining. The first calibration at
FMR 1e-2 produced 8 false matches in 497 held-out impostor pairs; at 1e-3 it produces 0,
and a test fails the build if that regresses.

A second test pins `DEFAULT_MATCH_THRESHOLD` to the value in `docs/benchmark.json`. Without
it, someone can hand-tune a threshold and the published accuracy table silently stops
describing the running code.

---

## 5. A Merkle tree, not a hash

Anchoring `SHA256(whole_document)` proves only that *something* changed. A tree over the
flattened fields proves **which** field changed, offline, from the bundle alone. That is
the difference between "verification failed" and "`post.canonical_url` was edited".

Details, including domain separation and the odd-node rule, are in
[EVIDENCE_SCHEMA.md](EVIDENCE_SCHEMA.md).

**Rejected: putting more on-chain.** Storing the post URL or a signature bundle on-chain
was considered and dropped. The contract holds a root, a submitter, a timestamp, and a
schema version, nothing else. Everything interesting is verifiable off-chain by anyone
holding the bundle, and biometric or personal data must never be written to an immutable
public ledger.

---

## 6. Failure can never look like success

Concrete mechanisms rather than good intentions:

| Risk | Mechanism |
|---|---|
| A zero/empty root anchored by a broken run | contract reverts `ZeroRoot`; client refuses locally too |
| An unknown root reading as a zero struct | `get()` reverts `UnknownRoot`; `verify()` returns an explicit boolean |
| Anchoring to the wrong network | chain ID asserted before signing; bytecode presence checked |
| A reverted transaction counted as success | receipt `status` checked, then the root is **re-read** before success is reported |
| A confirmation timeout causing a duplicate anchor | raises `CHAIN_PENDING`; `sigil anchor` resumes the existing bundle |
| A provider outage reported as "nothing found" | `SEARCH_UNAVAILABLE` and `SEARCH_EMPTY` are distinct codes |
| A silently over-spent search quota | budget `spend()` raises rather than returning a boolean a caller can ignore |
| Verification requiring the claimant's key | verification is a read-only `eth_call` |

Exit codes are stable: `0` verified, `1` honest negative, `2` configuration, `3`
verification failed.

**Stage ordering follows from this.** Discovery and evidence construction complete before
the chain is touched, so a chain outage costs the anchoring step and nothing else, the
bundle exists, has a root, and can be anchored later without repeating a single search.

---

## 7. Privacy

- Embeddings are `exclude=True` on the pydantic model, so they cannot reach serialization
  by accident. They are never logged and never written to a bundle.
- Nothing personal goes on-chain, only a hash.
- No cookies, no authentication to any platform, no scraping behind a login.
- `scripts/secret_scan.py` runs in CI and rejects a tracked `.env`, tracked evidence
  bundles, biometric artifacts, and key-shaped literals.
- Generated evidence is gitignored by default.

---

## Module map

| Module | Responsibility |
|---|---|
| `sigil/imaging.py` | validated ingest: magic bytes, EXIF, sRGB, bomb caps, digests |
| `sigil/face/engine.py` | ONNX sessions, SCRFD decode, NMS, batched embedding |
| `sigil/face/align.py` | Umeyama similarity transform to the canonical 112×112 frame |
| `sigil/face/quality.py` | explainable quality signals and warnings |
| `sigil/search/normalize.py` | URL canonicalization, hostname-parsed allowlist, post IDs |
| `sigil/search/serpapi.py` | provider client: retries, error classification, budget |
| `sigil/search/routes.py` | the four routes and progressive escalation |
| `sigil/search/cache.py` | content-addressed response cache |
| `sigil/search/quota.py` | account status, per-run budget, demo reserve |
| `sigil/candidates.py` | bounded async fetch and the media resolution ladder |
| `sigil/verify.py` | identity gate, three-state decision, ranking |
| `sigil/evidence/canonical.py` | RFC 8785 JCS |
| `sigil/evidence/merkle.py` | domain-separated tree, proofs, tamper localization |
| `sigil/evidence/bundle.py` | bundle layout, hashed/unhashed split, offline verify |
| `sigil/chain.py` | anchoring and key-free verification |
| `sigil/run.py` | stage orchestration and timing |
| `sigil/report.py` | contact sheet and self-contained HTML report |
| `sigil/bench.py` | calibration and measurement |
