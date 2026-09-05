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
| Embedding latency | not measurable, it does not run | 9.7 ms (CoreML) vs 33.8 ms (CPU) |

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

## 2. Four discovery routes with bounded fan-out

Reverse image search on a bare face crop rarely surfaces social posts. Four complementary
routes cover context, portrait crops, and named pages:

| Route | Question it asks | When it wins |
|---|---|---|
| R1 | Lens `type=all`, full image | photo context or an indexed entity is useful |
| R2 | Lens `type=all`, portrait crop | the whole image is dominated by background or clothing |
| R3 | Lens `type=all`, aligned face crop | only the face carries useful visual signal |
| R4 | entity pivot → `site:`-restricted web search | the person is *named* but the photo is not indexed on social |

**R4 deserves scrutiny.** It takes the identity Lens infers and searches social domains
for that name. It is a *recall* mechanism only, a name never establishes identity in
Sigil. Every post it surfaces still has to pass the same face gate as any other candidate.
Without that separation, R4 would be a way to launder a guess into a result.

The three Lens queries run when their crops are available, and R4 runs only when a name
was inferred and the per-run search budget allows it. Non-social result pages are harvested
with a small two-page cap so a portfolio can contribute more than the one image returned by
the search provider without turning page crawling into an unbounded crawl.

---

## 3. Identity and ranking are separate

The single most important invariant in the codebase:

> Search rank or agreement across routes can help order candidates that already passed the
> face gate. **None of them can grant identity.**

`sigil/verify.py` enforces it structurally: `rank()` filters to `matched` candidates
before scoring anything. There is no code path where a ranking signal can promote a
`NON_MATCH`.

**Ranking is the margin, and nothing else.** It used to be a weighted sum of five
hand-chosen coefficients over margin, media quality, route agreement, face size and
search rank. None of the five was measured, and four were proxies for the same thing:
how much signal the comparison had. Ordering decides which post gets anchored, so it
should be defensible on its own terms, and one measured quantity is easier to defend
than five invented ones. Two editorial tie-breaks survive and are labelled as such: a
social post outranks a news photograph because a social post is the deliverable, and an
independently found photograph outranks a repost of the submitted image.

Provider result buckets are treated as discovery hints only. A candidate still has to pass
the face gate after its downloaded media is fetched, regardless of which Lens bucket found
the URL.

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

### The resolution sweep, and a hypothesis that did not survive it

The one production false match on record, a stranger at 0.6888, was on a 112px thumbnail,
which suggested small faces should face a tighter threshold. `sigil benchmark` tests that
directly by downscaling LFW and re-detecting, which is what a search-provider thumbnail
actually is:

| face px | genuine p97 | impostor min |
|---:|---:|---:|
| 29 | 0.5220 | 0.8620 |
| 39 | 0.4923 | 0.8495 |
| 54 | 0.4695 | 0.8448 |
| 73 | 0.4699 | 0.8367 |
| 98 | 0.4669 | 0.8387 |

Impostors do not get closer as the face shrinks. What degrades is the genuine side, p97
climbing 0.467 to 0.522, so low resolution costs **recall, not precision**. A size penalty
would tighten a bar that is not the problem and spend recall where it is already worst,
so the shipped penalty is zero and a test pins it there.

The caveat is stated in the code and in `docs/benchmark.json`: LFW impostors are random
strangers, and a reverse image search returns look-alikes on purpose. This rules out the
simple explanation for the 0.6888 match, not the hard-negative one, which LFW cannot
sample. What the sweep does justify is `MIN_CANDIDATE_FACE_PX`: below roughly 48px the
genuine distribution starts colliding with the 0.55 operating point.

The same run calibrates the repost test, which had two invented constants. Re-encoding a
photo the way a search index does reaches p99 mean error 5.61 and dhash 0.0586, while the
nearest genuinely *different* photograph sits at 24.0 and 0.238. The shipped dhash bound
was 0.04, below the same-photo 99th percentile, so it was quietly showing genuine reposts
as independent discoveries. It is now 0.07, with over 3x separation on both signals.

---

## 5. Verification checks the contract, not just the answer

Reading `verify(root)` and getting `true` back proves that *some* contract said yes. It
does not prove which contract. That gap is reachable rather than theoretical: the address
can come from the bundle's own `receipt.json`, so a forged bundle can ship a pointer to a
contract with the same ABI that returns `true` for every root, and the verifier renders a
green pass.

So both verifiers hash the deployed runtime bytecode with `eth_getCode` and compare it to
the digest `deployments/sepolia.json` recorded at deploy time, before the read:

```console
$ sigil verify --bundle docs/example-bundle
  ✓ registry: runtime bytecode matches the recorded registry (c5708dce1ff2…)
╭─ On-chain verification PASS ─╮
```

```console
$ CONTRACT_ADDRESS=<some other real Sepolia contract> sigil verify --bundle docs/example-bundle
on-chain check failed: the contract at 0xfFf9…6B14 is not SigilRegistry:
runtime bytecode digest 9bbda01ae25d… does not match the recorded c5708dce1ff2…
$ echo $?
3
```

The check is deliberately **three-state**, not boolean: `verified`, `mismatch`, and
`unrecorded` for a chain with no committed deployment, such as a local test node. Folding
`unrecorded` into either edge would be the same mistake the rest of this document exists
to avoid, so it renders as `?` and is never reported as a pass.

The browser verifier does the identical check in JavaScript against a pinned digest, and a
test fails the build if that constant ever drifts from `deployments/sepolia.json` — without
it, a redeploy would leave the page rejecting the genuine registry, which reads as "the
evidence is bad" rather than "the page is stale".

---

## 6. A Merkle tree, not a hash

Anchoring `SHA256(whole_document)` proves only that *something* changed. A tree over the
flattened fields proves **which** field changed, offline, from the bundle alone. That is
the difference between "verification failed" and "`post.canonical_url` was edited".

Details, including domain separation and the odd-node rule, are in
[EVIDENCE.md](EVIDENCE.md).

**Rejected: putting more on-chain.** Storing the post URL or a signature bundle on-chain
was considered and dropped. The contract holds a root, a submitter, a timestamp, and a
schema version, nothing else. Everything interesting is verifiable off-chain by anyone
holding the bundle, and biometric or personal data must never be written to an immutable
public ledger.

---

## 7. Failure can never look like success

Concrete mechanisms rather than good intentions:

| Risk | Mechanism |
|---|---|
| A zero/empty root anchored by a broken run | contract reverts `ZeroRoot`; client refuses locally too |
| An unknown root reading as a zero struct | `get()` reverts `UnknownRoot`; `verify()` returns an explicit boolean |
| Anchoring to the wrong network | chain ID asserted before signing; bytecode presence checked |
| A reverted transaction counted as success | receipt `status` checked, then the root is **re-read** before success is reported |
| A confirmation timeout causing a duplicate anchor | raises `CHAIN_PENDING` carrying the transaction hash, which is written to `pending.json`; `sigil anchor --bundle` waits for *that* transaction instead of signing a second one |
| A look-alike contract answering `verify()` for every root | the deployed runtime bytecode is hashed and compared against `deployments/sepolia.json` **before** the read; a mismatch is exit `3`, not a pass |
| A provider outage reported as "nothing found" | `SEARCH_UNAVAILABLE` and `SEARCH_EMPTY` are distinct codes |
| A silently over-spent search quota | budget `spend()` raises rather than returning a boolean a caller can ignore |
| Verification requiring the claimant's key | verification is a read-only `eth_call` |

Exit codes are stable: `0` verified, `1` honest negative, `2` configuration, `3`
verification failed.

**Stage ordering follows from this.** Discovery and evidence construction complete before
the chain is touched, so a chain outage costs the anchoring step and nothing else, the
bundle exists, has a root, and can be anchored later without repeating a single search.

---

## 8. Privacy

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
| `sigil/search/providers/` | the four discovery sources behind one interface |
| `sigil/search/routes.py` | the four discovery routes and bounded page harvest |
| `sigil/search/cache.py` | content-addressed response cache |
| `sigil/search/quota.py` | account status, per-run budget, demo reserve |
| `sigil/candidates.py` | bounded async fetch and the media resolution ladder |
| `sigil/verify.py` | identity gate, three-state decision, ranking |
| `sigil/evidence/canonical.py` | RFC 8785 JCS |
| `sigil/evidence/merkle.py` | domain-separated tree, proofs, tamper localization |
| `sigil/evidence/bundle.py` | bundle layout, hashed/unhashed split, offline verify |
| `sigil/chain.py` | anchoring, registry identity, key-free verification |
| `sigil/run.py` | stage orchestration and timing |
| `sigil/report.py` | contact sheet and self-contained HTML report |
| `sigil/bench.py` | calibration and measurement |
| `sigil/prove.py` | checks all four task requirements against one real run |
| `sigil/preflight.py` | pre-demo environment and credential checks |
| `sigil/web.py` | localhost-only UI server for `sigil/static/` |
