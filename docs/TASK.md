# Task 3, requirement by requirement

The brief asks for one pipeline end to end:

> Face scan input → Web/social media search (find matching post) → Blockchain upload
> and verification of the discovered data

This document maps each stated requirement to the code that satisfies it, the test that
covers it, and the command that demonstrates it live. Everything here is checkable in a
few minutes from a clean clone.

**One command runs all four checks against a real run:**

```bash
sigil prove --image data/samples/public_figure.jpg
```

It prints a PASS or FAIL per requirement with the evidence it observed. Requirements 3 and
4 shell out to a **separate process with `PRIVATE_KEY` unset**, because a proof you can
only check from inside the program that produced it is not a proof.

---

## 1. Face identification

> Detect and encode a face from an input image.

| | |
|---|---|
| Detector | SCRFD `det_10g`, ONNX, decoded in [`sigil/face/engine.py`](../sigil/face/engine.py) |
| Encoder | ArcFace `w600k_r50`, 512-d, L2-normalised, with flip test-time augmentation |
| Alignment | Umeyama similarity transform onto the canonical 112x112 frame, [`sigil/face/align.py`](../sigil/face/align.py) |
| Quality gate | face size, Laplacian sharpness, exposure, pose, truncation |
| Tests | `tests/test_face.py` |
| Command | `sigil run --image <path>` |

Run directly on `onnxruntime`. There is no TensorFlow and no DeepFace: the project started
on DeepFace, which does not import against TensorFlow 2.21 in this environment, so the
ONNX weights DeepFace wraps are run directly instead.

**Measured**, by [`sigil benchmark`](../sigil/bench.py) on LFW, written to
[`benchmark.json`](benchmark.json):

| Metric | Value |
|---|---|
| ROC-AUC | 0.99555 |
| EER | 0.01204 |
| TAR @ FMR 1e-3 | 0.98594 |
| False matches | 1 / 499 impostor pairs |
| Coverage | 0.997 (3 of 1000 pairs had no detectable face) |
| Calibration / test | 8,191 pairs / 997 disjoint held-out pairs |

**Those figures are at the calibrated threshold of 0.795. The shipped threshold is 0.55.**
The distinction matters and is easy to gloss over. LFW's impostors are random pairs of
different people; this pipeline's impostors are candidates a reverse-image search returned
*because they resemble the query*, which is a much harder distribution. Calibrating on the
first and deploying against the second let a stranger through at 0.6888 in a real run.

| | Calibrated (LFW) | Shipped |
|---|---|---|
| Match threshold | 0.795 | **0.55** |
| Reject threshold | 0.845 | **0.75** |
| Genuine pairs kept | 98.6% | **97.2%** |
| Margin to nearest measured impostor | 0.00 | **0.24** |

The shipped point costs about 1.4 points of recall and moves the 0.55-0.75 overlap band,
where genuine and impostor distances really do mix, into `INCONCLUSIVE` rather than a
confident wrong name. A test asserts the shipped value can only ever be stricter than the
calibrated one.

Embeddings are biometric data. They stay in memory, and are never logged, written to the
evidence bundle, or put on chain.

---

## 2. Social media / web search

> Use the face to search the web and find at least one real, matching social media post.
> This should be a genuine search step, not a hardcoded result.

Discovery uses a bounded fan-out across four provider families plus a capped page harvest,
all of which fail in different ways.
The first Lens call seeds any inferred name; the crop, web, Exa, and Wikidata lookups then
run concurrently where credentials and the per-run budget allow. The stage is bounded by
the slowest active lookup rather than an unbounded crawl:

| Source | What it is good at | Needs |
|---|---|---|
| SerpApi Google Lens | the visual index: this photo, republished | `SERPAPI_KEY` |
| SerpApi Web | a name, restricted to social domains | `SERPAPI_KEY` |
| Exa | neural retrieval of pages *about* a person | `EXA_API_KEY` (optional) |
| Wikidata / Commons | curated full-resolution portraits | nothing |
| Page harvest | every photograph on a page, not just the one the engine picked | nothing |

Code: [`sigil/search/routes.py`](../sigil/search/routes.py),
[`sigil/search/providers/`](../sigil/search/providers/). Tests: `tests/test_search.py`,
`tests/test_providers.py`.

**How you can tell it is not hardcoded.** Every provider call is recorded with the
provider's own search id, shown in the UI's audit panel and in `receipt`/`context.json`,
and checkable against the provider's dashboard. `--no-cache` forces live calls. A
hardcoded answer has no search id.

**Genuine outcomes include finding nothing.** `SEARCH_EMPTY` (the provider returned
nothing) and `SEARCH_UNAVAILABLE` (the provider failed) are kept distinct, and
`NO_VERIFIED_MATCH` means candidates were examined and none passed the face gate. None of
the three is ever dressed up as a success.

**Search never decides identity.** Candidates are ranked only after they pass the face
gate, so a result that came first in the search cannot be promoted past verification. A
lookalike surfaced by the entity-pivot route is still rejected when its face distance is
outside the calibrated gate. Ranking among verified matches is the distance margin alone,
the only calibrated quantity available to it.

**The whole verified set is recorded, not only the winner.** The manifest commits to how
many candidates passed, how many are distinct photographs rather than reposts of the
submitted image, the best and runner-up distances, and the media digests of every verified
candidate. One candidate that scraped past the threshold and four that cleared it
comfortably are different claims, and anchoring only the winner could not tell them
apart.

---

## 3. Blockchain verification

> Upload the post, or a hash of it, to a blockchain to create a verifiable,
> tamper-evident record. Demonstrate re-verifying the data against the on-chain record.

| | |
|---|---|
| Chain | Ethereum Sepolia, chain id `11155111` |
| Contract | [`contracts/SigilRegistry.sol`](../contracts/SigilRegistry.sol) |
| Address | [`0xc92D2fDe2757b14e5EB6Aef849E41aBa1F395517`](https://sepolia.etherscan.io/address/0xc92D2fDe2757b14e5EB6Aef849E41aBa1F395517) |
| Source verified | [Sourcify **exact match**](https://repo.sourcify.dev/11155111/0xc92D2fDe2757b14e5EB6Aef849E41aBa1F395517/) on creation *and* runtime bytecode |
| Gas | under 80k per anchor, asserted in the contract test suite |
| Tests | `tests/contracts/SigilRegistry.test.ts` (9 tests), `tests/test_evidence.py` |

What is anchored is a **32-byte Merkle root** over the canonical evidence manifest.
Fields are serialised with RFC 8785 (JSON Canonicalization Scheme) and hashed into
domain-separated leaves, so the same evidence always produces the same root on any machine.

**No personal data goes on chain.** No image, no name, no post text, no embedding. The
contract stores a hash, a submitter address, a timestamp and a schema number. That is a
privacy requirement, not an optimisation: biometric data must not be written to an
immutable public ledger.

**Re-verification needs no private key.** It is a read-only `eth_call`:

```bash
sigil verify --bundle data/bundles/<run-id>
```

This works from a clean checkout with `CONTRACT_ADDRESS` and `SEPOLIA_RPC_URL` **both
unset**, falling back to a public Sepolia endpoint. A committed bundle is included at
[`docs/example-bundle/`](example-bundle/) so this can be checked without running a search.

**Which registry is read, and why the order matters.** Configuration wins, then the
checked-in `deployments/sepolia.json`, and only then the bundle's own `receipt.json`. The
receipt is last on purpose: it travels with the evidence, so a tampered one must not be
able to redirect an otherwise clean verifier.

**The contract's identity is checked before its answer is believed.** A `true` from
`verify(root)` proves only that *some* contract said yes, and a look-alike with the same
ABI can say yes to everything. Both verifiers hash the deployed runtime bytecode with
`eth_getCode` and compare it against the digest recorded at deploy time:

```console
$ sigil verify --bundle docs/example-bundle
  ✓ registry: runtime bytecode matches the recorded registry (c5708dce1ff2…)

$ CONTRACT_ADDRESS=<a different real Sepolia contract> sigil verify --bundle docs/example-bundle
on-chain check failed: the contract at 0xfFf9…6B14 is not SigilRegistry   # exit 3
```

The check is three-state. `unrecorded`, for a chain with no committed deployment such as a
local node, is reported as such and never as a pass.

---

## 4. Tamper evidence

The brief asks for a *tamper-evident* record. A single hash of the whole document would
tell you only that something changed. Per-field Merkle inclusion proofs tell you **which**
field changed:

```console
$ sigil verify --bundle data/bundles/<run-id>
╭─ Local verification FAIL ────────────────────────────────╮
│ FAIL, modified: post.canonical_url                       │
╰──────────────────────────────────────────────────────────╯
  ✗ tampered field: post.canonical_url
$ echo $?
3
```

Exit codes are stable: `0` verified, `3` verification failed, `2` configuration error.
A failure can never render as a green pass; when the root is genuinely anchored but the
bundle no longer matches it, the output says exactly that.

`sigil/static/verify.html` re-implements the canonicalization and Merkle construction **in JavaScript**,
opens from a `file://` URL with nothing installed, and reaches the same verdict. If two
independent implementations agree, the format is what is being verified rather than one
codebase's behaviour.

---

## 5. No website required

> You do not need to build or host a project website. Focus your time on the pipeline.

Understood, and the pipeline is the deliverable. Everything is reachable from the CLI:

```bash
sigil run       # face -> search -> verify -> evidence -> anchor
sigil verify    # re-verify a bundle, with or without the chain
sigil anchor    # anchor a bundle built earlier
sigil prove     # all four requirements, checked end to end
sigil benchmark # measured accuracy, with denominators
sigil preflight # environment checks before a demo
```

`sigil serve` starts a local interface. It is a convenience for demonstrating the pipeline
visually, not the product, and it **binds to localhost only** by design: publishing a
face-search interface would let anyone submit anyone's face.

---

---

## Honest limitations

Stating these is part of the work, not an apology for it.

- **Reverse image search is the ceiling, not the model.** Google Lens deliberately does not
  do face recognition. Handed a portrait it matches objects; handed a face crop it returns
  strangers who look similar. It finds a specific person only when that person is already
  indexed and captioned somewhere Google crawls. Given the right page, the face gate
  confirms a match at distance 0.24, so recognition is not the weak link.
- **LFW is easier than the open web.** It is frontal, well lit and celebrity-heavy. Numbers
  measured on it are an optimistic bound for social-media media.
- **Thresholds are calibrated against random impostors and deployed against lookalikes.**
  A similarity search surfaces hard negatives by construction, so the shipped operating
  point is deliberately tighter than the calibration suggests. See the comment in
  [`sigil/verify.py`](../sigil/verify.py).
- **Sepolia is a testnet.** Its history carries no economic guarantee. The same code anchors
  to mainnet unchanged if that guarantee is needed.
