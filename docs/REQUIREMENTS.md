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

Discovery fans out across five sources that fail in different ways, run concurrently, so
the stage costs the slowest source rather than the sum:

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
gate, so a result that came first in the search cannot be promoted past verification. On
one run the expansion round surfaced a *different* Colin Powell's LinkedIn profile; it was
rejected at distance 0.878.

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
| Tests | `test/SigilRegistry.test.ts` (9 tests), `tests/test_evidence.py` |

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
unset**: the registry address is read from the bundle's own receipt and a public Sepolia
endpoint is used. A committed bundle is included at
[`docs/example-bundle/`](example-bundle/) so this can be checked without running a search.

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

`verify.html` re-implements the canonicalization and Merkle construction **in JavaScript**,
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
sigil index     # local face index: build, evaluate, search
sigil preflight # environment checks before a demo
```

`sigil serve` starts a local interface. It is a convenience for demonstrating the pipeline
visually, not the product, and it **binds to localhost only** by design: publishing a
face-search interface would let anyone submit anyone's face.

---

## Beyond the brief: a local face index

The brief does not ask for this. It exists because "why not just do what lenso.ai does"
is the obvious question, and the honest answer needs the parts separated.

A face-search engine is four things. Three of them are code, and they are built and
measured here:

| Part | Status |
|---|---|
| Embed a face | ArcFace, 512-d, already required by requirement 1 |
| Store N embeddings | [`sigil/index.py`](../sigil/index.py) |
| Nearest-neighbour search | one matrix multiply; embeddings are unit vectors, so cosine distance is a dot product |
| **Crawl social media to fill it** | **not code** |

Measured over LFW, leave-one-out, so a query never counts its own row as the answer:

| Metric | Value |
|---|---|
| Vectors / identities | 2,000 / 934 |
| Recall @ 1 | **0.9975** |
| Recall @ 10 | 0.9975 |
| Query latency | p50 **0.034 ms**, p95 0.044 ms |
| Index size | 3.82 MB |
| Detection coverage | 1.000 |

Search is exact, not approximate: every query scores every vector, so there is no recall
traded away for speed. At this size a matrix multiply beats building a graph; beyond roughly
a million vectors an HNSW or IVF-PQ structure would earn its complexity.

Two corpora ship, both defensible: `lfw`, a public research dataset, used to measure the
index honestly rather than to demo it; and `runs`, faces this tool already fetched during
its own searches, so a repeat query costs no API call.

**What is deliberately absent is the crawl.** Filling an index at lenso.ai's scale means
scraping Instagram, Facebook and LinkedIn, which breaks their terms, needs authentication
bypass and rotating infrastructure, and builds a permanent biometric record of millions of
people who were never asked. That is the decision that would raise recall on private
individuals the most, and it is a decision rather than an engineering problem. The machine
is built; what goes into it is left to whoever runs it.

```bash
sigil index build --corpus lfw --limit 2000
sigil index eval  --corpus lfw
sigil index search --image data/samples/public_figure.jpg
```

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
