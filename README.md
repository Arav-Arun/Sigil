# Sigil

**A face, sealed.**

Sigil takes a face image, genuinely searches the live web for social-media posts showing
that person, independently verifies the match with face recognition, and anchors a
tamper-evident Merkle proof of the finding on Ethereum Sepolia.

```
face image → live reverse-image search → verified social post → Merkle evidence → Sepolia anchor → independent re-verification
```

Built for **HH Goa 2026 Shortlisting Task 3**.

**Every requirement, checked against a real run, in one command:**

```bash
sigil prove --image data/samples/public_figure.jpg
```

```
 1  face identification   PASS  SCRFD det_10g + ArcFace w600k_r50, 512-d
                                LFW ROC-AUC 0.99555 on 997 held-out pairs
 2  genuine web search    PASS  25 candidates fetched and face-checked, 13 passed
                                R1:lens-all-full: id 6a987bc2f73e4d260952920a
                                verified post youtube.com/watch?v=… at distance 0.2594
 3  blockchain record     PASS  root anchored on Sepolia, re-read from a separate
                                process with PRIVATE_KEY unset: matches
 4  tamper evidence       PASS  one character changed -> FAIL naming post.canonical_url
                                exit 3; restored -> PASS
```

Requirements 3 and 4 run in a **separate process with `PRIVATE_KEY` unset**, because a
proof you can only check from inside the program that produced it is not a proof.

[`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md) maps each requirement to its code, its test
and its command. [`docs/WHITEBOARD.md`](docs/WHITEBOARD.md) is the explainer script.

---

## What makes this different

Most pipelines of this shape can only tell you *that* something changed. Sigil tells you
**which field** changed, offline, from the bundle alone:

```console
$ sigil verify --bundle data/bundles/20260902T144231Z-a3f9c1d2
╭─ Local verification FAIL ────────────────────────────────────────╮
│ root 0x7c4e…b721                                                 │
│ modified: post.canonical_url                                     │
╰──────────────────────────────────────────────────────────────────╯
  ✗ tampered field: post.canonical_url
```

Three more properties the design insists on:

- **Verification needs no private key.** Re-verifying is a read-only `eth_call`. A proof
  that required the claimant's key would not be a proof.
- **Uncertainty is never converted into a result.** Identity is `MATCH`, `NON_MATCH`, or
  `INCONCLUSIVE`. A high search rank cannot promote a candidate that failed the face
  gate, and an outage (`SEARCH_UNAVAILABLE`) is never reported as "nothing found"
  (`SEARCH_EMPTY`).
- **Every accuracy claim below is measured**, by a script in this repo, with denominators.

---

## Measured results

Apple M4, 32 GB, Python 3.12.13. Reproduce with `sigil benchmark`.

| Metric | Value |
|---|---|
| Model | InsightFace `buffalo_l`, SCRFD `det_10g` + ArcFace `w600k_r50`, with flip TTA |
| Dataset | LFW (`sklearn.datasets.fetch_lfw_pairs`, funneled, colour) |
| Calibration / held-out | 8,191 pairs / 997 pairs (disjoint) |
| ROC-AUC | **0.99555** |
| EER | **0.01204** |
| TAR @ FMR 1e-2 | **0.9879** |
| TAR @ FMR 1e-3 | **0.9859** |
| False matches | **1 / 499** impostor pairs |
| False non-matches | 6 / 498 genuine pairs |
| Detection coverage | **0.997** (12 faces not found across all splits) |
| Detect latency | p50 **55 ms**, p95 97 ms (CPU) |
| Embed latency | p50 **34 ms**, p95 48 ms (CoreML, flip TTA = 2 passes) |
| Engine agreement | bit-exact with the InsightFace reference (cos **1.000000**) |
| CoreML vs CPU | numerically identical (max Δ **2.4e-07**) |

**The shipped operating point is deliberately tighter than this table.** The numbers above
answer "what does LFW allow", and LFW's impostors are *random pairs of different people*.
This pipeline never sees those. A reverse-image search returns candidates precisely because
they resemble the query, so every negative it must survive is drawn from the hard tail of
that distribution.

Measured on the held-out split: genuine pairs reach p95 **0.4867**, and the single closest
impostor sits at **0.7924**. Calibrating at FMR 1e-3 lands on 0.795, which puts the boundary
*on the impostor edge* with no margin, and a real run then matched a stranger at **0.6888**.
So the shipped threshold is **0.55**, just above the genuine 97th percentile:

| | Calibrated (LFW) | Shipped |
|---|---|---|
| Match threshold | 0.795 | **0.55** |
| Reject threshold | 0.845 | **0.75** |
| Genuine pairs kept | 98.6% | **97.2%** |
| Margin to nearest impostor | 0.00 | **0.24** |

It costs about 1.4 points of recall, and the 0.55-0.75 band where the two distributions
genuinely overlap now reports `INCONCLUSIVE` rather than a confident wrong name. Trading
recall to avoid naming the wrong person is the entire reason the gate has three states.

**Coverage is reported on purpose.** A detection failure is not a neutral event, the pair
silently leaves the evaluation, so a pipeline that detects *less* scores *better* on a
coverage-blind metric. Every figure above is conditional on the 99.7% of pairs where a
face was found.

**On the 1 false match.** With ~499 impostor pairs the smallest observable non-zero
rate is 1/499 ≈ 2e-3, so a held-out split cannot resolve an FMR of 1e-3 at all. The
operating point is chosen on the ~4,000 calibration negatives, where it can be. The test
suite enforces a 1e-2 ceiling on the held-out rate rather than asserting zero, because
asserting zero would be asserting luck.

Thresholds are calibrated on the LFW **train + 10-fold** splits and reported on the
**disjoint test** split. Tuning and reporting on the same pairs is the most common way a
face-recognition benchmark misleads, so it is avoided here. Full report:
[`docs/benchmark.json`](docs/benchmark.json).

> **Caveat, stated plainly.** LFW is a frontal, well-lit celebrity benchmark. Social-media
> media, compressed thumbnails, group shots, extreme pose, is harder, and real-world
> accuracy will be lower than the table above. The uncertainty band exists for exactly
> that reason.

---

## Architecture

```
input image
  │  magic bytes · MIME · EXIF transpose · sRGB · decompression-bomb caps · sha256
  ▼
SCRFD det_10g ──▶ every face: bbox + score + 5 landmarks
  │  select: single | --face-index N | --largest   (multi-face never silently guesses)
  │  quality gate: size · Laplacian sharpness · exposure · roll/yaw · truncation
  ▼
Umeyama similarity alignment → canonical 112×112 → ArcFace → L2-normalized 512-d
  │  (the embedding is biometric data: kept in memory, never logged, never on-chain)
  ▼
┌─── GENUINE SEARCH: one call, then a concurrent fan-out across five sources ───┐
│ wave 1   SerpApi Lens type=all on the full image                              │
│          → visual matches + pages about the image + the entity Lens inferred  │
│                                                                               │
│ wave 2   only if wave 1 is thin. All four run AT ONCE, so the stage costs the  │
│          slowest source, not the sum:                                         │
│   lens      Lens type=all on a head-and-shoulders crop     SERPAPI_KEY        │
│   serpweb   the inferred name, site:-restricted            SERPAPI_KEY        │
│   exa       neural retrieval of pages about the person     EXA_API_KEY  (opt) │
│   wikidata  curated full-resolution portrait               no credential      │
│                                                                               │
│ a source that is unconfigured is skipped, one that fails is recorded, and one │
│ that hangs is dropped at a timeout. None of them can take the run down.       │
└───────────────────────────────────────────────────────────────────────────────┘
  │  URL canonicalization · dedupe · social ranked first (never filtered first)
  ▼
bounded-concurrency fetch · redirect/size/content-type caps · sha256 BEFORE decode
  │  ladder: original → OpenGraph → oEmbed → thumbnail, each labelled by quality
  ▼
detect ALL faces per candidate → batch embed → cosine vs source → best-face-wins
  │  MATCH / NON_MATCH / INCONCLUSIVE
  ▼
RFC 8785 canonical JSON manifest → domain-separated Merkle leaves → 32-byte root
  │  H("SIGIL:LEAF:v1:" ‖ field ‖ canonical_value)
  ▼
SigilRegistry.anchor(root, schemaVersion) on Sepolia → receipt → Etherscan
  ▼
sigil verify --bundle   (fresh process, no private key, names the broken leaf on failure)
```

### Why not DeepFace

The project began on DeepFace. It does not import in this environment at all, `retinaface` calls `validate_for_keras3()` and raises against TensorFlow 2.21. Rather
than pin `tf-keras` and inherit the rest, Sigil runs the ONNX weights DeepFace wraps:

| | DeepFace | Sigil |
|---|---|---|
| Status | does not import | works |
| Install | ~600 MB (TensorFlow) | ~90 MB |
| Per-call cost | model reload | sessions warmed once |
| Alignment | library-internal | explicit, in this repo, unit-tested |
| Embedding |, | 3.5× faster on CoreML (33.8 → 9.7 ms) |

The SCRFD anchor decoding and NMS are implemented in
[`sigil/face/engine.py`](sigil/face/engine.py) and validated bit-exact against the
InsightFace reference.

---

## Setup

Requires **Python 3.12** and **Node 22**.

```bash
git clone https://github.com/Arav-Arun/HHgoa-FaceID.git && cd HHgoa-FaceID
make setup          # uv sync --all-extras --group dev && npm ci
cp .env.example .env
```

Fill in `.env`:

```bash
SERPAPI_KEY=...                                          # serpapi.com, free tier is 100 searches/month
SEPOLIA_RPC_URL=https://ethereum-sepolia-rpc.publicnode.com   # public, no key needed
PRIVATE_KEY=...                                          # a THROWAWAY wallet, faucet-funded only
CONTRACT_ADDRESS=...                                     # set after deploying
```

Deploy the contract:

```bash
npm run compile && npm run deploy:sepolia
```

Check everything before a demo:

```bash
sigil preflight --live
```

`--live` also queries remaining SerpApi quota and the wallet balance. Without it,
preflight makes no network calls.

---

## Usage

### Local web interface

```bash
sigil serve          # opens http://127.0.0.1:8420
```

Drop, paste, or pick an image. Every candidate the pipeline examined appears with its
cosine distance, the reason it passed or failed, which of the four search routes surfaced
it, and where the media came from. The sidebar carries the evidence root and a search
audit panel listing each provider call with its real search ID.

**Localhost only, by design.** `serve` refuses any other bind address with an error.
Publishing a face-search interface would let anyone submit anyone's face, which is exactly
the use the responsible-use section rules out.

### Independent verifier

`verify.html` opens straight from the filesystem, no server, no install. It re-implements
the RFC 8785 canonicalization and the Merkle construction in JavaScript and rebuilds the
root from scratch rather than trusting the Python that produced it, then reads the anchor
with a plain `eth_call`. Two independent implementations agreeing is what makes the
evidence format a format rather than one library's output.

### Command line

```bash
# Full pipeline: search, verify, anchor
sigil run --image data/samples/subject.jpg

# Force a genuinely live search (spends quota, use this for the recording)
sigil run --image data/samples/subject.jpg --no-cache

# Build evidence without touching the chain
sigil run --image data/samples/subject.jpg --skip-chain

# Independently verify a bundle, no private key required
sigil verify --bundle data/bundles/<run-id>

# Verify offline, without any RPC
sigil verify --bundle data/bundles/<run-id> --local-only

# Anchor a bundle built earlier (e.g. after an RPC outage)
sigil anchor --bundle data/bundles/<run-id>

# Reproduce the accuracy table
sigil benchmark
```

Exit codes are stable, because a failure must never render as a success:

| Code | Meaning |
|---|---|
| `0` | succeeded and verified |
| `1` | completed honestly with a negative result (no verified match) |
| `2` | configuration or environment problem |
| `3` | **verification FAILED**, tampered or not anchored |

See [`docs/COMPARISON.md`](docs/COMPARISON.md) for how this measures up against
InsightFace, DeepFace, commercial face-search products, and Palantir's provenance model, including what they do better.

---

### What reverse-image search can and cannot do

Worth stating plainly, because it sets the ceiling on every recall number below.

**Google Lens does not do face recognition.** It is suppressed deliberately, for privacy.
Handed a portrait of someone in distinctive glasses it returned sixty results, almost all
eyewear retailers;
handed a tight face crop it returned strangers who look similar. It finds *this person*
when the photo, or the person, is already indexed and captioned somewhere Google crawls.

That is why discovery fans out rather than wrapping one endpoint, and why a face with no
public photo presence returns `SEARCH_EMPTY` or a set of candidates that are all correctly
rejected. Neither is a bug, and neither is dressed up as a match. A dedicated face-search
index (the commercial people-search engines run their own) would raise recall on private
individuals, which is exactly the capability this project's responsible-use section
declines to build.

## Surviving the SerpApi free tier

100 searches/month, and a naive run spends four. Five mechanisms keep that workable:

1. **Content-addressed cache** keyed by image digest + route + params. Repeat runs cost
   nothing. `--no-cache` forces live for the recorded demo.
2. **`type=all` instead of two narrower calls.** One request returns visual matches, the
   pages that discuss the image, and the inferred entity. It halved the common path from
   two searches to one, and from 12.9s to 6.1s.
3. **Progressive escalation.** Wave two only runs when wave one is thin, so a typical
   successful run spends one search.
4. **Sources that cost nothing.** Wikidata needs no account and no quota, so it keeps
   contributing after SerpApi is exhausted.
5. **One upload per run.** The `image_id` is reused across every Lens route within its
   10-minute lifetime.
6. **Hard per-run budget** (`--search-budget`) that raises rather than overspending, plus
   a **reserve floor** live runs refuse to dip below, so the last searches of the month
   cannot be spent on a debugging loop.
7. **Fixture-driven tests.** The whole suite runs with zero API calls.

---

## Blockchain

| | |
|---|---|
| Chain | **Ethereum Sepolia** (chain ID `11155111`) |
| Contract | [`contracts/SigilRegistry.sol`](contracts/SigilRegistry.sol) |
| Address | [`0xc92D2fDe2757b14e5EB6Aef849E41aBa1F395517`](https://sepolia.etherscan.io/address/0xc92D2fDe2757b14e5EB6Aef849E41aBa1F395517) |
| Source verified | [Sourcify **exact match**](https://repo.sourcify.dev/11155111/0xc92D2fDe2757b14e5EB6Aef849E41aBa1F395517/), [Blockscout](https://eth-sepolia.blockscout.com/address/0xc92D2fDe2757b14e5EB6Aef849E41aBa1F395517?tab=contract) |
| Compiler | `solc 0.8.34+commit.80d5c536`, evm `osaka`, optimizer off |
| Anchored | 32-byte Merkle root, submitter, timestamp, schema version |
| Gas | under 80k per anchor (asserted in the contract test suite) |

Sourcify reports an **exact match** on both the creation and runtime bytecode, which is
the strong form: the contract at that address is byte-identical to what
[`contracts/SigilRegistry.sol`](contracts/SigilRegistry.sol) in this repository compiles
to, metadata included. A *partial* match would mean only the executable code agreed.

That distinction is not academic here. An earlier deployment fell to partial after a
one-line edit to a **comment** in the contract: solc hashes the source into the metadata
blob it appends to the runtime code, so the repository silently stopped reproducing the
deployed bytecode while every file still claimed otherwise. Nothing misbehaved. It just
quietly stopped being verifiable. `sigil preflight --live` now compares the deployed code
against the compiled artifact and reports which of the three cases holds:

```console
$ sigil preflight --live
│ chain    │ PASS │ chain 11155111, contract 0xc92D2fDe…95517, 3 root(s) anchored │
│ bytecode │ PASS │ deployed code matches contracts/SigilRegistry.sol exactly     │
```

[`deployments/sepolia.json`](deployments/sepolia.json) is written by the deploy script,
not by hand, and records the source commit, compiler settings, and the SHA-256 of the
runtime bytecode that was actually deployed. The deploy aborts if the code that lands
on chain is not the code this tree compiles.

**No personal data is ever written on-chain.** No images, no embeddings, no names, no post
text, only a hash. That is a privacy requirement, not an optimisation: biometric data
must not be written to an immutable public ledger.

The contract rejects the zero root, rejects duplicate roots (a re-anchor cannot overwrite
the original submitter or timestamp), and `get()` reverts on an unknown root so a
zero-struct read can never be mistaken for success.

`--chain local` runs the identical code path against a Hardhat node for CI and offline work.

---

## Testing

```bash
make test          # 263 Python tests + 9 Solidity tests
make lint          # ruff + mypy --strict
```

The end-to-end test in [`tests/test_e2e_local.py`](tests/test_e2e_local.py) runs the real
face engine, a real HTTP fetch, real Merkle construction, and a real EVM transaction
against a local Hardhat node, with no credentials. Start a node first:

```bash
npx hardhat node
```

Live SerpApi tests are opt-in behind `RUN_LIVE_TESTS=1` so CI consumes neither quota nor
test ETH.

---

## Limitations

Stated honestly, because the project's central claim is that it does not overstate things.

- **Search recall is the weakest link.** Reverse image search finds people who are already
  indexed. For someone with no public photo presence, Sigil correctly returns
  `SEARCH_EMPTY`, it will not invent a match. This is the intended behaviour, not a bug,
  and the recorded demo shows it.
- **LFW is easier than the open web.** See the caveat under Measured results.
- **Login-walled platforms** (Instagram, Facebook) often refuse direct media fetches.
  Sigil falls back through OpenGraph/oEmbed to the search provider's thumbnail and labels
  the evidence quality accordingly; it never fabricates page content and never
  authenticates to a platform.
- **A thumbnail-derived match is weaker evidence** than a full-resolution one. The bundle
  records which was used.
- **The InsightFace model weights are licensed for non-commercial research use.** Fine for
  this hackathon; a commercially licensed model would be required to productize.
- **SerpApi is a single point of failure.** The provider interface is abstracted, but only
  one provider is implemented.
- **Sepolia is a testnet.** Its history carries no economic guarantee; the same code
  anchors to mainnet unchanged if that guarantee is needed.
- **`image_id` expires after ten minutes** and is treated as ephemeral, never as proof.

## Responsible use

Sigil performs face search against public web data. Use it only on yourself or on
consenting subjects, or on public figures in a research context. It must not be used to
identify or locate private individuals without consent. Results are probabilistic. Face
recognition is regulated in many jurisdictions, check your local law before deploying
anything like this.

Face embeddings are treated as biometric secrets throughout: excluded from serialization,
never logged, never written to the evidence bundle, and never anchored on-chain.

## License

MIT. Model weights are governed by the [InsightFace](https://github.com/deepinsight/insightface)
license, which restricts them to non-commercial research use.
