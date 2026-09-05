# Demo runbook

A script for the screen recording, plus everything to check before pressing record.
Target length: **3–5 minutes**, one unedited take.

The recording has to show three things the task asks for, a face scan, a genuinely
discovered social post, and a blockchain record you can re-verify, and one thing that
makes the difference between a demo and a proof: **the system failing correctly when the
evidence is altered.**

---

## T-minus 30 minutes

```bash
sigil preflight --live
```

Every row must be `PASS` or a `WARN` you understand. In particular:

| Check | What to confirm |
|---|---|
| `serpapi` | enough searches left for **three cold runs plus the recording** (~12), above the reserve |
| `wallet` | non-zero Sepolia balance; a single anchor costs well under 0.001 ETH |
| `chain` | contract address resolves and reports its anchored count |
| `face_models` | ~190 MB present, so nothing downloads mid-take |

Then:

- [ ] `git status` clean; note the commit you are recording, you will tag it afterwards.
- [ ] `uv run --group dev pytest -q` and `npm test` green.
- [ ] Do Not Disturb on. Close Slack, mail, and anything that can raise a banner.
- [ ] Terminal at a readable size, aim for ~100 columns at 1080p, large font.
- [ ] Browser open with two blank tabs (one for the social post, one for Etherscan).
- [ ] `rm -rf data/bundles/*` so the run you record is unmistakably fresh.
- [ ] Dry-run the whole sequence once **without** recording. If the search comes back
      empty for the primary subject, switch to a backup before you start filming.

> **Quota discipline.** Only `--no-cache` spends searches. A cached rehearsal of an image
> you have already run costs nothing and prints `searches: 0 live, 2 cached`, so rehearse
> freely and drop the cache only for the take. Confirm the counter says `0 live` on your
> last dry run; if it says `2 live`, the image is new to the cache and you are spending.

---

## The take

### 1. Frame the problem (~20 s)

Show `README.md` briefly, or just say it:

> "Sigil takes a face, searches the live web for a matching social post, verifies the
> match with face recognition, and anchors a tamper-evident proof on Sepolia. The point
> isn't that it finds someone, it's that you can check afterwards that nothing was
> altered, and it tells you exactly what changed if something was."

### 2. Show the input (~15 s)

```bash
open data/samples/public_figure.jpg
```

Say who it is and why using them is appropriate, a public figure, or your own face.
This matters and takes five seconds.

> **Terminal or the UI?** Both are real, and they run the same pipeline through the same
> code path. The UI is better television and the terminal is better evidence, so the
> recommended take does discovery in the UI (steps 3 to 5) and verification in the
> terminal (steps 7 and 8). If you would rather keep it to one surface, the terminal
> alone is a complete demo; the UI alone is not, because the tamper test is a CLI moment.
>
> ```bash
> sigil serve          # http://127.0.0.1:8420, localhost only, by design
> ```
>
> Click a sample tile, or drop an image on the card. The progress panel shows the same
> stages the CLI prints, then the results grid fills in with every candidate, its
> platform, its cosine distance, and a link to the live post.

### 3. Run the pipeline live (~60 s)

```bash
sigil run --image data/samples/public_figure.jpg --no-cache
```

`--no-cache` is the important flag: it forces real provider calls, so nothing on screen
came from a stored result. Narrate the stages as they print, face, live search,
candidate fetch, verification, evidence, anchor.

When the table appears, narrate **what it actually shows**, which depends on the subject.
Check your dry run first and pick the line that is true:

- *Some candidates rejected:* point at them. This is the strongest moment in the video.
  > "It looked at N candidates and rejected M. That one is result number one from the
  > search and it still failed the face check, a good search rank cannot grant identity
  > here."
- *All candidates accepted* (common for a heavily photographed public figure):
  > "Every one of these passed the face gate at a threshold calibrated to a false match
  > rate of one in a thousand. Search rank never enters that decision; a candidate the
  > face check rejects cannot be promoted by ranking first."

Do not script a rejection count in advance. A well-indexed public figure often returns
twelve for twelve, and narrating rejections that are not on screen is the one thing that
would make the rest of the video less believable.

### 4. Open the real post (~20 s)

Copy the winning URL into the browser. Show that it is a live, real social-media post,
and that the person in it is the person in the input.

### 5. Show the evidence (~30 s)

```bash
open data/bundles/<run-id>/report.html
```

Scroll to the contact sheet, green box on the accepted match, red on the rejects, each
with its cosine distance. Then point at the root:

> "That 32-byte root is a Merkle tree over every field of the evidence, the post URL,
> the media digest, the model version, the decision, the distance."

### 6. Show it on-chain (~30 s)

Open the explorer link from the report or the terminal. Show the transaction, the
contract, and that the anchored value equals the root on screen.

> "No image, no embedding, no name is on the chain, only the hash. Biometric data must
> not go on a public immutable ledger."

### 7. Verify independently (~30 s)

```bash
sigil verify --bundle data/bundles/<run-id>
```

Say the part that matters:

> "This is a fresh process and it needs no private key, verification is a read-only
> call. Anyone with this bundle and a public RPC endpoint can do exactly this."

### 7b. Verify in a browser, with nothing installed (~20 s)

Open `sigil/static/verify.html` from a `file://` URL and drop the bundle folder onto it.

> "This is a second, independent implementation of the canonicalization and the Merkle
> tree, in JavaScript, in a page with no build step and no dependencies. If it agrees with
> the Python, the format is the thing being verified, not my code. It reads the registry
> address out of the bundle receipt and checks it with a plain `eth_call`."

### 7c. The wrong-contract demo (~20 s), optional but strong

Point the verifier at a different real contract and watch it refuse:

```bash
CONTRACT_ADDRESS=0xfFf9976782d46CC05630D1f6eBAb18b2324d6B14 \
  sigil verify --bundle data/bundles/<run-id>; echo "exit: $?"
```

> "The address a verifier uses can come out of the bundle itself, so 'the contract said
> yes' isn't enough — a look-alike with the same interface could say yes to anything. It
> hashes the deployed bytecode and compares it to what the deploy recorded. Wrong
> contract, exit 3."

### 8. The tamper demo (~45 s), the closer

Change one character in the post URL:

```bash
python3 - <<'PY'
import json
from pathlib import Path
from sigil.evidence.canonical import canonicalize

p = Path("data/bundles/<run-id>/manifest.json")
m = json.loads(p.read_bytes())
m["post"]["canonical_url"] = m["post"]["canonical_url"][:-1] + "9"
p.write_bytes(canonicalize(m))
print("changed the last character of the post URL")
PY
```

```bash
sigil verify --bundle data/bundles/<run-id>
echo "exit code: $?"
```

Expected output:

```
╭─ Local verification FAIL ────────────────────────────────────────╮
│ root 0x…                                                         │
│ FAIL, modified: post.canonical_url                              │
╰──────────────────────────────────────────────────────────────────╯
  ✗ tampered field: post.canonical_url
exit code: 3
```

> "It doesn't just say the evidence is broken, it names the field. That's what the Merkle
> tree buys over a single hash of the whole document."

Restore it and show `PASS` again, so the bundle you leave behind is the real one.

If you showed the browser verifier at step 7b, drop the tampered bundle on it too: the browser
names the same field, from a completely separate implementation. That is the difference
between "my code says it is fine" and "the evidence is checkable".

### 9. Close honestly (~20 s)

Say the limitation out loud. It costs nothing and it is the most credible twenty seconds
in the video:

> "Reverse image search only finds people who are already indexed. On a subject with no
> public photo presence this returns SEARCH_EMPTY, it will not invent a match. The
> accuracy numbers in the README come from a benchmark script in the repo, on LFW, which
> is easier than the open web."

If you have time, run the second demo with an unindexed face and let it return
`SEARCH_EMPTY` or `INCONCLUSIVE` on camera. A system that refuses to answer is worth more
than one that always answers.

---

## After the take

- [ ] Watch it back at full size, is the terminal text legible?
- [ ] Confirm no credential, `.env`, wallet address you care about, or private path is
      visible in any frame.
- [ ] Upload (YouTube unlisted / Drive / Loom).
- [ ] **Open the link in a private window** to confirm it plays without a sign-in.
- [ ] Tag the recorded commit: `git tag -a v1.0.0-hhgoa -m "recorded demo" && git push --tags`
- [ ] Submit the repo link and the video link.
- [ ] Do not push further changes to the tagged state.

---

## If something goes wrong mid-take

| Symptom | Do this |
|---|---|
| `SEARCH_EMPTY` | Stop and restart with a backup image. Do not retry the same input on camera. |
| `SEARCH_UNAVAILABLE` | Provider outage or quota. Check `sigil preflight --live`, then reschedule. |
| `NO_VERIFIED_MATCH` | The honest answer. Either narrate it as the abstention demo, or restart with a backup. |
| `CHAIN_PENDING` | Sepolia is slow. The transaction hash is already saved in the bundle's `pending.json`, so `sigil anchor --bundle <dir>` resumes that exact transaction rather than signing a new one. Don't re-run the search. |
| Wallet has no ETH | Faucet, then `sigil anchor --bundle <dir>`. The evidence is already built. |

The design principle behind that table: discovery and evidence complete before the chain
is touched, so a chain problem never costs you the run.
