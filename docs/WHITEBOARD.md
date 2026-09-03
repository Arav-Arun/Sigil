# Whiteboard explainer

A script for drawing and talking through the whole pipeline. Target **5 to 7 minutes**.

Draw left to right in six panels. Everything below is what to write on the board and what
to say while writing it, so you can do both at once. The numbers are real; they come from
[`benchmark.json`](benchmark.json) and from runs you can reproduce.

One framing line to open with, because it sets up everything else:

> "The task is face in, social post out, sealed on a blockchain. The interesting part isn't
> that it finds someone. It's that afterwards, anyone can check nothing was altered, and if
> something was, the system tells you exactly which field."

---

## Panel 1 · Input and encoding

**Draw**

```
  [photo]  ──▶  SCRFD  ──▶  align 112×112  ──▶  ArcFace  ──▶  512 numbers
                 detect        5 landmarks                     (unit vector)
```

**Say**

- "Detection finds every face and five landmarks. A similarity transform rotates and scales
  the face onto a canonical frame, so the same person photographed at a different angle
  lands in the same place."
- "ArcFace turns that crop into 512 numbers. Same person, nearby vectors. Different person,
  far apart. Distance is cosine."
- "Those numbers are biometric data. They stay in memory. Never logged, never written into
  the evidence, never put on the chain."

**If asked why not a library:** "It started on DeepFace. DeepFace doesn't import against
TensorFlow 2.21 here, so rather than pin around it, the ONNX weights DeepFace wraps are run
directly. That drops TensorFlow from the dependency tree entirely, and makes the failure
modes visible instead of buried three libraries deep."

---

## Panel 2 · The fan-out

**Draw** the vector arriving into a column of four boxes, all in parallel, converging again:

```
                    ┌── SerpApi Lens ──┐
                    ├── SerpApi Web ───┤
   512 numbers ──▶  ├── Exa (neural) ──┼──▶  candidates
                    ├── Wikidata ──────┤
                    └── page harvest ──┘
```

**Say**

- "One reverse-image API is a wrapper, and you inherit its blind spots whole."
- "**Google Lens does not do face recognition.** It's suppressed for privacy. Give it a
  portrait of someone in distinctive glasses it returned sixty results, almost all eyewear
  retailers. Given a face crop it returns strangers who look similar."
- "So the search fans out over sources that fail differently, all at once. Wall clock is
  the slowest source, not the sum: 15.5 seconds of work in 7.6 elapsed."
- "Wikidata needs no account and no quota, so it keeps working after the paid quota is gone.
  Page harvest is ours: a search engine hands back one picture per page, but a portfolio
  has several, so we fetch the page and take them all."

**The credibility line:** "Every provider call is recorded with the provider's own search
id. You can look it up on their dashboard. A hardcoded answer has no search id."

---

## Panel 3 · The gate

**Draw** a funnel, wide in, narrow out, with three exits:

```
   25 candidates ──▶  │ compare every face │ ──▶  MATCH   d ≤ 0.55
                      │  in every image    │      UNSURE  0.55 – 0.75
                      └────────────────────┘      NO      d ≥ 0.75
```

**Say**

- "Every candidate is downloaded and every face in it is compared. Group photos matter:
  comparing only the largest face is the classic way to miss a real match."
- "Three answers, not two. The middle band is where genuine and impostor distances actually
  overlap, so it reports uncertainty instead of resolving it."
- "**Search rank cannot grant identity.** Candidates are only ranked after they pass the
  gate. Something that came first in the search still has to pass."

**The moment to land, and the best thirty seconds in the talk:**

> "On one run the search surfaced a *different* Colin Powell's LinkedIn profile. Same name,
> different person. The face gate rejected it at distance 0.878."

**If you have time, the harder story:** "An earlier version matched a stranger at 0.6888.
The threshold was calibrated on LFW, where impostors are random pairs of different people.
But a reverse-image search returns candidates *because* they look like you, so the negatives
it faces are hard by construction. Calibrating on random pairs and deploying against
lookalikes errs in the optimistic direction. The operating point moved from 0.795 to 0.55.
It costs 1.4 points of recall and it stops the system naming the wrong person."

---

## Panel 4 · Sealing

**Draw** a small Merkle tree, four leaves to one root:

```
   post.url   platform   distance   model
      │          │          │         │
      └────┬─────┘          └────┬────┘
           └──────── root ───────┘        ← 32 bytes
```

**Say**

- "Every field is serialised to one canonical byte sequence, RFC 8785, so the same evidence
  gives the same bytes on any machine, in any language."
- "Each field is hashed into a domain-separated leaf, and the leaves combine into one root."
- "Why a tree instead of one hash of the document: a single hash tells you *something*
  changed. The tree tells you *which field*."

---

## Panel 5 · The chain

**Draw**

```
   root ──▶ SigilRegistry.anchor(root, v1) ──▶ Sepolia
                                                 │
                             stores: root · submitter · timestamp
```

**Say**

- "Only the root goes on chain. No image, no name, no post text, no embedding. That's a
  privacy requirement, not an optimisation: biometric data must not go on an immutable
  public ledger."
- "What the chain adds is the one thing you cannot fake locally: an independent witness that
  this exact root existed at this exact time."
- "The contract is verified on Sourcify as an exact match, so the bytecode at that address
  is what the source in the repo compiles to."

---

## Panel 6 · Re-verification, and the closer

**Draw** an arrow going *backwards* from the chain to a fresh box:

```
   bundle ──▶ recompute root ──▶ compare with chain ──▶ PASS
                                                        FAIL + which field
```

**Say**

- "Verification is a read-only call. **No wallet, no private key.** A proof that needed the
  claimant's key wouldn't be a proof."
- "It works from a clean checkout with no configuration at all: the registry address comes
  from the bundle's own receipt."

**Then do the live tamper demo. This is the ending.**

Change one character in the post URL and re-run:

```
  FAIL, modified: post.canonical_url
  exit code 3
```

> "One character. It doesn't just say the evidence is broken, it names the field. That is
> what the Merkle tree buys over a single hash."

Restore it, show PASS, and close:

> "And there's a second implementation of that check, in JavaScript, in a page that opens
> from a file:// URL with nothing installed. If two independent implementations agree, what
> you're trusting is the format, not my code."

---

## The four requirements, and where each is proved

Have this ready, because it is what you are being marked against.

| # | Requirement | Panel | One-line proof |
|---|---|---|---|
| 1 | Face identification | 1 | SCRFD + ArcFace, ROC-AUC 0.99555 on 997 held-out LFW pairs |
| 2 | Genuine web/social search | 2, 3 | five concurrent sources, every call carries a provider search id |
| 3 | Blockchain record | 4, 5 | Merkle root anchored on Sepolia, contract verified exact-match |
| 4 | Re-verification | 6 | read-only `eth_call`, no key, and a one-character edit names the field |

Or just run it: `sigil prove --image data/samples/public_figure.jpg` prints all four.

---

## Questions you will get, and short answers

**"Why Sepolia and not mainnet?"**
It is a testnet, so its history carries no economic guarantee, and the same code anchors to
mainnet unchanged if that guarantee is needed. Nothing about the design depends on which
chain it is.

**"Couldn't you just fake the search?"**
Every call carries the provider's own search id, checkable on their dashboard. And
`--no-cache` forces live calls, so the run on camera is unambiguously live.

**"What happens if it finds nothing?"**
It says so. `SEARCH_EMPTY` when the provider returned nothing, `SEARCH_UNAVAILABLE` when
the provider broke, `NO_VERIFIED_MATCH` when candidates were examined and none passed.
Three different failures, never one dressed as success.

**"How is this different from a people-search engine?"**
Deliberately, it does not run a face index over scraped social media. That is the thing
that would raise recall on private individuals most, and it is a choice not to build it.
The index code exists and is measured; what goes into it is left to whoever runs it.

**"Why is it slow?"**
Search dominates: about 6 seconds cached, 12 to 13 live. Detection is 70% of the per-image cost because the
detector letterboxes everything to 640×640 and runs on CPU while the embedder is on the
GPU. That split is deliberate; CoreML errors on the detector's dynamic shapes.

**"What can't it do?"**
Find someone who is not already indexed on the public web. That is the provider's ceiling,
not the model's. Given the right page, the face gate confirms a match at distance 0.24.
