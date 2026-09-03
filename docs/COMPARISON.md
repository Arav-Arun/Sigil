# How Sigil compares

Claims here come from reading the installed source, not from marketing pages. File paths
refer to the versions pinned in this repo's lockfile.

---

## 1. Face recognition libraries

### InsightFace 0.7.3, the weights Sigil uses

Sigil loads InsightFace's `buffalo_l` ONNX models directly. Reading its code turned up
three things worth acting on.

**`swapRB=True`, everywhere.** Both `scrfd.py:154` and `arcface_onnx.py` build their input
with `cv2.dnn.blobFromImage(..., swapRB=True)` from a BGR source, so the networks actually
see **RGB**. Sigil works in RGB natively and was swapping to BGR, which silently dropped
embedding agreement to cosine 0.94. Catching this required diffing against the reference,
not reading the docs. After the fix: bbox delta 0.000 px, embedding cosine 1.000000.

**No flip TTA.** `ArcFaceONNX.get_feat()` is a single forward pass:

```python
blob = cv2.dnn.blobFromImages(imgs, 1.0 / self.input_std, input_size, ..., swapRB=True)
net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
return net_out
```

But the accuracy figures published for these weights come from the ArcFace evaluation
protocol, which averages a crop's embedding with its mirror. Adding it lifted AUC
0.99537 → 0.99588 and widened positive/negative separation by +0.0107.

**No detection fallback.** `SCRFD.detect()` runs once at a fixed threshold and returns an
empty array on failure. On our sweep that silently discarded 11 of 1,000 test pairs. Our
cascade recovers 8 of them.

*Licence:* the model-zoo weights are research / non-commercial. Stated in our README.

### DeepFace 0.0.100, evaluated and rejected

**It does not import** in this environment. `retinaface/commons/package_utils.py:24` raises
against TensorFlow 2.21 because `tf-keras` is absent. The face stack had never run once.

**Its thresholds are constants.** `deepface/config/threshold.py` is a static lookup table:

```python
thresholds = {
    "ArcFace":   {"cosine": 0.68, ...},
    "Buffalo_L": {"cosine": 0.55, ...},
    ...
}
```

`find_threshold()` reads that table and returns. Nothing is ever calibrated against the
data it will be applied to, and no false-match rate is attached to the number. Sigil's
0.7898 was measured on 8,143 LFW pairs at FMR 1e-3, and a test fails the build if the
shipped constant drifts from the benchmark that produced it.

|  | DeepFace | Sigil |
|---|---|---|
| Imports here | ✗ | ✓ |
| Install size | ~600 MB (TensorFlow) | ~90 MB |
| Threshold | hardcoded constant | measured, FMR-targeted, drift-tested |
| Per-call cost | model reload | warmed sessions |
| Flip TTA | ✗ | ✓ |
| Detection fallback | ✗ | ✓ 3-stage cascade |
| Alignment | inside the wrapper | explicit, unit-tested here |

### Others considered

| Tool | Why not |
|---|---|
| `face_recognition` / dlib | HOG or CNN detector, 128-d encodings. Materially weaker than ArcFace on pose, age gap, and low resolution, the conditions that dominate social-media media |
| FaceNet (`facenet-pytorch`) | MTCNN + InceptionResnetV1. Solid, but ArcFace's margin loss is the stronger 512-d baseline and the ONNX weights avoid a Torch dependency |
| AWS Rekognition / Azure Face | Accurate and well-engineered, but a black box: no measurable threshold, no local reproducibility, per-call cost, and the face leaves the machine. Azure Face is additionally gated behind restricted access |
| CompreFace, FaceX | Self-hosted services wrapping similar models. A whole service to deploy for one embedding call |

---

## 2. Face-search products

lenso.ai, PimEyes, and FaceCheck.ID are the closest things to a commercial equivalent of
this pipeline. They are genuinely better at one thing and structurally worse at another.

**Where they win: index.** They crawl and maintain their own face index over billions of
images. Sigil has no index, it queries a general reverse-image API and pivots on inferred
entities. For an ordinary person with a thin public footprint, they will find results
where Sigil returns `SEARCH_EMPTY`. That is a real capability gap and no amount of
engineering closes it without an index.

**Where they lose: everything is unfalsifiable.**

| Question a reviewer would ask | Commercial face search | Sigil |
|---|---|---|
| How confident is this match? | not shown | cosine distance, threshold, and margin per candidate |
| What did it reject, and why? | not shown | every rejected candidate, with its reason |
| Where did the image come from? | source behind a paywall | post URL, media URL, and provenance label (original / OpenGraph / oEmbed / thumbnail) |
| Was the search actually run? | trust us | provider search ID per route, live vs cached |
| Can it say "I don't know"? | no, a ranked list only | `INCONCLUSIVE` is a first-class outcome |
| Can I prove the result later? | no | Merkle root anchored on-chain; tamper localized to the field |
| Where does my uploaded face go? | their servers | stays on your machine; the UI refuses to bind off-localhost |

The product framing is *"here are people who look like this."* Sigil's is *"here is a claim,
here is the evidence for it, here is what I rejected, and here is a proof you can check
without me."* Different jobs. For the task at hand, where the deliverable is a
*verifiable* record, the second is the right one.

**One design point borrowed deliberately:** their results grid is genuinely good UX, so
Sigil's local UI uses the same shape. The difference is what fills it, where they show a
locked "Unlock Sources" badge, Sigil shows the distance, the decision, the route, and the
link.

---

## 3. Palantir's methodology

Foundry is not a face-search tool, but the problem it solves, *making an analytical
conclusion auditable after the fact*, is exactly this project's problem. Four of its
principles map directly, and adopting them is most of what separates Sigil from a script
that prints a URL.

### Lineage and provenance

Foundry maintains an [automatic lineage graph](https://www.palantir.com/docs/foundry/data-lineage/overview):
every derived dataset traces to its sources, and you can ask of any value where it came
from.

Sigil's evidence manifest is a lineage record for a single conclusion. It binds the input
digest, the aligned-crop digest, the candidate-media digest, the sanitized search-response
digest, the model identifier, and a configuration digest, so "which bytes, which model,
which settings produced this decision" is answerable from the bundle alone. The Merkle
tree goes one step further than Foundry's graph in a narrow way: lineage is not just
*recorded*, it is *cryptographically enforced*, and a change to any node is localized to
the field.

### Immutability and the audit log

Foundry's answer is an append-only log inside a trusted platform. Sigil's is an
append-only public ledger: `SigilRegistry` rejects re-anchoring, so a root's submitter and
timestamp cannot be rewritten even by the person who wrote them. Verification needs no
credential, which removes the trusted platform from the trust chain entirely, a strictly
stronger property than an internal audit log, for the narrow thing it covers.

### Human-in-the-loop

Foundry's posture is that the system surfaces evidence and a human decides. Sigil's
`INCONCLUSIVE` state is the same idea with teeth: the pipeline is structurally unable to
resolve an uncertain distance into an answer, and `rank()` filters to verified matches
before scoring, so no ranking signal can promote something that failed the face gate.

### Data minimization and civil liberties

Palantir's stated position is that access controls and purpose limitation belong in the
architecture rather than in policy. The concrete analogues here:

- embeddings are `exclude=True` on the model, so biometric vectors cannot reach
  serialization by accident;
- nothing personal goes on-chain, only a hash, because an immutable public ledger is the
  worst possible place for biometric data;
- the UI refuses to bind off-localhost with an error, not a warning;
- no authentication to any platform and no scraping behind a login;
- CI rejects a tracked `.env`, tracked evidence, or key-shaped literals.

### What Foundry has that Sigil does not

Honest gaps, not oversights, they are out of scope for a single-operator tool:

- **Purpose-based access control.** Foundry can require a declared, logged justification
  before data is touched. Sigil has no notion of who is asking or why. *The cheapest
  meaningful addition would be a required `--purpose` string hashed into the manifest, so
  the reason for a search becomes part of the immutable record.*
- **Retention and deletion policy.** Foundry enforces lifecycles. Sigil writes bundles and
  leaves them; deletion is manual.
- **Multi-user governance.** No roles, no per-field access control, no review workflow.

---

## 4. Summary

Sigil is not the best face *finder*, a product with its own crawled index will find more
people, and that gap is structural.

It is, as far as this comparison found, the only one of these that can hand you a result
and let you prove afterwards, without trusting the tool that produced it, that the result
was not altered, and tell you exactly which field changed if it was.

The three ideas doing that work:

1. **Calibrate, don't guess.** Every threshold traces to a measurement with a stated
   protocol and a test that fails if the code drifts from it.
2. **Show the rejects.** A system that only shows successes cannot be audited.
3. **Make verification independent.** No key, no install, no server, and a second
   implementation of the format in a different language to prove it is really a format.
