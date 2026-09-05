# Example bundle

A real, anchored evidence bundle from a live run, committed so that anyone can check the
proof without credentials, without a wallet, and without running the search.

```bash
sigil verify --bundle docs/example-bundle
```

That reads the Merkle root out of `merkle-proofs.json`, rebuilds it from `manifest.json`,
and then reads the registry named in `receipt.json` over a public Sepolia endpoint. No
`.env` is needed for either half. `sigil/static/verify.html` does the same thing in a browser from a
`file://` URL, in an independent JavaScript implementation.

Three files, and nothing else, on purpose:

| File | Why it is here |
|---|---|
| `manifest.json` | the claim itself, in RFC 8785 canonical form; the bytes the root is built from |
| `merkle-proofs.json` | the root and a per-field inclusion proof, so a failure names the field |
| `receipt.json` | chain id, registry address, transaction, block: where to go looking |

**No media is committed.** A full run also writes the input image, the aligned face crop,
the fetched candidate media, a contact sheet, and a self-contained `report.html`. Those
stay local. The manifest records their SHA-256 digests, so verification reports
`artifact digests (no media files supplied)` here and checks them when the media is
present. Face embeddings appear in no bundle at all, ever.

To see the media half, run the pipeline yourself and verify that bundle instead.

This committed bundle is an immutable historical Sepolia proof. It was anchored by an
earlier pipeline revision (so its recorded route label and calibrated threshold are kept
as-is); new runs use the current bounded discovery flow and shipped thresholds.
