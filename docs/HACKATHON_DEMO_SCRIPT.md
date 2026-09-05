# Sigil Hackathon Demo Script

**Length:** approximately 3.5 to 4 minutes  
**Demo:** [http://127.0.0.1:8420](http://127.0.0.1:8420)

On the architecture diagram, change **“above match threshold”** to **“cosine distance ≤ 0.55.”** Lower distance means a closer face match.

## Speaker 1 — Problem and architecture

**Show the landing page, then the architecture diagram and draw its arrows in order.**

> Finding an exact copy of a photograph online is easy. The harder problem is finding a different photograph of the same person, verifying that the face actually matches, and proving that the result has not been changed later.
>
> This is Sigil. It takes a face photo, searches the public web for real posts, verifies the results with facial recognition, and uses the Ethereum Sepolia blockchain to create a tamper-evident record.
>
> The pipeline begins with SCRFD, which detects the face and creates an aligned crop. ArcFace converts that face into a 512-dimensional embedding.
>
> Sigil then searches the full image, a portrait crop, and a tight face crop through real search providers. We use multiple crops because whole-image search often focuses on a shirt, logo, or background instead of the person.
>
> Search only discovers possible results. It does not decide identity. Sigil downloads every candidate image, detects the faces inside it, and compares their ArcFace embeddings with the original using cosine distance. A distance at or below `0.55` is a match, while uncertain results are reported as inconclusive instead of being forced into a decision.
>
> We also distinguish an exact copy of the uploaded photo from a genuinely different photo of the same person. Exact copies are labeled clearly and ranked below independently verified social posts.
>
> After verification, Sigil hashes the evidence into a Merkle root and writes that root to a smart contract on Sepolia. This means blockchain is part of the working pipeline: it records the final verified result so it can be checked later for tampering.

**Return to the website, click How it works, and briefly scroll through the five stages. In the blockchain section, click Change one character and then Reset.**

> The website also explains each stage. This small interactive example shows why we use a Merkle root. Changing one character in the evidence immediately creates a different fingerprint and marks it as tampered. Resetting the field restores the original verified root.
>
> Now we’ll return to Face Search and run the real pipeline.

## Speaker 2 — Live demo and blockchain verification

**Open Face Search, upload the Salman Khan image, and start the search.**

> Let’s run the complete pipeline. I’ll upload this photograph and start the search. These results are discovered from the web; they are not selected from a hardcoded list.

**While it runs, point to the stage indicator.**

> Sigil is detecting the face, searching the web, downloading candidates, comparing faces, building the evidence bundle, and anchoring the result on the blockchain.

**When the results appear, show the selected result and any exact-photo card separately.**

> Sigil has found public pages and social results associated with Salman Khan. The search provider may also return an exact copy of the uploaded image. Sigil labels that as an exact photo instead of pretending it discovered a different photograph.
>
> Names and search rankings only help locate candidates. They do not prove identity. Every result must still pass the original ArcFace comparison.
>
> Here, Sigil has found a different photograph of Salman Khan on a real public source. The displayed cosine distance is below our `0.55` match threshold, so the face passes verification and the different photograph is ranked above any exact-image duplicate.

**Point to the Search audit. Then open Full report, scroll briefly, and close that tab.**

> The search audit shows every discovery source and whether each provider call was live or cached. Full report gives us the complete run rather than only the winner: every candidate, its image, distance, face count, decision reason, timing, evidence root, and blockchain receipt. If an image contains several people, Sigil checks every detected face. If there is no usable face or the result is uncertain, it reports that honestly instead of guessing.

**Point to the evidence root and click View transaction on Etherscan.**

> Once the face is verified, Sigil hashes the complete result into one Merkle root and submits it to our smart contract on Ethereum Sepolia. This Etherscan page shows the real successful blockchain transaction. Only the root is stored on-chain, not the photograph or face embedding.

**Return to Sigil, click Verify independently, and point to Evidence intact and Anchored on-chain.**

> Sigil now rebuilds the root from the saved evidence and checks the same root against the smart contract. Etherscan proves that the transaction exists, while these two green checks prove that our current evidence matches it. This read-only verification needs no wallet or private key.
>
> That completes the full pipeline: face identification, real web and social discovery, blockchain anchoring, and independent tamper verification.

**Finish on the two successful verification checks.**
