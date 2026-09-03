"""Demonstrate every task requirement in one run, with nothing pre-baked.

The brief lists four requirements. Each is easy to *claim* and each has a specific way of
being faked, so this module is built around defeating the fake rather than around printing
a green tick:

===  =========================  =============================================
 1   face identification        the numbers come from a benchmark in this repo
                                with denominators, not from a README sentence
 2   genuine web search         the provider's own search id is printed, so the
                                result can be looked up on their dashboard; a
                                hardcoded answer has no such id
 3   blockchain record          the root is read back **from a separate
                                process** with no private key, over a public
                                endpoint, because a proof you can only check
                                from inside the program that wrote it is not a
                                proof
 4   tamper evidence            one character is changed in a copy of the
                                bundle and the failure has to name the field
===  =========================  =============================================

Steps 3 and 4 shell out to `sigil verify` deliberately. Calling the verification functions
in-process would share loaded state with the code that produced the evidence, and the
whole claim is that a third party can check this without that state.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sigil.config import ROOT_DIR, Settings

logger = logging.getLogger(__name__)

EXIT_VERIFY_FAILED = 3


@dataclass(slots=True)
class Requirement:
    """One task requirement and what was actually observed for it."""

    number: int
    name: str
    passed: bool = False
    detail: list[str] = field(default_factory=list)
    error: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "requirement": self.number,
            "name": self.name,
            "passed": self.passed,
            "detail": list(self.detail),
            "error": self.error,
        }


@dataclass(slots=True)
class ProofReport:
    requirements: list[Requirement] = field(default_factory=list)
    bundle_dir: str = ""
    evidence_root: str = ""

    @property
    def passed(self) -> bool:
        return bool(self.requirements) and all(r.passed for r in self.requirements)

    def to_json(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "bundle_dir": self.bundle_dir,
            "evidence_root": self.evidence_root,
            "requirements": [r.to_json() for r in self.requirements],
        }


def _run_cli(*args: str, without_key: bool = False) -> subprocess.CompletedProcess[str]:
    """Invoke the CLI in a genuinely separate process.

    ``without_key`` blanks PRIVATE_KEY in the child's environment. Verification is
    supposed to need no signing key, and the only way to show that is to take the key
    away and watch it still work.
    """

    env = dict(os.environ)
    if without_key:
        env["PRIVATE_KEY"] = ""
    return subprocess.run(
        [sys.executable, "-m", "sigil.cli", *args],
        capture_output=True,
        text=True,
        cwd=str(ROOT_DIR),
        env=env,
        check=False,
    )


def _benchmark_numbers() -> dict[str, Any]:
    path = ROOT_DIR / "docs" / "benchmark.json"
    if not path.is_file():
        return {}
    try:
        loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return loaded
    except (OSError, json.JSONDecodeError):
        return {}


def _requirement_one(result: Any) -> Requirement:
    """Face identification: a face was detected, encoded, and the model is measured."""

    item = Requirement(1, "face identification")
    face = getattr(result, "face", None)
    if face is None:
        item.error = "no face was detected in the input image"
        return item

    report = _benchmark_numbers()
    item.detail.append(
        f"{face.detector_backend} + {face.model_name}, "
        f"{face.bounding_box.width}x{face.bounding_box.height}px face, "
        f"quality {face.quality.quality_score:.2f}"
    )
    if report:
        item.detail.append(
            f"LFW ROC-AUC {report.get('roc_auc')}, "
            f"TAR@FMR1e-3 {report.get('tar_at_fmr_1e3')} "
            f"on {report.get('test_pairs')} held-out pairs, "
            f"{report.get('false_matches')}/{report.get('test_negative')} false matches"
        )
    else:
        item.detail.append("no committed benchmark; run `sigil benchmark`")
    # The embedding is biometric and is never printed, only its shape.
    item.detail.append(f"{len(face.embedding)}-d embedding, held in memory only")
    item.passed = True
    return item


def _requirement_two(result: Any) -> Requirement:
    """Genuine search: real provider calls, with ids anyone can look up."""

    item = Requirement(2, "genuine web search")
    records = list(getattr(result, "search_records", []) or [])
    # `verifications` is every candidate that was fetched and face-checked. `candidates`
    # does not exist on the result; reading it reported "0 examined" on a run that had
    # examined twelve.
    examined = list(getattr(result, "verifications", []) or [])
    selected = getattr(result, "selected", None)

    if not records:
        item.error = "no provider call was made"
        return item

    live = [r for r in records if not r.get("from_cache")]
    sources = {str(r.get("route", "")).split(":")[0] for r in records}
    budget = getattr(result, "budget", {}) or {}

    matched = sum(1 for v in examined if v.matched)
    item.detail.append(
        f"{len(sources)} source(s), {len(examined)} candidate(s) fetched and face-checked, "
        f"{matched} passed the gate, "
        f"{budget.get('live_searches', 0)} live / {budget.get('cached_searches', 0)} cached"
    )
    for record in records[:3]:
        identifier = record.get("search_id") or "no id"
        state = "CACHED" if record.get("from_cache") else "LIVE"
        item.detail.append(f"{record.get('route')}: id {identifier} ({state})")
    if not live:
        item.detail.append(
            "every call was served from the local cache; use --no-cache to force live"
        )

    if selected is None:
        item.error = (
            f"{len(examined)} candidate(s) examined, none passed the face gate. "
            "That is an honest outcome, but it does not demonstrate a found post."
        )
        return item

    decision = selected.decision
    item.detail.append(
        f"verified post {selected.candidate.source_url} "
        f"({selected.candidate.platform}), distance {decision.distance:.4f} "
        f"<= {decision.threshold}"
    )
    item.passed = True
    return item


def _requirement_three(bundle: Path) -> Requirement:
    """Blockchain record: re-read in a separate process with no private key."""

    item = Requirement(3, "blockchain record")
    receipt_path = bundle / "receipt.json"
    if not receipt_path.is_file():
        item.error = "the run produced no chain receipt; was it started with --skip-chain?"
        return item

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    item.detail.append(
        f"root {receipt['evidence_root'][:18]}… anchored on chain {receipt['chain_id']} "
        f"at {receipt['contract_address'][:12]}…"
    )
    item.detail.append(f"tx {receipt['transaction_hash'][:18]}… block {receipt['block_number']}")

    # Separate process, and PRIVATE_KEY blanked so no signing key can be involved.
    completed = _run_cli("verify", "--bundle", str(bundle), without_key=True)
    if completed.returncode != 0:
        item.error = (
            f"re-verification failed in a fresh process (exit {completed.returncode}): "
            f"{completed.stdout.strip()[-200:] or completed.stderr.strip()[-200:]}"
        )
        return item

    item.detail.append(
        "re-read from a separate process with PRIVATE_KEY unset: the on-chain root matches"
    )
    item.passed = True
    return item


def _requirement_four(bundle: Path) -> Requirement:
    """Tamper evidence: change one character, the failure must name the field."""

    item = Requirement(4, "tamper evidence")
    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file():
        item.error = "the bundle has no manifest to tamper with"
        return item

    # Work on a copy. The real bundle is evidence and is never mutated to make a point.
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "bundle"
        shutil.copytree(bundle, copy)

        original = copy.joinpath("manifest.json").read_bytes()
        text = original.decode("utf-8")
        marker = '"canonical_url":"'
        start = text.find(marker)
        if start == -1:
            item.error = "manifest has no post.canonical_url to alter"
            return item
        cut = start + len(marker)
        end = text.index('"', cut)
        # Flip the final character of the URL. One character, nothing else.
        last = text[end - 1]
        swapped = "9" if last != "9" else "8"
        copy.joinpath("manifest.json").write_text(
            text[: end - 1] + swapped + text[end:], encoding="utf-8"
        )
        item.detail.append(
            f"changed the last character of post.canonical_url, {last!r} -> {swapped!r}"
        )

        completed = _run_cli("verify", "--bundle", str(copy), "--json", without_key=True)
        if completed.returncode != EXIT_VERIFY_FAILED:
            item.error = (
                f"a tampered bundle exited {completed.returncode}, expected "
                f"{EXIT_VERIFY_FAILED}. Tampering was not detected as a failure."
            )
            return item

        try:
            payload = json.loads(completed.stdout)
            modified = payload.get("tamper", {}).get("modified", [])
        except json.JSONDecodeError:
            modified = []

        if "post.canonical_url" not in modified:
            item.error = (
                "verification failed, but did not name post.canonical_url as the changed "
                f"field (named: {modified or 'nothing'})"
            )
            return item

        item.detail.append(
            f"verification failed with exit {EXIT_VERIFY_FAILED} and named the field: "
            f"{', '.join(modified)}"
        )

    # The untouched bundle must still pass, so the failure above was the edit and
    # nothing else.
    restored = _run_cli("verify", "--bundle", str(bundle), without_key=True)
    if restored.returncode != 0:
        item.error = "the untouched bundle no longer verifies; the tamper test was not isolated"
        return item
    item.detail.append("the untouched bundle still verifies, so the failure was the edit alone")
    item.passed = True
    return item


def prove(
    image: Path,
    *,
    settings: Settings,
    no_cache: bool = False,
    skip_chain: bool = False,
) -> ProofReport:
    """Run the pipeline once and check every requirement against what it produced."""

    from sigil.run import run_pipeline

    report = ProofReport()
    result = run_pipeline(
        image,
        settings=settings,
        face_index=None,
        select_largest=True,
        no_cache=no_cache,
        use_cache=not no_cache,
        skip_chain=skip_chain,
    )
    report.bundle_dir = str(result.bundle_dir or "")
    report.evidence_root = str(result.evidence_root or "")

    report.requirements.append(_requirement_one(result))
    report.requirements.append(_requirement_two(result))

    bundle = Path(result.bundle_dir) if result.bundle_dir else None
    if bundle is None or not bundle.is_dir():
        missing = "the run produced no evidence bundle"
        report.requirements.append(Requirement(3, "blockchain record", error=missing))
        report.requirements.append(Requirement(4, "tamper evidence", error=missing))
        return report

    if skip_chain:
        skipped = "--skip-chain was set, so nothing was anchored"
        report.requirements.append(Requirement(3, "blockchain record", error=skipped))
    else:
        report.requirements.append(_requirement_three(bundle))
    report.requirements.append(_requirement_four(bundle))
    return report


__all__ = ["ProofReport", "Requirement", "prove"]
