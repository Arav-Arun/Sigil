"""Judge-friendly command-line interface for Sigil.

Exit codes are stable and meaningful, because the most important property of this tool is
that a failure never looks like a success:

===== ==========================================================================
 0     the requested operation succeeded and was verified
 1     the operation completed honestly with a negative result (no match found)
 2     configuration or environment problem
 3     verification FAILED, evidence was tampered with or is not anchored
===== ==========================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sigil import __app_name__, __version__
from sigil.config import ConfigurationError, Settings, get_settings
from sigil.models import DecisionStatus, PipelineErrorCode
from sigil.preflight import preflight_passed, run_preflight

console = Console()

EXIT_OK = 0
EXIT_NO_MATCH = 1
EXIT_CONFIG = 2
EXIT_VERIFY_FAILED = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sigil",
        description="Find a face on the public web and record the result so it can be checked.",
    )
    parser.add_argument("--version", action="version", version=f"{__app_name__} {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show stage-level logging")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Discover, verify, and anchor a face")
    run.add_argument("--image", required=True, type=Path, help="Input face image")
    run.add_argument("--face-index", type=int, default=None, help="Which face to use")
    run.add_argument("--largest", action="store_true", help="Use the largest detected face")
    run.add_argument("--max-results", type=int, default=24)
    run.add_argument("--match-threshold", type=float, default=None)
    run.add_argument("--reject-threshold", type=float, default=None)
    run.add_argument("--search-budget", type=int, default=4, help="Max live searches per run")
    run.add_argument("--no-cache", action="store_true", help="Force live search (uses quota)")
    run.add_argument(
        "--name",
        default="",
        help="Optional name to search alongside the face. Decides where to look, never "
        "who is in the picture; every page it reaches still has to pass the face gate.",
    )
    run.add_argument("--skip-chain", action="store_true", help="Build evidence without anchoring")
    run.add_argument("--output", type=Path, default=None)
    run.add_argument("--json", action="store_true", dest="as_json")

    verify = sub.add_parser("verify", help="Independently verify an evidence bundle")
    verify.add_argument("--bundle", required=True, type=Path)
    verify.add_argument("--local-only", action="store_true", help="Skip the on-chain read")
    verify.add_argument("--json", action="store_true", dest="as_json")

    anchor = sub.add_parser("anchor", help="Anchor a bundle that was built earlier")
    anchor.add_argument("--bundle", required=True, type=Path)
    anchor.add_argument("--json", action="store_true", dest="as_json")

    proof = sub.add_parser(
        "prove", help="Run the pipeline and check every task requirement end to end"
    )
    proof.add_argument("--image", required=True, type=Path, help="Input face image")
    proof.add_argument("--no-cache", action="store_true", help="Force live search (uses quota)")
    proof.add_argument("--skip-chain", action="store_true", help="Skip anchoring (requirement 3)")
    proof.add_argument("--json", action="store_true", dest="as_json")

    pre = sub.add_parser("preflight", help="Check the local environment before a demo")
    pre.add_argument("--live", action="store_true", help="Also query SerpApi quota and the RPC")
    pre.add_argument("--json", action="store_true", dest="as_json")

    serve = sub.add_parser("serve", help="Open the local web interface")
    serve.add_argument("--port", type=int, default=8420)
    serve.add_argument("--host", default="127.0.0.1", help="localhost only, by design")
    serve.add_argument("--no-browser", action="store_true")

    bench = sub.add_parser("benchmark", help="Measure accuracy and latency on LFW")
    bench.add_argument("--limit", type=int, default=None, help="Cap pairs per split")
    bench.add_argument("--cpu", action="store_true", help="Force the CPU execution provider")
    bench.add_argument(
        "--quality-only",
        action="store_true",
        help="Measure only the resolution sweep and repost test, merging into the report",
    )
    bench.add_argument("--output", type=Path, default=Path("docs/benchmark.json"))
    bench.add_argument("--json", action="store_true", dest="as_json")
    return parser


# -- rendering --------------------------------------------------------------------


def _print_url(label: str, url: str) -> None:
    """Print a URL on its own line, unwrapped.

    Rich wraps to the console width, and inside a Panel that width is the border-inset
    one. A 79-character explorer link then breaks mid-address, which is worse than
    useless: `…1F395` on one line and `517` on the next cannot be clicked or copied, and
    a truncated hex address is the kind of thing someone pastes into a block explorer
    without noticing. So links are printed outside panels, with wrapping turned off.
    """

    if not url:
        return
    console.print(f"  {label}", style="dim", end=" ")
    console.print(url, soft_wrap=True, no_wrap=True, overflow="ignore")


def _render_candidates(result: Any) -> None:
    table = Table(title="Candidates examined", show_lines=False)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Platform")
    table.add_column("Decision")
    table.add_column("Distance", justify="right")
    table.add_column("Faces", justify="right")
    table.add_column("Media")
    table.add_column("Post", overflow="fold", max_width=52)

    colours = {
        DecisionStatus.MATCH: "bold green",
        DecisionStatus.NON_MATCH: "red",
        DecisionStatus.INCONCLUSIVE: "yellow",
    }
    for index, item in enumerate(result.verifications, start=1):
        status = item.decision.status
        distance = f"{item.decision.distance:.4f}" if item.decision.distance is not None else "-"
        table.add_row(
            str(index),
            item.candidate.platform,
            f"[{colours[status]}]{status}[/{colours[status]}]",
            distance,
            str(item.faces_detected),
            str(item.media.quality).lower(),
            str(item.candidate.source_url),
        )
    console.print(table)


def _render_result(result: Any) -> None:
    if result.selected:
        selected = result.selected
        console.print(
            Panel(
                f"[bold green]MATCH[/bold green]  distance "
                f"{selected.decision.distance:.4f} ≤ {selected.decision.threshold:.2f}"
                f"  (margin {selected.margin:+.4f})\n"
                f"[bold]{selected.candidate.source_url}[/bold]\n"
                f"{selected.decision.reason}",
                title="Verified social post",
                border_style="green",
            )
        )
    if result.evidence_root:
        console.print(f"  evidence root : [bold cyan]{result.evidence_root}[/bold cyan]")
        console.print(f"  bundle        : {result.bundle_dir}")
        if result.bundle_dir and (result.bundle_dir / "report.html").is_file():
            console.print(f"  report        : {result.bundle_dir / 'report.html'}")
    if result.receipt:
        console.print(f"  transaction   : {result.receipt.transaction_hash}")
        console.print(f"  block         : {result.receipt.block_number}")
        _print_url("explorer      :", result.receipt.explorer_url)

    if result.timings_ms:
        timings = "  ".join(f"{k} {v:.0f}ms" for k, v in result.timings_ms.items())
        console.print(f"  [dim]{timings}[/dim]")
    if result.budget:
        console.print(
            f"  [dim]searches: {result.budget.get('live_searches', 0)} live, "
            f"{result.budget.get('cached_searches', 0)} cached[/dim]"
        )


# -- commands ---------------------------------------------------------------------


def cmd_run(args: argparse.Namespace, settings: Settings) -> int:
    from sigil.run import run_pipeline
    from sigil.verify import DEFAULT_MATCH_THRESHOLD, DEFAULT_REJECT_THRESHOLD

    if not args.image.is_file():
        console.print(f"[red]Input image not found:[/red] {args.image}")
        return EXIT_CONFIG

    stages = {
        "warm_up": "Loading models",
        "face": "Detecting and encoding the face",
        "search": "Searching the live web",
        "acquire": "Fetching candidate media",
        "verify": "Verifying identity",
        "evidence": "Building evidence and Merkle root",
        "anchor": "Anchoring on-chain",
    }

    def on_stage(name: str) -> None:
        if not args.as_json:
            console.print(f"[dim]→[/dim] {stages.get(name, name)}…")

    result = run_pipeline(
        args.image,
        settings=settings,
        output_root=args.output,
        face_index=args.face_index,
        select_largest=args.largest,
        max_candidates=args.max_results,
        match_threshold=args.match_threshold or DEFAULT_MATCH_THRESHOLD,
        reject_threshold=args.reject_threshold or DEFAULT_REJECT_THRESHOLD,
        no_cache=args.no_cache,
        use_cache=not args.no_cache,
        name_hint=getattr(args, "name", "") or "",
        search_budget=args.search_budget,
        skip_chain=args.skip_chain,
        on_stage=on_stage,
    )

    if args.as_json:
        print(json.dumps(result.to_json(), indent=2, default=str))
    else:
        if result.verifications:
            _render_candidates(result)
        _render_result(result)
        if not result.ok:
            console.print(
                Panel(
                    f"[yellow]{result.error_code}[/yellow]\n{result.error_message}",
                    title="No verified result",
                    border_style="yellow",
                )
            )

    if result.ok:
        return EXIT_OK
    if result.error_code in (
        PipelineErrorCode.INVALID_CONFIGURATION,
        PipelineErrorCode.INVALID_INPUT,
    ):
        return EXIT_CONFIG
    return EXIT_NO_MATCH


def _read_chain(bundle: Path, root: str, settings: Settings) -> tuple[bool, dict[str, Any], str]:
    """Look one root up on chain. Returns ``(ok, payload, provenance)``.

    Any failure to reach or agree with the chain, an outage as much as a missing root,
    returns ``ok=False``. Verification that could not be completed is not verification.
    """

    from sigil.chain import SEPOLIA_CHAIN_ID, ChainClient, ChainError, resolve_registry
    from sigil.evidence.bundle import load_proofs

    try:
        rpc_url, contract, provenance = resolve_registry(bundle, settings)
        client = ChainClient(rpc_url, contract, expected_chain_id=SEPOLIA_CHAIN_ID)
        # Identity before the read, matching the browser verifier. An answer from the
        # wrong contract is worse than no answer, because it looks like a pass, and a
        # look-alike that reverts should be reported as the wrong contract rather than
        # as an RPC fault.
        identity, identity_detail = client.registry_identity()
        if identity == "mismatch":
            return False, {"contract_identity": identity, "error": identity_detail}, provenance
        record = client.read(root)
    except (ChainError, ConfigurationError) as exc:
        return False, {"error": str(exc)}, ""

    expected_schema = int(load_proofs(bundle).get("schema_version", 1))
    payload: dict[str, Any] = {
        "contract_identity": identity,
        "contract_identity_detail": identity_detail,
        "exists": record.exists,
        "submitter": record.submitter,
        "anchored_at": record.anchored_at.isoformat() if record.anchored_at else None,
        "schema_version": record.schema_version,
        "chain_id": record.chain_id,
        "contract": record.contract_address,
        "explorer": record.explorer_url(),
    }
    if record.exists and record.schema_version != expected_schema:
        payload["error"] = (
            f"schema version {record.schema_version} does not match bundle version "
            f"{expected_schema}"
        )
        return False, payload, provenance
    return record.exists, payload, provenance


def cmd_verify(args: argparse.Namespace, settings: Settings) -> int:
    from sigil.evidence.bundle import BundleError, verify_bundle

    try:
        outcome = verify_bundle(args.bundle)
    except BundleError as exc:
        console.print(f"[red]{exc}[/red]")
        return EXIT_CONFIG

    payload: dict[str, Any] = {
        "root": outcome.root,
        "local_ok": outcome.ok,
        "tamper": {
            "modified": list(outcome.tamper.modified),
            "added": list(outcome.tamper.added),
            "removed": list(outcome.tamper.removed),
            "computed_root": outcome.tamper.computed_root,
        },
        "artifact_failures": list(outcome.artifact_failures),
    }

    if not args.as_json:
        console.print(
            Panel(
                f"root {outcome.root}\n"
                + (
                    "evidence intact, every field hashes to the anchored root"
                    if outcome.ok
                    else outcome.summary()
                ),
                title="Local verification " + ("PASS" if outcome.ok else "FAIL"),
                border_style="green" if outcome.ok else "red",
            )
        )
        for field_name in outcome.tamper.modified:
            console.print(f"  [red]✗ tampered field:[/red] [bold]{field_name}[/bold]")

    chain_ok: bool | None = None
    if not args.local_only:
        chain_ok, chain, provenance = _read_chain(args.bundle, outcome.root, settings)
        payload["chain"] = chain
        if not args.as_json:
            if provenance:
                console.print(f"[dim]{provenance}[/dim]")
            mark = {"verified": "[green]✓[/green]", "unrecorded": "[yellow]?[/yellow]"}.get(
                str(chain.get("contract_identity", "")), "[red]✗[/red]"
            )
            if chain.get("contract_identity_detail"):
                console.print(f"  {mark} registry: {chain['contract_identity_detail']}")
            if chain.get("error"):
                console.print(f"[red]on-chain check failed:[/red] {chain['error']}")
            elif not chain_ok:
                console.print(
                    Panel(
                        f"root {outcome.root} is NOT anchored on chain {chain['chain_id']}",
                        title="On-chain verification FAIL",
                        border_style="red",
                    )
                )
            elif outcome.ok:
                console.print(
                    Panel(
                        f"anchored by {chain['submitter']}\n"
                        f"at {chain['anchored_at']} on chain {chain['chain_id']}",
                        title="On-chain verification PASS",
                        border_style="green",
                    )
                )
                _print_url("registry:", str(chain.get("explorer", "")))
            else:
                # The root is genuinely anchored, but this bundle no longer hashes to it.
                # A green PASS here would let a tampered bundle read as verified at a
                # glance, which is the exact failure this design exists to prevent.
                console.print(
                    Panel(
                        f"The anchored root exists on chain {chain['chain_id']}, "
                        f"submitted by {chain['submitter']}\n"
                        f"at {chain['anchored_at']}, but the bundle on disk no longer "
                        "matches it.\n"
                        "The on-chain record is intact; the local evidence is not.",
                        title="On-chain record found, but the evidence does NOT match",
                        border_style="red",
                    )
                )

    if args.as_json:
        print(json.dumps(payload, indent=2, default=str))
    return EXIT_OK if outcome.ok and chain_ok is not False else EXIT_VERIFY_FAILED


def cmd_anchor(args: argparse.Namespace, settings: Settings) -> int:
    from sigil.chain import SEPOLIA_CHAIN_ID, ChainClient, ChainError, resolve_registry
    from sigil.evidence.bundle import (
        BundleError,
        attach_receipt,
        clear_pending,
        load_pending,
        load_proofs,
        write_pending,
    )

    try:
        proofs = load_proofs(args.bundle)
    except BundleError as exc:
        console.print(f"[red]{exc}[/red]")
        return EXIT_CONFIG

    root = str(proofs["root"])
    pending = load_pending(args.bundle)
    try:
        if pending:
            # A previous attempt got the transaction on chain but timed out waiting.
            # Resuming needs no key, and signing again could anchor twice.
            rpc_url, contract, _ = resolve_registry(args.bundle, settings)
            client = ChainClient(rpc_url, contract, expected_chain_id=SEPOLIA_CHAIN_ID)
            console.print(f"[dim]resuming pending transaction {pending}[/dim]")
            receipt = client.await_transaction(pending, root)
        else:
            settings.require("chain-write")
            client = ChainClient(
                settings.sepolia_rpc_url.get_secret_value(),
                settings.contract_address,
                expected_chain_id=SEPOLIA_CHAIN_ID,
            )
            receipt = client.anchor(
                root, int(proofs.get("schema_version", 1)), settings.private_key.get_secret_value()
            )
    except (ChainError, ConfigurationError) as exc:
        if getattr(exc, "transaction_hash", ""):
            write_pending(args.bundle, exc.transaction_hash, SEPOLIA_CHAIN_ID)  # type: ignore[union-attr]
        console.print(f"[red]{exc}[/red]")
        return EXIT_CONFIG

    clear_pending(args.bundle)
    attach_receipt(args.bundle, receipt)
    if args.as_json:
        print(receipt.model_dump_json(indent=2))
    else:
        console.print(
            Panel(
                f"root {root}\ntx {receipt.transaction_hash}",
                title="Anchored",
                border_style="green",
            )
        )
        _print_url("explorer:", receipt.explorer_url)
    return EXIT_OK


def cmd_prove(args: argparse.Namespace, settings: Settings) -> int:
    """Demonstrate all four task requirements against one real run."""

    from sigil.prove import prove

    report = prove(
        args.image,
        settings=settings,
        no_cache=args.no_cache,
        skip_chain=args.skip_chain,
    )

    if args.as_json:
        print(json.dumps(report.to_json(), indent=2))
        return EXIT_OK if report.passed else EXIT_VERIFY_FAILED

    table = Table(title="Task requirements, checked against this run")
    table.add_column("#", justify="right")
    table.add_column("Requirement")
    table.add_column("", justify="center")
    table.add_column("Evidence", overflow="fold")
    for item in report.requirements:
        mark = "[green]PASS[/green]" if item.passed else "[red]FAIL[/red]"
        body = "\n".join(item.detail) if item.passed else (item.error or "no evidence")
        table.add_row(str(item.number), item.name, mark, body)
    console.print(table)

    if report.bundle_dir:
        console.print(f"  [dim]bundle {report.bundle_dir}[/dim]")
    console.print(
        "\n  [bold green]All four requirements demonstrated.[/bold green]"
        if report.passed
        else "\n  [bold red]Not every requirement was demonstrated.[/bold red]"
    )
    return EXIT_OK if report.passed else EXIT_VERIFY_FAILED


def cmd_preflight(args: argparse.Namespace, settings: Settings) -> int:
    results = run_preflight(settings, live=getattr(args, "live", False))
    if args.as_json:
        print(json.dumps([item.model_dump(mode="json") for item in results], indent=2))
    else:
        table = Table(title=f"{__app_name__} preflight")
        table.add_column("Check")
        table.add_column("Status")
        table.add_column("Detail", overflow="fold")
        for result in results:
            colour = {"PASS": "green", "WARN": "yellow", "FAIL": "red"}[result.status.value]
            table.add_row(result.name, f"[{colour}]{result.status}[/{colour}]", result.detail)
        console.print(table)
    return EXIT_OK if preflight_passed(results) else EXIT_CONFIG


def cmd_serve(args: argparse.Namespace, settings: Settings) -> int:
    from sigil.web import serve

    try:
        serve(args.host, args.port, settings=settings, open_browser=not args.no_browser)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return EXIT_CONFIG
    except OSError as exc:
        console.print(f"[red]Could not bind {args.host}:{args.port}[/red], {exc}")
        return EXIT_CONFIG
    return EXIT_OK


def cmd_benchmark(args: argparse.Namespace, settings: Settings) -> int:
    from sigil.bench import run_benchmark, run_quality_calibration

    if args.quality_only:
        payload = run_quality_calibration(prefer_coreml=not args.cpu, output=args.output)
        print(json.dumps(payload, indent=2))
        return EXIT_OK

    report = run_benchmark(limit=args.limit, prefer_coreml=not args.cpu, output=args.output)
    if args.as_json:
        print(json.dumps(report.to_json(), indent=2))
    else:
        table = Table(title="Sigil accuracy and latency")
        table.add_column("Metric")
        table.add_column("Value", justify="right")

        def measured(value: float, places: int = 5) -> str:
            """Render an unmeasurable statistic as such, never as a number."""

            return "not measurable" if value != value else f"{value:.{places}f}"

        rows = [
            ("model", report.model),
            ("providers", ", ".join(report.providers)),
            ("calibration pairs", str(report.calibration_pairs)),
            (
                "held-out test pairs",
                f"{report.test_pairs} ({report.test_positive} genuine, "
                f"{report.test_negative} impostor)",
            ),
            ("ROC-AUC", measured(report.roc_auc)),
            ("EER", measured(report.eer)),
            ("TAR @ FMR 1e-2", measured(report.tar_at_fmr_1e2, 4)),
            ("TAR @ FMR 1e-3", measured(report.tar_at_fmr_1e3, 4)),
            ("match threshold", f"{report.match_threshold:.4f}"),
            ("reject threshold", f"{report.reject_threshold:.4f}"),
            ("false matches", f"{report.false_matches} / {report.test_negative}"),
            ("false non-matches", f"{report.false_non_matches} / {report.test_positive}"),
            ("detection failures", str(report.detection_failures)),
            ("detect p50/p95 ms", f"{report.detect_latency.p50_ms}/{report.detect_latency.p95_ms}"),
            ("embed p50/p95 ms", f"{report.embed_latency.p50_ms}/{report.embed_latency.p95_ms}"),
        ]
        for name, value in rows:
            table.add_row(name, value)
        console.print(table)
        for note in report.notes:
            console.print(f"  [dim]{note}[/dim]")
    return EXIT_OK


def _silence_teardown_noise() -> None:
    """Keep the terminal clean at exit.

    The CoreML execution provider emits partitioning notices on file descriptor 2 while
    tearing down. They are harmless, but they would land at the end of a screen
    recording, so the descriptor is closed off once real output is finished.
    """

    import atexit
    import os

    def quiet() -> None:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
        except OSError:
            pass

    atexit.register(quiet)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _silence_teardown_noise()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        settings = get_settings()
    except Exception as exc:  # configuration errors must not look like crashes
        console.print(f"[red]Configuration error:[/red] {exc}")
        return EXIT_CONFIG

    handlers = {
        "run": cmd_run,
        "verify": cmd_verify,
        "anchor": cmd_anchor,
        "prove": cmd_prove,
        "preflight": cmd_preflight,
        "serve": cmd_serve,
        "benchmark": cmd_benchmark,
    }
    try:
        return handlers[args.command](args, settings)
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted[/yellow]")
        return EXIT_CONFIG
    except ConfigurationError as exc:
        console.print(f"[red]{exc}[/red]")
        return EXIT_CONFIG


if __name__ == "__main__":
    sys.exit(main())
