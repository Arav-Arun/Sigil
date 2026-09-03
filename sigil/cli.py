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
from sigil.config import ROOT_DIR, ConfigurationError, Settings, get_settings
from sigil.models import DecisionStatus, PipelineErrorCode
from sigil.preflight import preflight_passed, run_preflight

console = Console()

EXIT_OK = 0
EXIT_NO_MATCH = 1
EXIT_CONFIG = 2
EXIT_VERIFY_FAILED = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sigil", description="A face, sealed.")
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

    idx = sub.add_parser("index", help="Build and query a local face index")
    idx_sub = idx.add_subparsers(dest="index_command", required=True)
    idx_build = idx_sub.add_parser("build", help="Embed and index a corpus")
    idx_build.add_argument("--corpus", default="lfw", choices=["lfw", "runs"])
    idx_build.add_argument("--limit", type=int, default=None, help="Cap images indexed")
    idx_build.add_argument("--json", action="store_true", dest="as_json")
    idx_eval = idx_sub.add_parser("eval", help="Measure recall and query latency")
    idx_eval.add_argument("--corpus", default="lfw", choices=["lfw", "runs"])
    idx_eval.add_argument("--queries", type=int, default=300)
    idx_eval.add_argument("--json", action="store_true", dest="as_json")
    idx_search = idx_sub.add_parser("search", help="Find the nearest faces to an image")
    idx_search.add_argument("--image", required=True, type=Path)
    idx_search.add_argument("--corpus", default="lfw", choices=["lfw", "runs"])
    idx_search.add_argument("--top", type=int, default=10)
    idx_search.add_argument("--json", action="store_true", dest="as_json")

    pre = sub.add_parser("preflight", help="Check the local environment before a demo")
    pre.add_argument("--live", action="store_true", help="Also query SerpApi quota and the RPC")
    pre.add_argument("--json", action="store_true", dest="as_json")

    serve = sub.add_parser("serve", help="Open the local web interface")
    serve.add_argument("--port", type=int, default=8420)
    serve.add_argument("--host", default="127.0.0.1", help="localhost only, by design")
    serve.add_argument("--no-browser", action="store_true")

    bench = sub.add_parser("benchmark", help="Measure accuracy and latency on LFW")
    bench.add_argument("--limit", type=int, default=None, help="Cap pairs per split")
    bench.add_argument(
        "--resolution",
        action="store_true",
        help="Measure small-face recovery against a lanczos control instead",
    )
    bench.add_argument("--cpu", action="store_true", help="Force the CPU execution provider")
    bench.add_argument("--output", type=Path, default=Path("docs/benchmark.json"))
    bench.add_argument("--json", action="store_true", dest="as_json")
    return parser


# -- rendering --------------------------------------------------------------------


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
        console.print(f"  explorer      : [link]{result.receipt.explorer_url}[/link]")

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


# A public Sepolia endpoint, so re-verification needs no account anywhere. Reading a
# public ledger should not require a signup, and the whole claim of this project is that
# a third party can check the proof without asking us for anything.
PUBLIC_SEPOLIA_RPC = "https://ethereum-sepolia-rpc.publicnode.com"


def _chain_target(bundle: Path, settings: Settings) -> tuple[str, str, str]:
    """Decide which registry to read, preferring configuration over the bundle.

    A bundle names the registry it claims to be anchored in. Using that is not trusting
    it: the root either is or is not in the contract at that address, and a bundle that
    points somewhere convenient still fails, because the contract is public and so is the
    address it names. Falling back to it means a reviewer with an empty ``.env`` can still
    run the on-chain half, which is the difference between a checkable claim and one that
    requires our credentials to check.
    """

    rpc_url = settings.sepolia_rpc_url.get_secret_value() or ""
    contract = settings.contract_address or ""
    if rpc_url and contract:
        return rpc_url, contract, ""

    receipt_path = Path(bundle) / "receipt.json"
    receipt: dict[str, Any] = {}
    if receipt_path.is_file():
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            receipt = {}

    notes: list[str] = []
    if not contract:
        contract = str(receipt.get("contract_address") or "")
        if contract:
            notes.append(f"registry {contract} read from the bundle receipt")
    if not rpc_url:
        rpc_url = PUBLIC_SEPOLIA_RPC
        notes.append(f"using the public endpoint {PUBLIC_SEPOLIA_RPC}")

    if not contract:
        raise ConfigurationError(
            "Missing required configuration: CONTRACT_ADDRESS "
            "(and the bundle has no receipt.json to read it from)"
        )
    return rpc_url, contract, ", ".join(notes)


def cmd_verify(args: argparse.Namespace, settings: Settings) -> int:
    from sigil.chain import ChainClient, ChainError
    from sigil.evidence.bundle import BundleError, verify_bundle

    try:
        outcome = verify_bundle(args.bundle)
    except BundleError as exc:
        console.print(f"[red]{exc}[/red]")
        return EXIT_CONFIG

    payload: dict[str, Any] = {
        "root": outcome.root,
        "local_ok": outcome.tamper.ok and not outcome.artifact_failures,
        "tamper": {
            "modified": list(outcome.tamper.modified),
            "added": list(outcome.tamper.added),
            "removed": list(outcome.tamper.removed),
            "computed_root": outcome.tamper.computed_root,
        },
        "artifact_failures": list(outcome.artifact_failures),
    }

    local_ok = payload["local_ok"]
    if not args.as_json:
        style = "green" if local_ok else "red"
        console.print(
            Panel(
                f"root {outcome.root}\n"
                + (
                    "evidence intact, every field hashes to the anchored root"
                    if local_ok
                    else outcome.summary()
                ),
                title="Local verification " + ("PASS" if local_ok else "FAIL"),
                border_style=style,
            )
        )
        if outcome.tamper.modified:
            for field_name in outcome.tamper.modified:
                console.print(f"  [red]✗ tampered field:[/red] [bold]{field_name}[/bold]")

    chain_ok = None
    if not args.local_only:
        try:
            rpc_url, contract, provenance = _chain_target(args.bundle, settings)
            if not args.as_json and provenance:
                console.print(f"[dim]{provenance}[/dim]")
            client = ChainClient(rpc_url, contract, expected_chain_id=None)
            record = client.read(outcome.root)
            chain_ok = record.exists
            payload["chain"] = {
                "exists": record.exists,
                "submitter": record.submitter,
                "anchored_at": record.anchored_at.isoformat() if record.anchored_at else None,
                "schema_version": record.schema_version,
                "chain_id": record.chain_id,
                "contract": record.contract_address,
                "explorer": record.explorer_url(),
            }
            if not args.as_json:
                if record.exists and local_ok:
                    console.print(
                        Panel(
                            f"anchored by {record.submitter}\n"
                            f"at {record.anchored_at} on chain {record.chain_id}\n"
                            f"{record.explorer_url()}",
                            title="On-chain verification PASS",
                            border_style="green",
                        )
                    )
                elif record.exists:
                    # The root is genuinely anchored, but this bundle no longer hashes
                    # to it. Rendering a green PASS here would let a tampered bundle
                    # read as verified at a glance, which is the exact failure this
                    # whole design exists to prevent.
                    console.print(
                        Panel(
                            f"The anchored root exists on chain {record.chain_id}, "
                            f"submitted by {record.submitter}\n"
                            f"at {record.anchored_at}, but the bundle on disk no longer "
                            "matches it.\n"
                            "The on-chain record is intact; the local evidence is not.",
                            title="On-chain record found, but the evidence does NOT match",
                            border_style="red",
                        )
                    )
                else:
                    console.print(
                        Panel(
                            f"root {outcome.root} is NOT anchored on chain {record.chain_id}",
                            title="On-chain verification FAIL",
                            border_style="red",
                        )
                    )
        except (ChainError, ConfigurationError) as exc:
            payload["chain"] = {"error": str(exc)}
            if not args.as_json:
                console.print(f"[yellow]on-chain check skipped:[/yellow] {exc}")

    if args.as_json:
        print(json.dumps(payload, indent=2, default=str))

    if not local_ok:
        return EXIT_VERIFY_FAILED
    if chain_ok is False:
        return EXIT_VERIFY_FAILED
    return EXIT_OK


def cmd_anchor(args: argparse.Namespace, settings: Settings) -> int:
    from sigil.chain import ChainClient, ChainError
    from sigil.evidence.bundle import BundleError, attach_receipt, load_proofs

    try:
        proofs = load_proofs(args.bundle)
    except BundleError as exc:
        console.print(f"[red]{exc}[/red]")
        return EXIT_CONFIG

    root = str(proofs["root"])
    try:
        settings.require("chain-write")
        client = ChainClient(
            settings.sepolia_rpc_url.get_secret_value(),
            settings.contract_address,
            expected_chain_id=None,
        )
        receipt = client.anchor(
            root, int(proofs.get("schema_version", 1)), settings.private_key.get_secret_value()
        )
    except (ChainError, ConfigurationError) as exc:
        console.print(f"[red]{exc}[/red]")
        return EXIT_CONFIG

    attach_receipt(args.bundle, receipt)
    if args.as_json:
        print(receipt.model_dump_json(indent=2))
    else:
        console.print(
            Panel(
                f"root {root}\ntx {receipt.transaction_hash}\n{receipt.explorer_url}",
                title="Anchored",
                border_style="green",
            )
        )
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


def cmd_index(args: argparse.Namespace, settings: Settings) -> int:
    """Build, measure, or query the local face index."""

    from sigil.index import FaceIndex, build_index, evaluate_index, index_path

    path = index_path(args.corpus)

    if args.index_command == "build":
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            TimeRemainingColumn,
        )

        with Progress(
            *Progress.get_default_columns()[:2],
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Indexing {args.corpus}", total=args.limit)

            def advance(done: int, total: int) -> None:
                progress.update(task, completed=done, total=total)

            index = build_index(args.corpus, limit=args.limit, on_progress=advance)
        index.save(path)
        payload = index.stats.to_json()
        if args.as_json:
            print(json.dumps(payload, indent=2))
            return EXIT_OK
        table = Table(title=f"Face index built from {args.corpus}")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        for key, value in payload.items():
            table.add_row(key.replace("_", " "), str(value))
        console.print(table)
        console.print(f"  [dim]{path}[/dim]")
        return EXIT_OK

    if not path.is_file():
        console.print(
            f"[red]No index at {path}.[/red] Build one: sigil index build --corpus {args.corpus}"
        )
        return EXIT_CONFIG
    index = FaceIndex.load(path)

    if args.index_command == "eval":
        report = evaluate_index(index, queries=args.queries)
        if args.as_json:
            print(json.dumps(report, indent=2))
            return EXIT_OK
        table = Table(title="Face index retrieval, leave-one-out")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        for key, value in report.items():
            if key == "note":
                continue
            table.add_row(key.replace("_", " "), str(value))
        console.print(table)
        console.print(f"  [dim]{report.get('note', '')}[/dim]")
        return EXIT_OK

    # search
    import tempfile

    import numpy as np

    from sigil.face import detect_and_encode

    with tempfile.TemporaryDirectory() as tmp:
        face = detect_and_encode(args.image, tmp, select_largest=True)
    hits = index.search(np.asarray(face.embedding, dtype=np.float32), top=args.top)

    if args.as_json:
        print(json.dumps([hit.to_json() for hit in hits], indent=2))
        return EXIT_OK

    from sigil.verify import DEFAULT_MATCH_THRESHOLD, DEFAULT_REJECT_THRESHOLD

    table = Table(title=f"Nearest faces in {args.corpus} ({len(index):,} vectors)")
    table.add_column("#", justify="right")
    table.add_column("Identity")
    table.add_column("Distance", justify="right")
    table.add_column("Gate")
    for position, hit in enumerate(hits, start=1):
        if hit.distance <= DEFAULT_MATCH_THRESHOLD:
            gate = "[green]MATCH[/green]"
        elif hit.distance >= DEFAULT_REJECT_THRESHOLD:
            gate = "[red]NON_MATCH[/red]"
        else:
            gate = "[yellow]INCONCLUSIVE[/yellow]"
        table.add_row(str(position), hit.label, f"{hit.distance:.4f}", gate)
    console.print(table)
    console.print(
        "  [dim]Retrieval ranks; the same face gate still decides. A nearest neighbour "
        "is not a match.[/dim]"
    )
    return EXIT_OK


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


def _cmd_benchmark_resolution(args: argparse.Namespace) -> int:
    """Report every small-face treatment side by side, including when none of them win."""

    from sigil.bench_resolution import run_resolution_benchmark

    report = run_resolution_benchmark(
        limit=args.limit or 200,
        output=ROOT_DIR / "docs" / "benchmark-resolution.json",
    )
    if args.as_json:
        print(json.dumps(report.to_json(), indent=2))
        return EXIT_OK

    table = Table(title="Small-face recovery, measured against a lanczos control")
    table.add_column("Face size")
    table.add_column("Treatment")
    table.add_column("Decidable", justify="right")
    table.add_column("TAR @ FMR 1e-2", justify="right")
    table.add_column("Separation", justify="right")
    for row in report.results:
        tar, sep = row["tar_at_fmr_1e2"], row["separation"]
        table.add_row(
            f"{row['face_px']}px",
            row["treatment"],
            f"{row['decided']}/{row['pairs']}  ({row['coverage']:.0%})",
            "not measurable" if tar != tar else f"{tar:.4f}",
            "n/a" if sep != sep else f"{sep:.2f}",
        )
    console.print(table)
    console.print(f"\n  [bold]{report.verdict}[/bold]")
    for note in report.notes:
        console.print(f"  [dim]{note}[/dim]")
    return EXIT_OK


def cmd_benchmark(args: argparse.Namespace, settings: Settings) -> int:
    if getattr(args, "resolution", False):
        return _cmd_benchmark_resolution(args)

    from sigil.bench import run_benchmark

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
        "index": cmd_index,
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
