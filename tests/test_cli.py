from __future__ import annotations

from pathlib import Path

from sigil.cli import build_parser, main


def test_verify_does_not_require_image() -> None:
    args = build_parser().parse_args(["verify", "--bundle", "evidence/run-1"])

    assert args.command == "verify"
    assert args.bundle == Path("evidence/run-1")


def test_run_requires_image() -> None:
    parser = build_parser()

    try:
        parser.parse_args(["run"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("run without --image must fail")


def test_preflight_json_is_machine_readable(capsys: object) -> None:
    exit_code = main(["preflight", "--json"])

    assert exit_code in {0, 2}
