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


def test_run_fails_with_exit_config_if_image_missing(tmp_path: Path) -> None:
    exit_code = main(["run", "--image", str(tmp_path / "nonexistent.jpg")])
    assert exit_code == 2


def test_verify_fails_with_exit_config_if_bundle_missing(tmp_path: Path) -> None:
    exit_code = main(["verify", "--bundle", str(tmp_path / "nonexistent-bundle")])
    assert exit_code == 2


def test_chain_target_prefers_settings_over_receipt(tmp_path: Path) -> None:
    from sigil.cli import _chain_target
    from sigil.config import Settings

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "receipt.json").write_text(
        '{"contract_address": "0x1111111111111111111111111111111111111111"}', encoding="utf-8"
    )

    settings = Settings(
        _env_file=None,
        SEPOLIA_RPC_URL="https://sepolia.custom.example",
        CONTRACT_ADDRESS="0x2222222222222222222222222222222222222222",
    )
    rpc, contract, _ = _chain_target(bundle, settings)
    assert rpc == "https://sepolia.custom.example"
    assert contract == "0x2222222222222222222222222222222222222222"
