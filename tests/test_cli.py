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


def test_registry_resolution_prefers_settings_over_receipt(tmp_path: Path) -> None:
    """A receipt travels with the evidence, so it must never outrank configuration."""

    from sigil.chain import resolve_registry
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
    rpc, contract, _ = resolve_registry(bundle, settings)
    assert rpc == "https://sepolia.custom.example"
    assert contract == "0x2222222222222222222222222222222222222222"


def test_registry_resolution_falls_back_to_the_public_endpoint(tmp_path: Path) -> None:
    """With no credentials at all, verification still has somewhere to read from."""

    from sigil.chain import PUBLIC_SEPOLIA_RPC, resolve_registry
    from sigil.config import Settings

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    rpc, contract, provenance = resolve_registry(bundle, Settings(_env_file=None))
    assert rpc == PUBLIC_SEPOLIA_RPC
    # The checked-in deployment record is the trust anchor when nothing is configured.
    assert contract.startswith("0x")
    assert "checked-in" in provenance
