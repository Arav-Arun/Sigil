"""Pre-demo environment checks.

Everything that has ever broken a live demo gets a check here. The ordering is
deliberate: free local checks run first, and the two checks that cost something (a
SerpApi account lookup, an RPC round-trip) run last and are skipped when unconfigured.

Credential *values* are never printed, only whether they are present and whether they
work.
"""

from __future__ import annotations

import json
import platform
import shutil
import sys
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from sigil.config import ROOT_DIR, Settings, ensure_output_dir

if TYPE_CHECKING:
    from sigil.chain import ChainClient

REQUIRED_PYTHON = (3, 12)
MIN_FREE_DISK_GB = 2.0

# Compiled by `npx hardhat compile`. Absent in a fresh clone, which is a WARN, not a FAIL.
ARTIFACT_PATH = ROOT_DIR / "artifacts" / "contracts" / "SigilRegistry.sol" / "SigilRegistry.json"

# solc appends a CBOR metadata blob to the runtime code: the IPFS hash of the source
# metadata, the compiler version, and a two-byte length. 53 bytes, so 106 hex characters.
METADATA_HEX_LEN = 106


class CheckStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"


class CheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    status: CheckStatus
    detail: str


def _ok(name: str, detail: str) -> CheckResult:
    return CheckResult(name=name, status=CheckStatus.PASS, detail=detail)


def _warn(name: str, detail: str) -> CheckResult:
    return CheckResult(name=name, status=CheckStatus.WARN, detail=detail)


def _fail(name: str, detail: str) -> CheckResult:
    return CheckResult(name=name, status=CheckStatus.FAIL, detail=detail)


def _check_python() -> CheckResult:
    actual = sys.version_info[:2]
    detail = f"{platform.python_version()} (required {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}.x)"
    return _ok("python", detail) if actual == REQUIRED_PYTHON else _fail("python", detail)


def _check_disk() -> CheckResult:
    free_gb = shutil.disk_usage(ROOT_DIR).free / 1e9
    detail = f"{free_gb:.1f} GB free"
    if free_gb < MIN_FREE_DISK_GB:
        return _warn("disk_space", f"{detail}; models and bundles need headroom")
    return _ok("disk_space", detail)


def _check_output_dir() -> CheckResult:
    path = ensure_output_dir()
    try:
        probe = path / ".write-probe"
        probe.touch(exist_ok=True)
        probe.unlink()
    except OSError as exc:
        return _fail("output_directory", str(exc))
    return _ok("output_directory", str(path))


def _check_models() -> list[CheckResult]:
    """Confirm the ONNX runtime and weights are present before a demo, not during one."""

    results: list[CheckResult] = []
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        accelerated = [p for p in providers if p != "CPUExecutionProvider"]
        results.append(
            _ok(
                "onnxruntime",
                f"{ort.__version__}; providers: {', '.join(providers)}"
                + ("" if accelerated else " (CPU only)"),
            )
        )
    except ImportError:
        return [_fail("onnxruntime", "not installed; run `uv sync --all-extras`")]

    try:
        from sigil.face.engine import DETECTOR_FILE, EMBEDDER_FILE, model_pack_dir

        directory = model_pack_dir()
        missing = [f for f in (DETECTOR_FILE, EMBEDDER_FILE) if not (directory / f).is_file()]
        if missing:
            results.append(_fail("face_models", f"missing from {directory}: {', '.join(missing)}"))
        else:
            size_mb = sum((directory / f).stat().st_size for f in (DETECTOR_FILE, EMBEDDER_FILE))
            results.append(_ok("face_models", f"{directory} ({size_mb / 1e6:.0f} MB)"))
    except Exception as exc:
        results.append(_fail("face_models", f"unavailable: {exc}"))
    return results


def _check_serpapi(settings: Settings, *, live: bool) -> CheckResult:
    if not settings.serpapi_key.get_secret_value():
        return _warn("serpapi", "SERPAPI_KEY is not configured; the search stage will not run")
    if not live:
        return _ok("serpapi", "key configured (pass --live to query remaining quota)")

    try:
        from sigil.search.quota import DEFAULT_RESERVE
        from sigil.search.serpapi import SerpApiClient

        status = SerpApiClient(settings.serpapi_key.get_secret_value()).account()
    except Exception as exc:
        return _fail("serpapi", f"account lookup failed: {exc}")

    detail = status.describe(DEFAULT_RESERVE)
    if status.searches_left <= 0:
        return _fail("serpapi", detail + ", quota exhausted")
    if status.searches_left < DEFAULT_RESERVE:
        return _warn("serpapi", detail + ", below the demo reserve")
    return _ok("serpapi", detail)


def _check_bytecode(client: ChainClient) -> CheckResult:
    """Compare the deployed runtime code against the contract compiled from this tree.

    This check exists because the failure it catches is invisible otherwise. Editing a
    *comment* in the contract changes the source hash solc embeds in the metadata blob,
    so the repository stops reproducing the deployed bytecode while every file still
    claims the deployment is current. The executable code is identical in that case, so
    nothing misbehaves. It just quietly stops being verifiable.
    """

    if not ARTIFACT_PATH.is_file():
        return _warn("bytecode", "run `npx hardhat compile` to check the deployed code")

    try:
        artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
        expected = artifact["deployedBytecode"]
        if isinstance(expected, dict):
            expected = expected["object"]
        deployed = client.deployed_code()
    except Exception as exc:
        return _warn("bytecode", f"could not compare deployed code: {exc}")

    deployed = "0x" + deployed.removeprefix("0x").lower()
    expected = "0x" + str(expected).removeprefix("0x").lower()

    if deployed == expected:
        return _ok("bytecode", "deployed code matches contracts/SigilRegistry.sol exactly")
    if deployed[:-METADATA_HEX_LEN] == expected[:-METADATA_HEX_LEN]:
        return _warn(
            "bytecode",
            "executable code matches but solc metadata differs; the contract source has "
            "changed since deployment, so this tree no longer reproduces it. Redeploy.",
        )
    return _fail(
        "bytecode",
        "deployed code does NOT match contracts/SigilRegistry.sol; "
        "CONTRACT_ADDRESS points at a different contract",
    )


def _check_chain(settings: Settings, *, live: bool) -> list[CheckResult]:
    results: list[CheckResult] = []
    if not settings.sepolia_rpc_url.get_secret_value():
        return [_warn("chain", "SEPOLIA_RPC_URL is not configured; anchoring will not run")]
    if not settings.contract_address:
        return [_warn("chain", "CONTRACT_ADDRESS is not configured; deploy the contract first")]
    if not live:
        return [_ok("chain", f"RPC and contract {settings.contract_address} configured")]

    try:
        from sigil.chain import ChainClient

        client = ChainClient(
            settings.sepolia_rpc_url.get_secret_value(),
            settings.contract_address,
            expected_chain_id=None,
        )
        results.append(
            _ok(
                "chain",
                f"chain {client.chain_id}, contract {client.address}, "
                f"{client.total_anchored()} root(s) anchored",
            )
        )
    except Exception as exc:
        return [_fail("chain", str(exc))]

    results.append(_check_bytecode(client))

    key = settings.private_key.get_secret_value()
    if not key:
        results.append(_warn("wallet", "PRIVATE_KEY is not configured; verification still works"))
        return results

    try:
        from eth_account import Account

        address = Account.from_key(key).address
        balance = client._w3.eth.get_balance(address) / 1e18
        detail = f"{address} holds {balance:.4f} ETH"
        results.append(
            _ok("wallet", detail)
            if balance > 0.001
            else _fail("wallet", detail + ", fund from a faucet")
        )
    except Exception as exc:
        results.append(_fail("wallet", f"could not read wallet: {exc}"))
    return results


def run_preflight(settings: Settings, *, live: bool = False) -> list[CheckResult]:
    """Run every check. ``live`` enables the two that make network calls."""

    results: list[CheckResult] = [
        _check_python(),
        CheckResult(
            name="project_root",
            status=CheckStatus.PASS
            if (ROOT_DIR / "pyproject.toml").is_file()
            else CheckStatus.FAIL,
            detail=str(ROOT_DIR),
        ),
        _check_disk(),
        _check_output_dir(),
    ]
    results.extend(_check_models())
    results.append(_check_serpapi(settings, live=live))
    results.extend(_check_chain(settings, live=live))
    return results


def preflight_passed(results: list[CheckResult]) -> bool:
    return all(result.status is not CheckStatus.FAIL for result in results)


__all__ = ["CheckResult", "CheckStatus", "preflight_passed", "run_preflight"]
