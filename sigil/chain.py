"""Anchoring and independent verification against an EVM chain.

Two properties drive the design:

* **Verification must never need a private key.** Re-verifying is a read-only
  ``eth_call``; anyone with the bundle and a public RPC endpoint can do it. If proving
  the claim required the claimant's key, it would not be a proof.
* **A failure must never render as success.** Every path that could return a
  falsely-positive "verified" is closed: chain ID is asserted before signing, the
  receipt status is checked, a timed-out transaction is persisted as pending rather than
  resubmitted, and an unknown root reads as absent rather than as a zero struct.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sigil.config import ROOT_DIR, Settings, get_settings

if TYPE_CHECKING:
    from web3.types import TxParams
from sigil.models import ChainReceipt, PipelineErrorCode

logger = logging.getLogger(__name__)

SEPOLIA_CHAIN_ID = 11155111

EXPLORERS = {
    11155111: "https://sepolia.etherscan.io",
    1: "https://etherscan.io",
    84532: "https://sepolia.basescan.org",
    80002: "https://amoy.polygonscan.com",
}

# Minimal ABI. Checked in rather than read from Hardhat artifacts so that verification
# works from a clean clone without compiling anything.
REGISTRY_ABI: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "anchor",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "root", "type": "bytes32"},
            {"name": "schemaVersion", "type": "uint16"},
        ],
        "outputs": [],
    },
    {
        "type": "function",
        "name": "verify",
        "stateMutability": "view",
        "inputs": [{"name": "root", "type": "bytes32"}],
        "outputs": [
            {"name": "exists", "type": "bool"},
            {"name": "submitter", "type": "address"},
            {"name": "anchoredAt", "type": "uint64"},
            {"name": "schemaVersion", "type": "uint16"},
        ],
    },
    {
        "type": "function",
        "name": "isAnchored",
        "stateMutability": "view",
        "inputs": [{"name": "root", "type": "bytes32"}],
        "outputs": [{"name": "", "type": "bool"}],
    },
    {
        "type": "function",
        "name": "totalAnchored",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"name": "", "type": "uint256"}],
    },
    {
        "type": "event",
        "name": "Anchored",
        "anonymous": False,
        "inputs": [
            {"name": "root", "type": "bytes32", "indexed": True},
            {"name": "submitter", "type": "address", "indexed": True},
            {"name": "anchoredAt", "type": "uint64", "indexed": False},
            {"name": "schemaVersion", "type": "uint16", "indexed": False},
        ],
    },
]


class ChainError(RuntimeError):
    """A chain interaction failed. Carries a stable pipeline error code."""

    def __init__(self, code: PipelineErrorCode, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class AnchorRecord:
    """What the chain says about one root."""

    exists: bool
    submitter: str
    anchored_at: datetime | None
    schema_version: int
    chain_id: int
    contract_address: str

    def explorer_url(self) -> str:
        base = EXPLORERS.get(self.chain_id)
        return f"{base}/address/{self.contract_address}" if base else ""


def explorer_tx_url(chain_id: int, tx_hash: str) -> str:
    base = EXPLORERS.get(chain_id)
    return f"{base}/tx/{tx_hash}" if base else ""


def _root_bytes(root: str | bytes) -> bytes:
    value = root if isinstance(root, bytes) else bytes.fromhex(root.removeprefix("0x"))
    if len(value) != 32:
        raise ChainError(
            PipelineErrorCode.INVALID_INPUT, f"evidence root must be 32 bytes, got {len(value)}"
        )
    if value == b"\x00" * 32:
        raise ChainError(PipelineErrorCode.INVALID_INPUT, "refusing to use the zero root")
    return value


class ChainClient:
    """Thin web3 wrapper around :class:`SigilRegistry`."""

    def __init__(
        self,
        rpc_url: str,
        contract_address: str,
        *,
        expected_chain_id: int | None = SEPOLIA_CHAIN_ID,
        timeout: float = 30.0,
    ) -> None:
        try:
            from web3 import HTTPProvider, Web3
        except ImportError as exc:  # pragma: no cover - declared dependency
            raise ChainError(
                PipelineErrorCode.INVALID_CONFIGURATION,
                "web3 is not installed; run `uv sync --all-extras`",
            ) from exc

        if not rpc_url:
            raise ChainError(PipelineErrorCode.INVALID_CONFIGURATION, "RPC URL is not configured")

        self._w3 = Web3(HTTPProvider(rpc_url, request_kwargs={"timeout": timeout}))
        if not self._w3.is_connected():
            raise ChainError(
                PipelineErrorCode.SEARCH_UNAVAILABLE, "could not connect to the RPC endpoint"
            )

        self.chain_id = self._w3.eth.chain_id
        if expected_chain_id is not None and self.chain_id != expected_chain_id:
            raise ChainError(
                PipelineErrorCode.CHAIN_MISMATCH,
                f"connected to chain {self.chain_id}, expected {expected_chain_id}. "
                "Refusing to continue against the wrong network.",
            )

        self.address = self._w3.to_checksum_address(contract_address)
        code = self._w3.eth.get_code(self.address)
        if not code or code == b"":
            raise ChainError(
                PipelineErrorCode.CHAIN_MISMATCH,
                f"no contract bytecode at {self.address} on chain {self.chain_id}",
            )
        self.contract = self._w3.eth.contract(address=self.address, abi=REGISTRY_ABI)

    # -- read (no private key required) ------------------------------------------

    def read(self, root: str | bytes) -> AnchorRecord:
        """Read a root's anchor record. This is the whole verification path."""

        raw = _root_bytes(root)
        exists, submitter, anchored_at, schema_version = self.contract.functions.verify(raw).call()
        return AnchorRecord(
            exists=bool(exists),
            submitter=str(submitter),
            anchored_at=datetime.fromtimestamp(anchored_at, tz=UTC) if anchored_at else None,
            schema_version=int(schema_version),
            chain_id=self.chain_id,
            contract_address=self.address,
        )

    def total_anchored(self) -> int:
        return int(self.contract.functions.totalAnchored().call())

    def deployed_code(self) -> str:
        """The runtime bytecode actually deployed at this address, as lowercase hex."""

        return "0x" + self._w3.eth.get_code(self.address).hex().removeprefix("0x").lower()

    # -- write --------------------------------------------------------------------

    def anchor(
        self,
        root: str | bytes,
        schema_version: int,
        private_key: str,
        *,
        confirmations: int = 1,
        timeout: float = 180.0,
    ) -> ChainReceipt:
        """Anchor a root and wait for the receipt, verifying it actually succeeded."""

        from eth_account import Account
        from web3.types import Wei

        raw = _root_bytes(root)
        account = Account.from_key(private_key)
        sender = self._w3.to_checksum_address(account.address)

        balance = self._w3.eth.get_balance(sender)
        if balance == 0:
            raise ChainError(
                PipelineErrorCode.CHAIN_PENDING,
                f"{sender} has zero balance on chain {self.chain_id}; fund it from a faucet",
            )

        existing = self.read(raw)
        if existing.exists:
            raise ChainError(
                PipelineErrorCode.CHAIN_MISMATCH,
                f"root is already anchored by {existing.submitter} at {existing.anchored_at}",
            )

        latest = self._w3.eth.get_block("latest")
        base_fee = latest.get("baseFeePerGas") or self._w3.eth.gas_price
        priority = self._w3.eth.max_priority_fee
        params: TxParams = {
            "from": sender,
            "nonce": self._w3.eth.get_transaction_count(sender),
            "chainId": self.chain_id,
            "maxPriorityFeePerGas": Wei(priority),
            # Two base fees of headroom absorbs several blocks of congestion.
            "maxFeePerGas": Wei(base_fee * 2 + priority),
        }
        transaction = self.contract.functions.anchor(raw, int(schema_version)).build_transaction(
            params
        )
        transaction["gas"] = int(self._w3.eth.estimate_gas(transaction) * 1.2)

        signed = account.sign_transaction(transaction)
        tx_hash = self._w3.eth.send_raw_transaction(signed.raw_transaction)
        tx_hex = tx_hash.hex()
        if not tx_hex.startswith("0x"):
            tx_hex = "0x" + tx_hex
        logger.info("anchor transaction submitted: %s", tx_hex)

        try:
            receipt = self._w3.eth.wait_for_transaction_receipt(tx_hash, timeout=timeout)
        except Exception as exc:
            # The transaction may still confirm. Surfacing it as pending lets a later
            # `sigil anchor --resume` finish without submitting a duplicate.
            raise ChainError(
                PipelineErrorCode.CHAIN_PENDING,
                f"transaction {tx_hex} did not confirm within {timeout}s: {exc}",
            ) from exc

        if receipt["status"] != 1:
            raise ChainError(
                PipelineErrorCode.CHAIN_MISMATCH, f"transaction {tx_hex} reverted on chain"
            )

        if confirmations > 1:
            target = receipt["blockNumber"] + confirmations - 1
            while self._w3.eth.block_number < target:  # pragma: no cover - timing dependent
                pass

        record = self.read(raw)
        if not record.exists:
            raise ChainError(
                PipelineErrorCode.CHAIN_MISMATCH,
                "transaction succeeded but the root does not read back; refusing to claim success",
            )

        return ChainReceipt(
            chain_id=self.chain_id,
            contract_address=self.address,
            evidence_root="0x" + raw.hex(),
            transaction_hash=tx_hex,
            block_number=int(receipt["blockNumber"]),
            submitter=sender,
            anchored_at=record.anchored_at or datetime.now(UTC),
            gas_used=int(receipt["gasUsed"]),
            explorer_url=explorer_tx_url(self.chain_id, tx_hex),
        )


def deployment_path(chain_id: int = SEPOLIA_CHAIN_ID) -> Path:
    name = {11155111: "sepolia", 1: "mainnet", 84532: "base-sepolia", 80002: "amoy"}.get(
        chain_id, str(chain_id)
    )
    return ROOT_DIR / "deployments" / f"{name}.json"


def load_deployment(chain_id: int = SEPOLIA_CHAIN_ID) -> dict[str, Any] | None:
    path = deployment_path(chain_id)
    if not path.is_file():
        return None
    try:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data


def client_from_settings(
    settings: Settings | None = None, *, expected_chain_id: int | None = SEPOLIA_CHAIN_ID
) -> ChainClient:
    resolved = settings or get_settings()
    resolved.require("chain-read")
    return ChainClient(
        resolved.sepolia_rpc_url.get_secret_value(),
        resolved.contract_address,
        expected_chain_id=expected_chain_id,
    )


__all__ = [
    "EXPLORERS",
    "REGISTRY_ABI",
    "SEPOLIA_CHAIN_ID",
    "AnchorRecord",
    "ChainClient",
    "ChainError",
    "client_from_settings",
    "deployment_path",
    "explorer_tx_url",
    "load_deployment",
]
