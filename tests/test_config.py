from __future__ import annotations

import pytest

from sigil.config import ConfigurationError, Settings


def make_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "SERPAPI_KEY": "",
        "SEPOLIA_RPC_URL": "",
        "PRIVATE_KEY": "",
        "CONTRACT_ADDRESS": "",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_search_requires_only_serpapi_key() -> None:
    settings = make_settings(SERPAPI_KEY="search-secret")

    settings.require("search")


def test_chain_read_does_not_require_private_key() -> None:
    settings = make_settings(
        SEPOLIA_RPC_URL="https://rpc.example/secret",
        CONTRACT_ADDRESS="0x" + "1" * 40,
    )

    settings.require("chain-read")


def test_chain_write_lists_missing_fields_without_values() -> None:
    settings = make_settings()

    with pytest.raises(ConfigurationError) as exc_info:
        settings.require("chain-write")

    message = str(exc_info.value)
    assert "CONTRACT_ADDRESS" in message
    assert "PRIVATE_KEY" in message
    assert "SEPOLIA_RPC_URL" in message
    assert "secret" not in message.lower()


def test_redacted_summary_never_contains_secret_values() -> None:
    settings = make_settings(
        SERPAPI_KEY="search-secret",
        SEPOLIA_RPC_URL="https://rpc.example/rpc-secret",
        PRIVATE_KEY="wallet-secret",
    )

    rendered = repr(settings.redacted_summary())

    assert "search-secret" not in rendered
    assert "rpc-secret" not in rendered
    assert "wallet-secret" not in rendered
