"""Typed, redaction-safe configuration for Sigil."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
SAMPLES_DIR = DATA_DIR / "samples"
OUTPUTS_DIR = DATA_DIR / "outputs"

SOCIAL_DOMAINS = (
    "instagram.com",
    "twitter.com",
    "x.com",
    "facebook.com",
    "linkedin.com",
    "tiktok.com",
    "reddit.com",
    "youtube.com",
    "youtu.be",
    "threads.net",
    "pinterest.com",
)


class ConfigurationError(ValueError):
    """Raised when a requested stage is missing required configuration."""


class Settings(BaseSettings):
    """Environment-backed settings whose secret values are redacted by design."""

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    serpapi_key: SecretStr = Field(default=SecretStr(""), alias="SERPAPI_KEY")
    # Optional second discovery source. Absent is a normal state: the fan-out simply
    # runs with whatever sources are configured.
    exa_api_key: SecretStr = Field(default=SecretStr(""), alias="EXA_API_KEY")
    sepolia_rpc_url: SecretStr = Field(default=SecretStr(""), alias="SEPOLIA_RPC_URL")
    private_key: SecretStr = Field(default=SecretStr(""), alias="PRIVATE_KEY")
    contract_address: str = Field(default="", alias="CONTRACT_ADDRESS")

    mode: Literal["balanced", "accurate", "fast"] = "balanced"
    detector_backend: str = "scrfd_10g"
    model_name: str = "arcface_w600k_r50"
    distance_metric: Literal["cosine", "euclidean", "euclidean_l2"] = "cosine"
    # Wave one routinely returns 60+ results. Verifying more of them costs one batched
    # embed and a few concurrent fetches, and it is the difference between a grid that
    # ends at twelve and one a reviewer can actually work through.
    max_candidates: int = Field(default=24, ge=1, le=50)
    download_concurrency: int = Field(default=8, ge=1, le=10)
    http_timeout_seconds: float = Field(default=15.0, gt=0, le=60)
    http_retries: int = Field(default=2, ge=0, le=5)
    search_country: str = Field(default="in", min_length=2, max_length=2)
    search_language: str = Field(default="en", min_length=2, max_length=2)

    @field_validator("contract_address")
    @classmethod
    def validate_contract_address(cls, value: str) -> str:
        if value and (not value.startswith("0x") or len(value) != 42):
            raise ValueError("CONTRACT_ADDRESS must be a 20-byte 0x-prefixed address")
        return value

    def require(self, *stages: Literal["search", "chain-read", "chain-write"]) -> None:
        """Validate only the credentials required by the requested pipeline stages."""

        missing: list[str] = []
        if "search" in stages and not self.serpapi_key.get_secret_value():
            missing.append("SERPAPI_KEY")
        if {"chain-read", "chain-write"}.intersection(stages):
            if not self.sepolia_rpc_url.get_secret_value():
                missing.append("SEPOLIA_RPC_URL")
            if not self.contract_address:
                missing.append("CONTRACT_ADDRESS")
        if "chain-write" in stages and not self.private_key.get_secret_value():
            missing.append("PRIVATE_KEY")
        if missing:
            names = ", ".join(sorted(set(missing)))
            raise ConfigurationError(f"Missing required configuration: {names}")

    def redacted_summary(self) -> dict[str, object]:
        """Return diagnostics that reveal presence, never credential values."""

        return {
            "serpapi_configured": bool(self.serpapi_key.get_secret_value()),
            "exa_configured": bool(self.exa_api_key.get_secret_value()),
            "sepolia_rpc_configured": bool(self.sepolia_rpc_url.get_secret_value()),
            "private_key_configured": bool(self.private_key.get_secret_value()),
            "contract_address": self.contract_address or None,
            "mode": self.mode,
            "detector_backend": self.detector_backend,
            "model_name": self.model_name,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache application settings."""

    return Settings()


def ensure_output_dir() -> Path:
    """Create the outputs directory if it does not exist."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUTS_DIR
