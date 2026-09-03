#!/usr/bin/env python3
"""Reject credentials and biometric artifacts before they reach history.

A leaked wallet key or a committed face crop cannot be un-published, so this runs in CI
on every push and is worth running locally before a commit too:

    python scripts/secret_scan.py

Exits non-zero on the first category of problem found, printing GitHub Actions
annotations when running under CI.
"""

from __future__ import annotations

import re
import subprocess
import sys

# Test vectors use repeated bytes (0xaaaa…, 0x1111…). A real key does not.
REPEATED_BYTE_HEX = re.compile(r"0x(?:([0-9a-fA-F]{2})\1{31})\b")

# Hardhat and Anvil's first well-known development account. Public by design.
WELL_KNOWN_DEV_KEYS = {
    "ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
}

SECRET_PATTERNS = (
    r"0x[a-fA-F0-9]{64}",
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
)
API_KEY_PATTERN = r"""(SERPAPI_KEY|api_key)\s*[:=]\s*["']?[a-f0-9]{40,}"""

# Paths whose whole purpose is to contain hex fixtures.
# deployments/ holds public transaction and contract hashes by design; they are the
# opposite of secret, since the whole point is that anyone can look them up.
EXCLUDED = (
    ":!tests",
    ":!test",
    ":!*.lock",
    ":!docs",
    ":!*.md",
    ":!scripts/secret_scan.py",
    ":!deployments",
)


def _git(*args: str) -> list[str]:
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    return [line for line in result.stdout.splitlines() if line.strip()]


def _error(message: str) -> None:
    prefix = "::error::" if sys.stdin.isatty() is False else ""
    print(f"{prefix}{message}", file=sys.stderr)


def check_tracked_files() -> bool:
    """No .env, and no generated evidence, may be tracked."""

    ok = True
    tracked = _git("ls-files")

    if ".env" in tracked:
        _error(".env is tracked; remove it from the index and rotate the credentials")
        ok = False

    generated = [
        path
        for path in tracked
        if re.match(r"^data/(outputs|bundles)/", path) and not path.endswith(".gitkeep")
    ]
    if generated:
        _error(f"generated evidence is tracked: {', '.join(generated[:5])}")
        ok = False

    biometric = [p for p in tracked if re.search(r"(face_crop|aligned_crop|embedding)", p)]
    if biometric:
        _error(f"biometric artifacts are tracked: {', '.join(biometric[:5])}")
        ok = False

    return ok


def check_secrets() -> bool:
    """No private-key-shaped literal outside the fixture directories."""

    hits: list[str] = []
    for pattern in SECRET_PATTERNS:
        for line in _git("grep", "-nIE", pattern, "--", *EXCLUDED):
            if REPEATED_BYTE_HEX.search(line):
                continue
            if any(key in line for key in WELL_KNOWN_DEV_KEYS):
                continue
            hits.append(line)

    if hits:
        _error("possible private key or secret:")
        for line in hits[:10]:
            print(f"  {line}", file=sys.stderr)
        return False
    return True


def check_api_keys() -> bool:
    hits = _git("grep", "-nIE", API_KEY_PATTERN, "--", ":!tests", ":!docs", ":!scripts")
    hits = [line for line in hits if "your_serpapi_key_here" not in line]
    if hits:
        _error("possible API key committed:")
        for line in hits[:10]:
            print(f"  {line}", file=sys.stderr)
        return False
    return True


def main() -> int:
    checks = (
        ("tracked files", check_tracked_files),
        ("secret literals", check_secrets),
        ("api keys", check_api_keys),
    )
    ok = True
    for name, check in checks:
        passed = check()
        print(f"{'ok  ' if passed else 'FAIL'} {name}")
        ok &= passed
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
