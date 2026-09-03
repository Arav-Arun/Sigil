"""Is everything the package needs actually in the repository?

This exists because it was not. A `.gitignore` rule reading `evidence/`, intended for a
top-level directory of generated bundles, matches a directory of that name at *any* depth
and silently excluded `sigil/evidence/`: the canonical JSON serialiser, the Merkle tree and
the bundle writer. Every local check passed, because the files were on disk. A fresh clone
of the published repository could not import the package at all.

CI eventually noticed, but only by accident: ruff could not resolve `sigil.evidence` as a
first-party module in the checkout, so it reported an import-sorting error. A missing
quarter of the codebase should not surface as a lint nit.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _tracked() -> set[str]:
    result = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        pytest.skip("not a git checkout")
    return set(result.stdout.split())


def _on_disk(*roots: str, suffix: str = ".py") -> set[str]:
    found: set[str] = set()
    for name in roots:
        base = ROOT / name
        if not base.is_dir():
            continue
        for path in base.rglob(f"*{suffix}"):
            if "__pycache__" in path.parts or ".venv" in path.parts:
                continue
            found.add(str(path.relative_to(ROOT)))
    return found


class TestEverySourceFileIsCommitted:
    def test_no_python_module_is_missing_from_the_repository(self):
        missing = sorted(_on_disk("sigil", "tests", "scripts") - _tracked())
        assert not missing, (
            "these Python files exist on disk but are not tracked by git, so a clone will "
            f"not contain them: {missing}. Check .gitignore for an unanchored directory "
            "pattern; a bare `name/` matches that directory at any depth."
        )

    def test_no_contract_or_deploy_script_is_missing(self):
        missing = sorted(
            (_on_disk("contracts", suffix=".sol") | _on_disk("scripts", "test", suffix=".ts"))
            - _tracked()
        )
        assert not missing, f"untracked contract or script files: {missing}"

    def test_every_package_directory_has_an_init(self):
        """A missing __init__.py makes a namespace package that behaves subtly differently."""

        packages = {
            path.parent
            for path in (ROOT / "sigil").rglob("*.py")
            if "__pycache__" not in path.parts
        }
        missing = sorted(
            str(directory.relative_to(ROOT))
            for directory in packages
            if not (directory / "__init__.py").exists()
        )
        assert not missing, f"package directories without __init__.py: {missing}"


class TestTheEvidenceModuleSpecifically:
    """The module that went missing, named so a regression is unmistakable."""

    @pytest.mark.parametrize("module", ["__init__.py", "canonical.py", "merkle.py", "bundle.py"])
    def test_is_tracked(self, module):
        assert f"sigil/evidence/{module}" in _tracked()

    def test_no_ignore_rule_excludes_anything_under_the_package(self):
        """The precise property, rather than a rule about how rules are written.

        Plenty of unanchored patterns are correct: `__pycache__/` and `node_modules/`
        should match at any depth. What must never happen is an ignore rule excluding a
        directory inside the package, which is what `evidence/` did.
        """

        directories = sorted(
            {
                str(path.parent.relative_to(ROOT))
                for path in (ROOT / "sigil").rglob("*.py")
                if "__pycache__" not in path.parts
            }
        )
        result = subprocess.run(
            ["git", "check-ignore", "-v", *directories],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        # check-ignore exits 1 and prints nothing when no path is ignored, which is the
        # outcome we want.
        assert not result.stdout.strip(), (
            "an ignore rule excludes a package directory, so a clone will be missing "
            f"source:\n{result.stdout.strip()}"
        )
