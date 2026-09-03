"""The provider contract and the concurrent fan-out that drives it."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from sigil.models import SearchCandidate

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ProviderResult:
    """What one source returned, plus enough to audit it afterwards."""

    name: str
    candidates: list[SearchCandidate] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    def summary(self) -> dict[str, Any]:
        return {
            "source": self.name,
            "candidates": len(self.candidates),
            "entities": list(self.entities),
            "elapsed_ms": round(self.elapsed_ms, 1),
            "error": self.error,
        }


def run_providers(
    jobs: Sequence[tuple[str, Callable[[], ProviderResult]]],
    *,
    timeout: float = 25.0,
) -> list[ProviderResult]:
    """Run every job concurrently and collect what finishes in time.

    One slow or broken source must not decide the latency of the whole search, and it
    must not take the run down with it: a source that raises is recorded as an error
    result and the rest of the fan-out proceeds. Wall clock is therefore the slowest
    source rather than the sum of all of them.
    """

    if not jobs:
        return []

    results: list[ProviderResult] = []
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {pool.submit(_timed, name, fn): name for name, fn in jobs}
        try:
            for future in as_completed(futures, timeout=timeout):
                results.append(future.result())
        except TimeoutError:
            late = [name for future, name in futures.items() if not future.done()]
            logger.warning("provider(s) %s exceeded %.0fs and were dropped", late, timeout)
            for name in late:
                results.append(ProviderResult(name=name, error=f"timed out after {timeout:.0f}s"))
    return results


def _timed(name: str, fn: Callable[[], ProviderResult]) -> ProviderResult:
    started = time.perf_counter()
    try:
        result = fn()
    except Exception as exc:  # a source failing is data, not a crash
        logger.warning("source %s failed: %s: %s", name, type(exc).__name__, exc)
        return ProviderResult(
            name=name,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            error=f"{type(exc).__name__}: {exc}",
        )
    result.elapsed_ms = (time.perf_counter() - started) * 1000.0
    logger.info(
        "source %s: %d candidate(s) in %.0fms", name, len(result.candidates), result.elapsed_ms
    )
    return result


__all__ = ["ProviderResult", "run_providers"]
