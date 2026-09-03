"""SerpApi quota accounting and spend control.

Three separate failures are worth preventing, and they need different mechanisms:

1. *Silent exhaustion*, discovering at record time that the month's searches are gone.
   Handled by reading the account endpoint in preflight and before every live run.
2. *Reckless spend*, a debugging loop quietly burning fifty searches. Handled by a
   hard per-run budget that raises rather than continuing.
3. *Deadline starvation*, using the last search on a test instead of the recording.
   Handled by a reserve floor that live runs refuse to dip below.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

ACCOUNT_ENDPOINT = "https://serpapi.com/account"

# Searches held back for the final rehearsal and the recorded demo.
DEFAULT_RESERVE = 15


class QuotaError(RuntimeError):
    """Raised when a live search would exceed the configured budget or reserve."""


@dataclass(frozen=True, slots=True)
class AccountStatus:
    """A snapshot of SerpApi account health. Contains no credential material."""

    plan_name: str
    searches_left: int
    total_searches_left: int
    this_month_usage: int

    @property
    def healthy(self) -> bool:
        return self.searches_left > 0

    def describe(self, reserve: int = DEFAULT_RESERVE) -> str:
        return (
            f"{self.plan_name}: {self.searches_left} searches left "
            f"({self.this_month_usage} used this month, reserve {reserve})"
        )

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> AccountStatus:
        def as_int(key: str) -> int:
            try:
                return int(payload.get(key) or 0)
            except (TypeError, ValueError):
                return 0

        return cls(
            plan_name=str(payload.get("plan_name") or "unknown plan"),
            searches_left=as_int("plan_searches_left"),
            total_searches_left=as_int("total_searches_left") or as_int("plan_searches_left"),
            this_month_usage=as_int("this_month_usage"),
        )


@dataclass(slots=True)
class SearchBudget:
    """A hard ceiling on live searches for a single pipeline run.

    ``spend`` raises rather than returning False so that no caller can accidentally
    continue past the limit by ignoring a return value.
    """

    limit: int = 4
    spent: int = 0
    cached: int = 0

    def spend(self, route: str) -> None:
        if self.spent >= self.limit:
            raise QuotaError(
                f"search budget exhausted ({self.spent}/{self.limit}); "
                f"refusing to run route '{route}'. Raise --search-budget to allow more."
            )
        self.spent += 1
        logger.debug("search budget: %d/%d after %s", self.spent, self.limit, route)

    def record_cache_hit(self) -> None:
        self.cached += 1

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.spent)

    def summary(self) -> dict[str, int]:
        return {"live_searches": self.spent, "cached_searches": self.cached, "limit": self.limit}


def check_reserve(status: AccountStatus, *, needed: int, reserve: int = DEFAULT_RESERVE) -> None:
    """Refuse a live run that would eat into the reserved demo quota."""

    if status.searches_left - needed < reserve:
        raise QuotaError(
            f"only {status.searches_left} SerpApi searches remain; this run needs {needed} "
            f"and {reserve} are reserved for the final demo. "
            f"Use cached results (drop --no-cache), lower --search-budget, or top up the plan."
        )


__all__ = [
    "ACCOUNT_ENDPOINT",
    "DEFAULT_RESERVE",
    "AccountStatus",
    "QuotaError",
    "SearchBudget",
    "check_reserve",
]
