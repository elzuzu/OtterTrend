from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol


@dataclass
class RiskContext:
    symbol: str
    side: str
    amount_usd: float
    balance_usd: float
    market: Dict[str, float]


@dataclass
class RiskCheckResult:
    approved: bool
    reason: str | None = None


class RiskRule(Protocol):
    @property
    def name(self) -> str:  # pragma: no cover - protocol
        ...

    def check(self, ctx: RiskContext) -> RiskCheckResult:  # pragma: no cover - protocol
        ...


__all__ = ["RiskContext", "RiskCheckResult", "RiskRule"]
