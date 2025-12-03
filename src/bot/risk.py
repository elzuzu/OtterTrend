from __future__ import annotations

from typing import Dict, List

from typing import Dict, List

from src.interfaces.risk_rule import RiskCheckResult, RiskContext, RiskRule


class MaxOrderSizeRule:
    def __init__(self, max_order_usd: float = 20.0, max_equity_pct: float = 0.05) -> None:
        self._max_order_usd = max_order_usd
        self._max_equity_pct = max_equity_pct

    @property
    def name(self) -> str:
        return "max_order_size"

    def check(self, ctx: RiskContext) -> RiskCheckResult:
        limit = min(self._max_order_usd, ctx.balance_usd * self._max_equity_pct)
        if ctx.amount_usd > limit:
            return RiskCheckResult(
                approved=False,
                reason=f"Order ${ctx.amount_usd:.2f} exceeds limit ${limit:.2f}",
            )
        return RiskCheckResult(approved=True)


class LiquidityRule:
    def __init__(self, min_volume_24h: float = 50_000.0, min_liquidity_usd: float = 5_000.0) -> None:
        self._min_volume_24h = min_volume_24h
        self._min_liquidity_usd = min_liquidity_usd

    @property
    def name(self) -> str:
        return "liquidity_guard"

    def check(self, ctx: RiskContext) -> RiskCheckResult:
        volume = ctx.market.get("volume_24h", 0.0)
        liquidity = ctx.market.get("liquidity_1pct", 0.0)
        if volume < self._min_volume_24h or liquidity < self._min_liquidity_usd:
            return RiskCheckResult(
                approved=False,
                reason="Insufficient volume/liquidity for safe execution",
            )
        return RiskCheckResult(approved=True)


class RiskManager:
    def __init__(self) -> None:
        self.rules: List[RiskRule] = [MaxOrderSizeRule(), LiquidityRule()]

    def add_rule(self, rule: RiskRule) -> None:
        self.rules.append(rule)

    def check_risk(self, amount_usd: float, balance_usd: float, symbol: str, side: str, market: Dict[str, float]) -> RiskCheckResult:
        ctx = RiskContext(
            symbol=symbol,
            side=side,
            amount_usd=amount_usd,
            balance_usd=balance_usd,
            market=market,
        )
        for rule in self.rules:
            result = rule.check(ctx)
            if not result.approved:
                return result
        return RiskCheckResult(approved=True)


__all__ = ["RiskManager", "MaxOrderSizeRule", "LiquidityRule"]
