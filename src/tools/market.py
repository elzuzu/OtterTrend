from __future__ import annotations

from typing import Any, Dict, List

from src.interfaces.base_tool import BaseTool, ToolDefinition
from src.interfaces.exchange import BaseExchange
from src.tools.registry import register_tool


class MarketReader:
    def __init__(self, exchange: BaseExchange) -> None:
        self.exchange = exchange

    async def get_market_snapshot(self) -> Dict[str, Any]:
        tickers = await self.exchange.fetch_tickers()
        balance = await self.exchange.fetch_balance()
        return {"tickers": tickers, "balance": balance}

    async def get_new_listings(self, known_markets: List[str]) -> List[Dict[str, Any]]:
        markets = await self.exchange.fetch_markets()
        fresh = [m for m in markets if m.get("symbol") not in known_markets]
        return [
            {
                "symbol": m.get("symbol"),
                "base": m.get("base"),
                "quote": m.get("quote"),
                "listing_time_estimated": m.get("info", {}).get("launchTime"),
            }
            for m in fresh
        ]

    async def get_top_gainers_24h(self, limit: int = 5) -> Dict[str, Any]:
        tickers = await self.exchange.fetch_tickers()
        sorted_pairs = sorted(tickers.values(), key=lambda t: t.get("percentage", 0.0), reverse=True)
        top = []
        for ticker in sorted_pairs[:limit]:
            top.append(
                {
                    "symbol": ticker.get("symbol"),
                    "change_24h": ticker.get("percentage", 0.0),
                    "volume_usdt": ticker.get("quoteVolume") or ticker.get("baseVolume", 0.0),
                }
            )
        return {"exchange": self.exchange.id, "top_gainers": top}

    async def estimate_liquidity(self, symbol: str) -> Dict[str, Any]:
        ticker = await self.exchange.fetch_ticker(symbol)
        return {
            "symbol": symbol,
            "volume_24h": ticker.get("baseVolume", 0.0),
            "liquidity_1pct": ticker.get("quoteVolume", 0.0) * 0.01,
        }


@register_tool
class MarketSnapshotTool(BaseTool):
    def __init__(self, reader: MarketReader | None = None) -> None:
        self.reader = reader

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="market_snapshot",
            description="Retourne les tickers et le solde du portefeuille",
            parameters={"type": "object", "properties": {}},
            category="observer",
        )

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        if not self.reader:
            raise ValueError("MarketReader not provided")
        return await self.reader.get_market_snapshot()


@register_tool
class NewListingsTool(BaseTool):
    def __init__(self, reader: MarketReader | None = None) -> None:
        self.reader = reader

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_new_listings",
            description="Détecte les nouveaux listings MEXC",
            parameters={"type": "object", "properties": {"known_markets": {"type": "array", "items": {"type": "string"}}}},
            category="observer",
        )

    async def execute(self, **kwargs: Any) -> List[Dict[str, Any]]:
        if not self.reader:
            raise ValueError("MarketReader not provided")
        known_markets = kwargs.get("known_markets", [])
        return await self.reader.get_new_listings(known_markets)


@register_tool
class TopGainersTool(BaseTool):
    def __init__(self, reader: MarketReader | None = None) -> None:
        self.reader = reader

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_top_gainers_24h",
            description="Retourne les top gainers 24h",
            parameters={"type": "object", "properties": {"limit": {"type": "integer", "default": 5}}},
            category="observer",
        )

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        if not self.reader:
            raise ValueError("MarketReader not provided")
        limit = int(kwargs.get("limit", 5))
        return await self.reader.get_top_gainers_24h(limit)


@register_tool
class LiquidityTool(BaseTool):
    def __init__(self, reader: MarketReader | None = None) -> None:
        self.reader = reader

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="estimate_liquidity",
            description="Estime la liquidité d'une paire",
            parameters={"type": "object", "properties": {"symbol": {"type": "string"}}},
            category="observer",
        )

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        if not self.reader:
            raise ValueError("MarketReader not provided")
        symbol = kwargs.get("symbol")
        return await self.reader.estimate_liquidity(symbol)


__all__ = [
    "MarketReader",
    "MarketSnapshotTool",
    "NewListingsTool",
    "TopGainersTool",
    "LiquidityTool",
]
