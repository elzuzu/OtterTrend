from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from src.config import AppSettings, get_settings
from src.interfaces.base_tool import BaseTool, ToolDefinition
from src.interfaces.exchange import BaseExchange
from src.tools.registry import register_tool


class MarketReader:
    def __init__(self, exchange: BaseExchange, settings: AppSettings | None = None) -> None:
        self.exchange = exchange
        self.settings = settings or get_settings()

    async def get_market_price(self, symbol: str) -> float:
        ticker = await self.exchange.fetch_ticker(symbol)
        return float(ticker.get("last") or 0.0)

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        ticker = await self.exchange.fetch_ticker(symbol)
        return {
            "symbol": symbol,
            "last": float(ticker.get("last", 0.0)),
            "bid": float(ticker.get("bid", 0.0)),
            "ask": float(ticker.get("ask", 0.0)),
            "volume_24h": float(ticker.get("baseVolume", 0.0)),
            "quote_volume_24h": float(ticker.get("quoteVolume", 0.0)),
            "change_24h_pct": float(ticker.get("percentage", 0.0)),
            "high_24h": float(ticker.get("high", 0.0)),
            "low_24h": float(ticker.get("low", 0.0)),
        }

    async def get_tickers(self, symbols: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        tickers = await self.exchange.fetch_tickers()
        result: Dict[str, Dict[str, Any]] = {}
        for sym, data in tickers.items():
            if symbols and sym not in symbols:
                continue
            result[sym] = {
                "last": float(data.get("last", 0.0)),
                "volume_24h": float(data.get("baseVolume", 0.0)),
                "quote_volume_24h": float(data.get("quoteVolume", 0.0)),
                "change_24h_pct": float(data.get("percentage", 0.0)),
            }
        return result

    async def get_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> List[Dict[str, Any]]:
        ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        return [
            {
                "timestamp": candle[0],
                "open": candle[1],
                "high": candle[2],
                "low": candle[3],
                "close": candle[4],
                "volume": candle[5],
            }
            for candle in ohlcv
        ]

    async def get_orderbook(self, symbol: str, depth: int = 20) -> Dict[str, Any]:
        ob = await self.exchange.fetch_order_book(symbol, depth)
        bids = ob.get("bids", [])[:depth]
        asks = ob.get("asks", [])[:depth]
        best_bid = float(bids[0][0]) if bids else 0.0
        best_ask = float(asks[0][0]) if asks else 0.0
        mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0.0
        spread_pct = ((best_ask - best_bid) / mid_price * 100) if mid_price else 0.0
        return {
            "symbol": symbol,
            "timestamp": ob.get("timestamp"),
            "bids": bids,
            "asks": asks,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": mid_price,
            "spread_pct": round(spread_pct, 4),
            "low_liquidity_warning": spread_pct > 1.0,
        }

    async def get_balance(self) -> Dict[str, Any]:
        balance = await self.exchange.fetch_balance()
        return {
            "total": {k: float(v) for k, v in balance.get("total", {}).items() if v > 0},
            "free": {k: float(v) for k, v in balance.get("free", {}).items() if v > 0},
            "used": {k: float(v) for k, v in balance.get("used", {}).items() if v > 0},
        }

    async def get_portfolio_state(self) -> Dict[str, Any]:
        cfg = self.settings
        balance = await self.get_balance()
        base_bal = balance["free"].get("USDT", 0.0)
        positions: List[Dict[str, Any]] = []
        total_equity = base_bal
        for asset, qty in balance["total"].items():
            if asset.upper() == "USDT" or qty <= 0:
                continue
            symbol = f"{asset}/USDT"
            try:
                price = await self.get_market_price(symbol)
            except Exception:
                continue
            value_usdt = qty * price
            total_equity += value_usdt
            positions.append({
                "asset": asset,
                "symbol": symbol,
                "amount": qty,
                "price": price,
                "value_usdt": value_usdt,
            })
        return {
            "mode": "paper" if cfg.paper_trading else "live",
            "exchange": self.exchange.id,
            "balance_usdt": base_bal,
            "total_equity_usdt": total_equity,
            "positions": positions,
            "positions_count": len(positions),
        }

    async def _resolve_symbols(self, symbols: Optional[List[str]] = None) -> Optional[List[str]]:
        if symbols:
            return symbols
        if self.settings.watchlist_symbols:
            return self.settings.watchlist_symbols
        markets = await self.exchange.fetch_markets()
        inferred = [m.get("symbol") for m in markets if m.get("quote") == "USDT" and m.get("symbol")]
        return inferred[:10] if inferred else None

    async def get_market_snapshot(self, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        selected = await self._resolve_symbols(symbols)
        tickers = await self.get_tickers(selected)
        portfolio = await self.get_portfolio_state()
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "exchange": self.exchange.id,
            "mode": "paper" if self.settings.paper_trading else "live",
            "fees_info": "0% maker / 0.01% taker - profite pour scalper",
            "tickers": tickers,
            "portfolio": portfolio,
        }

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
            "volume_24h": float(ticker.get("baseVolume", 0.0)),
            "liquidity_1pct": float(ticker.get("quoteVolume", 0.0)) * 0.01,
        }

    def risk_constraints(self) -> Dict[str, Any]:
        cfg = self.settings
        return {
            "max_order_usd": cfg.risk_max_trade_usd,
            "max_equity_pct": cfg.risk_max_trade_pct_equity,
            "max_daily_loss_usd": cfg.risk_max_daily_loss_usd,
            "max_positions": cfg.risk_max_open_positions,
            "paper_mode": cfg.paper_trading,
            "exchange_fees": "0% maker / 0.01% taker",
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
            parameters={"type": "object", "properties": {"symbols": {"type": "array", "items": {"type": "string"}}}},
            category="observer",
        )

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        if not self.reader:
            raise ValueError("MarketReader not provided")
        return await self.reader.get_market_snapshot(kwargs.get("symbols"))


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


@register_tool
class RiskConstraintsTool(BaseTool):
    def __init__(self, reader: MarketReader | None = None) -> None:
        self.reader = reader

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="risk_constraints",
            description="Retourne les limites de risque hard-codées",
            parameters={"type": "object", "properties": {}},
            category="agir",
        )

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        if not self.reader:
            raise ValueError("MarketReader not provided")
        return self.reader.risk_constraints()


__all__ = [
    "MarketReader",
    "MarketSnapshotTool",
    "NewListingsTool",
    "TopGainersTool",
    "LiquidityTool",
    "RiskConstraintsTool",
]
