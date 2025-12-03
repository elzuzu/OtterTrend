from __future__ import annotations

from typing import Any, Dict, List

from src.interfaces.exchange import BaseExchange


class PaperExchange(BaseExchange):
    def __init__(self, exchange_id: str = "paper") -> None:
        self.id = exchange_id
        self.balance = {"free": {"USDT": 1000.0}}
        self.positions: Dict[str, Dict[str, Any]] = {}

    async def connect(self) -> None:
        return None

    async def fetch_markets(self) -> List[Dict[str, Any]]:
        return []

    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        return {"symbol": symbol, "last": 1.0, "percentage": 0.0, "baseVolume": 0.0}

    async def fetch_tickers(self) -> Dict[str, Any]:
        return {}

    async def fetch_order_book(self, symbol: str, limit: int | None = None) -> Dict[str, Any]:
        depth = limit or 20
        bids = [[0.99, 1.0]] * depth
        asks = [[1.01, 1.0]] * depth
        return {"symbol": symbol, "bids": bids, "asks": asks, "timestamp": None}

    async def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> List[List[float]]:
        return [[0, 1.0, 1.0, 1.0, 1.0, 0.0] for _ in range(limit)]

    async def fetch_balance(self) -> Dict[str, Any]:
        return self.balance

    async def create_order(
        self, symbol: str, side: str, amount: float, order_type: str = "market", price: float | None = None
    ) -> Dict[str, Any]:
        notional = (price or 1.0) * amount
        self.balance["free"]["USDT"] -= notional
        self.positions[symbol] = {"side": side, "amount": amount, "price": price or 1.0}
        return {
            "id": f"paper-{symbol}",
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "price": price,
            "type": order_type,
            "status": "closed",
        }

    async def close_position(self, symbol: str) -> Dict[str, Any] | None:
        position = self.positions.pop(symbol, None)
        if not position:
            return None
        return {
            "symbol": symbol,
            "closed": True,
            "side": position["side"],
            "amount": position["amount"],
        }

    async def close(self) -> None:
        return None


__all__ = ["PaperExchange"]
