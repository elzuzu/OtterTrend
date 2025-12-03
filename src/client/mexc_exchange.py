from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import ccxt.async_support as ccxt

from src.interfaces.exchange import BaseExchange


class MEXCExchange(BaseExchange):
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False, http_client: Optional[object] = None) -> None:
        self.id = "mexc"
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.http_client = http_client
        self._client: Optional[ccxt.Exchange] = None

    @property
    def client(self) -> ccxt.Exchange:
        if self._client is None:
            params = {
                "apiKey": self.api_key,
                "secret": self.api_secret,
                "enableRateLimit": True,
                "options": {"defaultType": "spot", "recvWindow": 60_000},
            }
            self._client = ccxt.mexc(params)
            if self.testnet:
                self._client.set_sandbox_mode(True)
        return self._client

    async def connect(self) -> None:
        await self.client.load_markets()

    async def fetch_markets(self) -> List[Dict[str, Any]]:
        return list(self.client.markets.values())

    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        await asyncio.sleep(self.client.rateLimit / 1000)
        return await self.client.fetch_ticker(symbol)

    async def fetch_tickers(self) -> Dict[str, Any]:
        await asyncio.sleep(self.client.rateLimit / 1000)
        return await self.client.fetch_tickers()

    async def fetch_order_book(self, symbol: str, limit: int | None = None) -> Dict[str, Any]:
        await asyncio.sleep(self.client.rateLimit / 1000)
        return await self.client.fetch_order_book(symbol, limit)

    async def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> List[List[float]]:
        await asyncio.sleep(self.client.rateLimit / 1000)
        return await self.client.fetch_ohlcv(symbol, timeframe, limit=limit)

    async def fetch_balance(self) -> Dict[str, Any]:
        await asyncio.sleep(self.client.rateLimit / 1000)
        return await self.client.fetch_balance()

    async def create_order(
        self, symbol: str, side: str, amount: float, order_type: str = "market", price: float | None = None
    ) -> Dict[str, Any]:
        await asyncio.sleep(self.client.rateLimit / 1000)
        return await self.client.create_order(symbol, order_type, side, amount, price)

    async def close_position(self, symbol: str) -> Dict[str, Any] | None:
        try:
            position = await self.client.fetch_position(symbol)
        except Exception:
            position = None
        if not position:
            return None
        side = "sell" if position.get("side") == "long" else "buy"
        amount = position.get("contracts") or position.get("amount") or 0.0
        if amount:
            return await self.create_order(symbol=symbol, side=side, amount=amount, order_type="market")
        return None

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None


__all__ = ["MEXCExchange"]
