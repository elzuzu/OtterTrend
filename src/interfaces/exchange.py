from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseExchange(ABC):
    id: str

    @abstractmethod
    async def connect(self) -> None:
        ...

    @abstractmethod
    async def fetch_markets(self) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        ...

    @abstractmethod
    async def fetch_tickers(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    async def fetch_balance(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    async def create_order(
        self, symbol: str, side: str, amount: float, order_type: str = "market", price: float | None = None
    ) -> Dict[str, Any]:
        ...

    @abstractmethod
    async def close_position(self, symbol: str) -> Dict[str, Any] | None:
        ...

    @abstractmethod
    async def close(self) -> None:
        ...


__all__ = ["BaseExchange"]
