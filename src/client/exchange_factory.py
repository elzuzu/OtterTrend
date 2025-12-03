from __future__ import annotations

import importlib
from typing import Optional

from src.client.mexc_exchange import MEXCExchange
from src.client.paper_exchange import PaperExchange
from src.config import AppSettings
from src.interfaces.exchange import BaseExchange


def get_exchange(settings: AppSettings, http_client: Optional[object] = None) -> BaseExchange:
    if settings.paper_trading:
        return PaperExchange(settings.exchange_id)

    exchange_id = settings.exchange_id.lower()
    if exchange_id == "mexc":
        return MEXCExchange(
            api_key=settings.mexc_api_key,
            api_secret=settings.mexc_api_secret,
            testnet=settings.exchange_testnet,
            http_client=http_client,
        )

    try:
        module = importlib.import_module(f"src.client.{exchange_id}_exchange")
        exchange_cls = getattr(module, f"{exchange_id.upper()}Exchange")
        return exchange_cls(api_key=settings.mexc_api_key, api_secret=settings.mexc_api_secret, http_client=http_client)
    except Exception as exc:  # pragma: no cover - fallback
        raise ValueError(f"Unsupported exchange: {exchange_id}") from exc


__all__ = ["get_exchange"]
