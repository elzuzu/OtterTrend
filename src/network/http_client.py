from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional

import httpx


class ResilientHTTPClient:
    def __init__(self, base_url: str | None = None, timeout: float = 10.0, max_retries: int = 3) -> None:
        self.client = httpx.AsyncClient(base_url=base_url, timeout=timeout, http2=True)
        self.max_retries = max_retries

    async def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        for attempt in range(self.max_retries):
            try:
                response = await self.client.request(method, url, **kwargs)
                response.raise_for_status()
                return response
            except Exception:
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(0.5 * (attempt + 1))
        raise RuntimeError("Request failed")

    async def close(self) -> None:
        await self.client.aclose()


__all__ = ["ResilientHTTPClient"]
