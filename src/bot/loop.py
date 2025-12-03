from __future__ import annotations

import asyncio
from typing import Any, Dict

from src.bot.brain import BotBrain
from src.bot.memory import BotMemory
from src.bot.risk import RiskManager
from src.bot.system_prompt import SYSTEM_PROMPT
from src.client.exchange_factory import get_exchange
from src.client.groq_adapter import GroqAdapter
from src.config import get_settings
from src.tools.market import (
    LiquidityTool,
    MarketReader,
    MarketSnapshotTool,
    NewListingsTool,
    RiskConstraintsTool,
    TopGainersTool,
)


class BotLoop:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.exchange = get_exchange(self.settings)
        self.risk = RiskManager()
        self.memory = BotMemory(self.settings.sqlite_db_path)
        self.llm = GroqAdapter(self.settings.groq_api_key)
        self.brain = BotBrain(self.llm, self.risk)
        self.reader = MarketReader(self.exchange)
        self.tools = [
            MarketSnapshotTool(self.reader),
            NewListingsTool(self.reader),
            TopGainersTool(self.reader),
            LiquidityTool(self.reader),
            RiskConstraintsTool(self.reader),
        ]

    async def observe(self) -> Dict[str, Any]:
        snapshot = await self.reader.get_market_snapshot()
        new_listings = await self.reader.get_new_listings([])
        top_gainers = await self.reader.get_top_gainers_24h()
        snapshot.update({"new_listings": new_listings, "top_gainers": top_gainers})
        return snapshot

    async def run_forever(self, interval: float = 10.0) -> None:
        await self.exchange.connect()
        self.memory.log_info("Bot démarré")
        while True:
            snapshot = await self.observe()
            actions = await self.brain.think(SYSTEM_PROMPT, snapshot)
            await self.brain.act(self.exchange, self.memory, snapshot, actions)
            await asyncio.sleep(interval)

    async def shutdown(self) -> None:
        await self.exchange.close()
        self.memory.close()


async def main() -> None:
    loop = BotLoop()
    try:
        await loop.run_forever()
    except KeyboardInterrupt:
        await loop.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
