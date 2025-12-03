import unittest
from unittest.mock import AsyncMock

from src.client.paper_exchange import PaperExchange
from src.config import AppSettings
from src.tools.market import MarketReader


class MarketReaderTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.exchange = PaperExchange()
        self.reader = MarketReader(self.exchange)

    async def test_market_snapshot(self) -> None:
        snapshot = await self.reader.get_market_snapshot()
        self.assertIn("tickers", snapshot)
        self.assertIn("portfolio", snapshot)
        self.assertEqual(snapshot["exchange"], self.exchange.id)

    async def test_market_snapshot_respects_watchlist(self) -> None:
        settings = AppSettings(watchlist_symbols=["DOGE/USDT", "SHIB/USDT"])
        reader = MarketReader(self.exchange, settings=settings)
        reader.exchange.fetch_tickers = AsyncMock(
            return_value={
                "DOGE/USDT": {"last": 0.1, "baseVolume": 1_000_000, "quoteVolume": 100_000, "percentage": 5.0},
                "BTC/USDT": {"last": 60_000, "baseVolume": 10, "quoteVolume": 600_000, "percentage": 1.0},
            }
        )
        snapshot = await reader.get_market_snapshot()
        self.assertSetEqual(set(snapshot["tickers"].keys()), {"DOGE/USDT"})

    async def test_orderbook(self) -> None:
        ob = await self.reader.get_orderbook("BTC/USDT", depth=5)
        self.assertEqual(len(ob["bids"]), 5)
        self.assertEqual(len(ob["asks"]), 5)
        self.assertIn("spread_pct", ob)


if __name__ == "__main__":
    unittest.main()
