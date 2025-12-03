import unittest

from src.client.paper_exchange import PaperExchange
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

    async def test_orderbook(self) -> None:
        ob = await self.reader.get_orderbook("BTC/USDT", depth=5)
        self.assertEqual(len(ob["bids"]), 5)
        self.assertEqual(len(ob["asks"]), 5)
        self.assertIn("spread_pct", ob)


if __name__ == "__main__":
    unittest.main()
