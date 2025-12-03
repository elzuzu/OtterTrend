from __future__ import annotations

import json
from typing import Any, Dict, List

from src.bot.risk import RiskManager
from src.client.groq_adapter import GroqAdapter
from src.tools.registry import get_registered_tools


class BotBrain:
    def __init__(self, llm: GroqAdapter, risk: RiskManager) -> None:
        self.llm = llm
        self.risk = risk
        self.tools = get_registered_tools()

    async def think(self, system_prompt: str, observations: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt = [
            {"role": "user", "content": f"Observations: {json.dumps(observations)}"},
            {
                "role": "user",
                "content": "Propose une liste d'actions JSON avec type, symbol, side, size_pct_equity",
            },
        ]
        raw = await self.llm.chat(system_prompt, prompt)
        try:
            actions = json.loads(raw)
            if isinstance(actions, dict):
                actions = actions.get("actions", [])
        except json.JSONDecodeError:
            actions = []
        return actions

    async def act(self, exchange: Any, memory: Any, snapshot: Dict[str, Any], actions: List[Dict[str, Any]]) -> None:
        balance = snapshot.get("balance", {}).get("free", {}).get("USDT", 0.0)
        for action in actions:
            atype = action.get("type", "").upper()
            symbol = action.get("symbol")
            if not symbol:
                continue
            if atype == "OPEN":
                side = action.get("side", "buy").lower()
                size_pct = float(action.get("size_pct_equity", 0.01))
                amount_usd = balance * size_pct
                market_info = {"volume_24h": 0.0, "liquidity_1pct": 0.0}
                risk_result = self.risk.check_risk(amount_usd, balance, symbol, side, market_info)
                if not risk_result.approved:
                    memory.log_info(f"Ordre rejeté: {risk_result.reason}", {"action": action})
                    continue
                ticker = snapshot.get("tickers", {}).get(symbol) or {"last": 1.0}
                price = ticker.get("last", 1.0)
                amount = amount_usd / price
                order = await exchange.create_order(symbol=symbol, side=side, amount=amount, order_type="market")
                memory.log_trade_open(order, snapshot, action)
            elif atype == "CLOSE":
                order = await exchange.close_position(symbol)
                if order:
                    memory.log_trade_close(order, snapshot, action)


__all__ = ["BotBrain"]
