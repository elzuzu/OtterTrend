from __future__ import annotations

from typing import Any, Dict, List

from pytrends.request import TrendReq

from src.interfaces.base_tool import BaseTool, ToolDefinition
from src.tools.registry import register_tool


class TrendsClient:
    def __init__(self) -> None:
        self.client = TrendReq(hl="en-US", tz=360)

    async def interest_over_time(self, keywords: List[str]) -> Dict[str, Any]:
        self.client.build_payload(keywords, cat=0, timeframe="now 7-d", geo="", gprop="")
        data = self.client.interest_over_time()
        return data.to_dict() if hasattr(data, "to_dict") else {}


@register_tool
class TrendsTool(BaseTool):
    def __init__(self, client: TrendsClient | None = None) -> None:
        self.client = client or TrendsClient()

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="google_trends",
            description="Récupère l'intérêt de recherche sur 7 jours",
            parameters={"type": "object", "properties": {"keywords": {"type": "array", "items": {"type": "string"}}}},
            category="observer",
        )

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        keywords = kwargs.get("keywords", [])
        return await self.client.interest_over_time(keywords)


__all__ = ["TrendsClient", "TrendsTool"]
