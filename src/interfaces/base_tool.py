from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: Dict[str, Any]
    category: str


class BaseTool:
    @property
    def definition(self) -> ToolDefinition:  # pragma: no cover - abstract
        raise NotImplementedError

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError


__all__ = ["BaseTool", "ToolDefinition"]
