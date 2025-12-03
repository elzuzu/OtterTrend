from __future__ import annotations

from typing import Dict, Type

from src.interfaces.base_tool import BaseTool


_TOOL_REGISTRY: Dict[str, Type[BaseTool]] = {}


def register_tool(cls: Type[BaseTool]) -> Type[BaseTool]:
    _TOOL_REGISTRY[cls().definition.name] = cls
    return cls


def get_registered_tools() -> Dict[str, BaseTool]:
    return {name: cls() for name, cls in _TOOL_REGISTRY.items()}


__all__ = ["register_tool", "get_registered_tools"]
