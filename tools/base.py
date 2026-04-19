"""Base Tool class. Every action DAIMON can take in the real world is a Tool."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from permissions.levels import PermissionLevel


class BaseTool(ABC):
    """Subclass this for every tool. Each instance defines its own schema and
    runs its own execute() when Claude picks it."""

    name: str = ""
    description: str = ""
    permission_level: PermissionLevel = PermissionLevel.AUTO
    cost_per_use: float = 0.0  # estimated USD per execute() call
    is_high_stakes: bool = False  # if True, dispatch runs self_critique first

    @abstractmethod
    def input_schema(self) -> dict[str, Any]:
        """JSON schema for the tool's input — fed to Claude tool-use."""
        raise NotImplementedError

    @abstractmethod
    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Run the tool. Always return a dict with at least
        {"ok": bool, "summary": str}. May include other fields."""
        raise NotImplementedError

    def anthropic_tool_def(self) -> dict[str, Any]:
        """Package as an Anthropic tool-use definition."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema(),
        }

    def __repr__(self) -> str:
        return f"<Tool {self.name} ({self.permission_level.value})>"


class ToolRegistry:
    """Holds instantiated tools the agent can offer to the brain this cycle."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if not tool.name:
            raise ValueError(f"Tool missing name: {tool!r}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def all(self) -> list[BaseTool]:
        return list(self._tools.values())

    def anthropic_defs(self) -> list[dict[str, Any]]:
        return [t.anthropic_tool_def() for t in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)
