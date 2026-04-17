"""Tool registry for agent_computer.

Tools are functions the LLM can call. Schema uses the OpenAI function-calling
format, which OpenRouter supports across all tool-capable models.
"""

from __future__ import annotations
import inspect
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

logger = logging.getLogger("agent_computer.tools")


@dataclass
class ToolParam:
    name: str
    type: str  # "string", "integer", "boolean", "number"
    description: str
    required: bool = True
    enum: list[str] | None = None


@dataclass
class Tool:
    name: str
    description: str
    params: list[ToolParam]
    handler: Callable[..., Awaitable[str]]
    require_approval: bool = False

    def to_openai_schema(self) -> dict:
        """Convert to OpenAI function-calling tool schema (used by OpenRouter)."""
        properties = {}
        required = []
        for p in self.params:
            prop: dict[str, Any] = {"type": p.type, "description": p.description}
            if p.enum:
                prop["enum"] = p.enum
            properties[p.name] = prop
            if p.required:
                required.append(p.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


class ToolRegistry:
    """Central registry for all tools available to the agent."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._accepts_context: dict[str, bool] = {}  # cached signature checks

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        # Cache whether this handler accepts _context
        try:
            sig = inspect.signature(tool.handler)
            self._accepts_context[tool.name] = "_context" in sig.parameters
        except (ValueError, TypeError):
            self._accepts_context[tool.name] = False
        logger.info(f"Registered tool: {tool.name}")

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def get_openai_tools(self, allowed: list[str] | None = None) -> list[dict]:
        """Get tool schemas for the OpenAI-compatible API, filtered by allow list."""
        tools = self._tools.values()
        if allowed is not None:
            tools = [t for t in tools if t.name in allowed]
        return [t.to_openai_schema() for t in tools]

    def set_approval_requirements(self, require_approval: list[str]) -> None:
        for name in require_approval:
            if name in self._tools:
                self._tools[name].require_approval = True

    async def execute(self, name: str, params: dict[str, Any],
                      context: dict[str, Any] | None = None) -> str:
        """Execute a tool by name. Returns result as string.

        If the tool handler accepts a ``_context`` keyword argument and
        *context* is provided, it will be forwarded automatically.
        """
        tool = self._tools.get(name)
        if not tool:
            return json.dumps({"error": f"Unknown tool: {name}"})

        try:
            if context is not None and self._accepts_context.get(name, False):
                result = await tool.handler(**params, _context=context)
            else:
                result = await tool.handler(**params)
            return result if isinstance(result, str) else json.dumps(result)
        except Exception as e:
            logger.error(f"Tool {name} failed: {e}")
            return json.dumps({"error": str(e)})
