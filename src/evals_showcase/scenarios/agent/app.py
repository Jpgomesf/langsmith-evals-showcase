"""Scenario 4 — a small ReAct agent (the app-under-test).

A LangGraph prebuilt ReAct agent with two deterministic tools. The async target
returns the final answer and the ordered list of tools the agent called, so the
trajectory evaluators can score the agent's process.
"""

from __future__ import annotations

import ast
import operator
from functools import lru_cache
from typing import Any

from langchain_core.tools import tool

from ...models import get_chat_model, message_text

_OPS: dict[type[ast.AST], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


_MAX_EXPONENT = 100  # cap to prevent huge-integer DoS like 10**10**10


def _safe_eval(node: ast.AST) -> float:
    """Evaluate a parsed arithmetic expression, allowing numbers and +-*/%** only."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression, returning a tidy string result."""
    try:
        result = _safe_eval(ast.parse(expression, mode="eval").body)
    except ZeroDivisionError:
        return "undefined (division by zero)"
    return str(int(result)) if result == int(result) else str(result)


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression such as '47 * 13' or '256 / 8'."""
    return calculate(expression)


@tool
def word_count(text: str) -> str:
    """Count the number of whitespace-separated words in the given text."""
    return str(len(text.split()))


TOOLS = [calculator, word_count]


@lru_cache
def _agent() -> Any:
    from langgraph.prebuilt import create_react_agent

    return create_react_agent(get_chat_model(), TOOLS)


def _trajectory(messages: list[Any]) -> list[str]:
    """Ordered names of every tool the agent called."""
    return [call["name"] for msg in messages for call in (getattr(msg, "tool_calls", None) or [])]


async def arun_agent(inputs: dict[str, str]) -> dict[str, Any]:
    """Async LangSmith target: ``{"question": ...}`` -> ``{answer, trajectory}``."""
    result = await _agent().ainvoke({"messages": [("user", inputs["question"])]})
    messages = result["messages"]
    return {"answer": message_text(messages[-1]), "trajectory": _trajectory(messages)}
