"""Opt-in, turn-local reasoning experiment; no messages leave this process."""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

# Binding to the particular agent prevents nested auxiliary/child work from inheriting it.
_turn: ContextVar[tuple[Any, str | None] | None] = ContextVar("adaptive_reasoning_turn", default=None)


def classify(message, history):
    """Only standalone, exact social utterances qualify; uncertainty keeps defaults."""
    if history or not isinstance(message, str):
        return None
    text = message.strip().lower().rstrip("!. ")
    if text in {"hi", "hello", "hey", "thanks", "thank you"}:
        return "low"
    return None


@contextmanager
def adaptive_turn(agent, message, history, enabled):
    decision = classify(message, history) if enabled is True else None
    token = _turn.set((agent, decision))
    try:
        yield
    finally:
        _turn.reset(token)


def reasoning_for_turn(agent):
    """Return a suggestion only absent explicit config/overrides on the current route."""
    turn = _turn.get()
    if not turn or turn[0] is not agent or turn[1] is None:
        return None
    if getattr(agent, "reasoning_config", None) is not None or getattr(agent, "request_overrides", None):
        return None
    return {"enabled": True, "effort": turn[1]}
