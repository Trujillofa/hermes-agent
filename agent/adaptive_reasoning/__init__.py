"""Opt-in, turn-local reasoning experiment; no messages leave this process."""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

# Binding to the particular agent prevents nested auxiliary/child work from inheriting it.
_turn: ContextVar[tuple[Any, str | None, bool] | None] = ContextVar("adaptive_reasoning_turn", default=None)


def classify(message, history):
    """Only exact social utterances qualify; malformed history fails closed."""
    if not isinstance(message, str) or (history is not None and (
        not isinstance(history, list) or any(not isinstance(item, dict) for item in history)
    )):
        return None
    # A pending tool result or unfinished exchange may carry obligations that a
    # terse user message does not reveal; do not demote those turns.
    if history and (history[-1].get("role") != "assistant" or history[-1].get("tool_calls")
                    or history[-1].get("function_call") or not isinstance(history[-1].get("content"), str)):
        return None
    text = message.strip().lower().rstrip("!. ")
    if text in {"hi", "hello", "hey", "thanks", "thank you"}:
        return "low"
    return None


def enabled_for_turn(override):
    """An explicit per-call boolean wins; otherwise read the active profile each turn.

    Only a literal boolean enables the feature. Bad/missing config fails closed.
    """
    if override is not None:
        return override is True
    try:
        from hermes_cli.config import load_config
        config = load_config()
        return config.get("agent", {}).get("adaptive_reasoning") is True
    except Exception:
        return False


@contextmanager
def adaptive_turn(agent, message, history, enabled):
    active = enabled_for_turn(enabled)
    decision = classify(message, history) if active else None
    token = _turn.set((agent, decision, active))
    try:
        yield
    finally:
        _turn.reset(token)


def active_for_agent(agent):
    """Whether this agent's current turn opted in, including unclassified turns."""
    turn = _turn.get()
    return bool(turn and turn[0] is agent and turn[2])


def reasoning_for_turn(agent):
    """Return a suggestion only for a profile baseline, never an explicit decision."""
    turn = _turn.get()
    if not turn or turn[0] is not agent or turn[1] is None:
        return None
    from hermes_constants import ProfileReasoningConfig
    cfg = getattr(agent, "reasoning_config", None)
    if cfg is not None and not (isinstance(cfg, ProfileReasoningConfig) and cfg.is_baseline()):
        return None
    if isinstance(cfg, dict) and (cfg.get("enabled") is False or cfg.get("effort") == "none"):
        return None
    if getattr(agent, "request_overrides", None):
        return None
    return {"enabled": True, "effort": turn[1]}
