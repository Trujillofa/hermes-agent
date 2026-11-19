"""Opt-in, turn-local reasoning experiment; no messages leave this process."""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

# Binding to the particular agent prevents nested auxiliary/child work from inheriting it.
_turn: ContextVar[tuple[Any, str | None, bool, Any] | None] = ContextVar("adaptive_reasoning_turn", default=None)


def classify(message, history):
    """Only exact social utterances qualify; malformed history fails closed."""
    if not isinstance(message, str) or (history is not None and (
        not isinstance(history, list) or history
    )):
        return None
    # Even a completed-looking exchange can contain an unresolved question or
    # decision. Any history (including malformed history) keeps the baseline.
    text = message.strip().lower().rstrip("!. ")
    if text in {"hi", "hello", "hey", "thanks", "thank you"}:
        return "low"
    return None


def effective_policy_config():
    """The current session's policy, excluding a copied delegation parent."""
    from agent.delegation_context import has_inherited_policy_binding
    if has_inherited_policy_binding():
        return None
    from gateway.session_policy import active_policy_config
    return active_policy_config()


def enabled_for_turn(override):
    """An explicit per-call boolean wins; otherwise use the effective turn policy.

    Only a literal boolean enables the feature. Bad/missing config fails closed.
    """
    if override is not None:
        return override is True
    try:
        config = effective_policy_config()
        if config is None:
            from hermes_cli.config import load_config
            config = load_config()
        return config.get("agent", {}).get("adaptive_reasoning") is True
    except Exception:
        return False


@contextmanager
def adaptive_turn(agent, message, history, enabled, *, suppress_decision=False):
    active = enabled_for_turn(enabled)
    # A social-looking message does not establish a trivial turn: system or
    # developer instructions (and tool descriptions) can direct substantial work
    # on "hi". Nor can we reliably infer that a prior exchange is complete.
    # Until there is an explicit, trustworthy contract for the entire request,
    # retain the baseline on *every* turn, including the first one.
    decision = None
    token = _turn.set((agent, decision, active, message))
    try:
        yield
    finally:
        _turn.reset(token)


def turn_opt_in(agent):
    """Return the bound per-call decision, or None outside this agent's turn."""
    turn = _turn.get()
    return turn[2] if turn and turn[0] is agent else None


def active_for_agent(agent):
    """Whether this agent's current turn opted in, including unclassified turns."""
    turn = _turn.get()
    return bool(turn and turn[0] is agent and turn[2])


def invalidate_decision(agent):
    """Permanently stop demoting this turn after newly introduced work."""
    turn = _turn.get()
    if turn and turn[0] is agent and turn[1] is not None:
        _turn.set((turn[0], None, turn[2], turn[3]))


def reasoning_for_turn(agent, baseline=None, messages=None):
    """Return a suggestion only for a profile baseline, never an explicit decision."""
    turn = _turn.get()
    if not turn or turn[0] is not agent or turn[1] is None:
        return None
    if messages is not None:
        # Verify the actual request, not only the turn admission: scripted
        # preludes, tool rounds, steering and compaction can introduce work.
        if not isinstance(messages, list) or any(not isinstance(m, dict) for m in messages):
            invalidate_decision(agent)
            return None
        conversation = [m for m in messages if m.get("role") not in ("system", "developer")]
        if (len(conversation) != 1 or conversation[0].get("role") != "user"
                or conversation[0].get("content") != turn[3]):
            invalidate_decision(agent)
            return None
    from hermes_constants import ProfileReasoningConfig
    cfg = getattr(agent, "reasoning_config", None)
    if cfg is not None and not (isinstance(cfg, ProfileReasoningConfig) and cfg.is_baseline()):
        return None
    if isinstance(cfg, dict) and (cfg.get("enabled") is False or cfg.get("effort") == "none"):
        return None
    if getattr(agent, "request_overrides", None):
        return None
    # Never promote minimal/none or an unknown baseline to low. Codex's
    # transport has a known medium default when no config is set.
    effective = cfg if cfg is not None else baseline
    if effective is None and getattr(agent, "api_mode", None) == "codex_responses":
        effective = {"enabled": True, "effort": "medium"}
    if (not isinstance(effective, dict) or effective.get("enabled") is False
            or effective.get("effort") not in ("low", "medium", "high", "xhigh", "max")):
        return None
    return {"enabled": True, "effort": turn[1]}
