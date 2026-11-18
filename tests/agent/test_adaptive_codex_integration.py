"""Exercise the real request builder with an isolated copy of the profile's route shape."""
from unittest.mock import patch

import pytest
import run_agent  # bootstrap during collection, before the home I/O guard

from agent.adaptive_reasoning import adaptive_turn
from agent.chat_completion_helpers import build_api_kwargs, _reresolve_fallback_reasoning_config
from hermes_constants import resolve_reasoning_config, ProfileReasoningConfig

PROFILE = {
    "model": {"default": "gpt-6.1-terra", "provider": "openai-codex"},
    "agent": {"adaptive_reasoning": True, "reasoning_effort": "medium",
              "reasoning_overrides": {"openai-codex/gpt-6.1-terra": "high",
                                      "opencode-go/deepseek-v4.1-flash": "high"}},
}

@pytest.fixture
def codex(monkeypatch):
    from tests.agent.test_run_agent_codex_responses import _build_agent
    agent = _build_agent(monkeypatch)
    agent.model = "gpt-6.1-terra"
    agent.provider = "openai-codex"
    agent.api_mode = "codex_responses"
    agent.reasoning_config = resolve_reasoning_config(PROFILE, agent.model)
    agent.request_overrides = None
    return agent


def payload(agent, text, history=None, enabled=True):
    with adaptive_turn(agent, text, history, enabled):
        return build_api_kwargs(agent, [{"role": "user", "content": text}], [])


def test_configured_codex_builder_multiturn(codex):
    assert codex.reasoning_config == {"enabled": True, "effort": "high"}
    first = payload(codex, "hi")
    history = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "yes"}]
    second = payload(codex, "thanks", history)
    complex_turn = payload(codex, "what about that?", history)
    off = payload(codex, "hi", history, False)
    assert [p["reasoning"]["effort"] for p in (first, second, complex_turn, off)] == ["low", "high", "high", "high"]
    assert first["model"] == complex_turn["model"] == "gpt-6.1-terra"
    assert codex.reasoning_config["effort"] == "high"


@pytest.mark.parametrize("explicit", [{"enabled": True, "effort": "high"}, {"enabled": False}])
def test_explicit_equal_to_baseline_or_disabled_survives_fallback(codex, explicit):
    codex.reasoning_config = explicit.copy()
    with adaptive_turn(codex, "hi", [], True):
        before = build_api_kwargs(codex, [{"role": "user", "content": "hi"}], [])
        with patch("hermes_cli.config.load_config", return_value={}):
            _reresolve_fallback_reasoning_config(codex)
        after = build_api_kwargs(codex, [{"role": "user", "content": "hi"}], [])
    assert codex.reasoning_config == explicit
    assert before.get("reasoning") == after.get("reasoning")
    if explicit.get("enabled"):
        assert after["reasoning"]["effort"] == "high"
    else:
        assert after.get("reasoning", {}).get("effort") != "low"


def test_baseline_fallback_and_unknown_provenance(codex):
    with adaptive_turn(codex, "hi", [], True):
        with patch("hermes_cli.config.load_config", return_value=PROFILE):
            _reresolve_fallback_reasoning_config(codex)
        assert isinstance(codex.reasoning_config, ProfileReasoningConfig)
        assert build_api_kwargs(codex, [{"role": "user", "content": "hi"}], [])["reasoning"]["effort"] == "low"
    codex.reasoning_config = dict(codex.reasoning_config)
    assert payload(codex, "hi")["reasoning"]["effort"] == "high"


def test_request_override_suppresses_adaptation(codex):
    codex.request_overrides = {"reasoning": {"effort": "high"}}
    assert payload(codex, "hi")["reasoning"]["effort"] == "high"


@pytest.mark.parametrize("effort", ["minimal", "none", "unknown"])
def test_low_does_not_promote_lower_or_unknown_profile_baseline(codex, effort):
    codex.reasoning_config = ProfileReasoningConfig({"enabled": True, "effort": effort})
    from agent.adaptive_reasoning import reasoning_for_turn
    with adaptive_turn(codex, "hi", [], True):
        assert reasoning_for_turn(codex, codex.reasoning_config, [{"role": "user", "content": "hi"}]) is None


def test_request_builder_rechecks_work_after_initial_low(codex):
    first = [{"role": "user", "content": "hi"}]
    tool = first + [{"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "search", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "new task"}]
    with adaptive_turn(codex, "hi", [], True):
        assert build_api_kwargs(codex, first, [])["reasoning"]["effort"] == "low"
        assert build_api_kwargs(codex, tool, [])["reasoning"]["effort"] == "high"
        # Even if later compaction hides the added messages, never re-demote.
        assert build_api_kwargs(codex, first, [])["reasoning"]["effort"] == "high"


def test_real_conversation_facade_codex_multiturn(codex, monkeypatch):
    from tests.agent.test_run_agent_codex_responses import _codex_message_response
    calls = []
    codex._disable_streaming = True
    codex.compression_enabled = False
    codex.save_trajectories = False
    def capture(kwargs):
        calls.append(kwargs)
        return _codex_message_response("Hello!")
    monkeypatch.setattr(codex, "_interruptible_api_call", capture)
    with patch("hermes_cli.config.load_config", return_value=PROFILE):
        first = codex.run_conversation("hi")
        assert first["completed"] is True
        second = codex.run_conversation("thanks", conversation_history=first["messages"])
        assert second["completed"] is True
        codex.run_conversation("Explain the risks", conversation_history=second["messages"])
    assert [call["reasoning"]["effort"] for call in calls] == ["low", "high", "high"]


@pytest.mark.parametrize("explicit", [{"enabled": True, "effort": "high"}, {"enabled": False}])
def test_actual_route_switch_preserves_explicit(explicit):
    from tests.agent.test_primary_runtime_restore import _make_agent, _mock_resolve
    agent = _make_agent(fallback_model={"provider": "openrouter", "model": "anthropic/claude-sonnet-4"})
    agent.reasoning_config = explicit.copy()
    with adaptive_turn(agent, "hi", [], True), \
         patch("hermes_cli.config.load_config", return_value=PROFILE), \
         patch("agent.auxiliary_client.resolve_provider_client", return_value=(_mock_resolve(), None)):
        assert agent._try_activate_fallback() is True
        assert agent.reasoning_config == explicit
        assert _reresolve_and_wire(agent) == explicit


def _reresolve_and_wire(agent):
    from agent.chat_completion_helpers import _reasoning_config_for_wire
    return _reasoning_config_for_wire(agent)
