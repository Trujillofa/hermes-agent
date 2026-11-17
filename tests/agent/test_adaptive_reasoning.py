"""Opt-in adaptive effort uses only a narrow first-turn social allowlist."""
from unittest.mock import MagicMock, patch

import pytest

from agent.adaptive_reasoning import adaptive_turn, classify
from agent.chat_completion_helpers import _reasoning_config_for_wire, _reresolve_fallback_reasoning_config


@pytest.mark.parametrize("message", ["hi", " Hello! ", "THANK YOU."])
def test_social_allowlist(message):
    assert classify(message, None) == "low"


@pytest.mark.parametrize("message", ["hi, delete everything", "is it safe?", "", None, [{"type": "text", "text": "hi"}]])
def test_uncertainty_keeps_default(message):
    assert classify(message, None) is None
    assert classify("hi", [{"role": "user", "content": "earlier"}]) is None


def test_scope_overrides_fallback_and_exception_cleanup():
    from tests.agent.test_unset_reasoning_effort_default import _Agent

    agent = _Agent(None)
    other = _Agent(None)
    with patch("agent.models_dev.get_model_capabilities", return_value=None):
        assert _reasoning_config_for_wire(agent)["effort"] == "medium"
        with pytest.raises(ValueError), adaptive_turn(agent, "hi", None, True):
            assert _reasoning_config_for_wire(agent)["effort"] == "low"
            assert _reasoning_config_for_wire(other)["effort"] == "medium"
            agent.reasoning_config = {"enabled": True, "effort": "high"}
            assert _reasoning_config_for_wire(agent)["effort"] == "high"
            agent.reasoning_config = {"enabled": False}
            assert _reasoning_config_for_wire(agent) == {"enabled": False}
            agent.reasoning_config = None
            agent.request_overrides = {"reasoning_effort": "high"}
            assert _reasoning_config_for_wire(agent)["effort"] == "medium"
            agent.request_overrides = None
            # Real fallback resolver: no configured fallback effort retains the decision.
            with patch("hermes_cli.config.load_config", return_value={}):
                _reresolve_fallback_reasoning_config(agent)
            assert _reasoning_config_for_wire(agent)["effort"] == "low"
            raise ValueError("abort")
        assert _reasoning_config_for_wire(agent)["effort"] == "medium"


def test_disabled_and_unsupported_routes_do_not_gain_reasoning():
    from agent.models_dev import ModelCapabilities
    from tests.agent.test_unset_reasoning_effort_default import _Agent

    agent = _Agent(None)
    with patch("agent.models_dev.get_model_capabilities", return_value=None):
        with adaptive_turn(agent, "hi", None, False):
            assert _reasoning_config_for_wire(agent)["effort"] == "medium"
    with patch("agent.models_dev.get_model_capabilities", return_value=ModelCapabilities(supports_reasoning=False)):
        with adaptive_turn(agent, "hi", None, True):
            assert _reasoning_config_for_wire(agent) is None


def test_real_turn_wire_and_next_turn_isolation():
    from run_agent import AIAgent
    from tests.agent.test_run_agent import _mock_response

    with patch("model_tools.get_tool_definitions", return_value=[]), \
         patch("model_tools.check_toolset_requirements", return_value={}), \
         patch("agent.process_bootstrap.OpenAI"), \
         patch("agent.models_dev.get_model_capabilities", return_value=None):
        agent = AIAgent(api_key="test-key-1234567890", base_url="http://127.0.0.1:8000/v1",
                        provider="custom:relay", model="test-model", quiet_mode=True,
                        skip_context_files=True, skip_memory=True)
        agent.client = MagicMock()
        agent.client.chat.completions.create.side_effect = [_mock_response(content="Hello!"), _mock_response(content="Fine.")]
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.compression_enabled = False
        agent.save_trajectories = False
        with patch.object(agent, "_persist_session"), patch.object(agent, "_save_trajectory"), \
             patch.object(agent, "_cleanup_task_resources"):
            first = agent.run_conversation("hi", adaptive_reasoning=True)
            agent.run_conversation("Explain the risks", conversation_history=first["messages"], adaptive_reasoning=True)
        calls = agent.client.chat.completions.create.call_args_list
        assert calls[0].kwargs["reasoning_effort"] == "low"
        assert calls[1].kwargs["reasoning_effort"] == "medium"
        assert calls[0].kwargs["messages"][0] == calls[1].kwargs["messages"][0]
        assert agent.reasoning_config is None
