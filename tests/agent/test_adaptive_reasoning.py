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


@pytest.mark.parametrize("configured,override,expected", [
    (True, None, True), (False, None, False), ("true", None, False),
    (True, False, False), (False, True, True), (True, "true", False),
])
def test_profile_opt_in_and_per_call_override(configured, override, expected):
    from agent.adaptive_reasoning import enabled_for_turn
    with patch("hermes_cli.config.load_config", return_value={"agent": {"adaptive_reasoning": configured}}):
        assert enabled_for_turn(override) is expected
    with patch("hermes_cli.config.load_config", side_effect=ValueError):
        assert enabled_for_turn(None) is False


def test_fallback_explicit_and_nested_scope():
    from tests.agent.test_unset_reasoning_effort_default import _Agent
    parent, child = _Agent(None), _Agent(None)
    with patch("agent.models_dev.get_model_capabilities", return_value=None):
        with adaptive_turn(parent, "hi", [], True):
            with adaptive_turn(child, "explain risks", [], True):
                assert _reasoning_config_for_wire(child)["effort"] == "medium"
                assert _reasoning_config_for_wire(parent)["effort"] == "medium"
            assert _reasoning_config_for_wire(parent)["effort"] == "low"
            with patch("hermes_cli.config.load_config", return_value={"agent": {"reasoning_effort": "high"}}):
                _reresolve_fallback_reasoning_config(parent)
            assert _reasoning_config_for_wire(parent)["effort"] == "high"


def test_telegram_gateway_to_real_payload():
    import asyncio
    from gateway.config import Platform
    from tests.gateway.test_background_child_approval_after_turn import _Adapter, _turn
    from run_agent import AIAgent
    from tests.agent.test_run_agent import _mock_response

    loop = asyncio.new_event_loop()
    try:
        with patch("model_tools.get_tool_definitions", return_value=[]), \
             patch("model_tools.check_toolset_requirements", return_value={}), \
             patch("agent.process_bootstrap.OpenAI"), \
             patch("agent.models_dev.get_model_capabilities", return_value=None), \
             patch("hermes_cli.config.load_config", return_value={"agent": {"adaptive_reasoning": True}}):
            agent = AIAgent(api_key="test-key-1234567890", base_url="http://127.0.0.1:8000/v1",
                            provider="custom:relay", model="test-model", quiet_mode=True,
                            skip_context_files=True, skip_memory=True)
            agent.client = MagicMock()
            agent.client.chat.completions.create.return_value = _mock_response(content="Hello!")
            agent._cached_system_prompt = "You are helpful."
            agent._use_prompt_caching = False
            agent.compression_enabled = agent.save_trajectories = False
            turn = _turn(Platform.TELEGRAM, _Adapter("telegram"), loop)
            turn._ctx.message = "hi"
            with patch.object(agent, "_persist_session"), patch.object(agent, "_save_trajectory"), \
                 patch.object(agent, "_cleanup_task_resources"):
                turn._run_conversation_with_approval(agent, [], None, None, None)
            assert agent.client.chat.completions.create.call_args.kwargs["reasoning_effort"] == "low"
    finally:
        loop.close()


def test_real_profile_config_in_isolated_subprocess(tmp_path):
    import os
    import subprocess
    import sys
    home = tmp_path / "home"
    profile = home / ".hermes"
    profile.mkdir(parents=True)
    config = profile / "config.yaml"
    config.write_text("agent:\n  adaptive_reasoning: true\n")
    env = {**os.environ, "HOME": str(home), "HERMES_HOME": str(profile)}
    code = ("from agent.adaptive_reasoning import enabled_for_turn; "
            "assert enabled_for_turn(None) is True; "
            "assert enabled_for_turn(False) is False")
    subprocess.run([sys.executable, "-c", code], env=env, check=True)
    config.write_text("agent:\n  adaptive_reasoning: false\n")
    subprocess.run([sys.executable, "-c", "from agent.adaptive_reasoning import enabled_for_turn; assert enabled_for_turn(None) is False"], env=env, check=True)


def test_thread_scopes_are_isolated():
    import threading
    from tests.agent.test_unset_reasoning_effort_default import _Agent
    agent = _Agent(None)
    barrier = threading.Barrier(2)
    results = {}
    def worker(name, message):
        with adaptive_turn(agent, message, [], True):
            barrier.wait(timeout=5)
            results[name] = _reasoning_config_for_wire(agent)["effort"]
    with patch("agent.models_dev.get_model_capabilities", return_value=None):
        threads = [threading.Thread(target=worker, args=("social", "hi")),
                   threading.Thread(target=worker, args=("complex", "explain risk"))]
        for thread in threads: thread.start()
        for thread in threads: thread.join(timeout=10)
    assert results == {"social": "low", "complex": "medium"}


@pytest.mark.parametrize("surface", ["cli", "cron", "delegate"])
def test_normal_cli_and_cron_turns_reach_payload(surface):
    from types import SimpleNamespace
    from run_agent import AIAgent
    from tests.agent.test_run_agent import _mock_response
    with patch("model_tools.get_tool_definitions", return_value=[]), \
         patch("model_tools.check_toolset_requirements", return_value={}), \
         patch("agent.process_bootstrap.OpenAI"), \
         patch("agent.models_dev.get_model_capabilities", return_value=None), \
         patch("hermes_cli.config.load_config", return_value={"agent": {"adaptive_reasoning": True}}):
        agent = AIAgent(api_key="test-key-1234567890", base_url="http://127.0.0.1:8000/v1",
                        provider="custom:relay", model="test-model", quiet_mode=True,
                        skip_context_files=True, skip_memory=True)
        agent.client = MagicMock()
        agent.client.chat.completions.create.return_value = _mock_response(content="Hello!")
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.compression_enabled = agent.save_trajectories = False
        with patch.object(agent, "_persist_session"), patch.object(agent, "_save_trajectory"), \
             patch.object(agent, "_cleanup_task_resources"):
            if surface == "cron":
                from cron.scheduler import _run_agent_with_watchdog
                _run_agent_with_watchdog(agent, "hi", {}, "job", "test", "task", None)
            elif surface == "delegate":
                from tools.delegate_tool_child_run import _ChildRun
                parent = SimpleNamespace(_current_task_id=None)
                run = _ChildRun(agent, parent, 0, "hi", None, None)
                result, failure, _ = run.await_child()
                assert failure is None, failure
                assert result["final_response"]
            else:
                from hermes_cli.cli_single_query import _run_quiet_single_query
                cli = SimpleNamespace(agent=agent, conversation_history=[], session_id=agent.session_id)
                with patch("hermes_cli.quiet_single_query.quiet_notify_linger_seconds", return_value=0), \
                     patch("cli._run_kanban_goal_loop_q"), \
                     patch("hermes_cli.quiet_single_query.continue_quiet_notify_completions", return_value=None):
                    with pytest.raises(SystemExit) as exc:
                        _run_quiet_single_query(cli, "hi")
                    assert exc.value.code == 0
        assert agent.client.chat.completions.create.call_args.kwargs["reasoning_effort"] == "low"


@pytest.mark.parametrize("config,overrides,expected", [
    ({"enabled": True, "effort": "high"}, None, "high"),
    ({"enabled": False}, None, "none"),
    (None, {"reasoning_effort": "high"}, "high"),
])
def test_explicit_reasoning_wins_in_actual_payload(config, overrides, expected):
    from run_agent import AIAgent
    from tests.agent.test_run_agent import _mock_response
    with patch("model_tools.get_tool_definitions", return_value=[]), \
         patch("model_tools.check_toolset_requirements", return_value={}), \
         patch("agent.process_bootstrap.OpenAI"), \
         patch("agent.models_dev.get_model_capabilities", return_value=None):
        agent = AIAgent(api_key="test-key-1234567890", base_url="http://127.0.0.1:8000/v1",
                        provider="custom:relay", model="test-model", quiet_mode=True,
                        skip_context_files=True, skip_memory=True)
        agent.reasoning_config = config
        agent.request_overrides = overrides
        agent.client = MagicMock()
        agent.client.chat.completions.create.return_value = _mock_response(content="Hello!")
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.compression_enabled = agent.save_trajectories = False
        with patch.object(agent, "_persist_session"), patch.object(agent, "_save_trajectory"), \
             patch.object(agent, "_cleanup_task_resources"):
            agent.run_conversation("hi", adaptive_reasoning=True)
        assert agent.client.chat.completions.create.call_args.kwargs["reasoning_effort"] == expected
