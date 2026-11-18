"""Session reasoning provenance through gateway, fallback, and restoration."""
import json
from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from agent.adaptive_reasoning import adaptive_turn
from agent.chat_completion_helpers import build_api_kwargs, _reresolve_fallback_reasoning_config
from gateway.session_policy import build_policy, pin_reasoning_effort, unpin_reasoning
from hermes_constants import ProfileReasoningConfig, resolve_reasoning_config
from tests.agent.test_adaptive_codex_integration import PROFILE
from tests.agent.test_primary_runtime_restore import _make_agent, _mock_resolve
from tests.agent.test_run_agent_codex_responses import _build_agent


@pytest.mark.parametrize('following', [False, True])
@pytest.mark.parametrize('pin', ['high', 'none'])
def test_gateway_launch_pin_is_explicit_on_wire_and_fallback(tmp_path, monkeypatch, following, pin):
    request = {'source': 'cli', 'cwd': str(tmp_path), 'model': 'gpt-6.1-terra',
               'reasoning': pin, 'follow_profile_config': following}
    policy = build_policy(request, PROFILE, profile_terminal=False)
    agent = _build_agent(monkeypatch)
    agent.model, agent.provider, agent.api_mode = 'gpt-6.1-terra', 'openai-codex', 'codex_responses'
    agent.reasoning_config = policy.current_reasoning_config(None)
    agent.request_overrides = None
    explicit = agent.reasoning_config.copy()
    assert type(agent.reasoning_config) is dict
    with adaptive_turn(agent, 'hi', [], True):
        before = build_api_kwargs(agent, [{'role': 'user', 'content': 'hi'}], [])
        with patch('hermes_cli.config.load_config', return_value={'agent': {'reasoning_effort': 'medium'}}):
            _reresolve_fallback_reasoning_config(agent)
        after = build_api_kwargs(agent, [{'role': 'user', 'content': 'hi'}], [])
    assert agent.reasoning_config == explicit
    assert before.get('reasoning') == after.get('reasoning')
    assert before.get('reasoning', {}).get('effort') != 'low'
    if pin == 'high':
        assert before['reasoning']['effort'] == 'high'


@pytest.mark.parametrize('following', [False, True])
def test_committed_pin_and_reset_provenance(tmp_path, following):
    request = {'source': 'cli', 'cwd': str(tmp_path), 'model': 'gpt-6.1-terra',
               'reasoning': 'high', 'follow_profile_config': following}
    policy = build_policy(request, PROFILE, profile_terminal=False)
    cfg = policy.config()
    unpin_reasoning(cfg, PROFILE)
    policy = replace(policy, config_json=json.dumps(cfg))
    assert isinstance(policy.current_reasoning_config(None), ProfileReasoningConfig)
    pin_reasoning_effort(cfg, 'high', session=True)
    policy = replace(policy, config_json=json.dumps(cfg))
    assert type(policy.current_reasoning_config(None)) is dict


@pytest.mark.parametrize('new_value', [{'enabled': False}, {'enabled': True, 'effort': 'low'}, None, {}])
@pytest.mark.parametrize('enabled', [False, True])
def test_edit_during_fallback_survives_restore(new_value, enabled):
    agent = _make_agent(fallback_model={'provider': 'openrouter', 'model': 'anthropic/claude-sonnet-4'})
    agent._primary_runtime['reasoning_config'] = resolve_reasoning_config(PROFILE, 'gpt-6.1-terra')
    with adaptive_turn(agent, 'hi', [], enabled), \
         patch('agent.auxiliary_client.resolve_provider_client', return_value=(_mock_resolve(), None)), \
         patch('hermes_cli.config.load_config', return_value=PROFILE):
        assert agent._try_activate_fallback()
        agent.reasoning_config = new_value
        with patch('agent.process_bootstrap.OpenAI', return_value=MagicMock()):
            assert agent._restore_primary_runtime()
    assert agent.reasoning_config == new_value


def test_unmodified_fallback_restores_primary_profile_provenance():
    agent = _make_agent(fallback_model={'provider': 'openrouter', 'model': 'anthropic/claude-sonnet-4'})
    primary = resolve_reasoning_config(PROFILE, 'gpt-6.1-terra')
    agent._primary_runtime['reasoning_config'] = primary
    with patch('agent.auxiliary_client.resolve_provider_client', return_value=(_mock_resolve(), None)), \
         patch('hermes_cli.config.load_config', return_value={'agent': {'reasoning_effort': 'medium'}}):
        assert agent._try_activate_fallback()
        assert agent.reasoning_config['effort'] == 'medium'
        with patch('agent.process_bootstrap.OpenAI', return_value=MagicMock()):
            assert agent._restore_primary_runtime()
    assert isinstance(agent.reasoning_config, ProfileReasoningConfig)
    assert agent.reasoning_config == primary


@pytest.mark.parametrize('before_fallback', [False, True])
def test_edit_survives_multiple_fallbacks(before_fallback):
    from agent.route_binding import bind_route_entry
    agent = _make_agent(fallback_model={'provider': 'openrouter', 'model': 'model-a'})
    agent._primary_runtime['reasoning_config'] = resolve_reasoning_config(PROFILE, 'gpt-6.1-terra')
    if before_fallback:
        agent.reasoning_config = {'enabled': False}
    with patch('agent.auxiliary_client.resolve_provider_client', return_value=(_mock_resolve(), None)), \
         patch('hermes_cli.config.load_config', return_value=PROFILE):
        assert agent._try_activate_fallback()
        if not before_fallback:
            agent.reasoning_config = {'enabled': False}
        assert bind_route_entry(agent, {'provider': 'openrouter', 'model': 'model-b'},
                                   'openrouter', 'model-b') is not None
        with patch('agent.process_bootstrap.OpenAI', return_value=MagicMock()):
            assert agent._restore_primary_runtime()
    assert agent.reasoning_config == {'enabled': False}


def test_none_edit_between_fallbacks_is_not_replaced_by_fallback_default():
    from agent.route_binding import bind_route_entry
    agent = _make_agent(fallback_model={'provider': 'openrouter', 'model': 'model-a'})
    agent._primary_runtime['reasoning_config'] = resolve_reasoning_config(PROFILE, 'gpt-6.1-terra')
    with patch('agent.auxiliary_client.resolve_provider_client', return_value=(_mock_resolve(), None)), \
         patch('hermes_cli.config.load_config', return_value=PROFILE):
        assert agent._try_activate_fallback()
        agent.reasoning_config = None
        assert bind_route_entry(agent, {'provider': 'openrouter', 'model': 'model-b'},
                                'openrouter', 'model-b') is not None
        with patch('agent.process_bootstrap.OpenAI', return_value=MagicMock()):
            assert agent._restore_primary_runtime()
    assert agent.reasoning_config is None


@pytest.mark.parametrize('reasoning', [None, {'enabled': False}, {'enabled': True, 'effort': 'high'}])
@pytest.mark.parametrize('message', ['hi', 'complex task'])
def test_disabled_turn_is_identical_to_no_adaptive_scope(monkeypatch, reasoning, message):
    agent = _build_agent(monkeypatch)
    agent.model, agent.provider, agent.api_mode = 'gpt-6.1-terra', 'openai-codex', 'codex_responses'
    agent.reasoning_config = reasoning
    agent.request_overrides = None
    messages = [{'role': 'user', 'content': message}]
    baseline = build_api_kwargs(agent, messages, [])
    with adaptive_turn(agent, message, [], False):
        disabled = build_api_kwargs(agent, messages, [])
    assert disabled == baseline
