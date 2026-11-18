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
def test_edit_during_fallback_survives_restore(new_value):
    agent = _make_agent(fallback_model={'provider': 'openrouter', 'model': 'anthropic/claude-sonnet-4'})
    agent._primary_runtime['reasoning_config'] = resolve_reasoning_config(PROFILE, 'gpt-6.1-terra')
    with adaptive_turn(agent, 'hi', [], True), \
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
    agent.reasoning_config = primary
    with adaptive_turn(agent, 'complex task', [], True), \
         patch('agent.auxiliary_client.resolve_provider_client', return_value=(_mock_resolve(), None)), \
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
    with adaptive_turn(agent, 'hi', [], True), \
         patch('agent.auxiliary_client.resolve_provider_client', return_value=(_mock_resolve(), None)), \
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
    with adaptive_turn(agent, 'hi', [], True), \
         patch('agent.auxiliary_client.resolve_provider_client', return_value=(_mock_resolve(), None)), \
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


@pytest.mark.parametrize('snapshot_key', [False, True])
def test_unset_primary_restored_after_real_fallback(snapshot_key):
    agent = _make_agent(fallback_model={'provider': 'openrouter', 'model': 'fallback'})
    agent.reasoning_config = None
    if snapshot_key:
        agent._primary_runtime['reasoning_config'] = None
    else:
        agent._primary_runtime.pop('reasoning_config', None)
    with adaptive_turn(agent, 'complex task', [], True), \
         patch('agent.auxiliary_client.resolve_provider_client', return_value=(_mock_resolve(), None)), \
         patch('hermes_cli.config.load_config', return_value={'agent': {'reasoning_effort': 'high'}}):
        assert agent._try_activate_fallback()
        assert agent.reasoning_config['effort'] == 'high'
    with patch('agent.process_bootstrap.OpenAI', return_value=MagicMock()):
        assert agent._restore_primary_runtime()
    assert agent.reasoning_config is None


@pytest.mark.parametrize('explicit', [None, {}, {'enabled': False}, {'enabled': True, 'effort': 'high'}])
def test_disabled_fallback_matches_legacy_reresolution(explicit):
    agent = _make_agent(fallback_model={'provider': 'openrouter', 'model': 'fallback'})
    agent.reasoning_config = explicit
    with adaptive_turn(agent, 'hi', [], False), \
         patch('agent.auxiliary_client.resolve_provider_client', return_value=(_mock_resolve(), None)), \
         patch('hermes_cli.config.load_config', return_value={'agent': {'reasoning_effort': 'medium'}}):
        assert agent._try_activate_fallback()
    assert agent.reasoning_config == {'enabled': True, 'effort': 'medium'}
    assert '_fallback_reasoning_at_bind' not in vars(agent)


@pytest.mark.parametrize('following', [False, True])
@pytest.mark.parametrize('pin', [None, 'low'])
def test_policy_uses_actual_turn_model_not_frozen_model(tmp_path, following, pin):
    config = {'model': {'default': 'model-a', 'provider': 'openrouter'},
              'agent': {'reasoning_effort': 'medium',
                        'reasoning_overrides': {'model-a': 'high', 'model-b': 'xhigh'}}}
    request = {'source': 'cli', 'cwd': str(tmp_path), 'model': 'model-a',
               'follow_profile_config': following}
    if pin:
        request['reasoning'] = pin
    policy = build_policy(request, config, profile_terminal=False)
    with patch('gateway.session_authorities.active_authority', return_value=None):
        assert policy.current_reasoning_config(None, model='model-a')['effort'] == (pin or 'high')
        assert policy.current_reasoning_config(None, model='model-b')['effort'] == (pin or 'xhigh')
        assert policy.current_reasoning_config(None, model='model-a')['effort'] == (pin or 'high')


@pytest.mark.parametrize('enabled,explicit,expected', [
    (True, True, 'low'), (True, False, 'high'), (False, True, 'high'),
])
def test_real_model_switch_rebinds_baseline_or_keeps_session_pin(enabled, explicit, expected):
    agent = _make_agent()
    config = {'agent': {'adaptive_reasoning': enabled, 'reasoning_effort': 'medium',
                        'reasoning_overrides': {'new-model': 'high'}}}
    agent.reasoning_config = ({'enabled': True, 'effort': 'low'} if explicit
                              else resolve_reasoning_config(config, 'old-model'))
    agent._fallback_reasoning_at_bind = {'enabled': True, 'effort': 'medium'}
    agent._fallback_reasoning_primary = None
    with patch('hermes_cli.config.load_config', return_value=config), \
         patch('agent.model_metadata.get_model_context_length', return_value=131072), \
         patch('agent.process_bootstrap.OpenAI', return_value=MagicMock()):
        agent.switch_model('new-model', 'custom', api_key='sk-test',
                           base_url='https://my-llm.example.com/v1')
    assert agent.reasoning_config['effort'] == expected
    assert agent._primary_runtime['reasoning_config']['effort'] == expected
    assert '_fallback_reasoning_at_bind' not in vars(agent)
    assert '_fallback_reasoning_primary' not in vars(agent)


@pytest.mark.parametrize('existing', [False, True])
def test_voice_route_restores_fallback_tracking(existing):
    from agent.voice_turn_route import _capture, _reinstall
    agent = _make_agent()
    if existing:
        agent._fallback_reasoning_at_bind = {'enabled': False}
        agent._fallback_reasoning_primary = None
    before = {key: value for key, value in vars(agent).items() if key.startswith('_fallback_reasoning_')}
    state = _capture(agent)
    agent._fallback_reasoning_at_bind = {'enabled': True}
    agent._fallback_reasoning_primary = {'effort': 'high'}
    agent._fallback_reasoning_session_edit = ({},)
    with patch('agent.route_binding.reinstall_runtime_snapshot'):
        _reinstall(agent, state)
    after = {key: value for key, value in vars(agent).items() if key.startswith('_fallback_reasoning_')}
    assert after == before
