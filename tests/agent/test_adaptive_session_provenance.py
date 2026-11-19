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


@pytest.mark.parametrize('reuse_object', [False, True])
@pytest.mark.parametrize('second_fallback', [False, True])
def test_equal_assignment_during_fallback_is_a_session_edit(reuse_object, second_fallback):
    from agent.route_binding import bind_route_entry
    agent = _make_agent(fallback_model={'provider': 'openrouter', 'model': 'fallback'})
    primary = resolve_reasoning_config({'agent': {'reasoning_effort': 'high'}}, agent.model)
    agent.reasoning_config = agent._primary_runtime['reasoning_config'] = primary
    config = {'agent': {'reasoning_effort': 'medium'}}
    with adaptive_turn(agent, 'work', [], True), \
         patch('agent.auxiliary_client.resolve_provider_client', return_value=(_mock_resolve(), None)), \
         patch('hermes_cli.config.load_config', return_value=config):
        assert agent._try_activate_fallback()
        fresh = agent.reasoning_config if reuse_object else resolve_reasoning_config(config, 'primary')
        agent.reasoning_config = fresh
        if second_fallback:
            assert bind_route_entry(agent, {'provider': 'openrouter', 'model': 'fallback-2'},
                                    'openrouter', 'fallback-2') is not None
        with patch('agent.process_bootstrap.OpenAI', return_value=MagicMock()):
            assert agent._restore_primary_runtime()
    assert agent.reasoning_config == fresh
    assert type(agent.reasoning_config) is type(fresh)


def test_reassigning_none_during_fallback_is_a_session_edit():
    agent = _make_agent(fallback_model={'provider': 'openrouter', 'model': 'fallback'})
    primary = resolve_reasoning_config({'agent': {'reasoning_effort': 'high'}}, agent.model)
    agent.reasoning_config = agent._primary_runtime['reasoning_config'] = primary
    with adaptive_turn(agent, 'work', [], True), \
         patch('agent.auxiliary_client.resolve_provider_client', return_value=(_mock_resolve(), None)), \
         patch('hermes_cli.config.load_config', return_value={}):
        assert agent._try_activate_fallback()
        assert agent.reasoning_config is None
        agent.reasoning_config = None
        with patch('agent.process_bootstrap.OpenAI', return_value=MagicMock()):
            assert agent._restore_primary_runtime()
    assert agent.reasoning_config is None


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


@pytest.mark.parametrize('fresh', ['medium', 'low', None, 'disabled'])
def test_cached_gateway_turn_setting_survives_fallback_restoration(fresh):
    from tests.gateway.test_display_null_turn_wiring import _wire
    agent = _make_agent(fallback_model={'provider': 'openrouter', 'model': 'fallback'})
    primary = resolve_reasoning_config(PROFILE, 'gpt-6.1-terra')
    agent.reasoning_config = agent._primary_runtime['reasoning_config'] = primary
    config = {'agent': {'reasoning_effort': 'medium'}}
    with adaptive_turn(agent, 'complex task', [], True), \
         patch('agent.auxiliary_client.resolve_provider_client', return_value=(_mock_resolve(), None)), \
         patch('hermes_cli.config.load_config', return_value=config):
        assert agent._try_activate_fallback()
        assert agent.reasoning_config['effort'] == 'medium'
    if fresh == 'disabled':
        value = {'enabled': False}
    elif fresh is None:
        value = None
    else:
        value = resolve_reasoning_config({'agent': {'reasoning_effort': fresh}}, 'gpt-6.1-terra')
    _wire({}, agent=agent, reasoning_config=value)
    with patch('agent.process_bootstrap.OpenAI', return_value=MagicMock()):
        assert agent._restore_primary_runtime()
    assert agent.reasoning_config == value
    assert type(agent.reasoning_config) is type(value)


@pytest.mark.parametrize('following', [False, True])
def test_gateway_loader_passes_effective_model_to_local_policy(tmp_path, following):
    from types import SimpleNamespace
    from gateway.run_config_loaders import GatewayConfigLoadersMixin
    config = {'model': {'default': 'model-a', 'provider': 'openrouter'},
              'agent': {'reasoning_effort': 'medium',
                        'reasoning_overrides': {'model-a': 'high', 'model-b': 'low'}}}
    policy = build_policy({'source': 'cli', 'cwd': str(tmp_path), 'model': 'model-a',
                           'follow_profile_config': following}, config, profile_terminal=False)
    runner = SimpleNamespace(_resolve_session_key_or_none=lambda *_: 'route',
                             _peek_session_state=lambda *_: None)
    with patch('gateway.session_setting_commands.local_policy', return_value=policy), \
         patch('gateway.session_authorities.active_authority', return_value=None):
        resolved = GatewayConfigLoadersMixin._resolve_session_reasoning_config(runner, model='model-b')
    assert resolved['effort'] == 'low'


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


def test_disabled_cached_gateway_fallback_retains_legacy_restoration():
    from tests.gateway.test_display_null_turn_wiring import _wire
    agent = _make_agent(fallback_model={'provider': 'openrouter', 'model': 'fallback'})
    primary = {'enabled': True, 'effort': 'high'}
    agent.reasoning_config = agent._primary_runtime['reasoning_config'] = primary
    with adaptive_turn(agent, 'hi', [], False), \
         patch('agent.auxiliary_client.resolve_provider_client', return_value=(_mock_resolve(), None)), \
         patch('hermes_cli.config.load_config', return_value={'agent': {'reasoning_effort': 'medium'}}):
        assert agent._try_activate_fallback()
    _wire({}, agent=agent, reasoning_config={'enabled': False})
    assert '_fallback_reasoning_session_edit' not in vars(agent)
    with patch('agent.process_bootstrap.OpenAI', return_value=MagicMock()):
        assert agent._restore_primary_runtime()
    assert agent.reasoning_config == primary


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


def test_voice_effort_only_restoration_does_not_create_session_edit():
    from agent.voice_turn_route import begin_voice_turn_route, end_voice_turn_route
    agent = _make_agent()
    agent._fallback_reasoning_at_bind = {'enabled': True, 'effort': 'medium'}
    agent.reasoning_config = {'enabled': True, 'effort': 'medium'}
    agent.__dict__.pop('_fallback_reasoning_session_edit')
    agent._voice_turn_pending = True
    with patch('agent.auxiliary_task_config._get_auxiliary_task_config', return_value={'reasoning_effort': 'low'}), \
         patch('agent.voice_turn_route._route_target', return_value=None):
        begin_voice_turn_route(agent, [], None)
        end_voice_turn_route(agent)
    assert '_fallback_reasoning_session_edit' not in vars(agent)


@pytest.mark.parametrize('override,process_enabled,policy_enabled,explicit,expected', [
    (True, False, False, True, 'low'),
    (False, True, True, True, 'high'),
    (None, False, True, True, 'low'),
    (None, False, True, False, 'xhigh'),
])
def test_switch_uses_bound_policy_and_per_call_override(override, process_enabled, policy_enabled, explicit, expected):
    agent = _make_agent()
    process = {'agent': {'adaptive_reasoning': process_enabled, 'reasoning_effort': 'high'}}
    policy = {'agent': {'adaptive_reasoning': policy_enabled, 'reasoning_effort': 'medium',
                        'reasoning_overrides': {'new-model': 'xhigh'}}}
    agent.reasoning_config = ({'enabled': True, 'effort': 'low'} if explicit
                              else resolve_reasoning_config(policy, 'old-model'))
    with patch('gateway.session_policy.active_policy_config', return_value=policy), \
         patch('hermes_cli.config.load_config', return_value=process), \
         patch('agent.model_metadata.get_model_context_length', return_value=131072), \
         patch('agent.process_bootstrap.OpenAI', return_value=MagicMock()), \
         adaptive_turn(agent, 'work', [], override):
        agent.switch_model('new-model', 'custom', api_key='sk-test',
                           base_url='https://my-llm.example.com/v1')
    assert agent.reasoning_config['effort'] == expected


def test_disabled_empty_switch_snapshot_keeps_fallback_legacy_behavior():
    from agent.primary_runtime_snapshot import _build_primary_runtime_snapshot
    agent = _make_agent(fallback_model={'provider': 'openrouter', 'model': 'fallback'})
    agent.reasoning_config = {}
    agent._primary_runtime = _build_primary_runtime_snapshot(agent, agent.api_mode)
    assert agent._primary_runtime['reasoning_config'] is None
    with adaptive_turn(agent, 'work', [], False), \
         patch('agent.auxiliary_client.resolve_provider_client', return_value=(_mock_resolve(), None)), \
         patch('hermes_cli.config.load_config', return_value={'agent': {'reasoning_effort': 'medium'}}):
        assert agent._try_activate_fallback()
    with patch('agent.process_bootstrap.OpenAI', return_value=MagicMock()):
        assert agent._restore_primary_runtime()
    assert agent.reasoning_config['effort'] == 'medium'


@pytest.mark.parametrize('parent_enabled,child_enabled', [(True, False), (False, True)])
def test_delegated_child_resolves_own_opt_in(parent_enabled, child_enabled):
    from agent.adaptive_reasoning import enabled_for_turn
    from agent.delegation_context import delegated_child_context
    with patch('gateway.session_policy.active_policy_config', return_value={'agent': {'adaptive_reasoning': parent_enabled}}), \
         patch('hermes_cli.config.load_config', return_value={'agent': {'adaptive_reasoning': child_enabled}}):
        assert enabled_for_turn(None) is parent_enabled
        with delegated_child_context('child'):
            assert enabled_for_turn(None) is child_enabled
        assert enabled_for_turn(None) is parent_enabled


def test_disabled_explicit_ollama_does_not_probe_capabilities():
    from tests.agent.test_unset_reasoning_effort_default import _Agent
    from agent.chat_completion_helpers import _reasoning_config_for_wire
    agent = _Agent(reasoning_config={'enabled': True, 'effort': 'high'})
    agent.api_mode = 'chat_completions'
    agent._ollama_num_ctx = 4096
    agent.reasoning_config = {'enabled': True, 'effort': 'high'}
    agent._ollama_supports_thinking_cached = MagicMock(side_effect=AssertionError('unexpected probe'))
    with adaptive_turn(agent, 'work', [], False):
        assert _reasoning_config_for_wire(agent) == agent.reasoning_config
    agent._ollama_supports_thinking_cached.assert_not_called()


def test_enabled_empty_switch_snapshot_preserves_explicit_mapping():
    from agent.primary_runtime_snapshot import _build_primary_runtime_snapshot
    agent = _make_agent()
    agent.reasoning_config = {}
    with adaptive_turn(agent, 'work', [], True):
        snapshot = _build_primary_runtime_snapshot(agent, agent.api_mode)
    assert snapshot['reasoning_config'] == {}
    assert snapshot['reasoning_config'] is not agent.reasoning_config


@pytest.mark.parametrize('enabled,expected', [(True, 'xhigh'), (False, 'low')])
def test_fallback_uses_frozen_policy_only_when_enabled(tmp_path, enabled, expected):
    from gateway.session_policy import policy_scope
    config = {'agent': {'adaptive_reasoning': enabled, 'reasoning_effort': 'medium',
                        'reasoning_overrides': {'fallback': 'xhigh'}}}
    policy = build_policy({'source': 'cli', 'cwd': str(tmp_path)}, config, profile_terminal=False)
    agent = _make_agent(fallback_model={'provider': 'openrouter', 'model': 'fallback'})
    agent.reasoning_config = resolve_reasoning_config(config, agent.model)
    with policy_scope(policy), adaptive_turn(agent, 'work', [], None), \
         patch('hermes_cli.config.load_config', return_value={'agent': {'reasoning_effort': 'low'}}), \
         patch('agent.auxiliary_client.resolve_provider_client', return_value=(_mock_resolve(), None)):
        assert agent._try_activate_fallback()
        assert agent.reasoning_config['effort'] == expected


def test_child_policy_binding_overrides_copied_parent_for_switch_and_fallback(tmp_path):
    from agent.adaptive_reasoning import enabled_for_turn
    from agent.delegation_context import delegated_child_context
    from gateway.session_policy import policy_scope
    parent_cfg = {'agent': {'adaptive_reasoning': False, 'reasoning_effort': 'high'}}
    child_cfg = {'agent': {'adaptive_reasoning': True, 'reasoning_effort': 'medium',
                           'reasoning_overrides': {'new-model': 'xhigh', 'fallback': 'xhigh'}}}
    request = {'source': 'cli', 'cwd': str(tmp_path)}
    parent = build_policy(request, parent_cfg, profile_terminal=False)
    child = build_policy(request, child_cfg, profile_terminal=False)
    process = {'agent': {'adaptive_reasoning': False, 'reasoning_effort': 'low'}}
    agent = _make_agent(fallback_model={'provider': 'openrouter', 'model': 'fallback'})
    agent.reasoning_config = resolve_reasoning_config(child_cfg, agent.model)
    with patch('hermes_cli.config.load_config', return_value=process), \
         patch('agent.model_metadata.get_model_context_length', return_value=131072), \
         patch('agent.process_bootstrap.OpenAI', return_value=MagicMock()), \
         patch('agent.auxiliary_client.resolve_provider_client', return_value=(_mock_resolve(), None)), \
         policy_scope(parent):
        with delegated_child_context('child'):
            assert enabled_for_turn(None) is False  # copied parent is not the child's policy
            with policy_scope(child):
                assert enabled_for_turn(None) is True
                with adaptive_turn(agent, 'work', [], None):
                    agent.switch_model('new-model', 'custom', api_key='sk-test',
                                       base_url='https://my-llm.example.com/v1')
                    assert agent.reasoning_config['effort'] == 'xhigh'
                    assert agent._try_activate_fallback()
                    assert agent.reasoning_config['effort'] == 'xhigh'
            assert enabled_for_turn(None) is False
        assert enabled_for_turn(None) is False


def test_delegation_worker_copies_parent_but_accepts_child_binding(tmp_path):
    from agent.adaptive_reasoning import enabled_for_turn
    from gateway.session_policy import policy_scope
    from tests.tools.test_delegate import _make_mock_parent
    from tests.tools.test_delegate_child_transcript_release import _fake_child
    from tools.delegate_tool import _run_single_child
    cfg = {'agent': {'adaptive_reasoning': True}}
    policy = build_policy({'source': 'cli', 'cwd': str(tmp_path)}, cfg, profile_terminal=False)
    child = _fake_child([])
    child._delegate_profile_home = None
    seen = []

    def run_child(**kwargs):
        seen.append(enabled_for_turn(None))
        # Even rebinding the very same policy is a new scope owned by this child.
        with policy_scope(policy):
            seen.append(enabled_for_turn(None))
        seen.append(enabled_for_turn(None))
        return {'final_response': 'ok', 'completed': True, 'api_calls': 1, 'messages': []}

    child.run_conversation.side_effect = run_child
    with patch('hermes_cli.config.load_config', return_value={'agent': {'adaptive_reasoning': False}}), \
         policy_scope(policy):
        entry = _run_single_child(task_index=0, goal='work', child=child, parent_agent=_make_mock_parent())
        assert enabled_for_turn(None) is True
    assert seen == [False, True, False]
    assert entry['summary'] == 'ok'


@pytest.mark.parametrize("second_fallback", [False, True])
def test_in_place_edit_during_fallback_survives_primary_restore(second_fallback):
    from agent.route_binding import bind_route_entry
    agent = _make_agent(fallback_model={"provider": "openrouter", "model": "fallback"})
    primary = resolve_reasoning_config({"agent": {"reasoning_effort": "high"}}, agent.model)
    agent.reasoning_config = agent._primary_runtime["reasoning_config"] = primary
    with adaptive_turn(agent, "work", [], True), \
         patch("agent.auxiliary_client.resolve_provider_client", return_value=(_mock_resolve(), None)), \
         patch("hermes_cli.config.load_config", return_value={"agent": {"reasoning_effort": "medium"}}):
        assert agent._try_activate_fallback()
        agent.reasoning_config["effort"] = "low"
        if second_fallback:
            assert bind_route_entry(agent, {"provider": "openrouter", "model": "fallback-2"},
                                    "openrouter", "fallback-2") is not None
        with patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):
            assert agent._restore_primary_runtime()
    assert agent.reasoning_config["effort"] == "low"


@pytest.mark.parametrize("separate_voice", [False, True])
@pytest.mark.parametrize("edit", [{"enabled": False}, None, {"enabled": True, "effort": "low"}])
def test_session_edit_during_voice_on_fallback_survives_all_restores(separate_voice, edit):
    from agent.voice_turn_route import begin_voice_turn_route, end_voice_turn_route
    agent = _make_agent(fallback_model={"provider": "openrouter", "model": "fallback"})
    primary = resolve_reasoning_config({"agent": {"reasoning_effort": "high"}}, agent.model)
    agent.reasoning_config = agent._primary_runtime["reasoning_config"] = primary
    target = {"provider": "openrouter", "model": "voice"} if separate_voice else None
    with adaptive_turn(agent, "work", [], True), \
         patch("agent.auxiliary_client.resolve_provider_client", return_value=(_mock_resolve(), None)), \
         patch("hermes_cli.config.load_config", return_value={"agent": {"reasoning_effort": "medium"}}), \
         patch("agent.auxiliary_task_config._get_auxiliary_task_config", return_value={"reasoning_effort": "none"}), \
         patch("agent.voice_turn_route._route_target", return_value=target), \
         patch("agent.voice_turn_route._fits", return_value=True), \
         patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):
        assert agent._try_activate_fallback()
        agent._voice_turn_pending = True
        begin_voice_turn_route(agent, [], None)
        agent.reasoning_config = edit
        end_voice_turn_route(agent)
        assert agent.reasoning_config == edit
        assert agent._restore_primary_runtime()
    assert agent.reasoning_config == edit


@pytest.mark.parametrize("separate_voice", [False, True])
@pytest.mark.parametrize("mode", ["unchanged", "in_place", "nested_fallback"])
def test_voice_override_does_not_hide_or_invent_session_edits(separate_voice, mode):
    from agent.route_binding import bind_route_entry
    from agent.voice_turn_route import begin_voice_turn_route, end_voice_turn_route
    agent = _make_agent(fallback_model={"provider": "openrouter", "model": "fallback"})
    primary = resolve_reasoning_config({"agent": {"reasoning_effort": "high"}}, agent.model)
    agent.reasoning_config = agent._primary_runtime["reasoning_config"] = primary
    target = {"provider": "openrouter", "model": "voice"} if separate_voice else None
    with adaptive_turn(agent, "work", [], True), \
         patch("agent.auxiliary_client.resolve_provider_client", return_value=(_mock_resolve(), None)), \
         patch("hermes_cli.config.load_config", return_value={"agent": {"reasoning_effort": "medium"}}), \
         patch("agent.auxiliary_task_config._get_auxiliary_task_config", return_value={"reasoning_effort": "low"}), \
         patch("agent.voice_turn_route._route_target", return_value=target), \
         patch("agent.voice_turn_route._fits", return_value=True), \
         patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):
        assert agent._try_activate_fallback()
        agent._voice_turn_pending = True
        begin_voice_turn_route(agent, [], None)
        if mode == "in_place":
            agent.reasoning_config["effort"] = "xhigh"
        elif mode == "nested_fallback":
            assert bind_route_entry(agent, {"provider": "openrouter", "model": "voice-fallback"},
                                    "openrouter", "voice-fallback") is not None
        end_voice_turn_route(agent)
        assert agent._restore_primary_runtime()
    assert agent.reasoning_config["effort"] == ("xhigh" if mode == "in_place" else "high")


@pytest.mark.parametrize("edit", [None, {}, {"enabled": False}, {"enabled": True, "effort": "low"}])
@pytest.mark.parametrize("mutate", [False, True])
def test_same_route_voice_edit_without_fallback(edit, mutate):
    from agent.voice_turn_route import begin_voice_turn_route, end_voice_turn_route
    agent = _make_agent()
    agent.reasoning_config = resolve_reasoning_config({"agent": {"reasoning_effort": "high"}}, agent.model)
    with adaptive_turn(agent, "work", [], True), \
         patch("agent.auxiliary_task_config._get_auxiliary_task_config", return_value={"reasoning_effort": "low"}), \
         patch("agent.voice_turn_route._route_target", return_value=None):
        agent._voice_turn_pending = True
        begin_voice_turn_route(agent, [], None)
        agent.reasoning_config = edit.copy() if isinstance(edit, dict) else edit
        if mutate and isinstance(edit, dict):
            agent.reasoning_config["custom"] = "edited"
        end_voice_turn_route(agent)
    expected = {**edit, "custom": "edited"} if mutate and isinstance(edit, dict) else edit
    assert agent.reasoning_config == expected
    assert type(agent.reasoning_config) is type(expected)


@pytest.mark.parametrize("separate_voice", [False, True])
@pytest.mark.parametrize("prior_fallback", [False, True])
def test_voice_assignment_then_in_place_edit_survives_nested_fallback(separate_voice, prior_fallback):
    from agent.route_binding import bind_route_entry
    from agent.voice_turn_route import begin_voice_turn_route, end_voice_turn_route
    agent = _make_agent(fallback_model={"provider": "openrouter", "model": "fallback"})
    primary = resolve_reasoning_config({"agent": {"reasoning_effort": "high"}}, agent.model)
    agent.reasoning_config = agent._primary_runtime["reasoning_config"] = primary
    target = {"provider": "openrouter", "model": "voice"} if separate_voice else None
    with adaptive_turn(agent, "work", [], True), \
         patch("agent.auxiliary_client.resolve_provider_client", return_value=(_mock_resolve(), None)), \
         patch("hermes_cli.config.load_config", return_value={"agent": {"reasoning_effort": "medium"}}), \
         patch("agent.auxiliary_task_config._get_auxiliary_task_config", return_value={"reasoning_effort": "low"}), \
         patch("agent.voice_turn_route._route_target", return_value=target), \
         patch("agent.voice_turn_route._fits", return_value=True), \
         patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):
        if prior_fallback:
            assert agent._try_activate_fallback()
        agent._voice_turn_pending = True
        begin_voice_turn_route(agent, [], None)
        agent.reasoning_config = {"enabled": True, "effort": "low"}
        agent.reasoning_config["effort"] = "xhigh"
        assert bind_route_entry(agent, {"provider": "openrouter", "model": "voice-fallback"},
                                "openrouter", "voice-fallback") is not None
        end_voice_turn_route(agent)
        assert agent.reasoning_config == {"enabled": True, "effort": "xhigh"}
        if prior_fallback:
            assert agent._restore_primary_runtime()
    assert agent.reasoning_config == {"enabled": True, "effort": "xhigh"}
