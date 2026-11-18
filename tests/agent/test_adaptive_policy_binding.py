"""Adaptive admission uses the effective owner-bound session, not the process profile."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import run_agent  # bootstrap before the home I/O guard

from agent.adaptive_reasoning import active_for_agent
from gateway.session_policy import build_policy, policy_scope
from tests.agent.test_primary_runtime_restore import _make_agent


def _policy(tmp_path, value, **flags):
    return build_policy({'source': 'cli', 'cwd': str(tmp_path), **flags},
                        {'agent': {'adaptive_reasoning': value}}, profile_terminal=False)


def _turn(agent, override=None):
    seen = []

    def loop(bound, *args, **kwargs):
        seen.append(active_for_agent(bound))
        return {'final_response': 'ok'}

    with patch('agent.conversation_loop.run_conversation', side_effect=loop):
        agent.run_conversation('hello', adaptive_reasoning=override)
    assert len(seen) == 1
    assert not active_for_agent(agent)
    return seen[0]


@pytest.mark.parametrize('session_value,process_value', [(False, True), (True, False), ('true', True)])
def test_frozen_session_facade_and_override(tmp_path, session_value, process_value):
    policy = _policy(tmp_path, session_value, ignore_user_config=True)
    agent = _make_agent()
    with patch('hermes_cli.config.load_config', return_value={'agent': {'adaptive_reasoning': process_value}}):
        with policy_scope(policy):
            assert _turn(agent) is (session_value is True)
            assert _turn(agent, False) is False
            assert _turn(agent, True) is True
        assert _turn(agent) is process_value


def test_following_session_uses_own_profile_per_turn(tmp_path):
    owner = tmp_path / 'owner'
    owner.mkdir()
    config = owner / 'config.yaml'
    config.write_text('agent:\n  adaptive_reasoning: true\n')
    authority = SimpleNamespace(db=SimpleNamespace(db_path=str(owner / 'state.db')), profile_id=str(owner))
    policy = _policy(tmp_path, False, follow_profile_config=True)
    agent = _make_agent()
    with patch('hermes_cli.config.load_config', return_value={'agent': {'adaptive_reasoning': False}}):
        with policy_scope(policy, authority=authority):
            assert _turn(agent) is True
            config.write_text('agent:\n  adaptive_reasoning: false\n')
            assert _turn(agent) is True  # one owning-profile snapshot per turn scope
        with policy_scope(policy, authority=authority):
            assert _turn(agent) is False


def test_unresolvable_bound_policy_fails_closed(tmp_path):
    policy = _policy(tmp_path, True, follow_profile_config=True)
    agent = _make_agent()
    with patch('hermes_cli.config.load_config', return_value={'agent': {'adaptive_reasoning': True}}):
        with policy_scope(policy):
            assert _turn(agent) is False
        with policy_scope(_policy(tmp_path, True)):
            with patch('gateway.session_policy.LocalSessionPolicy.config', side_effect=ValueError('bad policy')):
                assert _turn(agent) is False


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('bypass', [False, True])
def test_managed_worker_real_frozen_config_bind(tmp_path, enabled, bypass):
    import os
    import subprocess
    import sys

    # Worker policy must bind before runtime imports; exercise that contract in a fresh exec.
    code = '''
import sys
from agent.safe_worker_policy import _bind_worker_config, _bind_safe_worker_policy
enabled = sys.argv[2] == 'true'
bypass = sys.argv[3] == 'true'
config = {'agent': {'adaptive_reasoning': enabled}}
if bypass:
    _bind_safe_worker_policy(safe_mode=False, ignore_user_config=True, config=config)
else:
    _bind_worker_config(config)
from gateway.session_policy import build_policy, policy_scope
from tests.agent.test_adaptive_policy_binding import _turn
from tests.agent.test_primary_runtime_restore import _make_agent
policy = build_policy({'source': 'cli', 'cwd': sys.argv[1], 'follow_profile_config': not bypass,
                       'ignore_user_config': bypass},
                      config, profile_terminal=False)
with policy_scope(policy):
    assert _turn(_make_agent()) is enabled
'''
    env = {**os.environ, 'HOME': str(tmp_path), 'HERMES_HOME': str(tmp_path / '.hermes')}
    subprocess.run([sys.executable, '-c', code, str(tmp_path), str(enabled).lower(), str(bypass).lower()],
                   env=env, check=True, timeout=90)
