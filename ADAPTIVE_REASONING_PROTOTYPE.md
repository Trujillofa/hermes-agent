# Opt-in adaptive reasoning

Off by default. After installing this change, opt in **only in the intended profile** with
`hermes config set agent.adaptive_reasoning true`; use `hermes config get agent` to
check the value. Set it to `false` to disable. A Python caller can pass
`adaptive_reasoning=True` or `False` to `run_conversation`; a literal per-call boolean
wins over the active profile's setting. Omission/`None` reads the active profile config
at turn admission. Invalid values or a config read error fail closed.

## Scope and precedence

The common `AIAgent.run_conversation` facade covers ordinary Telegram gateway turns,
classic CLI, and agent-backed delegated/cron turns. Other callers of that same facade
also get the profile setting, but not every possible path has been tested. A child reads
its own active profile inside the child profile scope, not its parent's decision.
Decisions are scoped by `ContextVar` to the individual agent and turn, and reset even on
exceptions; no prompts, history, tool schemas, or persistent reasoning settings change.

The deterministic local classifier suggests `low` **only** when the complete user
message is `hi`, `hello`, `hey`, `thanks`, or `thank you`, ignoring case, whitespace
and terminal periods/exclamation marks, on turns with no prior history. Any history,
including malformed or apparently completed exchanges, retains the default. Scripted
preludes and requests with additional conversation messages also retain the default;
this includes follow-up model calls after tools or steering. A wrapped gateway message,
attachment, non-string or longer message retains the existing default.
There is no classifier network call. This is deliberately narrow, not a general task
complexity classifier; savings may be negligible.

Explicit session/API `reasoning_config` (including disabled) wins, and any nonempty
`request_overrides` suppresses adaptation. A profile-resolved enabled config is a
baseline, not an explicit turn decision; the profile's disabled setting remains off.
Profile provenance is tracked in memory; if a path strips it (for example, serializes
and reconstructs a plain dict), adaptation fails closed. In-place changes that alter a
baseline likewise make it explicit. Explicit reasoning is retained during fallback;
profile baselines re-resolve for the fallback model. The one-shot continuation
reasoning-off, route rejection handling and provider clamping retain precedence. A
baseline below `low`, or an unknown baseline, is never promoted to `low`.
Adaptation is limited to OpenAI Codex Responses and chat-completions routes where the
existing unset-reasoning resolver returns a default. The Codex transport clamps effort
to the route's declared vocabulary; request rejection can suppress reasoning.

Direct auxiliary calls (including compression, title generation, vision and direct MoA
calls) bypass this hook. Separate auxiliary/fork agents that run the facade use their
own inputs and profile config; they do not inherit the parent's decision. The Codex
app-server `turn/start.effort` path reads `agent.reasoning_config` directly and bypasses
the adaptive wire hook entirely. Other Responses providers, Anthropic, Bedrock and
paths without an unset chat-completions default are likewise unaffected. Frozen local
session policies may lose baseline provenance on serialization and fail closed.
Provider-specific defaults or API
rejection may erase any benefit, and effort may affect prompt-cache keys. No live
provider compatibility, response quality, savings, or production rollout has been
validated; mocked request tests are not evidence of those outcomes.

## Reproducing offline tests

Use a disposable test environment containing project dependencies and pytest; never use
real profile credentials. From this checkout, substitute its Python executable below:

```bash
mkdir -p "$HOME/.hermes/cache/scratch/adaptive-suite-home"
export HOME="$HOME/.hermes/cache/scratch/adaptive-suite-home"
export HERMES_HOME="$HOME/.hermes"
export HERMES_PYTHON=/absolute/path/to/test-venv/bin/python
scripts/run_tests.sh -j 1 \
  tests/agent/test_adaptive_reasoning.py tests/agent/test_adaptive_codex_integration.py \
  tests/agent/test_unset_reasoning_effort_default.py \
  tests/agent/test_fallback_reasoning_override.py \
  tests/gateway/test_background_child_approval_after_turn.py \
  tests/gateway/test_api_cutover_contract.py \
  tests/gateway/test_observed_context_persistence.py \
  tests/hermes_cli/test_cli_gateway_model_route.py \
  tests/hermes_cli/test_cli_gateway_model_scope.py \
  tests/hermes_cli/test_cli_provider_resolution.py \
  tests/tools/test_delegation_profile_route.py \
  tests/cron/test_cron_reasoning_effort.py \
  --basetemp="$HOME/pytest-tmp"
```

The wrapper runs each file in its own subprocess; `-j 1` avoids concurrent reuse of the
explicit pytest base directory. Adaptive tests exercise mock-provider request payloads
through the actual Telegram turn runner, CLI quiet single-query, delegated child worker,
and cron watchdog, plus explicit reasoning and next-turn isolation. Profile config is
also read in an isolated subprocess. The Codex test builds actual Responses payloads
and runs three conversation turns against a mocked dispatch with the default profile's
model/effort shape; another test performs actual fallback route activation. They do not
send a real Telegram update, contact a provider, or validate the gateway after deployment.
