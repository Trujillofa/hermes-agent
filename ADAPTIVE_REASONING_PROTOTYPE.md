# Adaptive reasoning prototype (not enabled by default)

Experimental Python API only: `agent.run_conversation(message, adaptive_reasoning=True)`.
No CLI, TUI, gateway, cron, delegation, auxiliary, MoA, or configuration wiring is added.

The local deterministic classifier suggests `low` only for a first turn with no supplied
history whose entire string is exactly `hi`, `hello`, `hey`, `thanks`, or `thank you`
(case-insensitive, with whitespace and terminal periods/exclamation marks ignored).
Non-string/multimodal, contextual, complex, ambiguous, or high-risk inputs keep the
existing route default, not an enforced `high`. No text is sent to a classifier and no
messages or prompts are rewritten. This intentionally favors false negatives.

Only primary calls going through the ordinary request builder, and only routes where
Hermes already emits an unset-reasoning default, are eligible. Explicit agent reasoning
(including disabled) or any request overrides suppress the suggestion. Existing transport
clamping, rejection recovery, one-shot continuation overrides, and fallback-specific
explicit settings remain authoritative. A fallback without explicit reasoning can reuse
the same scoped decision if it reaches the same request builder and has an eligible
unset default. The Codex app-server path and direct auxiliary calls bypass this hook;
separately invoked child agents do not inherit the parent agent's suggestion.

Reasoning parameters can themselves affect provider prompt-cache keys; this experiment
does not guarantee cache hits or lower cost. A provider may clamp, ignore, reject or
interpret effort differently. The offline mock transport tests are not live provider or
model compatibility tests, nor a quality/safety evaluation. Do not deploy without
surface integration, full fallback and concurrency coverage, and provider validation.
