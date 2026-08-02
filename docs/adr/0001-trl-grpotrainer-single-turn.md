# Use TRL's GRPOTrainer with single-turn episodes

GRPO for LLMs is canonically single-turn — one prompt, N sampled completions, one scalar
reward each, advantage normalised within the group — and we follow that rather than building
a multi-turn loop where the model sees execution feedback and retries. We use TRL's
`GRPOTrainer` rather than hand-rolling the loop, because the training mathematics is not
where this project's value lies and TRL's implementation is tested.

## Considered Options

Hand-rolling GRPO was seriously considered: the core is roughly 150 lines, and owning the
advantage computation would make swapping to PPO a change to one function rather than a
change of library. It was rejected to avoid re-deriving work that already exists — the
subtle failure modes (logprob alignment across the prompt/completion boundary, padding
masks, token-count normalisation) fail silently as "it just doesn't learn."

Multi-turn episodes were rejected as a much harder credit-assignment problem that multiplies
rollout cost and would likely prevent ever reaching a working end-to-end run.

## Consequences

Interchangeable RL algorithms means swapping the Trainer class, not swapping an internal
function. TRL's `environment_factory` is **not** used — in TRL it means multi-turn tool
calling, which is a different thing from this project's verifier.
