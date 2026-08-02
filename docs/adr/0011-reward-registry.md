# Reward functions are a config-selected registry, defaulting to binary

Rather than committing to one reward shape, reward functions are a named registry selected
per run from config. Each entry records the source it came from. `binary` is the default for
the first run and the baseline every other entry is compared against.

The literature genuinely disagrees, and the disagreement is not resolvable from the armchair
— so the project is built to answer it empirically instead of guessing.

Full detail — the shared input type, each function's formula and range, the weighting
arithmetic, and what is measured but never rewarded — is in
[`docs/design/rl-reward-functions.md`](../design/rl-reward-functions.md).

## The registry

| Key | Rule | Source |
| --- | --- | --- |
| `binary` | 1.0 if all tests pass, else 0.0 | DeepCoder/rLLM `check_correctness`; DeepSeek-R1 rule-based rewards, [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) |
| `pass_rate` | passed / total | open-r1 `code_reward` |
| `binary_threshold` | 1.0 if pass_rate > 0.99 | open-r1 `binary_code_reward` |
| `ladder` | 0 → .05 parses → .10 runs → .10 + .90×pass_rate | This project; shaped like DHRCL's hierarchical decomposition, [arXiv:2607.26457](https://arxiv.org/html/2607.26457) |
| `code_r1` | −1.1 format failure, +0.1 wrong, +1.1 correct | code-r1 `coder1/__init__.py` — trains on CodeContests, Python, stdin/stdout |
| `hierarchical` | syntax → execution → partial correctness → AST alignment | DHRCL, [arXiv:2607.26457](https://arxiv.org/html/2607.26457) |
| `verpo` | KDE density-calibrated per-test weights + binary global anchor | VeRPO, [arXiv:2601.03525](https://arxiv.org/html/2601.03525) |

Auxiliary terms, composable with any primary via `reward_weights`:

| Key | Rule | Source |
| --- | --- | --- |
| `extractability` | `parse_term + fence_term`, −1.0 … +1.0, weight 0.1 — see ADR-0012 | `docs/research/format-adherence.md` §8.3, adapted |
| `overlong` | 0 / linear decay over cache / −1 | DAPO, [arXiv:2503.14476](https://arxiv.org/pdf/2503.14476) |

## Why binary is the default

The closest published match to this setup — Qwen2.5-7B-Instruct with GRPO, 16 rollouts per
problem ([arXiv:2605.02944](https://arxiv.org/html/2605.02944)) — found pass-rate reward
matched binary at pass@1 (40.9 vs 40.6) and lost 2 points at pass@16 (55.6 vs 57.6). Its
mechanism is the concerning part: **57.4% of groups contained both harmful samples with
positive advantage and helpful samples with negative advantage.** Partial credit does not
merely add noise, it pulls in opposing directions inside a single group.

Corroborating: VeRPO found uncalibrated dense rewards underperform binary and only beat it
after adding Gaussian-KDE calibration; and a composite-reward study
([arXiv:2605.17174](https://arxiv.org/html/2605.17174)) found auxiliary terms *degraded*
correctness, because "easy-to-satisfy proxy terms can dominate optimization and improve the
total reward without commensurate gains in execution-level correctness."

## Consequences

Because the verifier and scorer are split (ADR-0004), a single execution feeds every reward
function. **Every registered reward is logged each step, including the ones not driving
training** — one run yields all the counterfactual curves for free.

Comparisons are only meaningful against a metric that is not the reward: binary pass@1 and
pass@10 on the full test suite. Everything else — seed, filters, group size, LoRA config —
must be held fixed across runs.

**Dynamic sampling is not in this registry.** Over-generating and discarding degenerate
groups is a rollout policy, not a reward function, and lives behind its own flag. It was
DAPO's single largest measured gain (+9 points), so it must be toggleable independently of
reward shape or it will confound every comparison.

Each real comparison costs a full training run on one GPU. Budget for two or three genuine
A/B runs, not seven.
