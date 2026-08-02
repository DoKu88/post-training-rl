# Reward functions

The central reference for what drives training. Every reward function this project can run,
what it consumes, what it returns, and where it came from.

Governed by [ADR-0011](../adr/0011-reward-registry.md) (the registry),
[ADR-0012](../adr/0012-syntax-gated-extraction.md) (extraction), and
[ADR-0013](../adr/0013-public-tests-are-a-diagnostic.md) (public tests are not a reward).

---

## 1. The shared input

**Every reward function takes exactly one argument and returns one float.** That uniformity is
the entire point — it is what makes them interchangeable at runtime from a config string.

```python
RewardFn = Callable[[RolloutOutcome], float]
```

A function that needs anything else does not belong in the registry.

### `RolloutOutcome` in full

```python
@dataclass(frozen=True)
class RolloutOutcome:
    report: VerificationReport
    completion_token_count: int
    completion_was_truncated: bool


@dataclass(frozen=True)
class VerificationReport:
    problem_id: str
    extraction: Extraction
    graded_results: tuple[TestResult, ...]    # private-first + generated filler, ≤15
    public_results: tuple[TestResult, ...]    # diagnostic only — ADR-0013


@dataclass(frozen=True)
class Extraction:
    code: str | None
    fence: Fence       # TAGGED | UNTAGGED | OTHER_TAG | NONE   — packaging
    parsed: bool       # ast.parse accepted it                  — content


@dataclass(frozen=True)
class TestResult:
    test_index: int
    pool: TestPool                # PUBLIC | PRIVATE | GENERATED
    outcome: TestOutcome          # PASSED | WRONG_OUTPUT | RUNTIME_ERROR
                                  # | TIMEOUT | SKIPPED_AFTER_TIMEOUT
    duration_seconds: float
    stdout_was_truncated: bool
    stderr_excerpt: str
```

### Why one type rather than a bag of arguments

Three consequences follow from every function sharing this input, and all three matter:

1. **One execution feeds all of them.** The verifier runs once per rollout; every reward
   function is a pure transformation of the same result. Without this, comparing reward
   shapes would cost one full execution pass per shape.
2. **Shadow logging is free.** Rewards that are not driving training can be computed and
   logged on the same outcome, so a single run produces the counterfactual curve for every
   registered shape.
3. **They are trivially testable.** A reward test constructs a `RolloutOutcome` literal and
   asserts a number. No sandbox, no subprocess, no model.

### Conventions every function obeys

- `SKIPPED_AFTER_TIMEOUT` counts as **not passed**. Decided once, here, so no individual
  function re-decides it.
- `public_results` is **never read** by any reward function (ADR-0013).
- Functions are pure: no I/O, no clock, no global state, no randomness.
- A problem always has ≥5 graded tests (guaranteed by the dataset builder), so
  `len(graded_results) == 0` is an apparatus failure and raises rather than returning 0.0.

---

## 2. How they reach TRL

TRL calls each entry in `reward_funcs` with the whole batch. Verified against the installed
`trl 1.9.2`:

```python
output = reward_func(
    prompts=prompts,
    completions=completions,
    completion_ids=completion_ids_list,
    **reward_kwargs,          # every extra dataset column, plus trainer_state,
)                             # log_metric, log_extra
```

Our adapter bridges batch-shaped TRL to single-rollout registry functions:

```python
def make_trl_reward(name: str, cache: VerificationCache):
    fn = REWARD_FUNCTIONS[name]

    def reward(prompts, completions, completion_ids,
               problem_id, graded_tests, public_tests, **kwargs) -> list[float]:
        # TRL forwards each dataset column as a parallel list, one entry per rollout.
        problems = [rebuild_problem(pid, g, p)
                    for pid, g, p in zip(problem_id, graded_tests, public_tests)]
        outcomes = cache.outcomes(completions, problems, completion_ids)
        return [fn(o) for o in outcomes]

    reward.__name__ = name
    return reward
```

The dataset's columns are `prompt`, `problem_id`, `graded_tests`, `public_tests` — there is no
single `problem` column, because `datasets` stores Arrow and cannot hold a frozen dataclass.
`rebuild_problem` is the one place that conversion happens, and it asserts shape and names the
offending `problem_id` on failure.

The cache guarantees the verifier runs once per rollout regardless of how many reward
functions ask (ADR-0004), which is sound only because execution is deterministic (ADR-0008).

**`trainer_state` is available**, carrying `global_step`. Any reward needing a training-progress
schedule reads it from there — no factory layer is required.

---

## 3. The registry

### Primary rewards — exactly one drives training

Sorted by **prior likelihood of working on this task**, strongest first. That ranking is a
bet on the evidence available today, not a result — see §3.1 for what each rests on, and the
warning at the end of that section.

| # | Key | Rule | Range | Source |
| --- | --- | --- | --- | --- |
| **1** | **`binary`** ◀ default | `1.0` if every graded test passed, else `0.0` | {0, 1} | DeepCoder/rLLM `check_correctness` → `all(passed)`; DeepSeek-R1 rule-based rewards ([2501.12948](https://arxiv.org/abs/2501.12948)) |
| **2** | `code_r1` | `−1.1` no code · `+0.1` wrong · `+1.1` all pass | {−1.1, 0.1, 1.1} | code-r1 `coder1/__init__.py` — trains on CodeContests, Python, stdin/stdout |
| **3** | `verpo` | KDE density-calibrated per-test weights + binary global anchor | [0, 1] | VeRPO ([2601.03525](https://arxiv.org/html/2601.03525)) |
| **4** | `binary_threshold` | `1.0` if `pass_rate > 0.99` else `0.0` | {0, 1} | open-r1 `binary_code_reward`, `BINARY_THRESHOLD = 0.99` |
| **5** | `pass_rate` | `passed / total` over graded tests | [0, 1] | open-r1 `code_reward` |
| **6** | `ladder` | `0` no code → `.05` parses → `.10` runs → `.10 + .90×pass_rate` | [0, 1] | This project; shaped like DHRCL's hierarchy ([2607.26457](https://arxiv.org/html/2607.26457)) |
| **7** | `hierarchical` | syntax → execution → partial correctness → AST alignment | [0, 1] | DHRCL ([2607.26457](https://arxiv.org/html/2607.26457)) |

**Built for the first run: ranks 1, 2, 4, 5, 6.** `verpo` (3) and `hierarchical` (7) are
recorded here to map the design space, not to be implemented yet — each needs machinery no
other entry needs, and neither has a behaviour spec. The same applies to `overlong` below.
They earn one when a run actually selects them.

### 3.1 Basis for the ranking

**1. `binary`** — the only shape with a controlled ablation on a setup close to ours.
[arXiv:2605.02944](https://arxiv.org/html/2605.02944) ran Qwen2.5-7B-Instruct with GRPO at 16
rollouts per prompt: binary matched pass-rate at pass@1 (40.6 vs 40.9) and **beat it by 2
points at pass@16** (57.6 vs 55.6). It is also the shape most robust to this dataset's test
noise — AlphaCode's own 46% false-positive-or-slow rate poisons partial credit far more than
it poisons an all-or-nothing check. *Its known weakness is maximum sparsity*, which is why it
ships composed with `extractability` and why dynamic sampling exists as a separate lever.

**2. `code_r1`** — the only implementation in the survey that trains on **this exact task**:
`deepmind/code_contests`, Python, stdin/stdout, single turn. It is binary at heart (`+1.1`
correct versus `+0.1` wrong) with a format penalty bolted on, so it inherits binary's evidence
and builds in a degenerate-group mitigation. Ranked below `binary` only because its specific
`−1.1 / +0.1 / +1.1` spacing was never ablated, and because our default composition of
`binary + extractability` already approximates it with a ratio we chose deliberately.

**3. `verpo`** — the single dense reward demonstrated to *beat* binary
([+1.26 average on Qwen3-8B single-turn, +8.83 on Codeforces multi-turn](https://arxiv.org/html/2601.03525)).
Ranked third rather than first on three counts: it needs Gaussian-KDE density calibration
machinery that nothing else here needs, it was validated on Qwen3-8B rather than Qwen2.5, and
its largest gains came from the multi-turn setting this project does not do. Highest upside,
highest implementation cost.

**4. `binary_threshold`** — *not* less likely to work than `binary`; it is functionally the
same function with a float-comparison guard at 0.99. It inherits binary's evidence intact.
Ranked here because it adds nothing over `binary` unless `pass_rate` accumulates floating-point
error, which with ≤15 integer test counts it will not. Effectively a redundant entry kept for
parity with open-r1.

**5. `pass_rate`** — the shape with direct evidence **against** it in our setup: 2 points worse
at pass@16, and the mechanism is nastier than the number. The same paper found **57.4% of
groups contained both harmful samples with positive advantage and helpful samples with
negative advantage** — partial credit does not merely add noise, it pulls in opposing
directions inside a single group. Kept in the registry because
[SWE-RL](https://arxiv.org/abs/2502.18449) found continuous beat discrete precisely in the
regime where exact match almost never fires, which is arguably a 3B model on CodeContests. It
is the highest-variance option: most likely to fail on the evidence, with a specific reason
that evidence might not transfer.

**6. `ladder`** — this project's own construction, and the only entry with no external
validation at all. It inherits every problem `pass_rate` has (its top rung *is* pass rate) and
adds unablated rung values below it. [arXiv:2605.17174](https://arxiv.org/html/2605.17174)
speaks directly to this shape: composite rewards degraded LiveCodeBench from ~15 to **8.4–10.8**,
because *"easy-to-satisfy proxy terms can dominate optimization and improve the total reward
without commensurate gains in execution-level correctness."*

**7. `hierarchical`** — most machinery, least independent validation: a single paper reporting
on its own benchmark. Its AST-structural-alignment term is a *similarity* proxy rather than a
correctness measure, which is exactly the category the composite-reward study found harmful.
Included for completeness of the design space, not because it is expected to win.

> **This ranking is a prior, and the registry exists because the prior might be wrong.**
> The two best sources point in opposite directions — DeepCoder argues sparse-binary prevents
> reward hacking and ran no ablation; SWE-RL ran a real ablation and found continuous won, but
> on patch similarity rather than test pass rate. Nobody has published the exact comparison
> this project needs. Ranks 5 and 6 are the ones most likely to be revised upward by our own
> measurements, because the all-fail regime a 3B model enters on CodeContests is the specific
> condition under which dense reward is supposed to help.

### Auxiliary rewards — composable, weighted against the primary

| Key | Rule | Range | Weight | Source |
| --- | --- | --- | --- | --- |
| **`extractability`** | `parse_term + fence_term` | [−1, 1] | 0.1 | ADR-0012 |
| `overlong` | `0` / linear decay over cache / `−1` | [−1, 0] | TBD | DAPO ([2503.14476](https://arxiv.org/pdf/2503.14476)) |

**`extractability` in full** (ADR-0012):

```
parse_term = +0.6 if parsed else −0.6
fence_term = +0.4 tagged │ +0.2 untagged │ 0.0 other_tag │ −0.4 none
```

|  | tagged | untagged | other_tag | none |
| --- | --- | --- | --- | --- |
| **parses** | +1.0 | +0.8 | +0.6 | +0.2 |
| **does not parse** | −0.2 | −0.4 | −0.6 | −1.0 |

The parse swing (1.2) deliberately exceeds the fence swing (0.8), so the worst parsing
rollout still outranks the best non-parsing one. A well-fenced broken program must never beat
a bare working one.

**`overlong`** (DAPO), if adopted:

```
R = 0                                     if |y| ≤ L_max − L_cache
  = ((L_max − L_cache) − |y|) / L_cache   if L_max − L_cache < |y| ≤ L_max
  = −1                                    if |y| > L_max
```

Reads `completion_token_count`. Not in the v1 run — `mask_truncated_completions=True` already
prevents truncated programs being scored as wrong answers, which is the main harm.

---

## 4. Composition

```yaml
# config/reward.yaml
functions:
  - name: binary
    weight: 1.0
  - name: extractability
    weight: 0.1
shadow_log:
  - pass_rate
  - binary_threshold
  - ladder
  - code_r1
```

Becomes `reward_funcs` and `reward_weights` on `GRPOConfig`. TRL's default
`multi_objective_aggregation="sum_then_normalize"` sums the weighted terms **before**
normalising within the group, so an auxiliary term genuinely rescues an otherwise-degenerate
group.

### What the weights mean

Weights are **not** a probability distribution and need not sum to 1. GRPO normalises
advantage by the group's standard deviation:

```
A_i = (r_i − mean(r)) / std(r)
```

which is affine-invariant — scaling every reward by a constant leaves advantages unchanged.
Only the **ratio** between terms carries information. `[1.0, 0.1]` and `[10.0, 1.0]` train
identically.

The meaningful quantity is each term's weighted span:

| Term | Raw range | Weight | Weighted span |
| --- | --- | --- | --- |
| `binary` | 0 → 1 | 1.0 | 1.0 |
| `extractability` | −1 → +1 | 0.1 | 0.2 |

So formatting can move total reward by at most one fifth of what correctness can. That is the
design statement; the sum is not.

*(Caveat: under `scale_rewards="none"` — the Dr. GRPO variant — the std division disappears
and absolute scale then affects step size.)*

### Worked example

Group of 4 on a hard problem, all failing every test, so `binary = 0.0` for all four:

| Rollout | fence | parsed | extractability | Total |
| --- | --- | --- | --- | --- |
| A | tagged | ✓ | +1.0 | **0.10** |
| B | tagged | ✗ | −0.2 | **−0.02** |
| C | none (prose) | ✗ | −1.0 | **−0.10** |
| D | none (bare code) | ✓ | +0.2 | **0.02** |

mean = 0.0, std ≈ 0.0748 → advantages ≈ **[+1.34, −0.27, −1.34, +0.27]**

Without the auxiliary term all four rewards are 0.0, the standard deviation is 0, and the
prompt yields **no gradient at all** having cost four full rollouts. That is the entire reason
the term exists.

---

## 5. What is measured but never rewarded

| Signal | Logged as | Why not a reward |
| --- | --- | --- |
| Public pass rate | `diag/public_pass_rate` | Ground truth independent of the reward. Public tests are printed in the problem statement, so partial credit over them is directly hackable — DeepCoder's stated failure mode (ADR-0013) |
| Fence distribution | `format/fence_*` | Diagnostic half of extraction; the reward reads it, but the histogram exists to answer "is formatting the problem?" |
| Parse rate | `format/parsed` | The other half — "is the model failing to write valid Python?" |
| `frac_reward_zero_std` | TRL built-in | Fraction of degenerate groups. Near 1.0 means compute is being burned for nothing |
| Private/generated test mix | `data/generated_share` | Feeds the deferred decision in ADR-0009 |

Public pass rate is the load-bearing one. The reward is a **proxy** for the task, and ADR-0011
makes the proxy deliberately swappable — so a rising reward curve is consistent with both real
improvement and a reward optimising itself. A signal that does not move with the reward is the
only way to tell them apart between full evaluations.

---

## 6. Choosing between them

Comparisons are only meaningful against a metric that is not the reward: **binary pass@1 and
pass@10 on the full test suite**, with seed, filters, group size and LoRA config held fixed.

Each real comparison costs a full training run on one GPU. Budget two or three, not seven.
Shadow logging narrows the field for free — if two shapes produce near-identical curves on the
same rollouts, only one needs a real run.

Start at the top of the §3 ranking and work down. The first real ablation worth spending a
run on is **`binary` versus `pass_rate`** — ranks 1 and 5 — because they sit at opposite ends
of the evidence and the disagreement between DeepCoder's reasoning and SWE-RL's result is
exactly what our all-fail regime would settle.

Watch `frac_reward_zero_std` alongside `diag/public_pass_rate` while doing it. If the first
sits near 1.0, the comparison is measuring nothing and the answer is dynamic sampling or a
difficulty curriculum, not a different reward shape.
