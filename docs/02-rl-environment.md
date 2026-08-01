# Step 2 — RL Environment: Sandbox, Environment, Reward, Loss

**This is the core of the project.** Everything else is plumbing around it.
It has four parts, layered by responsibility:

1. **Sandbox** — safely execute untrusted code against test cases (a swappable
   *executor* the environment owns).
2. **Environment** — the **central unit**. Owns the sandbox + dataset; takes a
   completion (the "action") and produces a structured **`Outcome`** (the
   evidence). Expensive, impure, security-critical, deterministic.
3. **Reward** — **pure, polymorphic** function `Outcome → float`. Cheap,
   swappable, knows nothing about the sandbox. This is where ablations happen.
4. **Loss** — what the RL algorithm computes from rewards (the *library's* job;
   documented so you know what the reward feeds).

This entire doc is **model-free** — build and unit-test everything here on a
laptop with **no GPU**. Package is `posttrain` (`posttrain.sandbox`,
`posttrain.env`, `posttrain.rewards`); tests live under `tests/`; run with
`pytest` in the `post-train` conda env. GPU-touching tests (there are none in
this step) would be marked `@pytest.mark.gpu` and excluded with `-m "not gpu"`.

---

## 2.0 Architecture: environment central, reward polymorphic

The classic RL loop is `action → environment → outcome → reward(outcome)`. We
follow it literally:

```
  RL loop (per completion):

     algorithm ──completion──▶  ENVIRONMENT.evaluate()  ──Outcome──▶  REWARD
         ▲                       (owns sandbox + data)                (pure)
         └────────────────────────── scalar ◀───────────────────────────┘

  ENVIRONMENT  ── uses ──▶  sandbox/executor   +   data (Problem set)
  REWARD (ABC) ── impls ─▶  Binary │ Fractional │ Staged │ Composite
```

### The separation of concerns — what each layer owns

The single most important design rule: **the environment produces evidence; the
reward judges it.** They never overlap.

| Concern | **Environment** owns it | **Reward** owns it |
|---------|:----------------------:|:------------------:|
| Extracting code from the completion | ✅ | |
| Running code in the sandbox | ✅ | |
| Which tests to run (hidden vs public) | ✅ | |
| Test pass/fail, status, timings, length | ✅ (records into `Outcome`) | |
| problem_id → Problem routing, batching | ✅ | |
| dataset choice / prompt construction | ✅ | |
| **Turning evidence into a number** | | ✅ |
| Reward weights / shaping / gating | | ✅ |
| Being swapped for an ablation | (rarely) | ✅ (the point) |

Because the reward is pure over the `Outcome`, you can score the **same** rollout
with a binary reward, a fractional reward, and a staged reward — no re-execution,
identical evidence, no confounds. That is the composability you want.

### Why single-step (bandit), not a gym `step()` loop

TRL's GRPO/PPO are **not** gym environments and our task is a **contextual
bandit**: one prompt → one completion → one terminal reward. There is no state
transition, so the environment exposes `evaluate(completion) -> Outcome`, not
`reset()/step()`.

> **Forward-compatibility:** an *agentic / self-repair* variant (model writes code
> → sees failing tests → revises) **is** a genuine multi-step MDP. The `Outcome`
> contract and the env interface are designed so that variant later becomes a real
> `step()` loop **without changing the reward**. We don't build it now.

---

## Sprints at a glance

Build strictly bottom-up: each sprint's unit tests are green before starting the
next. Everything is CPU-only, no model, no GPU.

| Sprint | Name | Deliverable | Depends on |
|--------|------|-------------|------------|
| **2.1** | Sandbox executor + comparators | `posttrain/sandbox/` (isolated runner, `ExecResult`, comparators) | — |
| **2.2** | `Outcome` contract + `CodeContestEnv.evaluate` | `posttrain/env/outcome.py`, `posttrain/env/code_contest_env.py` | 2.1 |
| **2.3** | Polymorphic reward (ABC + concrete + Composite) | `posttrain/rewards/` (pure `Outcome → float`) | 2.2 (contract only) |
| **2.4** | `TrlRewardBridge` (compose + cache + 2-reward proof) | `posttrain/env/reward_bridge.py` | 2.2, 2.3 |

Reference (no code, after the sprints): **2.5 Loss** — what TRL does with the
scalar the reward returns.

---

## Sprint 2.1 — Sandbox executor + comparators

**Deliverable:** `src/posttrain/sandbox/` — an isolated code runner exposing
`run_code` / `run_against_tests`, the `ExecResult` dataclass, and
`comparators.py` (`exact_match`, `float_match`).

**Depends on:** —

**Build:** A **swappable collaborator**, not the environment itself — its own
ablation axis (firejail → nsjail → remote executor) behind a stable interface.
You are executing **arbitrary model output**: treat it as hostile. Restrict
generation to **Python-only** so there is no compiler/toolchain surface to escape
through.

### Requirements

| Concern | Mechanism |
|---------|-----------|
| Isolation | Run under **firejail** / **bubblewrap** / **nsjail** (not a bare subprocess). No network, private tmp, read-only FS except a scratch dir. |
| Wall-clock timeout | Hard kill after `time_limit` (5–10s). `subprocess` with `timeout=`, then kill the *process group* (child may fork). |
| CPU/memory caps | `resource.setrlimit` in a `preexec_fn`: `RLIMIT_CPU`, `RLIMIT_AS` (1–2 GB), `RLIMIT_NPROC`. |
| Determinism | Fresh temp dir per run; test input on **stdin**; capture **stdout**. |
| No host escape | Restrict to **Python-only** generation → no compiler/toolchain surface. |

### Interface (the sandbox contract)

```python
@dataclass
class ExecResult:
    status: str      # "ok" | "wrong" | "runtime_error" | "timeout"
    stdout: str
    stderr: str
    wall_time: float

def run_code(code: str, stdin: str, *, time_limit=6.0, mem_limit_mb=1024
             ) -> ExecResult: ...

def run_against_tests(code: str, tests: list[TestCase], *, comparator, **limits
                      ) -> list[bool]: ...   # per-test pass/fail
```

The **environment** (2.2) calls this; the **reward** (2.3) never does.

### Comparators (`comparators.py`)

```python
def exact_match(expected: str, got: str) -> bool:      # strip trailing ws per line
def float_match(expected, got, *, rtol=1e-6) -> bool:  # token-wise numeric tol
```
Default to `exact_match` with whitespace normalization; fall back to `float_match`
for numeric-output problems.

### Performance (this runs a *lot*)

The environment invokes the sandbox for **every** completion:
`batch × group_size × steps × tests_per_problem` executions. This dominates
wall-clock if naive.

- **Parallelize** across CPU cores with a `ProcessPoolExecutor` (the 5090 is idle
  during CPU verification — overlap it with the next rollout if possible).
- **Cap tests per problem** (doc 1 already subsamples generated tests).
- Reuse a warm interpreter pool if startup cost dominates.

**Unit tests**

Model-free, no GPU. These are the **security gate** — they must pass before any
model touches the sandbox. `tests/test_sandbox.py`:

- `tests/test_sandbox.py::test_reference_solutions_pass_all_tests` — every doc-1
  **reference solution** run over its tests → `run_against_tests` returns all
  `True` (correctness anchor).
- `tests/test_sandbox.py::test_infinite_loop_times_out` — `while True: pass` →
  `status == "timeout"`, process group killed, no zombie/leftover processes.
- `tests/test_sandbox.py::test_network_blocked` — `import socket; socket.connect(...)`
  → non-`ok` status, connection refused/blocked (no network egress).
- `tests/test_sandbox.py::test_memory_cap_enforced` — a ~10 GB allocation →
  `status == "runtime_error"` via `RLIMIT_AS`, host does **not** OOM.
- `tests/test_sandbox.py::test_fork_bomb_contained` — `os.fork()` bomb → contained
  by `RLIMIT_NPROC`, wall-clock stays bounded, host stays responsive.
- `tests/test_sandbox.py::test_exact_match_normalizes_whitespace` —
  `exact_match("1 2\n3\n", "1 2 \n3")` → `True` (trailing-ws tolerant).
- `tests/test_sandbox.py::test_float_match_within_rtol` —
  `float_match("0.1000001", "0.1", rtol=1e-6)` → `True`; a `1e-3` gap → `False`.

**✅ Verify it works (you run)**

```bash
conda activate post-train
# Security gate — the important one. Run it and read the summary.
pytest tests/test_sandbox.py -v -m "not gpu"

# Eyeball a single hostile run: an infinite loop must be killed, not hang.
python -c "from posttrain.sandbox import run_code; \
print(run_code('while True: pass', '', time_limit=2.0))"
```
Expected: all `test_sandbox.py` cases PASS; the one-liner prints an `ExecResult`
with `status='timeout'` and `wall_time≈2.0` within ~2–3s (never hangs).

---

## Sprint 2.2 — `Outcome` contract + `CodeContestEnv.evaluate`

**Deliverable:** `src/posttrain/env/outcome.py` (the frozen `Outcome` dataclass)
and `src/posttrain/env/code_contest_env.py` (`CodeContestEnv` with
`as_prompt_dataset` and `evaluate`).

**Depends on:** Sprint 2.1 (the sandbox it injects).

**Build:** The environment is the central unit. It owns the dataset + sandbox and
turns a completion into structured evidence. **It contains no scoring logic.**

### The `Outcome` contract (`env/outcome.py`) — the composability keystone

This dataclass is the stable interface between "what happened" and "how good."
Rewards depend **only** on this; extend it (not the reward) when a new reward
needs more evidence.

```python
@dataclass(frozen=True)
class Outcome:
    problem_id: str
    parsed: bool                 # was a ```python block found?
    code: str | None             # extracted code (None if not parsed)
    status: str                  # "no_code"|"compile_error"|"runtime_error"|"timeout"|"ran"
    per_test: list[bool]         # pass/fail for each test run
    n_tests: int
    wall_time: float
    completion_chars: int        # for length-aware rewards
    difficulty: int              # carried through for difficulty-weighted rewards

    @property
    def n_passed(self) -> int:      return sum(self.per_test)
    @property
    def fraction_passed(self) -> float:
        return self.n_passed / self.n_tests if self.n_tests else 0.0
    @property
    def all_passed(self) -> bool:   return self.n_tests > 0 and self.n_passed == self.n_tests
    @property
    def ran(self) -> bool:          return self.status == "ran"
```

### The environment itself (`env/code_contest_env.py`)

```python
class CodeContestEnv:
    def __init__(self, problems, sandbox, tokenizer, *, eval_on="hidden"):
        self._by_id    = {p.id: p for p in problems}
        self._sandbox  = sandbox        # injected executor (swappable)
        self._tok      = tokenizer
        self._eval_on  = eval_on        # score on hidden tests, never public

    # JOB 1 — present problems as prompts; carry problem_id so completions
    #         can be routed back to their problem after rollout.
    def as_prompt_dataset(self) -> Dataset:
        return Dataset.from_list([
            {"prompt": to_chat_prompt(p, self._tok), "problem_id": p.id}
            for p in self._by_id.values()])

    # JOB 2 — the RL "step": action (completion) → Outcome (evidence).
    #         All execution lives here. No scoring.
    def evaluate(self, problem_id: str, completion: str) -> Outcome:
        p    = self._by_id[problem_id]
        code = extract_last_python_block(completion)
        if code is None:
            return Outcome(problem_id, False, None, "no_code", [], 0,
                           0.0, len(completion), p.difficulty)
        tests   = p.hidden() if self._eval_on == "hidden" else p.tests
        per     = self._sandbox.run_against_tests(code, tests,
                                                  comparator=pick_comparator(p))
        status  = classify_status(code, per)   # ran / compile_error / timeout / ...
        return Outcome(problem_id, True, code, status, per, len(tests),
                       wall_time=..., completion_chars=len(completion),
                       difficulty=p.difficulty)
```

`evaluate` is deterministic in `(problem, code)` → **cache it** keyed by
`(problem_id, hash(code))`. Then ablating rewards over the same rollouts is nearly
free, and multiple reward *components* reuse one execution. (2.4 wires the cache.)

**Reward-hacking defense owned here:** scoring runs on **hidden** tests
(`eval_on="hidden"`), never the public example, so matching the public case only
still produces a hidden-fail `Outcome`.

**Unit tests**

Model-free, no GPU. Use a **fake/stub sandbox** for the pure-logic cases so tests
stay fast and hermetic; use the real sandbox only for the determinism check.
`tests/test_env.py`:

- `tests/test_env.py::test_no_code_completion_yields_no_code_outcome` — a
  completion with no ```python block → `Outcome(parsed=False, status="no_code",
  n_tests=0)`.
- `tests/test_env.py::test_reference_solution_all_passed` — reference solution →
  `outcome.all_passed is True` and `status == "ran"` (env + real sandbox agree).
- `tests/test_env.py::test_extracts_last_python_block` — completion with prose +
  code → `outcome.code` equals the fenced block.
- `tests/test_env.py::test_evaluate_scores_on_hidden_tests` — code that matches
  only the public example → `all_passed is False` (proves `eval_on="hidden"`).
- `tests/test_env.py::test_evaluate_is_deterministic_and_cacheable` — two
  `evaluate(pid, code)` calls → identical `Outcome` field-for-field (same
  `per_test`), so `(problem_id, hash(code))` is a valid cache key.
- `tests/test_env.py::test_outcome_derived_properties` — hand-built `Outcome` with
  `per_test=[T,T,F,F]` → `fraction_passed == 0.5`, `all_passed is False`.
- `tests/test_env.py::test_outcome_is_frozen` — assigning to an `Outcome` field
  raises `FrozenInstanceError` (evidence is immutable).

**✅ Verify it works (you run)**

```bash
conda activate post-train
pytest tests/test_env.py -v -m "not gpu"

# Prove determinism by hand: same (problem, code) → identical Outcome twice.
python -c "
from posttrain.env import CodeContestEnv
from posttrain.sandbox import Sandbox
env = CodeContestEnv.demo()          # tiny built-in fixture, no model
code = '\`\`\`python\nimport sys; print(sum(map(int, sys.stdin.read().split())))\n\`\`\`'
a = env.evaluate('demo-1', code); b = env.evaluate('demo-1', code)
print('deterministic:', a == b, '| all_passed:', a.all_passed)"
```
Expected: all `test_env.py` cases PASS; the one-liner prints
`deterministic: True | all_passed: True` (or `False` if the demo code is wrong,
but always `deterministic: True`).

---

## Sprint 2.3 — Polymorphic reward (ABC + concrete + Composite)

**Deliverable:** `src/posttrain/rewards/` — `base.py` (the `RewardFunction` ABC)
and `code_reward.py` (`BinaryReward`, `FractionalReward`, `StagedReward`,
`CompositeReward`, `LengthPenalty`).

**Depends on:** Sprint 2.2 — but only the **`Outcome` contract**, never the
sandbox or env. Rewards are pure and can be unit-tested against hand-built
`Outcome`s with no execution at all.

**Build:** The reward is where ablations happen. It is an **abstract base class
with concrete instantiations**, each a pure `Outcome → float`. It never touches
the sandbox, the dataset, or a completion string.

### ABC (`rewards/base.py`)

```python
class RewardFunction(ABC):
    @abstractmethod
    def score(self, outcome: Outcome) -> float: ...
```

### Concrete rewards (the ablation menu, `rewards/code_reward.py`)

```python
class BinaryReward(RewardFunction):          # sparse baseline
    def score(self, o): return 1.0 if o.all_passed else 0.0

class FractionalReward(RewardFunction):      # dense: fraction of tests passed
    def score(self, o): return o.fraction_passed

class StagedReward(RewardFunction):          # dense + gated (recommended default)
    def score(self, o):
        if not o.parsed:                 return 0.0
        if o.status == "compile_error":  return 0.05
        if not o.ran:                    return 0.05     # timeout / crash
        f = o.fraction_passed
        return 1.0 if f == 1.0 else 0.1 + 0.8 * f

class CompositeReward(RewardFunction):       # weighted sum of sub-rewards
    def __init__(self, terms: list[tuple[float, RewardFunction]]):
        self.terms = terms
    def score(self, o):
        return sum(w * r.score(o) for w, r in self.terms)

class LengthPenalty(RewardFunction):         # composable shaping term
    def __init__(self, target_chars): self.t = target_chars
    def score(self, o): return -max(0, o.completion_chars - self.t) / self.t
```

**`CompositeReward` is the composability payoff:** a reward is a weighted sum of
smaller `RewardFunction`s, so an ablation is "drop a term" or "reweight," not a
rewrite. Config picks which:

```yaml
reward:
  type: composite
  terms:
    - {weight: 0.9,  fn: staged}
    - {weight: 0.0,  fn: length_penalty, target_chars: 2000}  # off by default
```

### Why the default is dense (`StagedReward`), not `BinaryReward`

A **binary** reward is too sparse — early in training almost nothing passes, so
GRPO's group-relative advantage (2.5) collapses (all rewards equal → zero
advantage → no learning). Dense partial credit gives each group *spread*, which
is the entire learning signal. Ablate `binary` vs `fractional` vs `staged` — but
**start on `staged`.**

### Reward-hacking defenses (design constraints on env + reward)

Note these split across the two layers — another reason the split matters:

| Hack | Defense | Lives in |
|------|---------|----------|
| Match the **public** example only | Score on **hidden** tests, never public | **env** (`eval_on="hidden"`) |
| Read expected file / exfiltrate | Sandbox isolation: no network, private FS | **sandbox** |
| Fake success via `exit(0)` | Reward reads `per_test`, never exit code | **env** records / **reward** reads |
| Ramble for length-correlated reward | `LengthPenalty` term + monitor length | **reward** |
| Overfit the K sampled generated tests | Resample hidden subset per epoch | **env / data** |

**Unit tests**

Model-free, no GPU, **no sandbox** — score hand-built `Outcome`s directly, so
these are microsecond-fast and hermetic. `tests/test_reward.py`:

- `tests/test_reward.py::test_staged_reference_solution_scores_one` — `StagedReward`
  on an all-pass `Outcome` → `1.0`.
- `tests/test_reward.py::test_staged_no_code_scores_zero` — `StagedReward` on a
  `parsed=False` `Outcome` → `0.0`.
- `tests/test_reward.py::test_staged_partial_pass_in_midband` — `StagedReward` on a
  50%-pass `Outcome` → `0.1 + 0.8*0.5 == 0.5` (dense mid-band).
- `tests/test_reward.py::test_staged_crash_gets_floor` — `status="timeout"`,
  `parsed=True` → `0.05` (gated floor, not 0).
- `tests/test_reward.py::test_binary_and_fractional_purity` — same `Outcome` scored
  twice returns the identical float; scoring does **not** mutate the `Outcome`
  (purity / no side effects).
- `tests/test_reward.py::test_composite_equals_weighted_sum` — `CompositeReward`
  over terms equals `Σ wᵢ·rᵢ.score(o)` computed independently.
- `tests/test_reward.py::test_length_penalty_only_over_target` — under-target
  `Outcome` → `0.0`; over-target → negative, magnitude `= (chars-t)/t`.
- `tests/test_reward.py::test_public_only_cheat_scores_low` — a hidden-fail
  `Outcome` (public-only cheat) → low reward under `StagedReward` (well below 1.0).

**✅ Verify it works (you run)**

```bash
conda activate post-train
pytest tests/test_reward.py -v -m "not gpu"

# Watch one Outcome score differently under three rewards, no execution:
python -c "
from posttrain.env import Outcome
from posttrain.rewards import BinaryReward, FractionalReward, StagedReward
o = Outcome('p', True, 'x', 'ran', [True,True,False,False], 4, 0.1, 300, 1)
for R in (BinaryReward(), FractionalReward(), StagedReward()):
    print(type(R).__name__, R.score(o))"
```
Expected: all `test_reward.py` cases PASS; the one-liner prints
`BinaryReward 0.0`, `FractionalReward 0.5`, `StagedReward 0.5` — three verdicts
from one identical piece of evidence.

---

## Sprint 2.4 — `TrlRewardBridge` (compose + cache + 2-reward proof)

**Deliverable:** `src/posttrain/env/reward_bridge.py` — `TrlRewardBridge`, the
single callable TRL consumes, plus the caching layer that makes reward ablations
free.

**Depends on:** Sprints 2.2 (`CodeContestEnv`) and 2.3 (`RewardFunction`).

**Build:** TRL wants a single callable
`fn(prompts, completions, **cols) -> list[float]`. The bridge is the **only**
place env and reward meet. It keeps them independent: you vary `reward` for an
ablation while holding `env` fixed.

```python
class TrlRewardBridge:
    def __init__(self, env: CodeContestEnv, reward: RewardFunction):
        self.env, self.reward = env, reward
        self._cache = {}                       # (pid, hash(code)) -> Outcome

    def __call__(self, prompts, completions, problem_id, **kw) -> list[float]:
        out = []
        for c, pid in zip(completions, problem_id):
            key = (pid, hash(c))
            o = self._cache.get(key) or self.env.evaluate(pid, c)
            self._cache[key] = o
            out.append(self.reward.score(o))   # pure scoring over evidence
        return out
```

The eval harness (doc 7) reuses `env.evaluate` + `reward.score` directly — **one
definition of "how a problem is scored," used in training and eval.** For a
reward ablation, run rollouts once, then re-score cached `Outcome`s with each
`RewardFunction` — no re-execution.

**Unit tests**

Model-free, no GPU. Inject a **counting/spy sandbox** so tests can assert the
number of executions directly. `tests/test_bridge.py`:

- `tests/test_bridge.py::test_returns_float_per_completion` — bridge over N
  completions → `list[float]` of length N (TRL contract).
- `tests/test_bridge.py::test_cache_avoids_reexecution` — calling the bridge twice
  on the same `(problem_id, completion)` batch → sandbox execution count
  increments only on the first call (cache hit on the second).
- `tests/test_bridge.py::test_same_rollout_scored_by_two_rewards_no_reexecution` —
  the **composability proof**: run rollouts once through
  `TrlRewardBridge(env, StagedReward())`, then re-score the identical cached
  `Outcome`s with `BinaryReward()` and `FractionalReward()` → three reward
  vectors, and the sandbox executes **zero** additional times for the 2nd and 3rd.
- `tests/test_bridge.py::test_reward_swap_holds_env_fixed` — two bridges sharing
  one `env` but different `RewardFunction`s produce different scores from the same
  `Outcome`s (ablation = swap reward, env unchanged).
- `tests/test_bridge.py::test_bridge_scoring_matches_direct` — bridge output equals
  `[reward.score(env.evaluate(pid, c)) for ...]` computed directly (bridge adds
  caching, not new semantics).

**✅ Verify it works (you run)**

```bash
conda activate post-train
pytest tests/test_bridge.py -v -m "not gpu"

# Composability proof by hand: score one rollout batch with 2 rewards,
# execution count must not rise for the second reward.
python -c "
from posttrain.env import CodeContestEnv, TrlRewardBridge
from posttrain.rewards import StagedReward, BinaryReward
env = CodeContestEnv.demo()
b1 = TrlRewardBridge(env, StagedReward())
prompts, comps, pids = env.demo_batch()
r_staged = b1(prompts, comps, pids)          # executes here
r_binary = [BinaryReward().score(b1._cache[(p, hash(c))]) for c, p in zip(comps, pids)]
print('staged:', r_staged); print('binary:', r_binary)"
```
Expected: all `test_bridge.py` cases PASS; the one-liner prints two reward lists
of equal length from a **single** execution pass — the second reward reuses cached
`Outcome`s and triggers no further sandbox runs.

---

## Definition of done (Step 2)

All CPU-only, model-free. Sprints 2.1→2.4 complete when:

- **Sandbox (2.1)** passes all security + correctness tests (`test_sandbox.py`):
  timeout/mem/network/fork all contained, and every reference solution passes all
  its tests.
- **Environment (2.2)** — `env.evaluate` returns a populated `Outcome`;
  deterministic + cacheable keyed by `(problem_id, hash(code))`; scores on hidden
  tests so the public-only cheat fails.
- **Reward (2.3)** — rewards are pure over `Outcome`; `CompositeReward` composes;
  `test_reward.py` green including the public-only cheat and the purity check.
- **Composability proof (2.4)** — **the same rollout can be scored by ≥2 different
  `RewardFunction`s with no re-execution**, verified by execution count.
- **Bridge (2.4)** — `TrlRewardBridge` returns a `list[float]` matching completion
  count.
- **Still zero model / zero GPU used up to this point.**

## Run all tests for this step

```bash
conda activate post-train
pytest tests/test_sandbox.py tests/test_env.py tests/test_reward.py tests/test_bridge.py \
       -v -m "not gpu"
```
Expected: every case PASS, no GPU touched. This is the full Step 2 gate.

---

## 2.5 Loss (reference — no code here) — what the reward feeds

**You do not implement this** — TRL does (doc 6). It is **not a sprint**; there is
no code to write here. Documented so the reward's role is unambiguous: the reward
returns one scalar per completion, and this is what the algorithm does with it.
Shown for **GRPO** (default); PPO/RLOO in doc 6.

### GRPO in one screen

For each prompt, sample a **group** of G completions `{o_1..o_G}`, get their
rewards `{r_1..r_G}` (from 2.3). GRPO uses **no value network** — baseline is the
group's own mean.

**1. Group-relative advantage:**
```
A_i = (r_i − mean(r_1..r_G)) / (std(r_1..r_G) + ε)
```
This is *exactly* why 2.3 defaults to **dense**: if all G rewards are equal,
`A_i = 0` → zero gradient. Partial credit gives the spread that is the signal.

**2. Policy-gradient loss with PPO-style clipping** (per token t in completion i):
```
ratio_{i,t} = π_θ(o_{i,t}|…) / π_θ_old(o_{i,t}|…)
L_policy = − (1/Σ|o_i|) Σ_i Σ_t  min( ratio·A_i, clip(ratio,1−ε,1+ε)·A_i )
```

**3. KL to a frozen reference** (keeps the policy near base → prevents gibberish):
```
L = L_policy + β · KL(π_θ ‖ π_ref)
```

**You tune:** `G` (8–16), `ε_clip` (~0.2), `β` (~0.04), LR. **TRL owns:** ratios,
clipping, KL, backprop.

### PPO contrast (doc 6)

PPO swaps the group-mean baseline for a **learned value head** + **GAE**, adding a
value-loss term:
```
L_PPO = L_policy(clip) + c_v·L_value + c_kl·KL − c_ent·entropy
```
The critic lowers variance but **doubles memory** — a real cost on 32GB. **GRPO is
the right default here**; PPO is available via the same interface for comparison.
