# Design: verifier and scorer

Stage: **exploring**. Locked-in rules (schema validation, design patterns) are deliberately
not applied. Governed by ADR-0004 (verifier ≠ scorer), ADR-0011 (reward registry),
ADR-0012 (extraction cascade), and the sandbox/grading ADRs 0005–0010.

---

## 1. Module map and seams

```
                    ┌──────────────────────────────────────┐
   TRL GRPOTrainer  │  trl_adapter                         │
   ────────────────▶│  reward_funcs, VerificationCache     │
                    └──────────────┬───────────────────────┘
                                   │ RolloutOutcome
                    ┌──────────────▼───────────┐  ┌────────────────┐
                    │  verifier                │  │  rewards       │
                    │  Verifier.verify_batch   │  │  REWARD_FUNCS  │
                    └──────────────┬───────────┘  └────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
       ┌──────▼──────┐   ┌─────────▼────────┐  ┌────────▼────────┐
       │ extraction  │   │  sandbox         │  │  comparator     │
       │ (pure fn)   │   │  (Protocol)      │  │  (pure fn)      │
       └─────────────┘   └────┬──────┬──────┘  └─────────────────┘
                              │      │
                     ┌────────▼─┐  ┌─▼──────────┐
                     │ Firejail │  │ Subprocess │
                     └──────────┘  └────────────┘
```

**Exactly one real seam: `Sandbox`.** It has two genuine adapters (firejail for training,
plain subprocess for CI and machines without firejail), which is what earns a seam.

**Extraction and comparison are pure functions, not seams.** Each has one implementation and
is deterministic, so injecting them would buy a hypothetical seam and cost a layer of
indirection. They are tested directly. If a second comparator ever appears (it should not —
see ADR-0007), that is when the seam is earned.

**The scorer is not a class.** Reward functions are plain pure functions in a dict. There is
no `Scorer` object because there is no state and no varying collaborator.

---

## 2. Types — the contract crossing the seams

`src/post_training_rl/types.py`. All frozen dataclasses; all fields carry units in the name.

```python
class TestPool(StrEnum):
    PUBLIC = "public"
    PRIVATE = "private"
    GENERATED = "generated"


class TestOutcome(StrEnum):
    PASSED = "passed"
    WRONG_OUTPUT = "wrong_output"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    SKIPPED_AFTER_TIMEOUT = "skipped_after_timeout"   # ADR-0006


class ExtractionTier(StrEnum):
    TAGGED = "tagged"
    UNTAGGED = "untagged"
    ANY = "any"
    ANY_INVALID = "any_invalid"
    BARE = "bare"
    NONE = "none"


@dataclass(frozen=True)
class TestCase:
    input_text: str
    expected_output: str
    pool: TestPool


@dataclass(frozen=True)
class Problem:
    problem_id: str
    description: str
    graded_tests: tuple[TestCase, ...]   # private-first + generated filler, capped (ADR-0009)
    public_tests: tuple[TestCase, ...]   # never feeds the primary reward


@dataclass(frozen=True)
class TestResult:
    test_index: int
    pool: TestPool
    outcome: TestOutcome
    duration_seconds: float
    stdout_was_truncated: bool
    stderr_excerpt: str                  # capped; for debugging a failing rollout only


@dataclass(frozen=True)
class Extraction:
    code: str | None
    tier: ExtractionTier


@dataclass(frozen=True)
class VerificationReport:
    problem_id: str
    extraction: Extraction
    graded_results: tuple[TestResult, ...]
    public_results: tuple[TestResult, ...]


@dataclass(frozen=True)
class RolloutOutcome:
    """A VerificationReport plus facts the verifier cannot know."""
    report: VerificationReport
    completion_token_count: int
    completion_was_truncated: bool       # hit max_completion_length
```

### Why `RolloutOutcome` wraps rather than extends

The verifier receives a string; it has no tokenizer and must not acquire one. But the
`overlong` reward (ADR-0011) needs a token count, and truncation status matters for
interpreting a failure. Those facts belong to the adapter, which has the tokenizer and TRL's
`completion_ids`. Wrapping keeps each type owned by the layer that can actually populate it.

**Reward functions take `RolloutOutcome`, never `VerificationReport`.** One signature for
every entry in the registry is what makes them interchangeable.

---

## 3. `sandbox` — the one real seam

```python
@dataclass(frozen=True)
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int | None                # None when killed by signal
    duration_seconds: float
    timed_out: bool
    stdout_was_truncated: bool


class Sandbox(Protocol):
    def run(self, source: str, stdin_text: str, timeout_seconds: float) -> SandboxResult:
        """Execute `source` as a Python program with `stdin_text` on stdin.

        Raises only on infrastructure failure (sandbox binary missing, temp dir
        unwritable). A program that crashes, hangs, or floods output is a normal
        result, not an exception.
        """
```

The sandbox takes **source text, not a path**, and owns its own temp directory lifecycle.
Callers never learn that files are involved — that is the depth.

### `FirejailSandbox`

Flags per ADR-0005, all sourced from config, none hardcoded:

```
firejail --quiet --private --noprofile
         --seccomp=socket
         --rlimit-nproc=32 --rlimit-nofile=32
         --rlimit-fsize=2m --rlimit-as=4g
         --timeout=00:00:10
         --whitelist=<tmpdir>
         python3 <tmpdir>/solution.py
```

Implementation obligations:

- **Cap stdout in the parent.** `--rlimit-fsize` does not apply to pipes, and Linux pipe
  capacity is 64 KiB, so an unbounded reader is how a runaway `print` loop OOMs the *training
  process*. Read at most `stdout_cap_bytes`, set `stdout_was_truncated`, kill the child, and
  compare the truncated output anyway (ADR-0009).
- **Use `subprocess.run(..., timeout=)`, never `Popen.communicate(timeout=)`** — the latter
  does not kill the child on expiry.
- **Belt and braces on the timeout.** firejail's `--timeout` has 1-second granularity; the
  Python-side timeout is set slightly higher so firejail wins normally and Python is the
  backstop if firejail itself wedges.
- **Prepend the determinism preamble** (ADR-0008) and record the line offset, so a traceback
  in `stderr_excerpt` can be mapped back to the model's own line numbers.
- `PYTHONHASHSEED` is passed through the environment, which firejail preserves.

### `SubprocessSandbox`

Same interface, `subprocess.run` with `RLIMIT_*` set via `preexec_fn`. Weaker: no network
block, no private filesystem. Exists for CI and for developing on a machine without firejail.

**Guard against silent downgrade:** `backend` is an explicit config value with no default. If
it says `firejail` and the binary is absent, startup raises naming the missing binary — it
never falls back.

### Startup self-test

Before the first training step, run four known-hostile programs through the configured
sandbox and assert containment: an infinite loop (must time out), a fork bomb (must be capped
by `--rlimit-nproc`), a socket connection (must fail under `--seccomp=socket`), and a
runaway `print` loop (must truncate without exhausting parent memory).

This is the difference between a misconfigured sandbox and one that looks like the model
producing wrong answers. It costs about four seconds once per run.

---

## 4. `extraction` — pure, per ADR-0012

```python
def extract_python(completion: str, prefill: str = "") -> Extraction:
    """Recover executable Python from a completion via a syntax-gated cascade."""
```

Tier order: `tagged` → `untagged` → `any` → `bare` → `none`, taking the **last**
syntactically valid candidate at the first tier that yields one, gated by `ast.parse`.
A tier that has candidates but none that parse returns `any_invalid`.

`prefill` is prepended before matching. It defaults to `""`, and ADR-0012 keeps prefill off
by default — but the parameter exists because forgetting to re-prepend it is documented as
"the single most likely way to get this wrong and silently score zero on everything." Making
it a parameter of the only function that could need it means the mistake is visible at the
call site.

Never falls back to executing the whole completion unguarded — the `bare` tier is
syntax-gated precisely so prose fails it.

---

## 5. `comparator` — pure, per ADR-0007

```python
def outputs_match(actual: str, expected: str) -> bool:
    """Token comparison with CodeContests semantics."""
```

Split both on any whitespace, drop empties, require equal token counts, compare
case-insensitively, and apply a 1e-5 **absolute** tolerance when either token parses as a
float. Two inherited sharp edges, both documented in ADR-0007 and worth a comment at the
tolerance constant: the tolerance is absolute rather than relative, and 64-bit integers fall
through to the float path.

---

## 6. `verifier` — the deep module

```python
class Verifier:
    def __init__(self, sandbox: Sandbox, config: VerifierConfig) -> None: ...

    def verify_batch(
        self, items: Sequence[tuple[str, Problem]]
    ) -> list[VerificationReport]: ...
```

**One public method.** Behind it: extraction, preamble injection, per-test execution, the
abort-on-timeout rule, output capping, comparison, and result assembly. `_verify_one` is
private; tests exercise `verify_batch` with a single item, because the interface is the test
surface.

Only `verify_batch` is exposed because the training loop is always batched, and a batch is
also where parallelism lives.

### Per-rollout algorithm

1. `extract_python(completion, prefill)`. If tier is `none`, return a report with no results —
   **no sandbox invocation at all**. This is the cheap early exit and it matters: it is the
   most common failure early in training.
2. Prepend the determinism preamble.
3. For each test in `problem.graded_tests`, in order:
   - `sandbox.run(source, test.input_text, timeout_seconds)`
   - Map to a `TestOutcome`: non-zero exit or signal → `RUNTIME_ERROR`; `timed_out` →
     `TIMEOUT`; otherwise `outputs_match` → `PASSED` or `WRONG_OUTPUT`.
   - **On `TIMEOUT`, stop.** Remaining tests are recorded as `SKIPPED_AFTER_TIMEOUT`
     (ADR-0006). They are recorded rather than omitted so the count stays stable — a reward
     function must not have to ask why a rollout has fewer results than the problem has tests.
4. Public tests run in the same way, into `public_results`.

### Parallelism

`verify_batch` fans out over a `ThreadPoolExecutor`. Threads, not processes: the work is
`subprocess.run`, which releases the GIL while waiting, so there is nothing to gain from
pickling across processes. Worker count comes from config.

**Deliberately deferred:** the source is written to a temp file once per test rather than
once per rollout. At 15 tests that is 15 small writes where 1 would do — but the cost is
microseconds against a 10-second timeout budget, and the standards require a measurement
before a performance change. If profiling later shows it matters, the fix is a
`run_many(source, stdin_texts)` method on `Sandbox` returning an iterator, which also keeps
the abort-on-timeout logic in the caller.

### Error contract

Stated on the class, because it is the single most important thing a caller must know:

> **Raises only on infrastructure failure.** A solution that crashes, hangs, floods output,
> or produces no parseable code is *data* — it becomes a `TestOutcome` or an
> `ExtractionTier`, never an exception. A missing sandbox binary, an unwritable temp
> directory, or a malformed `Problem` raises, because those are systematic and a run that
> continues past them produces meaningless rewards for every subsequent step.

This is the "errors surface at a boundary" rule applied honestly: the boundary is the
training step, and the distinction that makes it work is between *expected failure of the
subject* and *failure of the apparatus*.

---

## 7. `rewards` — the registry, per ADR-0011

```python
RewardFn = Callable[[RolloutOutcome], float]

def binary_reward(outcome: RolloutOutcome) -> float: ...
def pass_rate_reward(outcome: RolloutOutcome) -> float: ...
def binary_threshold_reward(outcome: RolloutOutcome) -> float: ...
def ladder_reward(outcome: RolloutOutcome) -> float: ...
def code_r1_reward(outcome: RolloutOutcome) -> float: ...
def extractability_reward(outcome: RolloutOutcome) -> float: ...

REWARD_FUNCTIONS: dict[str, RewardFn] = {
    "binary": binary_reward,
    ...
}
```

Every entry is a pure function with the same signature — that is what makes them
interchangeable, and it is why they take `RolloutOutcome` rather than a bag of arguments.
Each carries its source in its docstring, matching ADR-0011's table.

**No mode flag.** There is no `score(outcome, kind="binary")`; selection is a dict lookup
against a config string. A mode flag would be several functions wearing one name.

**Parameterised rewards are deferred.** `overlong` needs `L_max`/`L_cache`, and an annealed
`ladder` needs a step counter, so both need factories rather than bare functions. Neither is
in the v1 run. Introducing a `Callable[[Mapping], RewardFn]` factory layer now would be
speculative generality for one hypothetical caller; when the second parameterised reward is
actually needed, that is the case that earns it. `ladder` ships unannealed.

`SKIPPED_AFTER_TIMEOUT` counts as not-passed everywhere. Stated once here so every reward
function does not have to re-decide it.

---

## 8. `trl_adapter` — the glue

TRL's contract: each entry in `reward_funcs` is called with the batch's completions plus every
extra dataset column as a kwarg, and returns `list[float]`.

The problem: TRL calls each reward function separately, but ADR-0004 requires **one execution
feeding all of them**.

```python
class VerificationCache:
    def __init__(self, verifier: Verifier) -> None: ...
    def outcomes(
        self, completions: Sequence[str], problems: Sequence[Problem],
        completion_token_counts: Sequence[int],
    ) -> list[RolloutOutcome]: ...
    def reset(self) -> None: ...
```

Keyed on `(problem_id, completion_text)`. The first reward function to be called for a batch
triggers verification; the rest hit the cache. `reset()` is called from a TRL step-end
callback, with a size cap as a backstop.

**Determinism makes the cache sound.** Two rollouts with identical text for the same problem
*must* verify identically (ADR-0008), so sharing an entry is correct rather than merely
convenient. Without ADR-0008 this cache would be a bug.

```python
def build_reward_functions(
    verifier: Verifier, config: RewardConfig,
) -> tuple[list[Callable], list[float]]:
    """Returns (reward_funcs, reward_weights) ready for GRPOConfig."""
```

Shadow logging (ADR-0011's "log every registered reward") is a single extra reward function
appended with **weight 0.0**: it computes every name in `config.shadow_log` from the cached
outcomes, emits them via TRL's `log_metric` kwarg, and returns zeros. Weight zero means it
cannot affect training, and TRL's `sum_then_normalize` leaves the total unchanged.

The tier histogram (`format/frac_none`, `format/frac_tagged`, …) is emitted the same way.
That histogram is the measurement ADR-0012 says nobody has published.

### Serialisation boundary

`datasets` stores Arrow, not Python objects, so `Problem` cannot live in a dataset column.
Columns hold plain dicts and lists; the adapter reconstructs frozen dataclasses on the way
in. This conversion is the one place where the type discipline is enforced at runtime rather
than statically, so it asserts shape explicitly and names the offending problem id on
failure.

---

## 9. `dataset`

```python
def build_dataset(config: DatasetConfig, tokenizer: PreTrainedTokenizerBase) -> Dataset:
```

Emits columns `prompt`, `problem_id`, `graded_tests`, `public_tests`.

Obligations, each traceable to a source:

- **Load from `data/` on `main`.** The datasets-server view is silently truncated to 28% of
  train.
- **Decode `difficulty` with the proto mapping, never `ClassLabel.int2str()`**, which is wrong
  for every value ≥ 19. `source` uses the HF mapping, which does *not* match the proto.
- **Filter** per ADR-0010: multiple-output patterns, interactive, file-I/O
  (`input_file` non-empty), over-length prompts. Every filter logs its drop count and the
  total is reported — a filter that silently removes half the corpus must be visible.
- **Select tests** per ADR-0009: private-first, longest-input-first, generated as filler, cap
  15, drop problems under 5.
- Prompt template lives in YAML as a block scalar, because it is a value a run depends on and
  two runs with different prompts must be diffable.

---

## 10. Configuration

Plain attribute extraction, no schema — the stage rule is explicit that a schema is premature
while exploring. Required keys are read directly so a missing one raises immediately; no
`.get()` defaults papering over absence.

```
config/
├── model/qwen2.5-3b.yaml      # bf16 + LoRA, generation_backend: continuous_batching
├── model/qwen2.5-7b.yaml      # nf4 block, quantization explicit
├── verifier.yaml              # sandbox backend + limits, determinism.seed, test caps
├── reward.yaml                # functions + weights, shadow_log list
├── dataset.yaml               # paths, filter patterns, prompt template
└── training.yaml              # GRPOConfig fields
```

`verifier.yaml` sketch:

```yaml
sandbox:
  backend: firejail            # firejail | subprocess — no default, must be explicit
  timeout_seconds: 10.0
  memory_limit_gib: 4
  max_processes: 32
  max_open_files: 32
  max_file_size_mib: 2
  stdout_cap_bytes: 10485760
  stderr_excerpt_bytes: 2048
  worker_threads: 12
determinism:
  seed: 0                      # null disables preamble injection
tests:
  max_tests_per_rollout: 15
  min_tests_required: 5
extraction:
  prefill: ""                  # "" = no prefill (ADR-0012)
```

Every number that appears in a branch or a limit is here. Nothing is a literal in code.

---

## 11. Testability

The seams give three levels, and none of them require patching:

| Level | How | Covers |
| --- | --- | --- |
| Pure functions | Call directly | Extraction cascade tiers, comparator semantics, every reward function |
| Verifier | Inject a `FakeSandbox` returning scripted `SandboxResult`s | Timeout abort, outcome mapping, result assembly — with no subprocesses at all, so it is fast and deterministic |
| Sandbox adapters | Real execution of small fixture programs | Containment: timeout, fork bomb, network, output flood |

`FakeSandbox` is the third adapter at the `Sandbox` seam and the reason the seam pays for
itself: verifier tests need no firejail, no temp files, and no wall-clock waiting.

---

## 12. Known gaps

Carried into the sprint plan rather than guessed at here.

1. **TRL's exact reward-function signature and kwarg-forwarding behaviour must be read from
   the installed version** before the adapter is written. The design assumes extra dataset
   columns arrive as kwargs and that `log_metric` is available.
2. **Arrow round-tripping of nested test structures** is assumed to work and is unverified.
3. **`private_tests` count distribution is unmeasured** (flagged in ADR-0009). If most
   problems carry fewer than 15, generated filler does more work than intended.
4. **Prompt token-length distribution is unmeasured**, so `max_prompt_length` and the
   resulting drop rate are unknown.
5. **Format-failure rate is unmeasured**, and it decides whether `extractability` is worth
   keeping at all.
6. **Sandbox throughput is unmeasured.** firejail startup × 15 tests × group size may make
   `ThreadPoolExecutor` insufficient; the fallback is a process pool or a persistent worker.
7. **Continuous batching with `Linear4bit`** is untested upstream and only bites at 7B.
8. **Dynamic sampling** (ADR-0011) would wrap generation via `rollout_func`. Nothing here
   precludes it — the cache is keyed per rollout, not per step — but it is not designed yet.
