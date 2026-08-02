# Sprint 1 — A verifier and scorer that can be trusted

**Objective:** grade a rollout correctly, in isolation, with no model involved.

No GPU, no checkpoint, no training loop. Everything here is a pure function or subprocess
work, which means all of it can be verified before anything expensive runs. This is the part
where a silent bug corrupts every number downstream — a comparator that requires exact string
matches scores correct solutions as failures, and nothing about that looks like a bug from the
outside.

**Module stage: exploring.** Locked-in rules do not apply — no config schema validation
(plain attribute extraction), no design patterns reached for in anticipation. See
`CODING_STANDARDS.md` §Stage.

---

## Working agreement

**Test-driven, vertical slices.** For each task: write one failing test for one behaviour,
make it pass, move to the next. Do **not** write a task's whole test list and then implement
against it — that verifies imagined behaviour and commits to test structure before the
implementation is understood.

**Only the tests listed here get written.** Each task names its tests exactly. If a behaviour
seems to need a test that is not listed, stop and say so rather than adding it.

**Pre-authorised dependencies.** These may be added without stopping to ask:

```
pytest, pytest-timeout          # test runner; timeout guards the sandbox suite
pyyaml                          # config loading
```

Everything else — including `trl`, `peft`, `datasets`, `transformers` — is **out of scope for
this sprint** and must not be imported. If a task appears to need one, that is a signal the
task is wrong.

**Three suites, separated by marker.** `pyproject.toml` registers both markers and excludes
them by default, so the command you run on every red-green cycle stays sub-second and spawns
nothing:

```toml
[tool.pytest.ini_options]
markers = [
    "subprocess_backend: spawns real subprocesses",
    "containment: requires firejail",
]
addopts = '-m "not containment and not subprocess_backend"'
```

```bash
conda activate post-train
pytest -q                         # 63 unit tests, no subprocess, sub-second
pytest -q -m subprocess_backend   # 4 tests, seconds
pytest -q -m containment          # 9 tests, requires firejail
```

**Tests pass a short `timeout_seconds` explicitly — 1.0, never the production 10.0.** One
timeout test at the production value would dominate the suite it lives in. 1.0 s is the floor,
because firejail's `--timeout` has one-second granularity.

**Not tested in this sprint, deliberately** (confirmed seam set, `behavior.md`): config
loading, log emission, and `SubprocessSandbox`'s hostile cases. Do not add tests for these.

---

## User stories

1. **As the training loop, I need output comparison to follow CodeContests semantics**, so a
   correct solution is never scored wrong for printing `YES` instead of `yes` or a trailing
   newline.
2. **As the training loop, I need code recovered from a completion with fence and parse
   recorded separately**, so I can later tell "the model cannot format" from "the model cannot
   write valid Python" — two failures with opposite fixes.
3. **As the machine this runs on, I need generated code contained**, so an infinite loop, a
   fork bomb, or a runaway `print` cannot take down the training run or the host.
4. **As any reward function, I need a rollout graded into a structured report**, so I can score
   it without knowing anything about subprocesses.
5. **As a researcher, I need interchangeable reward functions over one shared outcome type**,
   so reward shapes can be compared without re-executing the code.

---

## Task 1 — Scaffold, types, and config loading

### Behaviour

Every tunable value is read from YAML. A required key that is absent fails immediately and
names both the key and the file it was missing from. Nothing is defaulted into existence.

### Files

```
pyproject.toml
src/post_training_rl/__init__.py
src/post_training_rl/types.py
src/post_training_rl/config.py
config/verifier.yaml
config/reward.yaml
```

### Public interface

`types.py` — frozen dataclasses and enums exactly as specified in
[`verifier-scorer.md`](../design/verifier-scorer.md) **§2**: `TestPool`, `TestOutcome`,
`Fence`, `TestCase`, `Problem`, `TestResult`, `Extraction`, `VerificationReport`,
`RolloutOutcome` — plus `SandboxResult`, which is specified in **§3** alongside the `Sandbox`
protocol but lives here with the other types.

```python
def load_verifier_config(path: Path) -> VerifierConfig: ...
def load_reward_config(path: Path) -> RewardConfig: ...
```

Both read named attributes directly. **No Pydantic, no schema** — the attribute set is still
moving and the stage rule defers validation until it stops.

`config/verifier.yaml` carries every constant the sprint needs: sandbox backend, all rlimits,
`timeout_seconds`, `stdout_cap_bytes`, `stderr_excerpt_bytes`, `worker_threads`,
`determinism.seed`, `tests.max_tests_per_rollout`, `tests.min_tests_required`,
`extraction.prefill`. No literal governing behaviour appears anywhere in code.

**Only two of the six config files in [`verifier-scorer.md` §10](../design/verifier-scorer.md)
are created here.** `dataset.yaml` arrives in sprint 2; `training.yaml` and `model/*.yaml` in
sprint 3. The design is unchanged — this sprint simply needs a subset.

### Tests

**None.** Config loading is outside the confirmed test surface — a test would assert that
Python subscripting raises on a missing key. Types are data with no behaviour.

### Done when

`pytest -q` collects zero tests and exits clean; `config/verifier.yaml` contains every
constant later tasks reference; no module imports anything outside the standard library and
`pyyaml`.

---

## Task 2 — Comparator

### Behaviour

Splits both sides on any whitespace and discards empty tokens, so line structure is
irrelevant. Compares token by token, case-insensitively. Treats a pair as numeric when either
side parses as a float and accepts within 1e-5 **absolute**. Token counts must be equal.

Two sharp edges are inherited from DeepMind's reference implementation and preserved
deliberately: the tolerance is absolute rather than relative, and values beyond 32-bit integer
range fall through to the float path.

Source of truth: [`behavior.md`](../design/behavior.md) §1, [ADR-0007](../adr/0007-codecontests-token-comparator.md).

### Files

`src/post_training_rl/comparator.py` — `def outputs_match(actual: str, expected: str) -> bool`

### Unit tests

| Test | Asserts |
| --- | --- |
| `test_identical_output_matches` | Baseline |
| `test_trailing_newline_ignored` | `"3\n"` vs `"3"` |
| `test_line_structure_ignored` | `"1 2\n3"` vs `"1\n2 3"` — tokens, not lines |
| `test_comparison_is_case_insensitive` | `"YES"` vs `"yes"` |
| `test_extra_token_fails` | `"1 2 3"` vs `"1 2"` |
| `test_short_output_does_not_match_long_expected` | `"1"` vs `"1 2 3"` — pins the `zip`-truncation bug other implementations have |
| `test_float_within_tolerance_matches` | `"1.000001"` vs `"1.0"` |
| `test_float_outside_tolerance_fails` | `"1.001"` vs `"1.0"` |
| `test_integer_and_float_forms_match` | `"1"` vs `"1.0"` |
| `test_large_magnitude_uses_absolute_tolerance` | Pins the known sharp edge so a change is visible |
| `test_empty_matches_empty` | |
| `test_empty_does_not_match_nonempty` | |

Expected values come from the CodeContests semantics above, never from re-deriving what the
implementation does.

### Done when

All twelve pass. The 1e-5 constant is named and carries a comment explaining that it is
absolute, not relative.

---

## Task 3 — Extraction

### Behaviour

Recovers executable Python via a syntax-gated cascade and records **two independent facts**:
which fence the code arrived in, and whether it parses. They are never collapsed, because a
flawless fence can wrap broken code and correct code can arrive unfenced.

Cascade: tagged → untagged → other-tag → bare, taking the **last** syntactically valid
candidate at the first tier yielding one, gated by `ast.parse`. Prose is never returned as
code. It never falls back to the whole completion unguarded.

Source of truth: [`behavior.md`](../design/behavior.md) §2, [ADR-0012](../adr/0012-syntax-gated-extraction.md).

### Files

`src/post_training_rl/extraction.py` —
`def extract_python(completion: str, prefill: str = "") -> Extraction`

### Unit tests

| Test | Asserts |
| --- | --- |
| `test_python_tagged_fence_extracted` | `fence=TAGGED, parsed=True` |
| `test_last_tagged_fence_wins` | Two blocks; second returned |
| `test_untagged_fence_extracted_when_no_tagged` | `fence=UNTAGGED` |
| `test_tagged_fence_preferred_over_untagged` | Both present; tagged wins |
| `test_language_tag_aliases_accepted` | `py`, `python3`, `py3` |
| `test_non_python_tag_reports_other_tag` | ` ```cpp ` → `fence=OTHER_TAG` |
| `test_unparseable_code_reports_parsed_false_with_its_fence` | A ```` ```python ```` block with a syntax error → `fence=TAGGED, parsed=False`. **The conflation this design exists to prevent** |
| `test_syntax_gate_prefers_earlier_valid_block` | Last block invalid, earlier parses → earlier wins |
| `test_unterminated_fence_extracts_to_end` | The truncation case |
| `test_trailing_space_after_fence_marker_tolerated` | |
| `test_indented_closing_fence_tolerated` | |
| `test_bare_valid_python_without_fence` | `fence=NONE, parsed=True` |
| `test_prose_without_fence_reports_no_code` | `code is None, fence=NONE` — prose must fail the syntax gate |
| `test_empty_completion_reports_no_code` | |
| `test_prefill_is_reprepended` | Completion opening mid-block extracts correctly when `prefill` supplied |
| `test_missing_prefill_reports_no_code` | Same completion **without** `prefill` → no code. Pins the documented silent-zero trap |

### Done when

All sixteen pass, and `test_unparseable_code_reports_parsed_false_with_its_fence` demonstrably
fails against a single-tier implementation.

---

## Task 4 — Sandbox seam: protocol, fake, subprocess

### Behaviour

A sandbox accepts program **source text** — not a path — delivers stdin, captures stdout and
stderr, and reports duration, exit status, timeout, and truncation. Callers never learn that
temporary files are involved.

It sets `PYTHONHASHSEED` in the child's environment, so `set` and `dict` iteration order is
stable across runs. **It does not modify the source it was given** — seeding the *program* is
the verifier's half of ADR-0008 (task 6). The sandbox's contract is "run exactly this", and
the startup self-test depends on that: it runs hostile programs through this same sandbox and
must get them unmodified.

Infrastructure failures raise. A program that crashes, hangs, or floods is a normal result.

Source of truth: [`behavior.md`](../design/behavior.md) §4 items 1–3, 9, 11;
[ADR-0008](../adr/0008-deterministic-execution.md).

### Files

```
src/post_training_rl/sandbox/__init__.py     # Sandbox Protocol
src/post_training_rl/sandbox/fake.py         # FakeSandbox — scripted results + call log
src/post_training_rl/sandbox/subprocess_.py  # SubprocessSandbox
tests/test_sandbox_subprocess.py
```

### Public interface

```python
class Sandbox(Protocol):
    def run(self, source: str, stdin_text: str, timeout_seconds: float) -> SandboxResult: ...
```

`FakeSandbox` is a **test fixture** and lives in `src/`, not `tests/`, because the verifier's
tests and the startup self-test both use it. It records every call so tests can assert the
sandbox was *not* invoked.

### Unit tests

**None.** The `Sandbox` protocol is a type declaration, and `FakeSandbox` is a fixture
exercised throughout task 6. There is nothing here to assert that is not covered by the
integration tests below.

### Integration tests — marked `subprocess_backend`, in `tests/test_sandbox_subprocess.py`

| Test | Asserts |
| --- | --- |
| `test_program_output_is_captured` | Echo round-trips |
| `test_stdin_is_delivered` | |
| `test_infinite_loop_times_out` | `timed_out=True` within timeout + slack |
| `test_hash_seed_is_deterministic` | A program printing `hash("x")` gives the same value across two separate runs |

**No hostile-containment tests here.** `SubprocessSandbox` is documented as the weaker adapter
(ADR-0005); testing it as a security boundary would imply a guarantee it does not make.

### Done when

All four integration tests pass. `Sandbox` has exactly one method, and nothing in
`sandbox/` rewrites the source it is handed.

These four names are **reused verbatim in task 5** for the firejail adapter. They live in
separate modules — `tests/test_sandbox_subprocess.py` and `tests/test_sandbox_firejail.py` —
because two same-named tests in one module means the second silently shadows the first, and a
test that quietly stops existing is worse than one that fails.

---

## Task 5 — Firejail sandbox and containment

### Behaviour

Every containment guarantee in [`behavior.md`](../design/behavior.md) §4 items 4–8, 10, 12:

- A program that loops forever is killed and reported as timed out.
- A program that spawns processes without limit is capped and cannot affect the host.
- A program that opens a network connection fails.
- A program that writes output without limit has it truncated, and the truncation never
  exhausts memory in the **calling** process. Pipe capacity is 64 KiB and `--rlimit-fsize`
  does not apply to pipes, so the cap is enforced by the reader.
- A program that writes files without limit is capped.
- When the configured backend's binary is absent, startup fails naming it. It **never**
  silently downgrades.

Source of truth: [ADR-0005](../adr/0005-firejail-sandbox.md),
[ADR-0006](../adr/0006-flat-timeout-ignore-dataset-limit.md).

### Files

```
src/post_training_rl/sandbox/firejail.py
tests/test_sandbox_firejail.py        # separate module — see task 4's done-when
tests/fixtures/hostile_programs.py    # shared with task 8
```

### Implementation constraints

- Flags come from config, never hardcoded: `--private`, `--seccomp=socket`,
  `--rlimit-nproc`, `--rlimit-nofile`, `--rlimit-fsize`, `--rlimit-as`, `--timeout`.
- Use `subprocess.run(..., timeout=)`. **Never** `Popen.communicate(timeout=)` — it does not
  kill the child on expiry.
- The Python-side timeout is set above firejail's so firejail wins normally and Python is the
  backstop. `timed_out` is true if either fires. A child killed by a signal for any other
  reason reports `exit_code=None, timed_out=False` and becomes `RUNTIME_ERROR` — "hung" and
  "crashed" never collapse into one outcome.
- Read at most `stdout_cap_bytes`, then kill the child and set `stdout_was_truncated`.

### Integration tests — marked `containment`, gate CI

| Test | Asserts |
| --- | --- |
| `test_program_output_is_captured` | Echo round-trips under firejail |
| `test_stdin_is_delivered` | |
| `test_infinite_loop_times_out` | |
| `test_fork_bomb_is_contained` | Process cap holds; host unaffected |
| `test_network_access_is_blocked` | Socket connect fails under `--seccomp=socket` |
| `test_output_flood_is_truncated` | `stdout_was_truncated=True`; parent memory bounded |
| `test_file_write_beyond_limit_is_contained` | |
| `test_hash_seed_is_deterministic` | |
| `test_missing_firejail_binary_raises_naming_it` | Error message contains the binary name |

When firejail is absent these **skip loudly** with an explicit message. A containment test
that passes vacuously is worse than no test — it is false assurance about the one thing
protecting the machine.

CI installs firejail (`ppa:deki/firejail`) and runs this suite on every push.

### Done when

All nine pass in CI with firejail installed, and skip with a visible message without it.

---

## Task 6 — Verifier

### Behaviour

Extracts code, prepends the preamble, runs each graded test in order, classifies each result,
runs public tests separately, and processes a batch concurrently returning reports in input
order.

The guarantees that matter:

- **No extractable code ⇒ no execution at all.** The most common early outcome must cost
  nothing.
- **A timeout abandons the remaining tests**, which are still *recorded* as skipped, so the
  result count always equals the test count.
- **Truncated output is compared, not auto-failed.**
- **It assigns no rewards** and holds no opinion about what an outcome is worth.
- **It raises only on infrastructure failure.**

It also owns the **seeding preamble** — the verifier's half of ADR-0008. The sandbox fixes the
environment (`PYTHONHASHSEED`, task 4); the verifier seeds the program. Neither may assume the
other does it, because if both do the preamble appears twice, and if neither does determinism
vanishes with no symptom except reward noise inside groups.

Source of truth: [`behavior.md`](../design/behavior.md) §5,
[ADR-0004](../adr/0004-verifier-scorer-split.md),
[ADR-0008](../adr/0008-deterministic-execution.md).

### Files

```
src/post_training_rl/verifier.py      # Verifier, and the preamble builder it owns
tests/test_verifier.py
```

```python
class Verifier:
    def __init__(self, sandbox: Sandbox, config: VerifierConfig) -> None: ...
    def verify_batch(self, items: Sequence[tuple[str, Problem]]) -> list[VerificationReport]: ...
```

One public method. `_verify_one` is private; tests exercise `verify_batch` with a single item,
because the interface is the test surface. Fan out over a `ThreadPoolExecutor` — the work is
`subprocess.run`, which releases the GIL, so there is nothing to gain from processes.

### Unit tests — `FakeSandbox`, no subprocesses

| Test | Asserts |
| --- | --- |
| `test_no_extractable_code_skips_execution_entirely` | Sandbox call count is **0** |
| `test_all_passing_tests_report_passed` | |
| `test_mismatched_output_reports_wrong_output` | |
| `test_nonzero_exit_reports_runtime_error` | |
| `test_timeout_aborts_remaining_tests` | Timeout on test 2 of 5 → test 2 `TIMEOUT`, 3–5 `SKIPPED_AFTER_TIMEOUT` |
| `test_result_count_always_matches_test_count` | Even after an abort |
| `test_truncated_stdout_is_still_compared` | |
| `test_public_tests_are_reported_separately` | `public_results` populated, `graded_results` unaffected |
| `test_determinism_preamble_is_prepended` | Inspect the source `FakeSandbox` received |
| `test_preamble_seeds_random_and_hash` | The built preamble contains the configured seed and seeds both `random` and, when importable, numpy |
| `test_preamble_line_offset_is_recorded` | The offset needed to map a traceback back to the model's own line numbers |
| `test_batch_preserves_input_order` | |
| `test_infrastructure_error_propagates` | A raising sandbox does **not** become a `TestOutcome` |

### Done when

All thirteen pass in under a second — no subprocess should be spawned by this file's tests.

---

## Task 7 — Reward registry

### Behaviour

Every registered function has the identical signature `RolloutOutcome -> float` and is pure.
A test skipped after a timeout counts as not passed, everywhere. Public results are **never**
read by any reward function — they are a diagnostic (ADR-0013).

Source of truth: [`rl-reward-functions.md`](../design/rl-reward-functions.md),
[`behavior.md`](../design/behavior.md) §3, [ADR-0011](../adr/0011-reward-registry.md).

### Files

`src/post_training_rl/rewards.py`

Implement **five** primary entries plus **one** auxiliary: `binary`, `pass_rate`,
`binary_threshold`, `ladder`, `code_r1`, `extractability`. `hierarchical`, `verpo`, and
`overlong` are in the registry's design but are **not built** — each needs machinery no other
entry needs, and none is in the first run.

`ladder` ships unannealed.

### Unit tests

Per function, using `RolloutOutcome` fixtures built by a helper:

| Function | Tests |
| --- | --- |
| `binary` | `test_binary_rewards_all_passing`, `test_binary_rejects_single_failure`, `test_binary_rejects_timeout` |
| `pass_rate` | `test_pass_rate_is_fraction_passed`, `test_pass_rate_is_one_when_all_pass`, `test_pass_rate_is_zero_when_none_pass` |
| `binary_threshold` | `test_threshold_accepts_full_pass`, `test_threshold_rejects_partial_pass` |
| `ladder` | `test_ladder_rung_no_code`, `test_ladder_rung_parses`, `test_ladder_rung_runs`, `test_ladder_rung_partial_pass` |
| `code_r1` | `test_code_r1_penalises_missing_code`, `test_code_r1_scores_wrong_answer`, `test_code_r1_scores_correct` |
| `extractability` | `test_extractability_scores_each_fence_at_equal_parse`, `test_extractability_parsing_always_outranks_non_parsing` |

Shared, parameterised over the registry:

| Test | Asserts |
| --- | --- |
| `test_skipped_after_timeout_counts_as_not_passed` | Over every registered function |
| `test_every_registered_function_accepts_rollout_outcome` | The interchangeability contract, as an executable assertion |
| `test_public_results_never_affect_any_reward` | Vary `public_results`; every reward is unchanged |

`test_extractability_parsing_always_outranks_non_parsing` is the invariant ADR-0012 requires:
the worst parsing rollout must score above the best non-parsing one. It must hold for **all
sixteen** fence × parse combinations, not a sampled few.

### Done when

All pass, and the registry dict contains exactly the six implemented names.

---

## Task 8 — Startup self-test

### Behaviour

Before the first training step, the configured sandbox is verified against known-hostile
programs, and startup aborts if any is not contained.

It exists because a misconfigured sandbox is otherwise indistinguishable from the model
producing wrong answers — the expensive kind of bug. It costs a few seconds once per run.

Source of truth: [`behavior.md`](../design/behavior.md) §9 item 3.

### Files

`src/post_training_rl/startup.py` — `def verify_sandbox_or_raise(sandbox: Sandbox) -> None`

Reuses `tests/fixtures/hostile_programs.py` from task 5. Same fixtures, two call sites.

### Unit tests — `FakeSandbox`

| Test | Asserts |
| --- | --- |
| `test_self_test_passes_when_all_programs_contained` | Scripted containing sandbox → returns |
| `test_self_test_raises_naming_the_uncontained_program` | Scripted sandbox that lets the fork bomb through → raises, message names which check failed |

### Done when

Both pass, and running the self-test against the real `FirejailSandbox` succeeds locally.

---

## Sprint definition of done

- [ ] `pytest -q` green — **63 unit tests**, no subprocesses, sub-second
- [ ] `pytest -q -m subprocess_backend` green — **4 tests**
- [ ] `pytest -q -m containment` green in CI with firejail installed — **9 tests**
- [ ] A rollout can be scored end to end by all six reward functions with **no model loaded**
- [ ] Every constant lives in `config/verifier.yaml`; no behaviour-governing literal in code
- [ ] Nothing imports `trl`, `transformers`, `peft`, or `datasets`

## Review checkpoints

`/write-code` implements one task per invocation and reports between them. Review after
**task 3** (the cascade is the subtlest logic in the sprint), after **task 5** (containment is
the only thing protecting the host), and at the sprint gate.

## Known risks carried into sprint 2

- **Sandbox throughput is unmeasured.** firejail startup × 15 tests × group size may make a
  thread pool insufficient. The fallback is a process pool or a persistent worker, and the
  standards require a measurement before making that change.
- **The `torchvision` defect** (`CLAUDE.md` §Environment) blocks `from transformers import
  TrainerCallback`. It does not affect this sprint — nothing here imports transformers — but
  it must be fixed before sprint 3.
