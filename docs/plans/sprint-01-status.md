# Sprint 1 — status report

Implementation record for [`sprint-01.md`](sprint-01.md). Covers what was built, which
objectives were met, what was tested, what deviates from the plan, and what is carried
forward.

**Sprint 1 is complete. All four gates pass.** Branch `worktree-sprint-01`, 10 commits (README + 8 tasks + gate review).

---

## 1. Gate

| Gate | Required | Actual |
| --- | --- | --- |
| Unit suite green, no subprocess, sub-second | **63** | ✅ **63 passed in 0.03 s** |
| `-m subprocess_backend` green | 4 | ✅ **4 passed in 1.05 s** |
| `-m containment` green with firejail | 9 | ✅ **9 passed in 2.33 s** |
| A rollout scored end to end by all six rewards, no model loaded | — | ✅ §4 below |
| Every behaviour-governing constant in config | — | ✅ see §6.1 |
| Nothing imports `trl`, `transformers`, `peft`, `datasets` | — | ✅ stdlib + `pyyaml` + `pytest` only |

Per-file test counts match the plan exactly:

| File | Plan | Actual |
| --- | --- | --- |
| `test_comparator.py` | 12 | 12 |
| `test_extraction.py` | 16 | 16 |
| `test_verifier.py` | 13 | 13 |
| `test_rewards.py` | 20 | 20 |
| `test_startup.py` | 2 | 2 |
| `test_sandbox_subprocess.py` | 4 | 4 |
| `test_sandbox_firejail.py` | 9 | 9 |

No test was written that the sprint does not name, and none it names is missing.

---

## 2. What was built

```
config/verifier.yaml           sandbox limits, seed, test caps, prefill
config/reward.yaml             which rewards run, at what weight
requirements.txt               pyyaml, pytest, pytest-timeout
src/post_training_rl/
  types.py                     10 frozen types crossing every seam
  config.py                    YAML loading, no schema (stage: exploring)
  comparator.py                CodeContests token matching
  extraction.py                syntax-gated cascade, fence + parse as two facts
  verifier.py                  the deep module, plus the seeding preamble it owns
  rewards.py                   six-entry registry
  startup.py                   hostile-program self-test
  sandbox/
    __init__.py                the Sandbox Protocol — one method
    firejail.py                training adapter
    subprocess_.py             weaker CI/dev adapter
    fake.py                    scripted results + call log
    hostile_programs.py        five hostile programs, two call sites
```

Task-by-task, each committed separately and reviewed before commit:

| # | Task | Commit |
| --- | --- | --- |
| 1 | Scaffold, types, config | `395d6a8` |
| 2 | Comparator | `26f2271` |
| 3 | Extraction cascade | `cb70e37` |
| 4 | Sandbox seam, fake, subprocess | `0dc8f74` |
| 5 | Firejail, containment, ADR-0014 | `815da31` |
| 6 | Verifier | `1782de4` |
| 7 | Reward registry | `2a5e026` |
| 8 | Startup self-test | `29f1c77` |
| — | Gate review fixes | `b49b756` |

---

## 3. Measurements this sprint produced

Both were listed as unknowns. Both are now answered.

### 3.1 Sandbox throughput — the risk the roadmap carried into sprint 2

`verifier-scorer.md` §12 item 6 and the sprint's "known risks" both flagged this as
unmeasured, with a process pool as the contingency. **A thread pool is sufficient, by a wide
margin.**

The measurement found a defect first. Firejail's `--timeout` costs a flat ~2 s per execution,
*independent of its value*:

| Invocation | Elapsed, for a program taking 0.02 s |
| --- | --- |
| `--timeout=00:00:01` | 1.12 s, exit 1 — **killed despite having finished** |
| `--timeout=00:00:03` | 2.03 s |
| `--timeout=00:00:05` | 2.02 s |
| `--timeout=00:00:10` | 2.02 s |
| `--timeout=00:01:00` | 2.02 s |
| no `--timeout` | 0.02 s |

It polls rather than waking on child exit, so no choice of timeout escapes the floor. At 17
executions per rollout, group size 8, 12 worker threads, that is ~23 s of dead time per
optimizer step — roughly three hours across a 500-step run.

[**ADR-0014**](../adr/0014-external-wall-clock-timer.md) replaces it with `timeout(1)`
wrapping the firejail invocation. Result: **0.029 s per execution, 71× faster.** The
containment suite dropped from 17.4 s to 2.3 s. Twenty consecutive executions and five killed
infinite loops leaked zero firejail and zero python3 processes.

### 3.2 Startup self-test cost

3.13 s against the real `FirejailSandbox`, against the "about four seconds" budgeted in
`verifier-scorer.md` §3.

---

## 4. End-to-end demonstration

Five rollouts against one problem (5 graded + 1 public test), real `FirejailSandbox`, no
model loaded:

| Rollout | fence | parsed | graded | `binary` | `pass_rate` | `binary_threshold` | `ladder` | `code_r1` | `extractability` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| correct | tagged | ✓ | 5/5 | 1.00 | 1.00 | 1.00 | 1.00 | **1.10** | 1.00 |
| off by one | tagged | ✓ | 0/5 | 0.00 | 0.00 | 0.00 | 0.10 | 0.10 | 1.00 |
| crashes | tagged | ✓ | 0/5 | 0.00 | 0.00 | 0.00 | 0.05 | 0.10 | 1.00 |
| unparseable | tagged | ✗ | 0/5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.10 | −0.20 |
| prose only | none | ✗ | not executed | 0.00 | 0.00 | 0.00 | 0.00 | **−1.10** | −1.00 |

24 sandbox executions in 0.36 s — the prose rollout contributes none, which is the
short-circuit working. Every distinction the design exists to protect is visible:

- `binary` collapses everything below all-pass to 0.0 — its known sparsity, and why it ships
  composed with `extractability`.
- `ladder` separates "ran but wrong" (0.10) from "parsed but crashed" (0.05) from
  "unparseable" (0.00).
- `code_r1` separates prose (−1.10) from unparseable code (+0.10) — a 1.2 swing, and the
  reason for the fix in §5.2.
- `extractability`'s parse swing exceeds its fence swing: every parsing row scores above
  every non-parsing row, which is the invariant ADR-0012 requires.
- Rows 2–5 all have `binary = 0.0`. With `extractability` at weight 0.1 the totals differ
  (0.10, 0.10, −0.02, −0.10), so the group has non-zero variance and produces a gradient
  instead of being degenerate — `rl-reward-functions.md` §4's worked example, reproduced.

---

## 5. Defects found and fixed

Every task went through a two-axis review — standards and spec, run as independent agents so
neither masks the other — before its work was committed. Reviews ran after tasks 3, 5 and 6
and at the gate, each covering the preceding task too, matching the review checkpoints
`sprint-01.md` itself specifies. **Four real bugs were caught that the tests did not**, and in
two cases both reviewers found the same defect independently.

### 5.1 Extraction reported valid Python as unparseable — **task 3**

The opening-fence pattern tolerated leading indentation but never dedented, so a block nested
in a numbered list captured `"    print(1)"`, which `ast.parse` rejects. It reported
`(TAGGED, parsed=False)` — *claiming the model cannot write valid Python about a model that
just did*, which is the precise false signal ADR-0012 exists to prevent, feeding directly into
the parse histogram nobody has published.

Fixed by requiring the opening marker at column 0, which is what `behavior.md` §2.4 implies by
asking only for the *closing* fence to be indent-tolerant. See §6.2 for the residual gap.

### 5.2 The verifier collapsed "no code" into "unparseable code" — **task 6**

The short-circuit gated on `not extraction.parsed`, so recovered-but-unparseable code was
never executed and produced empty results — indistinguishable from prose. `behavior.md` §2.8
keeps those two apart at extraction *specifically* so the distinction survives, and `code_r1`
scores them 1.2 apart (−1.1 no code, +0.1 wrong).

The gate is now `extraction.code is None`. This was only affordable because of ADR-0014:
fifteen guaranteed `SyntaxError`s cost 0.44 s at 0.029 s per execution, against 30 s before.
**The measurement changed which reading of the spec was practical.**

### 5.3 A successful solution was reported as a timeout — **task 5**

`timed_out` was derived from elapsed time, and firejail's reap overhead is billed to the
program, so a solution finishing at 1.5 s under a 3 s limit showed ~3 s elapsed and was
recorded as `TIMEOUT`. At the production 10 s limit, every solution running past ~8 s was
misclassified — and since a timeout abandons all remaining tests (ADR-0006), **one
slow-but-correct solution lost its entire test suite.**

Signal death is now the discriminator; a program that merely crashes exits with a positive
status. Verified across the matrix: quick success, slow success, infinite loop, crash, and
slow-crash all classify correctly.

### 5.4 The self-test could pass with only the wall clock working — **task 8**

Three of the four containment predicates accepted a timeout as proof. A fork bomb that merely
ran out the clock satisfied the process-cap check; a flood that timed out satisfied the
truncation check. **A sandbox with nothing working except the timer would have passed** — the
exact failure the self-test exists to catch, and its test passed only because the predicates
were lax enough to accept its own fixture.

Each predicate now insists on the mechanism that program is meant to be stopped by.

### 5.5 Smaller fixes

- The bounded reader flagged truncation when output landed *exactly* on the cap, then killed a
  correct solution and reported `RUNTIME_ERROR`.
- `Popen` was unguarded: a parent-side failure could leave a sandboxed process running.
- `process.wait()` was unbounded, reachable by a child that closes stdout and hangs — the
  wedge case the backstop exists for.
- `FakeSandbox` had a check-then-pop race under the verifier's 12-thread fan-out.
- `Verifier.preamble` violated "one public method"; removed.
- `test_preamble_line_offset_is_recorded` restated the implementation's own formula and could
  not fail; it now asserts the model's first line lands at `line_offset`.

---

## 6. Deviations from the plan

Each is deliberate and recorded in code as well as here.

### 6.1 Every tunable now lives in YAML — including some the plan did not enumerate

Task 1 enumerated a fixed list for `config/verifier.yaml`, and the first pass kept everything
else in code. That was revisited: **all behaviour-governing values are now loaded from
config**, on the grounds that they are research parameters and two runs that graded
differently must be diffable as files.

Added to `config/verifier.yaml`:

| Key | Was |
| --- | --- |
| `sandbox.kill_after_seconds`, `sandbox.backstop_slack_seconds`, `sandbox.reap_timeout_seconds` | ADR-0014's three timers, hardcoded in `firejail.py` |
| `comparator.absolute_float_tolerance` | `ABSOLUTE_FLOAT_TOLERANCE = 1e-5` in `comparator.py` |
| `startup.self_test_timeout_seconds` | `_SELF_TEST_TIMEOUT_SECONDS = 2.0` in `startup.py` |

Added to `config/reward.yaml` under `shapes:` — every number defining a reward: the
`binary_threshold` cut-off, `code_r1`'s three rungs, `ladder`'s three, and `extractability`'s
parse and fence terms.

Two signature changes follow, both recorded as deviations:

- `outputs_match(actual, expected, absolute_tolerance)` — `verifier-scorer.md` §5 shows two
  parameters. The verifier passes the configured value.
- `verify_sandbox_or_raise(sandbox, timeout_seconds)` — task 8 shows one parameter. It takes
  the float rather than a config object, per "accept the least specific input that works".

The registry is now built by `build_reward_functions(shapes)`, with each entry closing over
the configured values — the mechanism `verifier-scorer.md` §7 already sanctions for a
parameterised entry. Every entry stays a plain `RolloutOutcome -> float`, so the
interchangeability contract is untouched.

**A caution kept in the config file itself:** treat a change to a shape value as defining a
*new* reward rather than tuning an existing one. `binary_threshold` at 0.95 is not a tuned
`binary_threshold`; a run using it is not comparable to one that did not. The values ship
matching their published sources.

What remains in code is not tunable: unit conversions (`_BYTES_PER_MIB`), an I/O buffer size
(`_READ_CHUNK_BYTES`), and `_TIMEOUT_EXIT_CODE = 124`, which is a GNU coreutils fact rather
than a choice.

### 6.2 Indented opening fences are not recognised

Confirmed as spec-exact by decision during the sprint. A ```` ```python ```` fence nested
inside a numbered list reports `(NONE, no code)` and scores as a format failure.

**Proposed ADR-0012 amendment:** tolerate an indented opening fence and `textwrap.dedent` the
captured block. Deferred because it needs a test outside task 3's list of sixteen. The fence
histogram in sprint 3 will show whether this ever actually happens.

### 6.3 `Popen` rather than `subprocess.run` in the firejail adapter

Task 5 says to use `subprocess.run(..., timeout=)`. Its two constraints cannot both be met:
`subprocess.run` buffers output without bound, so it cannot "read at most `stdout_cap_bytes`,
then kill the child", and `behavior.md` §4.7 requires the truncation never exhaust memory in
the calling process. The stated *reason* for the rule — `Popen.communicate(timeout=)` does not
kill the child — is honoured by an explicit `killpg` on the child's session, which is
stricter. Recorded in the module docstring.

### 6.4 `hostile_programs.py` is in `src/`, not `tests/fixtures/`

Task 8 puts `verify_sandbox_or_raise` in `src/`, and production code importing from `tests/`
breaks the moment the package is installed. The sprint makes this identical argument itself
for `FakeSandbox`. `sprint-01.md:503` still points at the old path and should be corrected.

### 6.5 The firejail suite briefly ran at 3.0 s, now back at 1.0 s

The plan's 1.0 s floor was justified by firejail's one-second `--timeout` granularity. Under
that flag the floor was actually ~2 s and a 1 s limit killed finished programs, so the suite
ran at 3.0 s. ADR-0014 removed the constraint — `timeout(1)` takes fractional seconds — and
the suite is back at the specified 1.0 s.

---

## 7. Open items and nits

### 7.1 The self-test refuses the documented CI backend

`SubprocessSandbox` fails the network check, because it cannot block network access. That is
correct per `behavior.md` §4.13 — the weaker adapter is explicitly not held to guarantees
5–8 — but it means `backend: subprocess` cannot start a run at all. This may be intended
(training uncontained is a bad idea) or may block CI. **Needs a decision before sprint 3.**

### 7.2 `ladder`'s timeout rung is unspecified

A rollout whose tests all time out scores 0.05, tied with a program that crashes on import,
despite having demonstrably run. `rl-reward-functions.md` §3 defines neither boundary and no
named test pins it. Wants a spec line, not a code change.

### 7.3 Public tests deliberately do NOT inherit the timeout abort — measured, and reversed

An earlier draft of this report recommended carrying the graded timeout across to public
tests, to save the 11 s a hung rollout spends there. **Measurement reversed that.**

The 11 s is `timeout_seconds` + `kill_after_seconds` — the full wall-clock limit burned by a
program that will not finish. Whether that cost buys anything depends on *why* the solution
is slow:

| Failure mode | Public-test cost | What it yields |
| --- | --- | --- |
| Infinite loop | 11 s (times out too) | Nothing. It was never going to finish |
| Too-slow algorithm, e.g. O(n²) | **0.03 s — passes** | "Algorithmically correct but too slow" |

Graded tests are ordered **longest-input-first** (ADR-0009), so a timeout fires on the
*largest* input, while public tests are the statement's small examples. Measured directly: an
O(n²) solution timed out on a 100,000-element graded input and **passed the 3-element public
test in 0.03 s**.

Carrying the abort would delete exactly that signal — for a 3B model on CodeContests,
"correct but too slow" is among the most informative diagnostics available — while saving
nothing in the case where it fires. The worst case is bounded anyway: public tests abort
internally after their own first timeout, so an infinite-loop rollout pays one extra 11 s, not
one per test.

**Current behaviour is correct. No change.**

### 7.4 Smaller notes

- `_read_capped` both reads and kills, so `exit_code` depends on a side effect of the read.
  Correct, but not a clean computation/effect split.
- `SubprocessSandbox` applies limits via `preexec_fn`, documented-unsafe in a multi-threaded
  process, and the verifier fans out over 12 threads. It is the CI/dev adapter only, and
  firejail sets its own limits, so the training path is unaffected.
- `SubprocessSandbox` slices a character count against a byte cap. Identical for the ASCII
  contest output almost always is.
- `_parses` catches `SyntaxError` and `ValueError`. Deeply nested input can raise
  `RecursionError` or `MemoryError` from `ast.parse` and would propagate as an infrastructure
  failure. Not observed; worth a guard if a run ever dies there.
- The residual timeout ambiguity in `firejail.py`: a solution killed by something other than
  our timers — the OOM killer — also dies by signal and reads as a timeout. `--rlimit-as`
  makes the child raise `MemoryError` instead, so this needs system-wide pressure.

---

## 8. What this unblocks

Sprint 2's dataset builder can be graded against a verifier that is now trustworthy, and its
format-failure and base-pass-rate measurements need exactly the extractor and verifier this
sprint delivered. The `frac_reward_zero_std` question sprint 4 turns on depends on the reward
registry behaving as §4's table shows it does.

Carried into sprint 2 unchanged: the **`torchvision 0.26.0` / CUDA 13.0 defect** blocking
`from transformers import TrainerCallback`. Nothing in this sprint imports transformers, so it
did not bite — but it must be fixed before sprint 3, where the cache reset hooks into exactly
that callback.
