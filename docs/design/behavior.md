# Intended behaviour

Companion to [`verifier-scorer.md`](./verifier-scorer.md). This is the specification each
module is built against: what it does, what it guarantees, and what it refuses. Tests are
derived from it one slice at a time during the red→green loop — not transcribed in bulk.

Vocabulary follows [`CONTEXT.md`](../../CONTEXT.md).

**Confirmed:** the seams under test are the nine modules below. Config loading, log emission,
and TRL's own training loop are deliberately outside the test surface. Containment behaviour
(§4) gates CI, with firejail installed in the runner.

**Build order** — dependencies first, so each module can be driven to green without stubbing
the next: comparator → extraction → rewards → verifier → sandbox → dataset → cache →
adapter → smoke.

---

## 1. Comparator

*Decides whether a program's output counts as matching the expected output (ADR-0007).*

### Does

1. Splits both sides on any whitespace and discards empty tokens, so line structure and
   indentation are irrelevant.
2. Compares token by token, case-insensitively — `YES` and `yes` are the same answer.
3. Treats a token pair as numeric when either side parses as a float, and accepts them within
   an absolute tolerance of 1e-5.
4. Treats integer and float spellings of the same value as equal — `1` matches `1.0`.

### Guarantees

5. Token counts must be equal. A short output never matches a longer expected output, and a
   long output never matches a shorter one.
6. Empty output matches empty expected output, and nothing else.
7. The comparison is symmetric and has no hidden state — the same pair always gives the same
   answer.

### Known sharp edges, preserved deliberately

8. The float tolerance is **absolute**, not relative, so it is lax at large magnitudes. This
   mirrors the reference implementation and is pinned so a change to it is visible.
9. Values that exceed 32-bit integer range fall through to the float path and inherit its
   tolerance. Also inherited from the reference implementation.

---

## 2. Extraction

*Recovers executable Python from a completion and reports how it found it (ADR-0012).*

### Does

1. Prefers a Python-tagged fenced block; failing that an untagged fence; failing that any
   fenced block; failing that bare code with no fence at all.
2. Within the winning tier, returns the **last** syntactically valid candidate — models
   routinely quote the problem's example or sketch a naive version before the real one.
3. Accepts `python`, `python3`, `py`, and `py3` as Python tags.
4. Tolerates trailing whitespace after a fence marker and an indented closing fence.
5. Treats an unterminated fence as running to the end of the completion, which is what a
   truncated generation looks like.
6. Prepends the configured prefill before matching, when one is configured.

### Guarantees

7. Every result carries **two independent facts**: which fence the code arrived in, and
   whether that code parses. They are never collapsed into a single value, because a flawless
   fence can wrap broken code and correct code can arrive unfenced — and the two failures call
   for opposite responses.
8. A candidate is only selected if `ast.parse` accepts it. When candidates exist but none
   parse, the last candidate is still returned with `parsed = False`, so the caller can
   distinguish "malformed code" from "no code at all".
9. Prose is never returned as code. The bare tier is syntax-gated precisely so that text
   which is not a program fails it.
10. It never falls back to returning the whole completion unguarded.

### Refuses

11. Returns fence `none` with no code for an empty completion, or for one containing nothing
    that parses and no fence.
12. When a prefill is configured but not supplied at the call site, a completion that opens
    mid-block yields fence `none`. This failure is deliberate and specified: it is the
    documented way this component gets silently misused, and it must fail loudly rather than
    return plausible-looking partial code. The prefill value has **two** consumers — the
    dataset builder that renders the prompt and the reward function that re-prepends before
    extraction — and they must be fed from the same config key.

---

## 3. Reward functions

*Turn a rollout outcome into a number (ADR-0011).*

### Guarantees across the registry

1. Every registered function has the identical signature `RolloutOutcome -> float`. That
   uniformity is the interchangeability contract; a function that needs anything else does
   not belong in the registry.
2. Every function is pure — no I/O, no execution, no clock, no global state. The same outcome
   always yields the same number.
3. A test skipped because an earlier test timed out counts as not passed, everywhere.
4. Public test results never influence **any** reward, primary or auxiliary. They are
   executed and logged as a diagnostic only (ADR-0013). A reward computed over tests the
   model can read is directly hackable, and the registry deliberately contains no entry that
   consumes them.
5. Each function's docstring names the source it came from.

### Per function

6. **`binary`** — 1.0 when every graded test passes, 0.0 otherwise. A single failure, timeout,
   or runtime error collapses it to 0.0.
7. **`pass_rate`** — the fraction of graded tests that passed.
8. **`binary_threshold`** — 1.0 when the pass rate exceeds 0.99, else 0.0. The threshold is a
   float-comparison guard, not a real tolerance.
9. **`ladder`** — graded rungs: no code, code that parses, code that runs, then pass rate.
   Ships unannealed. An annealed variant is possible without extra machinery, since the
   trainer passes `trainer_state` and its `global_step` to every reward function; it is simply
   not part of the first run.
10. **`code_r1`** — a format failure scores below a wrong answer, which scores below a correct
    one. The only surveyed design where failing to produce code is distinguishable from
    producing wrong code.
11. **`extractability`** — sums a parse term and a fence term, so both dimensions are rewarded
    independently. The parse swing is deliberately larger than the fence swing, guaranteeing
    that the worst parsing rollout outranks the best non-parsing one — a well-fenced broken
    program must never beat a bare working one. Carries weight 0.1 against the primary reward,
    and exists to give an otherwise-degenerate group some variance.

12. **`hierarchical`, `verpo`, and `overlong`** are in ADR-0011's registry but are **not
    specified here and not built for the first run.** Each needs machinery no other entry
    needs — AST structural alignment, Gaussian-KDE density calibration, and a length schedule
    respectively. They earn a behaviour spec when one is actually selected for a run. Listing
    them in the registry records the design space; it does not commit to building them.

---

## 4. Sandbox

*Runs one program against one input, and contains it (ADR-0005).*

### Does

1. Accepts program source as text and returns what happened. Callers never learn that
   temporary files are involved.
2. Delivers the supplied text on the program's stdin and captures stdout and stderr.
3. Reports duration, exit status, whether the program timed out, and whether output was
   truncated.

### Guarantees — containment

4. A program that loops forever is killed and reported as timed out.
5. A program that spawns processes without limit is capped and cannot affect the host.
6. A program that opens a network connection fails to do so.
7. A program that writes output without limit has that output truncated, and the truncation
   never exhausts memory in the calling process. Pipe capacity is 64 KiB and the file-size
   rlimit does not apply to pipes, so the cap is enforced by the reader.
8. A program that writes files without limit is capped.
9. The **execution environment** is deterministic: `PYTHONHASHSEED` is fixed, so `set` and
   `dict` iteration order is stable across runs. This is the sandbox's half of ADR-0008. It
   does **not** modify the source it was given — seeding the program is the verifier's half
   (§5), because the sandbox's contract is "run exactly this".

### Refuses

10. When the configured backend's binary is absent, startup fails with an error naming the
    missing binary. It never silently downgrades to a weaker backend — the backend is an
    explicit config value with no default.
11. Infrastructure failures — unwritable temp directory, missing binary — are raised. A
    program that crashes, hangs, or floods is a normal result, not an exception.

### Adapters

12. The firejail adapter provides every containment guarantee above.
13. The subprocess adapter provides the functional behaviour and resource limits, but **not**
    network isolation or a private filesystem. It exists for development convenience and is
    documented as the weaker option; it is not held to guarantees 5–8.

---

## 5. Verifier

*Executes a rollout's code against a problem's tests and reports what happened (ADR-0004).*

### Does

1. Extracts code, prepends the determinism preamble, and runs each graded test in order.
   Prepending is the verifier's half of ADR-0008 — the sandbox fixes the environment (§4), the
   verifier seeds the program. Neither may assume the other does it.
2. Classifies each test as passed, wrong output, runtime error, timed out, or skipped.
3. Runs the problem's public tests separately and reports them separately. No reward reads
   them; their pass rate is logged each step as ground truth that does not move with whatever
   reward is driving training (ADR-0013).
4. Processes a batch concurrently and returns reports in the order the batch was given.

### Guarantees

5. When no code can be extracted, **no execution happens at all**. This is the most common
   outcome early in training and it must cost nothing.
6. When a test times out, the remaining tests are abandoned — a solution that exceeds the
   limit on one input almost always exceeds it on the rest (ADR-0006).
7. Abandoned tests are still **recorded** as skipped. The number of results always equals the
   number of tests, so no reward function ever has to reason about why results are missing.
8. Truncated output is compared rather than auto-failed. Because the comparator requires
   matching token counts, truncation almost always fails anyway — but through the normal path,
   with no special case.
9. Every execution gets the same flat timeout. The dataset's own `time_limit` is never read.
10. It assigns no rewards and holds no opinion about what an outcome is worth.

### Refuses

11. **Raises only on infrastructure failure.** A solution that crashes, hangs, floods output,
    or produces no parseable code is data — it becomes a test outcome or an extraction
    result, never an
    exception. A missing sandbox, an unwritable directory, or a malformed problem raises,
    because those are systematic and every subsequent reward would be meaningless.

---

## 6. Verification cache

*Ensures one execution feeds every reward function (ADR-0004).*

### Guarantees

1. A given rollout is verified exactly once per step, however many reward functions ask for
   it.
2. Two rollouts with identical text for the same problem share one verification. This is
   correct rather than merely convenient, because execution is deterministic — without
   ADR-0008 it would be a bug.
3. Distinct rollouts are verified independently.
4. Cached outcomes do not survive a step boundary.

---

## 7. Dataset builder

*Turns the raw corpus into what training actually sees (ADR-0009, ADR-0010).*

### Does

1. Loads from the repository's own data files, never the truncated preview API, which serves
   28% of the training split.
2. Decodes difficulty using the protocol definition rather than the shipped label list, which
   is wrong for every value above a certain index.
3. Emits a prompt, a problem identifier, the graded tests, and the public tests.

### Guarantees — problem filtering

4. Problems whose statements admit multiple valid answers are removed from **both** training
   and evaluation. A correct solution that prints a different valid answer would otherwise be
   scored as wrong, which for RL is not a lost point but a gradient pointing away from
   correctness.
5. Interactive problems are removed — the harness reads input once and writes output once, so
   they can never pass.
6. Problems using file I/O rather than stdin/stdout are removed.
7. Problems whose prompt exceeds the token budget are removed rather than truncated. A
   truncated statement loses the constraints or the I/O format and is unsolvable by
   construction.
8. Every filter reports how many problems it dropped, and the totals are surfaced. A filter
   that quietly removes half the corpus must be visible.

### Guarantees — test selection

9. Graded tests are drawn from private tests first, topped up from generated tests only when
   a problem has too few. Generated tests are mutation-derived and can be invalid.
10. Tests are ordered longest-input-first, because longer inputs catch more bugs per
    execution.
11. No more than the configured maximum are kept, bounding per-rollout cost.
12. Problems that cannot reach the minimum test count are dropped entirely.
13. Public tests are kept in a separate field and never mixed into the graded set.

---

## 8. TRL adapter

*Presents the verifier and the registry in the shape TRL expects.*

### Does

1. Reconstructs problems from the dataset's plain columns, since the dataset stores Arrow
   rather than Python objects.
2. Builds the reward function list and matching weight list that the trainer consumes.
3. Supplies each rollout's token count and truncation status, which the verifier cannot know.

### Guarantees

4. Every reward function named in configuration is computed and logged each step, including
   those not driving training. One run therefore yields the counterfactual curves for every
   reward shape in the registry.
5. Shadow-logged rewards carry weight zero and provably cannot influence training.
6. The fence and parse distributions are logged separately every step. No published source
   reports a
   code parse-failure rate for any model on any benchmark, so this is a measurement the
   project makes for itself.
7. Reconstruction asserts the shape it expects and names the offending problem when it fails.
   This is the one place where type discipline is enforced at runtime rather than statically.

---

## 9. Configuration and startup

### Guarantees

1. Every tunable value — limits, caps, thresholds, seeds, model identity, quantisation,
   prompt template — is read from YAML. No literal that governs behaviour appears in a branch.
2. A required key that is absent fails immediately and names the key. Nothing is defaulted
   into existence.
3. Before the first training step, the configured sandbox is verified against known-hostile
   programs — an infinite loop, a fork bomb, a network connection, an output flood — and
   startup aborts if any is not contained.

The self-test exists because a misconfigured sandbox is otherwise indistinguishable from the
model producing wrong answers, which is the expensive kind of bug. It costs a few seconds
once per run and reuses the same fixtures as the containment tests.

---

## 10. End to end

### Guarantees

1. A short training run completes, produces finite rewards, logs the fence and parse
   histograms, and
   reports every shadow-logged reward alongside the training reward.

The trainer contract it rests on — reward-function signature, kwarg forwarding of dataset
columns, `log_metric`, and the step-end callback — has been verified against the installed
`trl 1.9.2` by reading its source. This test guards against those changing under a version
bump. It verifies the contract, not learning.
