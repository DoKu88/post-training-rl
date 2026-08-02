# Sprint 2 — The corpus, and the numbers we have been guessing

**Objective:** turn 13,328 raw problems into the filtered training set, and replace four
guesses with measurements.

Still no GPU and no training loop. The dataset builder is the last piece of the environment
box ([`rl-loop.md` §2a](../design/rl-loop.md)) — it supplies `S_t`, the state distribution,
and per [`rl-loop.md` §4.1](../design/rl-loop.md) it is also where the classic loop's `S_t+1`
arrow does *not* apply: the next prompt is the next row, drawn independently of what the model
just wrote.

**Module stage: exploring.** Locked-in rules do not apply — no config schema validation, no
design patterns reached for in anticipation. See `CODING_STANDARDS.md` §Stage.

**Why this sprint is dangerous.** [ADR-0010](../adr/0010-aggressive-problem-filtering.md)
makes the filters aggressive enough to break comparability with every published CodeContests
number. That is an accepted cost — but it means **the filtered corpus *is* the experiment**,
and a filter bug is invisible in the reward curve. A regex that removes 60% of the corpus
instead of 25% looks exactly like a smaller dataset, not like a defect.

---

## Environment — verified before writing this plan

| Fact | Status |
| --- | --- |
| Corpus downloaded | ✅ 7.2 GB at `~/.cache/huggingface/hub/datasets--deepmind--code_contests` |
| Row counts via `load_dataset` | ✅ **train 13,328 · valid 117 · test 165** — the full corpus, matching `dataset_infos.json` |
| `datasets` | ✅ 5.0.1 imports |
| `from transformers import AutoTokenizer` | ✅ **works** — the torchvision defect does *not* block this sprint |
| `from transformers import TrainerCallback` | ❌ still raises — **sprint 3's blocker, not ours** |
| Disk headroom | ✅ 1.6 TB free |

**The truncation trap does not apply to `load_dataset`.** `rlvr-stack.md` §4.4 warns that the
viewer, `/rows`, `/statistics` and `refs/convert/parquet` serve 3,762 of 13,328 train rows.
Measured: `load_dataset("deepmind/code_contests")` returns all 13,328. The trap is real but it
bites the *preview APIs*, not the loader. **Task 1 pins this with an assertion anyway**, because
a silent 28% corpus is exactly the failure ADR-0010 says would be invisible.

---

## Design — the shape, and why

Applying the deep-module vocabulary: a lot of behaviour behind a small interface, at a seam
that something actually varies across.

### The deep module

```
                     ┌──────────────────────────────────────────┐
   Iterable[Mapping] │  build_dataset(rows, config, tokenizer)  │  BuildResult
   ─────────────────▶│                                          │─────────────▶
   (raw corpus rows) │   decode · filter · select · render      │  (dataset, report)
                     └──────────────────────────────────────────┘
```

One public function. Behind it: proto decoding, four content filters, test-pool selection,
prompt rendering, the length filter, and drop accounting. A caller learns one signature and
gets all of it — that is the leverage.

### Two seams, and why exactly two

**Seam 1 — the corpus is a parameter, not a dependency.** `verifier-scorer.md` §9 specifies
`build_dataset(config, tokenizer)`, loading internally. **This plan takes `rows` as a
parameter instead**, for two reasons that are not style:

1. **Two real consumers.** The builder needs rows; the measurement pass (task 6) needs the
   *same* rows to compute distributions over the corpus *before* filtering. A function that
   loads internally forces the measurement pass to load a second time — 13,328 rows twice.
2. **The interface is the test surface.** With rows as a parameter, every builder test passes
   a list of dict literals. With loading inside, every test needs 7.2 GB on disk. The sprint-1
   `FakeSandbox` argument, reached by a different route.

`load_corpus(config)` remains as the thin impure loader, so nothing is lost — the composition
root calls `build_dataset(load_corpus(cfg), cfg, tok)`. Recorded as a deviation in task 5.

**Seam 2 — the tokenizer stays injected**, exactly as §9 specifies. It is the one collaborator
with a real alternative (3B and 7B tokenize differently), and it is what makes the length
filter's threshold meaningful rather than a character count.

### What is deliberately *not* a seam

Filters, test selection, prompt rendering, and the decoders are **pure functions**, tested
directly. Each has one implementation and no varying collaborator. Injecting them would buy a
hypothetical seam and cost a layer of indirection — the same call made for `comparator` and
`extraction` in sprint 1, and for the same reason.

### The pipeline, in cost order

Cheap rejections first, so 13,328 rows are not tokenized to discover half of them were
already dropped:

```
row ──▶ ① multiple-output regex   ──drop──▶ ┐
    ──▶ ② interactive             ──drop──▶ │
    ──▶ ③ file I/O (input_file)   ──drop──▶ ├──▶ FilterReport
    ──▶ ④ select_tests            ──drop──▶ │    counts per DropReason
    ──▶ ⑤ prompt length (tokenizer)──drop──▶ ┘    + total in / total out
    ──▶ kept: {prompt, problem_id, graded_tests, public_tests}
```

**Each drop is attributed to the first rule that fired**, so the counts partition the corpus
and sum to the total dropped. A problem that is both interactive and over-length counts once.
That is what makes `behavior.md` §7.8's "every filter reports how many problems it dropped"
a checkable statement rather than a set of overlapping tallies.

---

## Working agreement

Unchanged from sprint 1, and it earned its keep — four real defects were caught by review that
the tests did not catch.

**Test-driven, vertical slices.** One failing test for one behaviour, make it pass, move on.
Never write a task's whole test list and then implement against it.

**Only the tests listed here get written.** If a behaviour seems to need a test that is not
listed, stop and say so.

**Fixtures are dict literals shaped like real rows.** `rlvr-stack.md` §4.2 documents the exact
shape — `public_tests` is `{"input": [...], "output": [...]}`, *parallel lists*, because HF
`Sequence` transposes the proto's `repeated Test`. A fixture that uses a list of dicts tests a
shape the corpus never produces, which is precisely how sprint 1's truncation bug survived
(`sprint-01-status.md` §5.6). **Task 1 delivers one fixture builder; every later task uses it.**

**Pre-authorised dependencies** — beyond sprint 1's `pytest`, `pytest-timeout`, `pyyaml`:

```
datasets        # corpus loading; already installed at 5.0.1
transformers    # AutoTokenizer only — NOT TrainerCallback, which is still broken
```

`trl` and `peft` remain **out of scope**. If a task appears to need one, the task is wrong.

**Four suites now.** `pyproject.toml` gains a third marker:

```toml
markers = [
    "subprocess_backend: spawns real subprocesses",
    "containment: requires firejail",
    "corpus: requires the 7.2 GB CodeContests download",
]
addopts = '-m "not containment and not subprocess_backend and not corpus"'
```

```bash
pytest -q                    # unit — sprint 1's 63 plus this sprint's, still sub-second
pytest -q -m corpus          # real corpus, minutes
pytest -q -m subprocess_backend
pytest -q -m containment
```

**Not tested in this sprint, deliberately:** config loading (as in sprint 1), and
`load_corpus` itself, whose only logic is a `load_dataset` call and one assertion the `corpus`
suite covers.

---

## User stories

1. **As the training loop, I need every problem I am shown to be winnable**, so a gradient
   never points away from correctness because the harness could not judge a correct answer.
2. **As a researcher, I need to know how many problems each filter removed**, so a regex that
   silently eats half the corpus is visible as a number rather than as a smaller dataset.
3. **As ADR-0009, I need the private/generated split recorded per problem**, so the guard it
   deliberately deferred becomes a decision with data behind it.
4. **As `max_prompt_length`, I need the real token-length distribution**, so the value is
   measured rather than guessed and the drop rate it implies is known before it costs a run.
5. **As sprint 3, I need the dataset to rebuild reproducibly from a config file**, so two runs
   that trained differently can be diffed as files.

---

## Task 1 — Config, types, corpus loading, and the two decode traps

### Behaviour

Loads the corpus from the real data files and refuses a truncated one. Decodes `difficulty`
with the **proto** mapping and `source` with the **HF** mapping, because the dataset ships
these broken in opposite directions.

`rlvr-stack.md` §4.3 is the source of truth and is worth restating, because getting it
backwards is silent:

- **`difficulty` stores raw proto values while the `ClassLabel` name list is dense.** The
  proto skips 18 entirely (`K=17`, then `L=19`). So `int2str()` is wrong for **every value
  ≥ 19** — `1575_L` stores 19 and decodes as `M`. The label `"L"` has count 0 in all three
  splits, which is the tell.
- **`source` was remapped to dense indices** — `ATCODER` is 6 in the proto but **5** in HF,
  `AIZU` is 7 but **6**. Here `int2str()` *is* correct, but the integers must never be
  compared against the proto.

Net: **`difficulty` is unsafe to decode but safe to compare; `source` is safe to decode but
unsafe to compare.**

### Files

```
config/dataset.yaml
src/post_training_rl/dataset/__init__.py
src/post_training_rl/dataset/types.py        # DropReason, TestSelection, FilterReport, BuildResult
src/post_training_rl/dataset/decode.py       # difficulty + source
src/post_training_rl/dataset/corpus.py       # load_corpus
tests/fixtures/corpus_rows.py                # THE shared row-fixture builder
tests/test_dataset_decode.py
```

`config.py` gains `load_dataset_config(path: Path) -> DatasetConfig`, alongside the two
existing loaders.

### Public interface

```python
def decode_difficulty(stored: int) -> str      # proto mapping; raises on an unknown value
def decode_source(stored: int) -> str          # HF mapping
def load_corpus(config: DatasetConfig) -> Iterable[Mapping[str, Any]]
```

`config/dataset.yaml` carries: `hf_dataset_id`, `split`, `expected_row_counts` (the
truncation guard), `filters.multiple_output_patterns`, `filters.interactive_patterns`,
`max_prompt_tokens`, `tests.max_tests_per_rollout`, `tests.min_tests_required`, and
`prompt_template` as a **block scalar**. Two runs with different prompts must be diffable
([verifier-scorer §9](../design/verifier-scorer.md)).

`tests.max_tests_per_rollout` and `min_tests_required` already exist in `verifier.yaml`. They
belong to the dataset builder, which is what enforces them. **Task 1 moves them** and leaves
`verifier.yaml` without them — the verifier never read them.

### Unit tests

| Test | Asserts |
| --- | --- |
| `test_difficulty_decodes_low_values` | `17 -> "K"`, the region where dense and proto agree |
| `test_difficulty_decodes_past_the_skipped_index` | `19 -> "L"`, `20 -> "M"`. **The trap.** Fails against `ClassLabel.int2str()` |
| `test_difficulty_rejects_the_skipped_index` | `18` is not a proto value and must raise, not silently decode |
| `test_source_uses_the_dense_hf_mapping` | `5 -> "ATCODER"`, `6 -> "AIZU"` — *not* the proto's 6 and 7 |

### Done when

Four tests pass; `test_difficulty_decodes_past_the_skipped_index` demonstrably fails against
an `int2str()`-style dense implementation; `config/dataset.yaml` holds every constant the
later tasks reference; `verifier.yaml` no longer carries the two test caps.

---

## Task 2 — Problem filters

### Behaviour

Four content filters, each a pure predicate over one raw row, each attributing its drop to a
named reason. Removed from **both** training and evaluation splits — ADR-0010 is explicit that
filtering only the training split would leave evaluation measuring problems the harness cannot
judge.

- **Multiple-output** problems admit more than one correct answer but ship one expected
  output. AlphaCode §A.2 puts these at **~1/4 of the validation set**. For supervised learning
  this is a lost point; for RL it is *a gradient pointing away from correctness*, which is why
  it is filtered rather than tolerated.
- **Interactive** problems need bidirectional exchange with a judge. The harness reads stdin
  once and writes stdout once, so they can never pass.
- **File I/O** problems set `input_file` / `output_file` non-empty. DeepMind's own runner never
  reads these fields, so they would silently fail.
- **Over-length prompts** are dropped, never truncated — task 4, once a tokenizer exists.

Source of truth: [`behavior.md`](../design/behavior.md) §7 items 4–8,
[ADR-0010](../adr/0010-aggressive-problem-filtering.md).

### Files

```
src/post_training_rl/dataset/filters.py
tests/test_dataset_filters.py
```

### Public interface

```python
def content_drop_reason(
    row: Mapping[str, Any], config: DatasetConfig
) -> DropReason | None
```

One function, not four. The caller wants "should this go, and why" — three separate predicates
would make every call site re-encode the precedence order, and the precedence is what makes
the drop counts partition the corpus.

Patterns come from `config/dataset.yaml`, never from literals: they are the single most
likely thing to need tuning after the first measurement.

### Unit tests

| Test | Asserts |
| --- | --- |
| `test_clean_problem_is_kept` | Returns `None` — the baseline |
| `test_multiple_output_phrasing_is_dropped` | `"if there are multiple answers, print any"` → `MULTIPLE_OUTPUT` |
| `test_multiple_output_matching_is_case_insensitive` | Statements are prose; casing varies |
| `test_interactive_problem_is_dropped` | → `INTERACTIVE` |
| `test_file_io_problem_is_dropped` | `input_file="input.txt"` → `FILE_IO` |
| `test_empty_input_file_is_not_file_io` | The common case: `input_file=""` must be kept. Pins the truthiness bug |
| `test_first_matching_rule_wins` | A row matching two rules reports the earlier one, so counts partition |
| `test_patterns_come_from_config` | A row kept under the shipped patterns is dropped under an extended set |

### Done when

All eight pass, and no filter pattern appears as a literal in `filters.py`.

---

## Task 3 — Test-pool selection

### Behaviour

Draws graded tests **private-first, longest-input-first**, tops up from generated tests only
when a problem has too few, caps at the configured maximum, and drops problems that cannot
reach the floor. Public tests are kept in a separate field and never mixed in.

The pools are not equally trustworthy, and the numbers are worth carrying in the plan:
generated tests were made by *mutating* existing inputs and validated only by consensus among
30 human solutions. AlphaCode's own measurement puts the shipped suites at a **46%
false-positive-or-slow rate** — read that column, not the 4% one. The maintainers concede
generated tests may be invalid, and the generation code was never released.

**Record the private/generated split per problem.** ADR-0009 deliberately declined to guard
the mix — *"a problem with 2 private and 3 generated tests clears the floor, and 60% of its
graded signal then comes from the pool measured at a 46% false-positive-or-slow rate"* — on
the grounds that choosing a threshold without the distribution would be guessing. Task 6 is
where that guess becomes a decision.

Source of truth: [`behavior.md`](../design/behavior.md) §7 items 9–13,
[ADR-0009](../adr/0009-test-pool-selection.md).

### Files

```
src/post_training_rl/dataset/selection.py
tests/test_dataset_selection.py
```

### Public interface

```python
def select_tests(
    row: Mapping[str, Any], config: DatasetConfig
) -> TestSelection | None      # None when the floor cannot be met
```

`TestSelection` carries `graded: tuple[TestCase, ...]`, `public: tuple[TestCase, ...]`,
`private_count: int`, `generated_count: int`. The last two exist solely so task 6 can measure
the mix — they are the deferred guard's evidence, and the type is where they belong rather
than a side channel.

### Unit tests

| Test | Asserts |
| --- | --- |
| `test_private_tests_are_preferred` | 20 private, 20 generated, cap 15 → all 15 private |
| `test_generated_tests_fill_only_the_shortfall` | 3 private, 20 generated, cap 15 → 3 private + 12 generated |
| `test_tests_are_ordered_longest_input_first` | rLLM's heuristic: longer inputs catch more bugs per execution |
| `test_selection_is_capped` | Never more than `max_tests_per_rollout` |
| `test_problem_below_the_floor_is_dropped` | 2 private + 2 generated, floor 5 → `None` |
| `test_floor_counts_both_pools` | 2 private + 3 generated, floor 5 → kept, not dropped |
| `test_public_tests_are_never_mixed_into_graded` | The hackability guarantee (ADR-0013) |
| `test_pool_provenance_is_recorded` | `private_count` and `generated_count` match the selection |
| `test_parallel_list_row_shape_is_handled` | Reads `{"input": [...], "output": [...]}`, the shape HF actually produces |

### Done when

All nine pass, and `TestSelection` carries enough for task 6 to compute the generated share
without re-reading the corpus.

---

## Task 4 — Prompt rendering and the length filter

### Behaviour

Renders the problem statement into the prompt template from config, then drops — never
truncates — any problem whose prompt exceeds the token budget. A truncated statement loses the
constraints or the I/O format and is unsolvable by construction, so training on it teaches the
model that some problems are simply impossible.

The template is a block scalar in YAML because it is a value a run depends on. It must also
render the `extraction.prefill` consumer correctly if prefill is ever enabled — ADR-0012 warns
the prefill value has **two** consumers, the dataset builder that renders the prompt and the
reward path that re-prepends before extraction, and they must be fed from one config key.

Source of truth: [`behavior.md`](../design/behavior.md) §7 item 7.

### Files

```
src/post_training_rl/dataset/prompt.py
tests/test_dataset_prompt.py
```

### Public interface

```python
def render_prompt(description: str, template: str) -> str
def prompt_token_count(prompt: str, tokenizer: PreTrainedTokenizerBase) -> int
```

Two small pure-ish functions rather than one that does both, because the measurement pass
needs the count over *every* row including dropped ones, while the builder needs the rendered
prompt only for rows that survive.

### Unit tests

A stub tokenizer — whitespace-splitting, with a `.encode` — keeps these sub-second and off the
network. The real tokenizer appears in task 6.

| Test | Asserts |
| --- | --- |
| `test_description_is_substituted_into_the_template` | |
| `test_template_comes_from_config_not_code` | Two templates render differently |
| `test_prompt_within_budget_is_kept` | |
| `test_over_length_prompt_is_dropped_not_truncated` | → `PROMPT_TOO_LONG`, and the prompt is not shortened |

### Done when

All four pass, and no prompt text appears in `prompt.py`.

---

## Task 5 — `build_dataset` and the drop report

### Behaviour

Runs the pipeline in cost order, attributes every drop to the first rule that fired,
emits the four columns, and returns the counts alongside the dataset.

Emits exactly `prompt`, `problem_id`, `graded_tests`, `public_tests`. Not `difficulty` —
it is decoded for the measurement pass but nothing in the loop consumes it, and adding a
column the trainer does not read is scope the plan does not have.

Source of truth: [`behavior.md`](../design/behavior.md) §7 items 3, 8;
[verifier-scorer §9](../design/verifier-scorer.md).

### Files

```
src/post_training_rl/dataset/build.py
tests/test_dataset_build.py
tests/test_dataset_corpus.py        # marked `corpus`
```

### Public interface

```python
def build_dataset(
    rows: Iterable[Mapping[str, Any]],
    config: DatasetConfig,
    tokenizer: PreTrainedTokenizerBase,
) -> BuildResult
```

**Deviation from `verifier-scorer.md` §9, recorded deliberately.** §9 specifies
`build_dataset(config, tokenizer)` loading internally. Taking `rows` gives two things the
original shape cannot: the measurement pass reuses one corpus pass instead of loading 13,328
rows twice, and every unit test here passes dict literals instead of needing 7.2 GB on disk.
`load_corpus(config)` still exists, so the composition root reads
`build_dataset(load_corpus(cfg), cfg, tok)`.

`BuildResult(dataset, report)` returns both because `behavior.md` §7.8 makes the drop counts a
*guarantee*, not a log line. A caller that cannot see them cannot check them.

### Unit tests

| Test | Asserts |
| --- | --- |
| `test_clean_corpus_survives_intact` | No filter fires; row count preserved |
| `test_emitted_columns_are_exactly_the_four` | No extra column creeps in |
| `test_each_drop_is_counted_against_its_reason` | One row per reason → one count each |
| `test_drop_counts_partition_the_corpus` | `sum(drops) + kept == total`. The invariant that makes the report checkable |
| `test_a_row_matching_two_rules_is_counted_once` | Attributed to the earlier rule |
| `test_report_surfaces_the_total` | §7.8's "the totals are surfaced" |
| `test_graded_and_public_round_trip_through_the_columns` | The Arrow-shape assumption `verifier-scorer.md` §12 item 2 flags as unverified |

### Integration test — marked `corpus`

| Test | Asserts |
| --- | --- |
| `test_full_corpus_is_not_truncated` | train **13,328**, valid **117**, test **165**. The 28% trap, pinned |

### Done when

Seven unit tests plus the corpus test pass, and `test_drop_counts_partition_the_corpus` fails
against an implementation that lets two rules both claim the same row.

---

## Task 6 — The measurement pass

### Behaviour

**This is the sprint's actual objective.** Everything before it is machinery to make it
possible. One pass over the real corpus, producing the four distributions, written to a report
that becomes the evidence for config values currently set by guess.

Source of truth: [`roadmap.md`](roadmap.md) §Sprint 2 "The four unknowns".

| Unknown | Decides | Recorded in |
| --- | --- | --- |
| `private_tests` count distribution | Whether the 15-cap and 5-floor are right, and whether generated filler dominates the graded signal | ADR-0009 |
| Private/generated share per problem | The guard ADR-0009 **deferred rather than rejected** | ADR-0009 |
| Prompt token-length distribution | `max_prompt_tokens`, and how many problems are dropped rather than truncated | ADR-0010 |
| Filter drop rate per rule | Whether the multi-output regex is too greedy — §4.6 puts multi-output at **~25%** of the validation set, so a filter removing far more is over-matching | ADR-0010 |

Two figures to check against, both from `rlvr-stack.md`: train averages **79.1** generated
tests per problem against ~190 for valid/test, and the ≥5-test filter was applied by DeepMind
**only to valid/test** — so the training split has had no such filter applied before ours.

### Files

```
scripts/measure_corpus.py
docs/measurements/codecontests.md      # the generated report, committed
tests/test_dataset_histogram.py
```

### Public interface

```python
def summarise(values: Sequence[float]) -> Summary   # count, min, p10, median, p90, max, mean
```

A script, not a module, because it runs once and its output is a document. The one piece with
logic worth pinning is the summary — percentiles are easy to get subtly wrong, and every
number in the report rests on them.

### Unit tests

| Test | Asserts |
| --- | --- |
| `test_summary_reports_the_percentiles` | Against a known sequence |
| `test_summary_of_a_single_value` | Degenerate case; no divide-by-zero |

### Done when

`docs/measurements/codecontests.md` exists and is committed, carrying all four distributions
for the training split with the tokenizer named. **Every config value the measurement
contradicts is updated from measurement rather than estimate, and the change is noted in the
report.** If the multi-output filter drops far more than ~25%, that is a finding to surface,
not a number to accept.

---

## Task 7 — Carried from sprint 1

Three items `sprint-01-status.md` §7 leaves open. None blocks sprint 2; all block sprint 3.

### 7a — The `backend` → adapter factory *(needs a decision first)*

`sandbox.backend` is loaded into `SandboxConfig` and **never read** — nothing maps
`"firejail"` to `FirejailSandbox`. Separately, the startup self-test asserts all four
containment properties regardless of backend, and `SubprocessSandbox` fails exactly one of
them — the network check, which `behavior.md` §4.13 already exempts it from while §9.3 aborts
on it unconditionally.

**Blocked on your answer to one question: do you ever want to run the loop without firejail?**

- *No* → wire the factory, and let it refuse `subprocess` outright. `backend` becomes a guard
  rather than a choice.
- *Yes* → wire the factory, and make the self-test assert only the guarantees the configured
  backend claims, with a loud warning that network isolation is absent and the configuration
  must not be used for training.

| Test | Asserts |
| --- | --- |
| `test_backend_name_selects_the_adapter` | `"firejail"` → `FirejailSandbox` |
| `test_unknown_backend_raises_naming_it` | No silent default (ADR-0005) |

### 7b — Uninstall torchvision

```bash
conda activate post-train && pip uninstall torchvision
```

Re-verify `from transformers import TrainerCallback` afterwards. Confirmed still broken as of
this plan. Sprint 2 does not need it — `AutoTokenizer` works — but sprint 3's cache reset
hooks into exactly that callback.

### 7c — Correct two stale references

- `sprint-01.md:503` points `hostile_programs.py` at `tests/fixtures/`; it is in `src/`
  (`sprint-01-status.md` §6.4).
- `rl-reward-functions.md` §3 does not define `ladder`'s timeout rung, so a rollout whose tests
  all time out scores 0.05, tied with a program that crashes on import. A spec line, not a code
  change (§7.2).

### Done when

Both tests pass, `TrainerCallback` imports, and the two documents say what the code does.

---

## Sprint definition of done

- [ ] `pytest -q` green — sprint 1's **63** plus this sprint's **36**, still sub-second
- [ ] `pytest -q -m corpus` green — the full corpus builds, and is not truncated
- [ ] The three sprint-1 suites still green — **63 / 4 / 9**
- [ ] The filtered dataset builds reproducibly from `config/dataset.yaml`
- [ ] Drop counts reported per rule, and they **partition** the corpus
- [ ] `docs/measurements/codecontests.md` committed with all four distributions
- [ ] Every config value the measurement contradicts updated, and the change noted
- [ ] Nothing imports `trl` or `peft`

## Review checkpoints

`/write-code` implements one task per invocation and reports between them. Review after
**task 3** (test selection carries ADR-0009's deferred decision), after **task 5** (the drop
accounting is what makes every later number trustworthy), and at the sprint gate.

Sprint 1's evidence for this: four real defects were caught by review that the tests did not,
and two of them were found independently by both review axes.

## Known risks carried into sprint 3

- **The multi-output filter is the only lever, and it is a keyword regex.** No field marks
  these problems. `project-status.md` already lists residue as an accepted risk; task 6 will
  say how large it is.
- **Arrow round-tripping of nested test structures is assumed, not verified**
  (`verifier-scorer.md` §12 item 2). `test_graded_and_public_round_trip_through_the_columns`
  is the first thing that touches it; if it fails, the adapter shape changes in sprint 3.
- **`TrainerCallback` remains blocked** until 7b runs.
