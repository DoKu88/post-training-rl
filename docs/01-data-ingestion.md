# Step 1 — Data Ingestion and Verification

**Dataset:** DeepMind **CodeContests** — HuggingFace `deepmind/code_contests`.

**Goal:** turn the raw CodeContests dataset into a clean, normalized, verified
set of training/eval records — with **no GPU and no model involved**.

**Deliverables:** `src/posttrain/data/schema.py`, `src/posttrain/data/ingest.py`,
`src/posttrain/data/prompts.py`, `scripts/inspect_data.py`, `tests/test_data.py`,
plus a cached processed dataset on disk.

> **Source note (applies to every sprint):** use the HuggingFace mirror — do
> **not** rebuild from the repo's protobuf/Riegeli files.
>
> ```python
> from datasets import load_dataset
> ds = load_dataset("deepmind/code_contests")   # splits: train / valid / test
> ```
>
> Each row has (fields we care about):
>
> | Field | Meaning |
> |-------|---------|
> | `name`, `description` | Problem title + full statement (stdin/stdout format) |
> | `public_tests`  | `{input: [...], output: [...]}` — visible to solver |
> | `private_tests` | hidden tests |
> | `generated_tests` | machine-generated hidden tests (large) |
> | `solutions` | `{language: [...], solution: [...]}` — reference solutions |
> | `difficulty` | integer difficulty band |
> | `cf_rating`, `cf_tags` | Codeforces metadata |
>
> **Critical fact:** these problems are **stdin → stdout**, not function-call
> based (unlike HumanEval/MBPP). The sandbox feeds a string to stdin and compares
> captured stdout. This shapes the reward in doc 2.

---

## Sprints at a glance

| Sprint | Name | Deliverable |
|--------|------|-------------|
| 1.1 | Normalized schema | `schema.py` — `Problem` / `TestCase` dataclasses |
| 1.2 | Ingestion pipeline | `ingest.py` — load, flatten, filter, cap, bucket, split, cache |
| 1.3 | Prompt formatting | `prompts.py` — model-agnostic chat template |
| 1.4 | Verification & inspection gate | `scripts/inspect_data.py` + data-quality checks |

Sprints are dependency-ordered: build 1.1 → 1.2 → 1.3 → 1.4.

---

## Sprint 1.1 — Normalized schema

**Deliverable:** `src/posttrain/data/schema.py` defining `TestCase` and `Problem`.

**Depends on:** —

**Build:** Define frozen dataclasses that are *our own* schema (not raw HF rows).
Everything downstream depends only on `Problem`, which is what makes doc 4's
"any dataset / any model" requirement tractable.

```python
from dataclasses import dataclass, field

@dataclass(frozen=True)
class TestCase:
    input: str
    output: str
    kind: str            # "public" | "private" | "generated"

@dataclass(frozen=True)
class Problem:
    id: str              # == name, slugified
    statement: str       # cleaned description
    tests: list[TestCase]
    difficulty: int
    reference_solutions: list[str]   # Python only, for verification
    source: str = "code_contests"
    def public(self):  return [t for t in self.tests if t.kind == "public"]
    def hidden(self):  return [t for t in self.tests if t.kind != "public"]
```

**Unit tests**
- `tests/test_data.py::test_testcase_is_frozen` — mutating a `TestCase` field raises `FrozenInstanceError`.
- `tests/test_data.py::test_problem_public_hidden_partition` — `public()` returns only `kind=="public"`, `hidden()` returns the rest, and together they cover all tests with no overlap.
- `tests/test_data.py::test_problem_defaults` — `source` defaults to `"code_contests"` when omitted.

**✅ Verify it works (you run)**
```bash
conda activate post-train
python -c "from posttrain.data.schema import Problem, TestCase; \
p = Problem(id='x', statement='s', tests=[TestCase('1','2','public'), TestCase('3','4','private')], difficulty=1, reference_solutions=['print(1)']); \
print('public:', len(p.public()), 'hidden:', len(p.hidden()), 'source:', p.source)"
```
Expected: prints `public: 1 hidden: 1 source: code_contests` with no traceback.

---

## Sprint 1.2 — Ingestion pipeline (`ingest.py`)

**Deliverable:** `src/posttrain/data/ingest.py` exposing `ingest(split, ...)`,
`save(problems, path)`, `load_cached(path)`, plus a cached processed dataset on disk.

**Depends on:** Sprint 1.1

**Build:** Convert raw HF rows into `list[Problem]` and cache them. Steps, in order:

1. **Flatten tests.** Merge `public_tests`, `private_tests`, `generated_tests`
   into a single `list[TestCase]`, tagging `kind`. Zip the parallel
   `input`/`output` lists.
2. **Extract Python reference solutions.** From `solutions`, keep only
   `language == PYTHON3` (enum value `3`). These are your *ground truth* for
   verifying the sandbox in doc 2 — a correct reference must score 1.0.
3. **Filter for trainability.** Drop a problem if it has:
   - zero tests, OR zero Python reference solutions (can't verify the pipeline),
   - a statement > N tokens (keep prompts manageable; e.g. 2048),
   - interactive / special-judge problems (output isn't a deterministic string
     match — detect via tags/description keywords; these break simple comparison).
4. **Cap tests per problem.** `generated_tests` can be huge. Keep *all* public +
   private, and **subsample generated to ≤ K** (e.g. 15) with a fixed seed so
   reward computation stays fast and deterministic. Log how many were dropped.
5. **Difficulty buckets.** Attach a coarse `easy/medium/hard` tag — doc 3's smoke
   test and curriculum learning use this.
6. **Split discipline.** Keep HF's `train`/`valid`/`test` split boundaries. The
   eval harness (doc 7) *only* ever touches `test`. Never leak.
7. **Cache.** Save processed `Problem` records to disk (`datasets` arrow or
   jsonl). Re-ingestion should be one cached load, not a re-download.

```python
def ingest(split: str, *, max_generated=15, max_stmt_tokens=2048,
           seed=0) -> list[Problem]: ...
def save(problems, path): ...
def load_cached(path) -> list[Problem]: ...
```

**Unit tests** (feed synthetic raw-row dicts — no network/GPU; keep the
per-step helpers importable so they can be tested in isolation)
- `tests/test_data.py::test_flatten_tests_tags_and_zips` — merging public/private/generated tags each `kind` correctly and zips input/output pairs 1:1.
- `tests/test_data.py::test_extract_python_solutions_filters_language` — only `language==3` (PYTHON3) solutions are kept; other languages dropped.
- `tests/test_data.py::test_filter_drops_untrainable` — rows with zero tests or zero Python refs are dropped; a valid row survives.
- `tests/test_data.py::test_cap_generated_is_deterministic` — generated tests capped at K, public+private always kept, and same seed yields identical subsample.
- `tests/test_data.py::test_difficulty_bucketing` — difficulty ints map to expected `easy/medium/hard` tags at bucket boundaries.
- `tests/test_data.py::test_save_load_roundtrip` — `load_cached(save(problems))` returns equal `Problem` records (round-trip preserves fields).

**✅ Verify it works (you run)**
```bash
conda activate post-train
python -m posttrain.data.ingest --split train --cache data/processed/train
```
Expected: logs kept-vs-dropped counts with reasons, writes a cache under
`data/processed/train`, and a re-run loads from cache without re-downloading.

---

## Sprint 1.3 — Prompt formatting

**Deliverable:** `src/posttrain/data/prompts.py` with `to_chat_prompt(p, tokenizer)`.

**Depends on:** Sprint 1.1

**Build:** A separate function, kept out of ingestion so prompt tweaks don't force
re-ingest. Using the tokenizer's own chat template is what keeps this
**model-agnostic** (doc 4) — Qwen, Llama, etc. each apply their own format.

```python
def to_chat_prompt(p: Problem, tokenizer) -> str:
    system = ("You are an expert competitive programmer. Read input from stdin, "
              "write the answer to stdout. Put your final solution in a single "
              "```python code block.")
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user",   "content": p.statement}],
        tokenize=False, add_generation_prompt=True)
```

**Unit tests** (use a lightweight fake tokenizer implementing
`apply_chat_template` — no model download, no GPU)
- `tests/test_data.py::test_prompt_includes_statement` — the returned prompt contains `p.statement`.
- `tests/test_data.py::test_prompt_passes_system_and_user_roles` — the fake tokenizer receives a system message then a user message with the problem statement.
- `tests/test_data.py::test_prompt_requests_add_generation_prompt` — `apply_chat_template` is called with `add_generation_prompt=True` and `tokenize=False`.

**✅ Verify it works (you run)**
```bash
conda activate post-train
python -c "from transformers import AutoTokenizer; from posttrain.data.ingest import load_cached; from posttrain.data.prompts import to_chat_prompt; \
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct'); \
p = load_cached('data/processed/train')[0]; \
print(to_chat_prompt(p, tok)[:600])"
```
Expected: prints a formatted chat prompt (system + user turns) ending with the
model's generation-prompt marker, with no template error.

---

## Sprint 1.4 — Verification & inspection gate

**Deliverable:** `scripts/inspect_data.py` (`--split --n`) and the data-quality
checks in `tests/test_data.py`. This is the "and verification" in the step title.

**Depends on:** Sprints 1.2, 1.3

**Build:** Automated invariant checks over a cached split, plus a human-readable
inspection script. **Manual gate:** do not proceed to doc 2 until reference
solutions look real — doc 2's sandbox test *depends* on them scoring 1.0.

**Unit tests** (run over a small cached fixture split; all model-free)
- `tests/test_data.py::test_every_problem_has_tests_and_ref` — every `Problem` has ≥1 test and ≥1 Python reference solution.
- `tests/test_data.py::test_test_io_lengths_aligned` — no misaligned zips: each test has a paired input and output (no empty/dangling side).
- `tests/test_data.py::test_no_train_test_leak` — no `test`-split id appears in `train` (leak check).
- `tests/test_data.py::test_prompt_roundtrips_through_tokenizer` — prompt formatting round-trips through the (fake) tokenizer without error for the target model.

**✅ Verify it works (you run)**
```bash
conda activate post-train
python scripts/inspect_data.py --split train --n 5
```
Expected: prints 5 random problems — statement + a reference solution + first
test each — that read as coherent, plus a per-difficulty-bucket count table.
Eyeball that reference solutions look real before moving to doc 2.

---

## Definition of done

- [ ] `schema.py` defines frozen `Problem` / `TestCase`; `public()`/`hidden()` partition tests correctly.
- [ ] Ingestion produces a cached processed dataset on disk; re-ingestion is one cached load, not a re-download.
- [ ] Counts logged: kept vs dropped, and why.
- [ ] Every `Problem` has ≥1 test and ≥1 Python reference solution; test input/output lengths aligned.
- [ ] No `test`-split id appears in `train` (no leak).
- [ ] Prompt formatting is model-agnostic and round-trips through the tokenizer without error.
- [ ] `scripts/inspect_data.py` output eyeballed — reference solutions look real.
- [ ] You can answer: "how many trainable problems per difficulty bucket?"
- [ ] `tests/test_data.py` green.

## Run all tests for this step

```bash
conda activate post-train
pytest tests/test_data.py -m "not gpu"
```
