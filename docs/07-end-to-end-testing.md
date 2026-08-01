# Step 7 — End-to-End Testing & Evaluation

**Goal:** prove RL *actually improved coding ability* on held-out problems — not
just that training reward went up. This doc delivers the test infrastructure and
the eval harness themselves: a repeatable test pyramid + an honest eval harness.

---

## Sprints at a glance

| Sprint | Name | Delivers | Depends on |
|--------|------|----------|------------|
| **7.1** | Test conventions & pyramid infra | `tests/conftest.py`, `gpu` marker, `-m "not gpu"` laptop subset, CI wiring | docs 01–06 test files |
| **7.2** | End-to-end integration test | `tests/test_e2e.py` — 3 steps of the real loop, asserts data→reward→trainer contract | 7.1 |
| **7.3** | Eval harness `evaluate.py` | pass@1 / unbiased pass@k / per-difficulty / format% / length | 7.1, 7.2 |
| **7.4** | Experiment protocol & reproducibility | baseline↔trained compare, regression guards, config/seed/SHA pinning | 7.3 |

**Test pyramid (fast → slow)** — `tests/` mirrors this. Everything up to
model-swap runs on a **laptop with no GPU** (`-m "not gpu"`), the payoff of the
model-free design in docs 1–2.

| Layer | Scope | GPU? | Runs when |
|-------|-------|------|-----------|
| **Unit** | schema, comparators, prompt formatting | no | every commit (CI) |
| **Sandbox security** | timeout/mem/network/fork containment | no | every commit |
| **Reward** | reference→1.0, cheat→low, partial→mid | no | every commit |
| **Model-swap** | loader over tiny+real models | small GPU | pre-merge |
| **Algo-swap** | GRPO/RLOO/PPO each run N steps | small GPU | pre-merge |
| **Smoke integration** | 3 steps of the *real* pipeline, 5 problems | GPU | nightly |
| **Training probe** | rising reward on easy subset | GPU | on demand |

---

## Sprint 7.1 — Test conventions & pyramid infra

**Deliverable:** repo-wide pytest conventions — the `gpu` marker, `tests/conftest.py`,
the `-m "not gpu"` laptop-runnable subset, and CI wiring — that underpin the
per-sprint tests written back in docs 01–06.

**Depends on:** the test files authored in docs 01–06 (they import these conventions).

**Build:**
- Register a `gpu` marker in `pyproject.toml` (`[tool.pytest.ini_options] markers`)
  and a `tests/conftest.py` with shared fixtures (tiny-model path, tmp sandbox).
- Convention: any test touching a real GPU model is decorated `@pytest.mark.gpu`;
  laptop CI runs `pytest -m "not gpu"`, nightly runs the full set.
- CI: a `test` job runs `-m "not gpu"` on every push; a `gpu-nightly` job runs all.

**Unit tests**
- `tests/test_conventions.py::test_gpu_marker_registered` — the `gpu` marker is in
  `config.getini("markers")` (no "unknown marker" warning).
- `tests/test_conventions.py::test_not_gpu_selection_excludes_gpu` — using pytest's
  `pytester`, a dummy `@pytest.mark.gpu` test is **deselected** under `-m "not gpu"`
  and a plain test is **selected**.
- `tests/test_conventions.py::test_conftest_fixtures_available` — `tmp_sandbox` /
  tiny-model fixtures resolve without a GPU.

**✅ Verify it works (you run)**
```bash
conda activate post-train
pytest -m "not gpu" -q                     # laptop subset
pytest --markers | grep gpu                # marker is registered
pytest -m "gpu" --collect-only -q | tail   # shows the GPU-only set that CI defers
```
`Expected:` `-m "not gpu"` collects and passes the laptop subset green; `--markers`
lists `@pytest.mark.gpu`; the `gpu` collection lists only GPU-decorated tests.

---

## Sprint 7.2 — End-to-end integration test

**Deliverable:** `tests/test_e2e.py` — the whole loop, minimized, guarding the
data→reward→trainer contract.

**Depends on:** 7.1 (marker + conftest fixtures).

**Build:** wire the minimal real pipeline and assert the contract holds:
```
ingest(5 easy problems)
  → CodeContestEnv(reward)
  → load_policy(tiny or 1.5B, config)
  → build_trainer("grpo", ...)
  → trainer.train(max_steps=3)
  → assert: no crash, reward_fn was called, checkpoint written,
            metrics logged, reward values ∈ [0,1]
```
This is the guardrail: if a refactor breaks the contract, this fails in minutes.

**Unit tests** *(the fast, model-free half of the contract)*
- `tests/test_e2e.py::test_reward_fn_invoked_and_bounded` — with a stubbed policy,
  `reward_fn` is called ≥1× and every returned reward ∈ [0, 1].
- `tests/test_e2e.py::test_contract_ingest_to_env` — 5 ingested problems produce a
  `CodeContestEnv` whose prompts/tests are well-formed (schema-valid, model-free).
- `tests/test_e2e.py::test_e2e_smoke` — **`@pytest.mark.gpu`** — 3 real steps on a
  tiny/1.5B model: no crash, checkpoint written, metrics logged.

**✅ Verify it works (you run)**
```bash
conda activate post-train
pytest tests/test_e2e.py -m "not gpu" -q    # contract checks, no GPU
pytest tests/test_e2e.py -m gpu -q          # 3 real steps (needs GPU)
```
`Expected:` model-free contract tests green on a laptop; the GPU smoke test runs 3
steps, writes a checkpoint, and logs reward values ∈ [0,1] with no crash.

---

## Sprint 7.3 — Evaluation harness (`src/posttrain/eval/evaluate.py`)

**Deliverable:** `evaluate.py` measuring held-out capability: pass@1, unbiased
pass@k, per-difficulty, format-validity %, mean completion length.

**Depends on:** 7.1 (markers), 7.2 (same env/reward the contract test exercises).

**Build:** **Training reward ≠ capability.** Measure on the **held-out `test`
split**, which training never touched (doc 1.6).
```python
def evaluate(model, problems, reward, *, k=1, temperature=...) -> EvalReport:
    # generate k completions per problem via vLLM (merged model),
    # score with the SAME reward/sandbox used in training,
    # aggregate pass@1, pass@k (unbiased), per-difficulty, format%, length.
```
Metrics:
- **pass@1** — greedy (or 1 sample) solve rate. Primary headline number.
- **pass@k** — `k∈{1,5,10}`; the **unbiased** pass@k estimator
  `1 − C(n−c, k)/C(n, k)`. Shows whether RL sharpened the distribution or just
  reallocated probability mass.
- **By difficulty** — easy/medium/hard breakdown (RL often helps easy/medium most).
- **Format validity** — % completions with a parseable code block (a drop signals
  degeneration).
- **Mean completion length** — watch for length hacking / rambling.

**Use the merged model** (doc 5.5) and the **same sandbox+comparator** as training
— consistency is the point; a separate eval judge invites disagreement.

**Unit tests** *(estimator + report schema tested model-free on synthetic data)*
- `tests/test_eval_harness.py::test_pass_at_k_unbiased_known_values` — the
  estimator on hand-computed `(n, c, k)` cases (e.g. n=5,c=2,k=1 → 0.4;
  n=5,c=0,k=5 → 0.0; c≥k with c=n → 1.0) matches `1 − C(n−c,k)/C(n,k)`.
- `tests/test_eval_harness.py::test_pass_at_k_le_one_ge_pass_at_1` — for random
  synthetic `(n,c)`, pass@k ∈ [0,1] and monotonically non-decreasing in k.
- `tests/test_eval_harness.py::test_eval_report_schema` — `EvalReport` carries
  pass@1, pass@k dict, per-difficulty, format%, length with correct types.
- `tests/test_eval_harness.py::test_format_and_length_aggregation` — on synthetic
  completions (some with/without code blocks), format% and mean length are exact.
- `tests/test_eval_harness.py::test_evaluate_real_small_model` —
  **`@pytest.mark.gpu`** — `evaluate()` on a small merged model over a few
  `test` problems returns a populated, well-formed `EvalReport`.

**✅ Verify it works (you run)**
```bash
conda activate post-train
pytest tests/test_eval_harness.py -m "not gpu" -q          # estimator + schema
python -m posttrain.eval.evaluate \
    --model <merged_ckpt> --split test --k 1,5,10           # real report (GPU)
```
`Expected:` `-m "not gpu"` green (estimator matches closed form); the CLI prints a
pass@1/pass@k report with per-difficulty rows, format%, and mean length.

---

## Sprint 7.4 — Experiment protocol & reproducibility

**Deliverable:** the trustworthy-results workflow — baseline→train→eval→compare→
regression-guard — plus full reproducibility pinning.

**Depends on:** 7.3 (the eval harness produces the numbers being compared).

**Build:** the experiment protocol (what makes results trustworthy):
1. **Baseline first.** Eval the *untrained* Qwen2.5-7B-Instruct on `test`. The
   number to beat. Record it before any training.
2. **Train** with the chosen config (doc 4/5/6).
3. **Eval the merged checkpoint** on the *same* `test` set.
4. **Compare** pass@1/pass@k vs baseline, per difficulty. Report the delta.
5. **Guard against regressions:** confirm format-validity and length didn't degrade
   (RL can raise reward while quietly breaking output structure).
6. **Optional sanity:** eval on a general benchmark (e.g. a small HumanEval slice)
   to check the model didn't overfit to code_contests I/O style — LoRA should limit
   this (doc 5.4).

**Reproducibility:**
- Every run pinned by its resolved config (log the fully-merged YAML) + git SHA +
  dataset cache hash + seed.
- Checkpoints named `{model}-{algo}-{step}`; keep the config alongside.
- Log the exact test-subsample seed — reward depends on which generated tests were
  sampled (doc 1.4).

**Unit tests** *(guards & pinning are testable model-free)*
- `tests/test_protocol.py::test_regression_guard_flags_format_drop` — a guard given
  a trained report with lower format% than baseline **fails**; equal/higher passes.
- `tests/test_protocol.py::test_regression_guard_flags_length_blowup` — length
  above a threshold multiple of baseline is flagged.
- `tests/test_protocol.py::test_compare_reports_delta` — comparator emits correct
  per-difficulty pass@1/pass@k deltas for two synthetic `EvalReport`s.
- `tests/test_protocol.py::test_run_manifest_pins_config_sha_seed` — a completed run
  writes a manifest containing merged-config hash, git SHA, dataset hash, and seed.
- `tests/test_protocol.py::test_baseline_vs_trained_end_to_end` —
  **`@pytest.mark.gpu`** — baseline and 3-step-trained checkpoints both eval and the
  compare step produces a signed delta.

**✅ Verify it works (you run)**
```bash
conda activate post-train
pytest tests/test_protocol.py -m "not gpu" -q               # guards, compare, manifest
python -m posttrain.eval.evaluate --model <baseline> --split test --k 1,5,10
python -m posttrain.eval.evaluate --model <trained>  --split test --k 1,5,10
# compare + regression guard step prints the delta table
```
`Expected:` guard/compare/manifest tests green; baseline and trained reports print,
the compare step shows a per-difficulty pass@1/pass@k delta, and the regression
guard passes (no format/length degradation).

---

## Run all tests for this step

```bash
conda activate post-train
pytest -m "not gpu" -q                 # full laptop subset (docs 01–07)
pytest -q                              # everything, incl. @pytest.mark.gpu (needs GPU)
pytest tests/test_e2e.py tests/test_eval_harness.py tests/test_protocol.py -q
```

## Definition of done

- [ ] Repo-wide conventions in place: `gpu` marker registered, `tests/conftest.py`
      present, `pytest -m "not gpu"` selects the laptop subset, CI runs it per push.
- [ ] `pytest -m "not gpu"` green across the pyramid (unit → e2e contract → eval
      estimator → protocol guards); full `pytest` green on GPU.
- [ ] `tests/test_e2e.py` guards the data→reward→trainer contract (reward_fn
      invoked, rewards ∈ [0,1], checkpoint written, metrics logged).
- [ ] pass@k uses the **unbiased** estimator, unit-tested against closed-form values.
- [ ] Baseline vs. trained **pass@1 on held-out `test`** reported, with the
      **trained model higher** on at least easy+medium.
- [ ] pass@k reported for k∈{1,5,10}.
- [ ] No regression in format-validity or a blow-up in completion length (enforced
      by the regression guard).
- [ ] Eval uses the **merged** model + the **same sandbox/comparator** as training.
- [ ] The *entire* pipeline reproduces from a single command +
      `configs/experiment/train_7b.yaml`, with config/SHA/dataset-hash/seed pinned
      in a run manifest.
- [ ] Swapping model (doc 4) and algorithm (doc 6) demonstrated to be config-only.

---

## Wrap-up: the through-line

```
doc1 data ─┐
           ├─▶ doc2 env+reward+loss ─▶ doc3 prove wiring (3B) ─▶ doc4 swap 7B
sandbox ───┘                                                        │
                                                                    ▼
   doc7 e2e + eval  ◀── doc6 algo swap (GRPO/PPO/RLOO) ◀── doc5 LoRA/QLoRA fits 7B
```

Build the **model-free** core first (docs 1–2), prove the **loop** on something
tiny (doc 3), then vary **model** (4), **memory strategy** (5), and **algorithm**
(6) independently — each is one isolated variable plugging into a loop you already
trust. Doc 7 keeps you honest that it worked.
