# Step 6 — Interchangeable RL Algorithms (GRPO / PPO / RLOO)

**Goal:** make the learning algorithm a **config choice**, using **standard
libraries** (TRL) — so you can run the *same* env/reward/model with GRPO today,
PPO or RLOO tomorrow, and compare.

**Requirement (verbatim):** *"make this interchangeable so that we can try
multiple of these"* + *"use standard libraries."*

## The seam: a trainer factory

All algorithms share the same inputs — **(model, tokenizer, prompt-dataset,
reward_fn)** — and differ only in the trainer class + its config. So a factory is
enough; no custom RL math. `train.py` never names an algorithm — it calls
`build_trainer(...).train()`, and switching algorithm is a **one-line config
change** (`algo._file: algo/ppo.yaml`).

```python
# src/posttrain/algos/factory.py
from trl import GRPOTrainer, GRPOConfig, PPOTrainer, PPOConfig, RLOOTrainer, RLOOConfig

def build_trainer(cfg, model, tokenizer, env):
    common = dict(model=model, processing_class=tokenizer,
                  train_dataset=env.as_prompt_dataset())
    if cfg.algo.name == "grpo":
        return GRPOTrainer(args=GRPOConfig(**cfg.algo.grpo),
                           reward_funcs=[env.reward_fn], **common)
    if cfg.algo.name == "rloo":
        return RLOOTrainer(args=RLOOConfig(**cfg.algo.rloo),
                           reward_funcs=[env.reward_fn], **common)
    if cfg.algo.name == "ppo":
        return PPOTrainer(args=PPOConfig(**cfg.algo.ppo),
                          reward_model=RewardFnAdapter(env.reward_fn),
                          value_model=make_value_head(model), **common)
    raise ValueError(cfg.algo.name)
```

**The three algorithms, and what changes.** All optimize the **same reward**
(doc 2.2); they differ only in how reward → advantage → loss.

| | **GRPO** (default) | **RLOO** | **PPO** |
|---|---|---|---|
| Baseline | group mean | leave-one-out group mean | **learned value head (critic)** |
| Advantage | `(r−mean)/std` over group | `r_i − mean(r_{j≠i})` | GAE over value estimates |
| Extra model in VRAM | none | none | **value model (~2× memory)** |
| Loss terms | clip policy + KL | clip policy + KL | clip policy + **value loss** + KL (+ entropy) |
| Fit on 32GB? | ✅ best | ✅ | ⚠️ tight — critic competes with the double base copies |
| Tuning burden | low | low | higher |

**Loss recap (from doc 2.3):**

- **GRPO / RLOO:** `L = −E[min(ratio·A, clip(ratio,1±ε)·A)] + β·KL(π‖π_ref)`
  — the only difference is how `A` is computed (group-std vs leave-one-out).
- **PPO:** `L = L_policy(clip) + c_v·L_value + c_kl·KL − c_ent·entropy`, with a
  critic producing GAE advantages.

**Recommendation for this hardware:** **GRPO is the default** — no critic, lowest
memory, well-suited to verifiable binary-ish rewards. **RLOO** is a cheap
alternative baseline. **PPO** is included for completeness/comparison but its
value model competes for the ~20 GB headroom you have after the double 4-bit base
copies — expect to shrink batch/KV-cache to fit it, or run PPO on 3B only.

---

## Sprints at a glance

| Sprint | Name | Deliverable | Depends on |
|---|---|---|---|
| 6.1 | Factory + GRPO | `build_trainer` factory dispatching on `algo.name`; GRPO path trains | Step 5 (env/reward), config loader |
| 6.2 | RLOO via same interface | RLOO trainer through the same factory + `algo._file` | 6.1 |
| 6.3 | PPO via same interface | PPO trainer (value head/critic) through the same factory | 6.1 |
| 6.4 | Shared reference/KL model + swap regression | Adapter-disabled `π_ref` (no third copy) + config-only swap proof | 6.1, 6.2, 6.3 |

---

## Sprint 6.1 — Factory + GRPO

**Deliverable:** `build_trainer(cfg, model, tokenizer, env)` returns the correct
TRL trainer for `algo.name`, and the GRPO path trains end-to-end on the smoke
config.

**Depends on:** Step 5 (env exposing `as_prompt_dataset()` + `reward_fn`), config
loader that resolves `algo._file`.

**Build:**
- Implement `posttrain/algos/factory.py::build_trainer` with the GRPO branch and
  a `ValueError` for unknown names.
- Add `configs/algo/grpo.yaml`; ensure `train.py` calls the factory and never
  names an algorithm.

**Unit tests**
- `tests/test_algos_factory.py::test_dispatch_grpo` — with a mocked/stubbed
  `trl.GRPOTrainer`, `build_trainer(cfg(name="grpo"), …)` constructs and returns a
  `GRPOTrainer` instance (model-free, no GPU).
- `tests/test_algos_factory.py::test_dispatch_unknown_raises` — `algo.name="xyz"`
  raises `ValueError`.
- `tests/test_algos_factory.py::test_grpo_receives_reward_funcs` — the GRPO branch
  passes `reward_funcs=[env.reward_fn]` and `train_dataset=env.as_prompt_dataset()`
  through (assert on mock call kwargs).
- `tests/test_grpo_run.py::test_grpo_trains_few_steps` `@pytest.mark.gpu` — GRPO
  runs N steps on the 1.5B smoke config and mean reward at step N ≥ step 0.

**✅ Verify it works (you run)**
```bash
conda run -n post-train python -m posttrain.train algo._file=algo/grpo.yaml \
    model._file=model/1p5b.yaml train.max_steps=20
```
Expected: GRPO runs 20 steps with no code changes; logged reward trends upward on
the easy set.

---

## Sprint 6.2 — RLOO via same interface

**Deliverable:** RLOO trains through the **same** `build_trainer` factory,
selected purely by config (`algo._file: algo/rloo.yaml`), using the leave-one-out
group-mean baseline.

**Depends on:** 6.1.

**Build:**
- Add the RLOO branch to `build_trainer` (`RLOOTrainer` + `RLOOConfig`), reusing
  the shared `common` kwargs and `reward_funcs=[env.reward_fn]`.
- Add `configs/algo/rloo.yaml` mirroring `grpo.yaml`'s knobs.

**Unit tests**
- `tests/test_algos_factory.py::test_dispatch_rloo` — with `trl.RLOOTrainer`
  mocked, `build_trainer(cfg(name="rloo"), …)` returns an `RLOOTrainer` (model-free).
- `tests/test_algos_factory.py::test_rloo_same_common_kwargs` — RLOO branch passes
  the identical `model/processing_class/train_dataset` `common` kwargs as GRPO
  (assert equal call kwargs across both mocked branches).
- `tests/test_rloo_run.py::test_rloo_trains_few_steps` `@pytest.mark.gpu` — RLOO
  runs N steps on the 1.5B smoke config and mean reward at step N ≥ step 0.

**✅ Verify it works (you run)**
```bash
conda run -n post-train python -m posttrain.train algo._file=algo/rloo.yaml \
    model._file=model/1p5b.yaml train.max_steps=20
```
Expected: RLOO runs 20 steps via the `algo._file` change **only** (no code diff
from the GRPO run); logged reward rises on the easy set.

---

## Sprint 6.3 — PPO via same interface

**Deliverable:** PPO trains through the **same** factory, wiring the reward-fn
adapter + value head (critic) so the value-model path is exercised end-to-end.

**Depends on:** 6.1.

**Build:**
- Add the PPO branch: `RewardFnAdapter(env.reward_fn)` as `reward_model` and
  `make_value_head(model)` as `value_model` alongside `PPOConfig`.
- Add `configs/algo/ppo.yaml`; note the critic's ~2× memory — PPO may run on 3B
  only (shrink batch/KV-cache to fit the value model in the ~20 GB headroom).

**Unit tests**
- `tests/test_algos_factory.py::test_dispatch_ppo` — with `trl.PPOTrainer` mocked,
  `build_trainer(cfg(name="ppo"), …)` returns a `PPOTrainer` (model-free).
- `tests/test_algos_factory.py::test_ppo_wires_value_and_reward_model` — PPO branch
  passes non-null `reward_model` and `value_model` kwargs (assert both present in
  the mock call; `make_value_head`/`RewardFnAdapter` invoked).
- `tests/test_ppo_run.py::test_ppo_trains_few_steps` `@pytest.mark.gpu` — PPO runs
  N steps on the 3B smoke config without error (proves the value-model path is
  wired; reward need not rise).

**✅ Verify it works (you run)**
```bash
conda run -n post-train python -m posttrain.train algo._file=algo/ppo.yaml \
    model._file=model/3b.yaml train.max_steps=20
```
Expected: PPO runs 20 steps via the `algo._file` change **only**; the value-model
path executes without OOM on 3B (shrink batch/KV-cache if tight).

---

## Sprint 6.4 — Shared reference/KL model + swap regression

**Deliverable:** all three algorithms regularize against a frozen `π_ref` that is
the **base weights with the LoRA adapter disabled** — no *third* model copy — and
a regression proving the algorithm switch is **config-only**.

**Depends on:** 6.1, 6.2, 6.3.

**Build:**
- Confirm/configure TRL's PEFT adapter toggling so `π_ref` = base with adapter
  disabled (no separate reference model instantiated) for GRPO/RLOO/PPO.
- Add `tests/test_algo_swap.py` covering the config-only swap across all three.

**Unit tests**
- `tests/test_ref_model.py::test_no_third_model_copy` — when a PEFT/LoRA model is
  passed, the built trainer's reference is the adapter-disabled base (assert
  `trainer.ref_model is None` / adapter-toggle path taken, not a new copy).
- `tests/test_algo_swap.py::test_swap_is_config_only` — for
  `name in {grpo, rloo, ppo}`, `build_trainer` dispatches to the matching TRL class
  via `algo._file` differences alone, with identical `train.py`/factory call path
  (mocked trainers, model-free).
- `tests/test_algo_swap.py::test_metrics_keys_comparable` — each trainer config
  logs the same reward/eval metric keys, so algorithms are side-by-side comparable.
- `tests/test_algo_swap.py::test_all_three_run_few_steps` `@pytest.mark.gpu` — each
  of grpo/rloo/ppo runs N steps on the smoke config via `algo._file` only; GRPO &
  RLOO show rising reward, PPO at least completes.

**✅ Verify it works (you run)**
```bash
for algo in grpo rloo ppo; do
  conda run -n post-train python -m posttrain.train \
      algo._file=algo/${algo}.yaml model._file=model/3b.yaml train.max_steps=20
done
```
Expected: each of grpo/rloo/ppo runs 20 steps via the `algo._file` change **only**
(no code edits between runs); GRPO & RLOO show rising reward, PPO completes; no
third model copy is loaded for `π_ref` (adapter-disabled base is reused).

---

## Definition of done

- Algorithm is a **one-line config switch** (`algo._file`); `train.py` is
  algorithm-agnostic and calls only `build_trainer(...).train()`.
- `build_trainer` returns the correct TRL trainer for each `algo.name` and raises
  on unknown names.
- GRPO (primary) trains 7B; RLOO and PPO at least run end-to-end (PPO possibly on
  3B only due to the critic's memory cost).
- `π_ref` is the adapter-disabled base — no third model copy — for all three.
- You can produce a side-by-side reward/eval comparison of ≥2 algorithms on the
  same data + model (comparable, same-keyed metrics).

## Run all tests for this step

```bash
# Full suite (includes GPU few-steps-per-algorithm runs)
conda run -n post-train pytest tests/test_algos_factory.py tests/test_grpo_run.py \
    tests/test_rloo_run.py tests/test_ppo_run.py tests/test_ref_model.py \
    tests/test_algo_swap.py

# Fast, CPU-only subset (factory dispatch + config-only swap, no GPU)
conda run -n post-train pytest tests/test_algos_factory.py tests/test_algo_swap.py \
    tests/test_ref_model.py -m "not gpu"
```
