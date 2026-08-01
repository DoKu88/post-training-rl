# Step 3 — Dummy Test: 3B Model, No LoRA

**Goal:** prove the **RL wiring** end-to-end — rollout → reward → optimizer step →
reward-goes-up — on a small model that fits comfortably in 32GB with **full
fine-tuning (no LoRA)**. This is a *smoke test of the loop*, not a real training
run.

**Why 3B, no LoRA:** we want to remove *every* confound (LoRA correctness,
quantization, memory pressure) so that if the reward curve doesn't move, the bug
is in **our** code, not the config. See README principle #3.

**Model:** `Qwen/Qwen2.5-3B-Instruct` (fallback `Qwen/Qwen2.5-1.5B-Instruct`).

Package: `posttrain`. Tests: `tests/`. Env: conda `post-train` active. This is a
**wiring** smoke test — the point is proving the loop, not training a good coder.

---

## Memory sanity (3B full FT + colocated vLLM on 32GB)

| Component | ~Memory |
|-----------|---------|
| Training weights (bf16) | 6 GB |
| Grads (bf16) | 6 GB |
| Adam states | ~12 GB (this is why 7B full-FT is impossible; 3B just fits) |
| vLLM base copy (bf16) | 6 GB |
| KV cache + activations | remainder |

3B full-FT + Adam is ~24 GB before the vLLM copy — **tight**. Two clean options:

- **(a)** Full FT with vLLM in a **short generation window** and low
  `gpu_memory_utilization` (~0.25), tiny batch. Doable but fiddly.
- **(b) Recommended:** for the smoke test, use **8-bit Adam** (`bitsandbytes`)
  or gradient checkpointing to buy headroom, OR simply run the smoke test on
  **Qwen2.5-1.5B-Instruct** which fits trivially full-FT.

> The point of this step is *wiring correctness*, so if 3B full-FT + vLLM is
> tight, drop to 1.5B rather than adding LoRA — keep LoRA as the *one new
> variable* introduced in doc 5.

---

## Sprints at a glance

| Sprint | Name | Deliverable | Depends on |
|--------|------|-------------|------------|
| 3.1 | Config + model + vLLM generate | Smoke config loads, model + vLLM load, rollouts emit ```` ```python ```` blocks | — |
| 3.2 | Wire env + reward + trainer (3 steps) | Reward fn called, per-group rewards non-constant, 3 steps run with no OOM | 3.1 |
| 3.3 | Full short smoke run (100 steps) | 100-step run on easy subset; mean reward rises, KL finite, eval pass@1 ≥ step-0, W&B/TensorBoard logging | 3.2 |

---

## Sprint 3.1 — Config + model + vLLM generate

**Deliverable:** the smoke config (`configs/experiment/smoke_3b.yaml`) parses into a
typed config object, the model + tokenizer load, and vLLM produces completions that
contain ```` ```python ```` code blocks.

**Depends on:** —

**Build:**
- Add `configs/experiment/smoke_3b.yaml` (below) and a loader
  `posttrain.config.load_experiment(path)`.
- Add `posttrain.rollout.build_vllm(cfg)` + a `generate(prompts)` helper honoring
  `gpu_memory_utilization` and `max_completion_tokens`.

Config (`configs/experiment/smoke_3b.yaml`):

```yaml
defaults: [base]
model:
  name_or_path: Qwen/Qwen2.5-3B-Instruct
  quantization: none
  lora: {enabled: false}          # <-- no LoRA in this step
algo:
  name: grpo
  num_generations: 8              # group size G
  learning_rate: 1.0e-6
  kl_coef: 0.04
  max_prompt_tokens: 1024
  max_completion_tokens: 1024
data:
  dataset: deepmind/code_contests # CodeContests (default dataset)
  split: train
  difficulty: easy                # easy problems → non-zero reward density
  max_problems: 200               # tiny subset for a fast loop
train:
  per_device_batch_size: 1
  gradient_accumulation_steps: 8
  max_steps: 100                  # short — just watch the curve
  gradient_checkpointing: true
rollout:
  engine: vllm
  gpu_memory_utilization: 0.3
eval:
  every_steps: 25
  n_problems: 30
```

**Deliberate choices:**
- `difficulty: easy` — a 3B model must solve *something*, or all rewards are 0
  and there's no gradient (the sparse-reward trap from doc 2.3). Easy problems
  guarantee reward spread.
- `max_problems: 200`, `max_steps: 100` — this is a *minutes-to-hours* run, not
  days. You're debugging, not training.

**Unit tests**
- `tests/test_config.py::test_smoke_config_loads` — `load_experiment("configs/experiment/smoke_3b.yaml")` returns without error; **model-free**.
- `tests/test_config.py::test_smoke_config_no_lora` — asserts `cfg.model.lora.enabled is False` and `cfg.algo.name == "grpo"`; **model-free**.
- `tests/test_config.py::test_smoke_config_group_and_steps` — asserts `cfg.algo.num_generations == 8`, `cfg.train.max_steps == 100`, `cfg.data.difficulty == "easy"`; **model-free**.
- `tests/test_rollout.py::test_vllm_generates_python_block` — `@pytest.mark.gpu`; loads the model via vLLM, generates on 2 easy prompts, asserts each completion contains ```` ```python ````.

**✅ Verify it works (you run)**

```bash
conda activate post-train
# config parses (fast, no GPU)
python -c "from posttrain.config import load_experiment; c=load_experiment('configs/experiment/smoke_3b.yaml'); print(c.model.name_or_path, c.algo.num_generations)"
# smoke a few generations through vLLM
python -m posttrain.rollout.smoke --config configs/experiment/smoke_3b.yaml --n 2
```

Expected: config print shows `Qwen/Qwen2.5-3B-Instruct 8`; the rollout smoke prints 2
completions, each containing a ```` ```python ```` block, and does not OOM (drop to
`Qwen2.5-1.5B-Instruct` if VRAM is tight).

---

## Sprint 3.2 — Wire env + reward + trainer (3 steps)

**Deliverable:** the full loop is connected (data → prompt → rollout → reward →
advantage → update); running **3 steps** proves the reward fn is actually invoked,
returns non-constant values within a group, and the step completes with no OOM.

**Depends on:** 3.1

**Build:**
- Wire `posttrain.reward.code_reward(completion, tests)` and register it with the
  GRPO trainer so it's called once per completion.
- Assemble `posttrain.train.build_trainer(cfg)`; support a `--max-steps` override so
  a 3-step run is possible.

**Unit tests**
- `tests/test_reward.py::test_code_reward_passes_and_fails` — reward is high for a correct solution, low/0 for a wrong one; **model-free** (feeds canned completions).
- `tests/test_reward.py::test_group_rewards_non_constant` — given a group of mixed correct/incorrect completions, `reward_std > 0`; **model-free**.
- `tests/test_train.py::test_reward_fn_called` — with a stub rollout, running 1 trainer step asserts the reward fn was invoked `num_generations` times per prompt; **model-free** (monkeypatched generation).
- `tests/test_train.py::test_three_steps_no_oom` — `@pytest.mark.gpu`; runs the real loop for 3 steps and asserts it completes, logs `reward_mean`/`reward_std`, and per-group `reward_std > 0` for at least one group.

**✅ Verify it works (you run)**

```bash
conda activate post-train
# run just 3 steps of the real loop
python -m posttrain.train --config configs/experiment/smoke_3b.yaml --max-steps 3 2>&1 | tee /tmp/smoke_3step.log
# inspect the per-step reward stats
grep -E "step=|reward_mean|reward_std|frac_solved" /tmp/smoke_3step.log
```

Expected: 3 steps complete with no OOM; logs show `reward_mean`, `reward_std`, and
`frac_solved`; `reward_std > 0` for most groups (constant per-group reward → all
advantages 0 → nothing to learn).

---

## Sprint 3.3 — Full short smoke run (100 steps, easy subset)

**Deliverable:** the full 100-step smoke run on the 200-problem easy subset, with
W&B/TensorBoard logging, showing mean reward rising, KL finite/controlled, and eval
pass@1 on the 30-problem probe ≥ its step-0 value.

**Depends on:** 3.2

**Build:**
- Wire **Weights & Biases** or TensorBoard from day one:
  `reward_mean`, `reward_std`, `frac_solved`, `completion_length`, `kl`,
  `grad_norm`, `pass@1(eval)`. These same metrics carry through docs 4–7.
- Wire the eval probe (`eval.every_steps: 25`, `eval.n_problems: 30`) to log pass@1.

**Unit tests**
- `tests/test_metrics.py::test_reward_trend_upward` — given a synthetic ascending reward series, the trend helper reports a positive slope; **model-free**.
- `tests/test_metrics.py::test_kl_finite_guard` — the KL guard flags non-finite/exploding KL and passes finite KL; **model-free**.
- `tests/test_logging.py::test_logger_records_required_keys` — the logger emits all required keys (`reward_mean`, `reward_std`, `frac_solved`, `completion_length`, `kl`, `grad_norm`, `pass@1`); **model-free** (in-memory backend).
- `tests/test_train.py::test_smoke_reward_rises` — `@pytest.mark.gpu`; runs the 100-step smoke and asserts final-window `reward_mean` > initial-window, KL finite throughout, and eval pass@1 at the last probe ≥ step-0.

**✅ Verify it works (you run)**

```bash
conda activate post-train
# full short smoke run (minutes-to-hours)
python -m posttrain.train --config configs/experiment/smoke_3b.yaml 2>&1 | tee /tmp/smoke_full.log
# watch the curve live (TensorBoard) or check W&B run page
tensorboard --logdir runs/ &   # then open the reward_mean / kl / pass@1 panels
# quick post-hoc check from the log
grep -E "reward_mean|kl=|pass@1" /tmp/smoke_full.log | tail -n 40
```

Expected: **mean reward increases** over the 100 steps on the easy subset
(`reward_mean` trends upward); `reward_std > 0` for most groups; `kl` stays finite and
controlled (not exploding → gibberish); eval pass@1 at the final probe is ≥ its step-0
value.

If reward is flat: check (in order) — is the reward non-constant per group? are
completions well-formed? is the advantage non-zero? is the LR too small? This
ordering isolates env bug vs. optimizer config.

---

## Definition of done

This step is **not** about getting a good coder. Success = the plumbing is correct.
Verify **all** of:

- [ ] Training starts without OOM; a step completes. *(3.2)*
- [ ] vLLM rollouts produce completions containing ```` ```python ```` blocks. *(3.1)*
- [ ] Reward function is actually called and returns varied (non-constant) values
      within a group — log `reward_mean`, `reward_std`, `frac_solved`. *(3.2)*
- [ ] `reward_std > 0` for most groups (else advantages are all 0 → no learning). *(3.2)*
- [ ] **Mean reward trends upward** over ~100 steps on this tiny easy set
      (it should — the model can memorize/adapt to 200 easy problems). *(3.3)*
- [ ] KL to reference stays finite and controlled (not exploding → gibberish). *(3.3)*
- [ ] Eval pass@1 on the 30-problem probe is ≥ its step-0 value. *(3.3)*
- [ ] W&B/TensorBoard logs `reward_mean`, `reward_std`, `frac_solved`,
      `completion_length`, `kl`, `grad_norm`, `pass@1(eval)`. *(3.3)*

You now trust: data → prompt → rollout → reward → advantage → update → repeat.
Everything after this changes only *what model* and *what algorithm* plug into this
proven loop.

## Run all tests for this step

```bash
conda activate post-train
# everything (includes GPU-marked model tests)
pytest tests/ -v

# fast, model-free subset only (config parsing + wiring, no GPU)
pytest tests/ -v -m "not gpu"

# only the GPU model tests
pytest tests/ -v -m gpu
```

> Register the marker in `pyproject.toml`/`pytest.ini`:
> `markers = ["gpu: requires a CUDA GPU + model weights"]`.
