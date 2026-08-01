# Step 5 — LoRA / QLoRA for Qwen2.5

**Goal:** make 7B actually *trainable* on a single 32GB card by freezing the base
and training small low-rank adapters — with 4-bit quantization (QLoRA) to fit the
double base-weight copies.

**Why mandatory:** full-FT of 7B needs ~85–110 GB (weights + grads + Adam). LoRA
freezes the base (no grads/optimizer on 7B params) → trainable memory drops to
the adapters (~sub-GB). Combined with 4-bit base, 7B fits with ~20 GB headroom.

---

## Sprints at a glance

| Sprint | Name | Deliverable | Depends on |
|--------|------|-------------|-----------|
| 5.1 | Adapter + quant plumbing | `apply_lora` + `build_quant_config` in the loader; base frozen, only adapters trainable, config-only toggles | Doc-4 loader |
| 5.2 | 7B QLoRA fits & steps under 32GB | 7B QLoRA trains a few steps within the memory budget + adapter↔vLLM hot-swap sanity | 5.1 |
| 5.3 | Merge for eval + reward ablation | `merge_and_unload` for eval + rising-reward probe on the easy subset | 5.2 |

---

## Sprint 5.1 — Adapter + quant plumbing

**Deliverable:** `apply_lora` and `build_quant_config` live in the doc-4 loader.
With `lora.enabled=true` the base is frozen and only adapters carry gradients; the
trainable-param fraction is <2%. Toggling `lora.enabled` / `quantization` is
config-only.

**Depends on:** Doc-4 loader (`posttrain.models.loader`).

**Build:**
- `apply_lora(model, lc)` — wraps the model with PEFT LoRA (QLoRA-aware).
- `build_quant_config()` — returns the 4-bit `BitsAndBytesConfig`.
- Wire both into the loader so `lora` / `quantization` are pure config switches.

```python
# src/posttrain/models/loader.py
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def apply_lora(model, lc: LoraCfg):
    if model.is_quantized:                       # QLoRA path
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True)
    peft_cfg = LoraConfig(
        r=lc.r, lora_alpha=lc.alpha, lora_dropout=lc.dropout,
        target_modules=lc.target_modules, bias="none",
        task_type="CAUSAL_LM")
    return get_peft_model(model, peft_cfg)
```

`build_quant_config` (also in loader) for the 4-bit base:
```python
BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                   bnb_4bit_compute_dtype=torch.bfloat16,
                   bnb_4bit_use_double_quant=True)
```

**LoRA settings that matter *for RL* (not SFT).** RLVR is a gentle nudge, but give
it capacity — RL benefits from more than SFT:

| Hyperparam | SFT-typical | **RLVR here** | Why |
|-----------|------------|--------------|-----|
| rank `r` | 8–16 | **32–64** | more capacity for the policy shift |
| `alpha` | `2r` | `2r` (64–128) | standard scaling |
| `target_modules` | `q,v` | **all linear** (`q,k,v,o,gate,up,down`) | RL needs to move MLPs too |
| dropout | 0.05 | 0.0–0.05 | small |
| LR | 1e-4 | **1e-5 – 1e-4** | higher than full-FT LR |

**Unit tests** (tiny model / CPU, model-free — no 7B load):
- `tests/test_lora.py::test_only_adapters_require_grad` — after `apply_lora`, every
  param with `requires_grad=True` is a LoRA param; all base params are frozen.
- `tests/test_lora.py::test_trainable_fraction_under_2pct` — trainable-param count
  is <2% of total params.
- `tests/test_lora.py::test_build_quant_config_nf4` — `build_quant_config()` returns
  `load_in_4bit=True`, `bnb_4bit_quant_type=="nf4"`, double-quant on, bf16 compute.
- `tests/test_lora.py::test_target_modules_all_linear` — RL config expands
  `target_modules` to `q,k,v,o,gate,up,down`.
- `tests/test_lora.py::test_lora_toggle_is_config_only` — `lora.enabled=false`
  returns an unwrapped base model; `true` returns a PEFT model (no code path change).

**✅ Verify it works (you run)**
```bash
conda run -n post-train pytest tests/test_lora.py -m "not gpu" -q
```
Expected: structural tests pass on a tiny stand-in model; only adapter params are
trainable and the trainable fraction prints <2%.

---

## Sprint 5.2 — 7B QLoRA fits & steps under 32GB

**Deliverable:** the real 7B model loads in 4-bit with LoRA adapters, runs a few
training steps inside the 32GB budget, and the adapter weights hot-swap into vLLM
so generation reflects the current policy.

**Depends on:** Sprint 5.1.

**Build:**
- Load 7B with the 4-bit `BitsAndBytesConfig` + `apply_lora`, grad checkpointing on.
- Run a handful of optimizer steps; capture peak VRAM.
- Exercise TRL's LoRA-aware vLLM path (base loaded once, adapter hot-swapped).

**Memory budget, 7B QLoRA on 32GB:**

| Component | ~Memory |
|-----------|---------|
| Trainer base (4-bit nf4) | ~5 GB |
| LoRA adapters + grads + Adam(8-bit) | <1 GB |
| Activations (grad checkpointing on) | ~2–4 GB |
| **vLLM base (4-bit AWQ/GPTQ)** | ~5–6 GB |
| KV cache (`gpu_mem_util≈0.45`) | remainder |
| **Total** | **fits with headroom** |

**Adapter sync to vLLM:** GRPO updates adapters each step; vLLM must generate with
the *current* policy. Use TRL's LoRA-aware vLLM path (loads base once, hot-swaps
adapter weights) rather than reloading the full model. Confirm your TRL/vLLM
versions support LoRA hot-swap in colocate mode — this is the most
version-sensitive integration point in the whole project.

**Known caveats (be honest):**
- **QLoRA is ~20–30% slower** per step than bf16, and 4-bit slightly lowers base
  quality. You trade speed/quality for "fits at all."
- **"LoRA learns less"** on *large* distribution shifts. RLVR is small-shift, so
  LoRA is generally adequate; if reward **plateaus low**, the first lever is
  **raise `r`** (32→64→128), then widen `target_modules` (already all-linear),
  then revisit KL coef.
- LoRA also **resists catastrophic forgetting** of the base instruct abilities —
  a genuine plus when the reward only covers competitive programming.

**Unit tests** (`@pytest.mark.gpu`):
- `tests/test_lora_gpu.py::test_7b_qlora_loads_under_budget` — 7B loads in 4-bit
  with adapters; post-load peak VRAM is well under 32GB.
- `tests/test_lora_gpu.py::test_7b_qlora_few_steps_under_32gb` — a few real steps
  run; peak `torch.cuda.max_memory_allocated()` stays under the 32GB budget.
- `tests/test_lora_gpu.py::test_adapter_hotswap_changes_generation` — after
  hot-swapping updated adapter weights into vLLM, generation reflects the new
  policy (output differs from the pre-swap base).

**✅ Verify it works (you run)**
```bash
conda run -n post-train pytest tests/test_lora_gpu.py -m gpu -q
nvidia-smi --query-gpu=memory.used --format=csv -l 1   # watch during the run
```
Expected: 7B QLoRA trains a few steps within 32GB — log peak `nvidia-smi`; adapter
hot-swap into vLLM succeeds and generations track the current adapter, not the base.

---

## Sprint 5.3 — Merge for eval + reward ablation

**Deliverable:** `merge_and_unload()` produces a dense model whose generations
match the adapter model's, and a 7B QLoRA ablation probe shows rising reward on the
easy subset.

**Depends on:** Sprint 5.2.

**Build:**
- Merge adapters into base for eval/deploy; keep adapters as the training checkpoint.
- Run the easy-subset probe (same curve check as doc 3, now on the real 7B target).

**Merge for eval/deploy.** Eval (doc 7) and any deployment should run the
**merged** model:
```python
merged = peft_model.merge_and_unload()   # base + adapter → dense weights
```
Keep adapters as the training checkpoint; merge only for inference/eval so vLLM
runs at full speed without the adapter path.

**Unit tests** (`@pytest.mark.gpu`):
- `tests/test_lora_gpu.py::test_merge_matches_adapter_generation` —
  `merge_and_unload()` output matches the adapter model under greedy decode
  (token-for-token sanity).
- `tests/test_lora_gpu.py::test_merged_model_no_adapter_path` — merged model has no
  remaining PEFT/adapter modules (runs at full inference speed).
- `tests/test_lora_gpu.py::test_easy_probe_reward_rises` — 7B QLoRA shows rising
  reward on the easy subset (later-window mean reward > early-window mean).

**✅ Verify it works (you run)**
```bash
conda run -n post-train pytest tests/test_lora_gpu.py::test_easy_probe_reward_rises -m gpu -q
conda run -n post-train pytest tests/test_lora_gpu.py::test_merge_matches_adapter_generation -m gpu -q
```
Expected: merged model matches the adapter model on greedy decode; 7B QLoRA reward
rises on the easy probe subset (same curve shape as doc 3, on the real target model).

---

## Definition of done

- With `lora.enabled=true`, **only adapter params** are trainable and the base is
  frozen; trainable-param count is <2% of total.
- 7B QLoRA trains within 32GB, and you can state peak VRAM and step time for the
  7B config.
- Adapter weights hot-swap into vLLM so generation reflects the current policy.
- `merge_and_unload()` produces a model whose generations match the adapter model's
  (greedy sanity).
- 7B QLoRA shows **rising reward** on the easy probe subset.
- Toggling `lora.enabled` / `quantization` is **config-only** (loader handles it).

## Run all tests for this step

```bash
# Everything (needs a 32GB GPU for the gpu-marked tests):
conda run -n post-train pytest tests/test_lora.py tests/test_lora_gpu.py -q

# CPU / structural subset only (no GPU required):
conda run -n post-train pytest tests/test_lora.py -m "not gpu" -q
```
