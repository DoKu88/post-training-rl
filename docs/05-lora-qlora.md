# Step 5 — LoRA / QLoRA for Qwen2.5

**Goal:** make 7B actually *trainable* on a single 32GB card by freezing the base
and training small low-rank adapters. The base **must be quantized** (bf16 doesn't
fit — see below), so we **test both 8-bit and 4-bit** QLoRA and **prefer 8-bit if
it fits with a usable KV cache and acceptable step time** (8-bit keeps more of the
base model's quality); **4-bit is the fallback**.

**Why quantization is mandatory (and plain bf16 LoRA is not enough):** full-FT of
7B needs ~85–110 GB. LoRA removes the base's grad/optimizer memory (→ sub-GB
adapters) but does **nothing** about the *two resident base copies* GRPO keeps on
the GPU at once — the transformers **trainer** copy and the vLLM **rollout** copy.
Two **bf16** copies alone are ~30 GB → OOM. Quantizing the base is the only lever
that shrinks them:

| Base precision | 2 copies | +adapters+activations | KV-cache room (of ~30GB usable) | Fits 32GB? |
|---|---|---|---|---|
| bf16 (plain LoRA) | ~30 GB | ~34 GB | negative | ❌ **OOM** |
| **8-bit (preferred)** | ~15 GB | ~19 GB | ~11 GB | ✅ |
| **4-bit nf4 (fallback)** | ~10 GB | ~14 GB | ~16 GB | ✅ comfortable |

> **Policy — decided empirically in Sprint 5.2:** use **8-bit** if its fit-check
> passes with a usable KV cache and acceptable step time; **otherwise fall back to
> 4-bit**. (If you want a full-precision base with plain LoRA, that only fits at
> **3B** — the doc-3 smoke path — not 7B.)

---

## Sprints at a glance

| Sprint | Name | Deliverable | Depends on |
|--------|------|-------------|-----------|
| 5.1 | Adapter + quant plumbing | `apply_lora` + `build_quant_config` in the loader; base frozen, only adapters trainable, config-only toggles | Doc-4 loader |
| 5.2 | 7B QLoRA fits & steps (8-bit vs 4-bit) | Fit-check **both** precisions under 32GB, pick 8-bit if it fits + adapter↔vLLM hot-swap sanity | 5.1 |
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

`build_quant_config` (also in loader) supports **both** quantized bases — the
config field `quantization` selects `"8bit"` or `"4bit"` (or `"none"`):
```python
def build_quant_config(spec: str):        # "none" | "8bit" | "4bit"
    if spec == "none":  return None
    if spec == "8bit":                    # preferred — higher fidelity base
        return BitsAndBytesConfig(load_in_8bit=True)
    if spec == "4bit":                    # fallback — smallest footprint
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.bfloat16,
                                  bnb_4bit_use_double_quant=True)
    raise ValueError(spec)
```
`apply_lora`'s `prepare_model_for_kbit_training` handles **both** 8-bit and 4-bit
(it's k-bit-agnostic), so the LoRA path is identical regardless of which we pick.

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
- `tests/test_lora.py::test_build_quant_config_8bit` — `build_quant_config("8bit")`
  returns a `BitsAndBytesConfig` with `load_in_8bit=True`.
- `tests/test_lora.py::test_build_quant_config_nf4` — `build_quant_config("4bit")`
  returns `load_in_4bit=True`, `bnb_4bit_quant_type=="nf4"`, double-quant on, bf16 compute.
- `tests/test_lora.py::test_quant_toggle_is_config_only` — `"none"|"8bit"|"4bit"`
  are all selectable via the `quantization` config field with no code-path change
  (`"none"` → `None`).
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

## Sprint 5.2 — 7B QLoRA fits & steps (8-bit vs 4-bit)

**Deliverable:** the real 7B model loads with LoRA adapters in **both** 8-bit and
4-bit, each runs a few training steps, and we **measure peak VRAM + KV headroom +
step time for both** to decide which to use. **Pick 8-bit if it fits with a usable
KV cache and acceptable step time; else 4-bit.** The chosen adapter also hot-swaps
into vLLM so generation reflects the current policy.

**Depends on:** Sprint 5.1.

**Build:**
- Load 7B twice over — once `quantization=8bit`, once `quantization=4bit` — each
  with `apply_lora`, grad checkpointing on.
- For each: run a handful of optimizer steps; capture peak
  `torch.cuda.max_memory_allocated()`, remaining KV-cache room, and step time.
- Apply the **decision rule** (8-bit preferred) and record the winner in config.
- Exercise TRL's LoRA-aware vLLM path (base loaded once, adapter hot-swapped).

**Memory budget, 7B on 32GB — the two candidates:**

| Component | **8-bit (preferred)** | **4-bit nf4 (fallback)** |
|-----------|:---------------------:|:------------------------:|
| Trainer base | ~7.6 GB | ~5 GB |
| LoRA adapters + grads + Adam(8-bit) | <1 GB | <1 GB |
| Activations (grad checkpointing on) | ~2–4 GB | ~2–4 GB |
| vLLM base copy | ~7–8 GB (bnb int8) | ~5–6 GB (AWQ/GPTQ) |
| KV cache (`gpu_mem_util`) | ~11 GB remainder | ~16 GB remainder |
| **Total** | **fits (tighter)** | **fits (comfortable)** |

**8-bit is the tighter fit** — the empirical question is whether its ~11 GB of KV
headroom sustains a usable rollout batch at your prompt/completion lengths. If KV
pressure forces a tiny batch or OOMs, fall back to 4-bit.

**Adapter sync to vLLM:** GRPO updates adapters each step; vLLM must generate with
the *current* policy. Use TRL's LoRA-aware vLLM path (loads base once, hot-swaps
adapter weights) rather than reloading the full model. Confirm your TRL/vLLM
versions support LoRA hot-swap in colocate mode — this is the most
version-sensitive integration point in the whole project.

> **vLLM-side quantization caveat:** the vLLM rollout copy must be quantized to the
> chosen precision too (else it eats 14 GB in bf16). vLLM's **4-bit AWQ/GPTQ** path
> is the most mature; its **8-bit (bnb int8)** support is newer and more
> version-sensitive — part of what Sprint 5.2 is empirically checking. If vLLM
> can't serve the 8-bit copy on your versions, that alone forces the 4-bit fallback.

**Known caveats (be honest):**
- **8-bit vs 4-bit trade:** 8-bit keeps more of the base model's quality but is the
  **tighter memory fit** (less KV headroom) and its vLLM path is less mature; 4-bit
  gives more headroom and a proven vLLM path at a small quality cost. That's exactly
  why 5.2 measures both — **prefer 8-bit, fall back to 4-bit**.
- **QLoRA is ~20–30% slower** per step than bf16 (8-bit int8 matmuls tend to be
  slower than 4-bit's dequant path), and quantization slightly lowers base quality.
  You trade speed/quality for "fits at all."
- **"LoRA learns less"** on *large* distribution shifts. RLVR is small-shift, so
  LoRA is generally adequate; if reward **plateaus low**, the first lever is
  **raise `r`** (32→64→128), then widen `target_modules` (already all-linear),
  then revisit KL coef.
- LoRA also **resists catastrophic forgetting** of the base instruct abilities —
  a genuine plus when the reward only covers competitive programming.

**Unit tests** (`@pytest.mark.gpu`; the load/steps tests are **parametrized over
`quantization ∈ {"8bit", "4bit"}`** so both are exercised):
- `tests/test_lora_gpu.py::test_7b_qlora_loads_under_budget[8bit]` /
  `[4bit]` — 7B loads with adapters at that precision; post-load peak VRAM is under 32GB.
- `tests/test_lora_gpu.py::test_7b_qlora_few_steps_under_32gb[8bit]` /
  `[4bit]` — a few real steps run; peak `torch.cuda.max_memory_allocated()` stays
  under the 32GB budget for each precision.
- `tests/test_lora_gpu.py::test_8bit_leaves_usable_kv_headroom` — the 8-bit run
  sustains at least the target rollout batch (KV cache doesn't force batch→1 or OOM);
  this is the gate that decides whether 8-bit is viable.
- `tests/test_lora_gpu.py::test_precision_decision_prefers_8bit` — the selection
  rule returns `"8bit"` when its fit-check passes, else `"4bit"` (encodes "get away
  with 8-bit if we can").
- `tests/test_lora_gpu.py::test_adapter_hotswap_changes_generation` — after
  hot-swapping updated adapter weights into vLLM, generation reflects the new policy
  (output differs from the pre-swap base).

**✅ Verify it works (you run)**
```bash
# Fit-check BOTH precisions and watch VRAM; 8-bit is the tighter one.
conda run -n post-train pytest tests/test_lora_gpu.py -m gpu -q
nvidia-smi --query-gpu=memory.used --format=csv -l 1   # watch during the run

# Explicitly A/B the two bases at the real prompt/completion lengths:
conda run -n post-train python -m posttrain.train --config configs/experiment/train_7b.yaml \
    --set model.quantization=8bit --max-steps 5    # try 8-bit first
conda run -n post-train python -m posttrain.train --config configs/experiment/train_7b.yaml \
    --set model.quantization=4bit --max-steps 5    # fallback
```
Expected: **both** precisions train a few steps within 32GB (log peak `nvidia-smi`
for each). If 8-bit holds a usable rollout batch with acceptable step time, **keep
8-bit** (`model.quantization: 8bit` in the experiment config); otherwise fall back
to 4-bit. Adapter hot-swap into vLLM succeeds and generations track the current
adapter, not the base.

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

## Considerations & failure modes — validate by hand

QLoRA under **GRPO + colocated vLLM on one 32GB card** has several sharp edges.
Every risk below has a **🔧 hands-on check** you run to confirm the setup *before*
trusting a long run. The `@pytest.mark.gpu` tests catch the coarse failures (does it
fit, does it step); these manual probes catch the subtle ones that only bite at 7B
on real sequence lengths or after many steps. Run them in order — this is the manual
acceptance playbook for Step 5.

### 1. vLLM silently can't serve the quantized base (the most likely 8-bit blocker)
**Risk:** vLLM fails to load the **8-bit** (bnb int8) base, or silently keeps it in
bf16 (~15 GB) and blows the budget. Its **4-bit AWQ/GPTQ** path is mature; int8 is
newer and version-gated.
**Why:** the vLLM rollout copy is quantized independently of the trainer copy, and
int8 support lags 4-bit.
**🔧 Validate by hand:**
```bash
conda activate post-train
python -c "
from vllm import LLM
llm = LLM('Qwen/Qwen2.5-7B-Instruct', quantization='bitsandbytes',
          load_format='bitsandbytes', gpu_memory_utilization=0.5)
print(llm.generate(['2+2=']))"
nvidia-smi --query-gpu=memory.used --format=csv
```
Expected: it generates **and** the weight footprint is ~7–8 GB (int8), not ~15 GB.
If it raises, or memory looks bf16-sized, vLLM isn't really serving int8 → **fall
back to 4-bit AWQ/GPTQ.** This single check is the most common reason 8-bit is
rejected.

### 2. LoRA hot-swap unsupported in colocate → full reload every step
**Risk:** TRL reloads the whole model into vLLM each step instead of swapping just
the adapter — steps take tens of seconds, or it errors outright.
**Why:** LoRA hot-swap in `vllm_mode=colocate` is the most version-sensitive
integration point in the stack (TRL × vLLM × PEFT).
**🔧 Validate by hand:**
```bash
# Time 5 steps; a hot-swap step is seconds, a full-reload step is tens of seconds.
conda run -n post-train python -m posttrain.train --config configs/experiment/train_7b.yaml \
    --set model.quantization=8bit --max-steps 5 --log-step-time 2>&1 | grep step_time
```
Expected: `step_time` is stable and modest (no per-step multi-second reload spikes).
If every step pays a reload, pin known-good TRL/vLLM versions or use the 4-bit path.

### 3. Rollout policy ≠ training policy (precision mismatch biases GRPO)
**Risk:** the vLLM copy is a *separately quantized approximation* (possibly a
**different** scheme than the trainer — e.g. AWQ vLLM vs nf4 trainer). GRPO's
importance ratio assumes π_old (rollout) ≈ π_θ (trainer). If they diverge, ratios
and advantages are biased → unstable training that "looks fine" but doesn't learn.
**Why:** two independently quantized bases + LoRA applied to only one of them.
**🔧 Validate by hand:** sample a completion from vLLM, then score those exact token
ids under the trainer model and compare per-token logprobs:
```bash
conda run -n post-train python -m posttrain.tools.compare_logprobs \
    --config configs/experiment/train_7b.yaml --set model.quantization=8bit --n 8
```
Expected: mean `|Δ logprob|` per token is small (rule of thumb **< ~0.3 nats**);
`clip_fraction` in training stays moderate and KL is controlled. Large gaps → expect
noisy reward and high KL; prefer the precision whose rollout best matches the trainer.

### 4. KV cache too small at 8-bit → micro-batch or mid-run OOM
**Risk:** 8-bit's ~11 GB KV headroom can't hold `G` completions × your prompt+
completion lengths; effective batch collapses to 1, throughput craters, or it OOMs
several steps in (fragmentation).
**Why:** KV cache scales with `batch × G × (prompt+gen) tokens`; long CodeContests
statements are big.
**🔧 Validate by hand:** run at the **real** `num_generations` and max lengths for
enough steps to hit the longest sequences, watching memory:
```bash
conda run -n post-train python -m posttrain.train --config configs/experiment/train_7b.yaml \
    --set model.quantization=8bit algo.num_generations=8 algo.max_completion_tokens=1024 \
    --max-steps 50 &
nvidia-smi --query-gpu=memory.used --format=csv -l 2   # watch for creep / spikes
```
Expected: memory plateaus with headroom and survives 50 steps. A slow creep to the
ceiling or a step-N OOM means 8-bit's KV budget is too tight → 4-bit.

### 5. OOM only at the longest sequences / later steps (the 5-step smoke lies)
**Risk:** a 5-step smoke passes; step 200 OOMs because peak memory is set by the
single longest prompt+completion and by optimizer/grad-accum state that builds up.
**🔧 Validate by hand:** deliberately stress the tail — longest prompts, max gen,
grad accumulation on — for ~100 steps (covered by the #4 command with a higher
`--max-steps`; also try the hardest-difficulty subset). Expected: no OOM at the tail.

### 6. int8 is slower than 4-bit (may make 8-bit impractical even if it fits)
**Risk:** bnb int8 matmuls (with outlier decomposition) can be materially slower than
4-bit's dequant path — 8-bit might fit but train too slowly to be worth it.
**🔧 Validate by hand:** time tokens/sec for both and compare:
```bash
for q in 8bit 4bit; do
  echo "== $q =="; conda run -n post-train python -m posttrain.train \
    --config configs/experiment/train_7b.yaml --set model.quantization=$q \
    --max-steps 10 --log-step-time 2>&1 | grep -E "step_time|tokens_per_s"
done
```
Expected: 8-bit within a tolerable factor of 4-bit (e.g. ≤ ~1.5× slower). If 8-bit is
far slower, take 4-bit even though 8-bit "fits."

### 7. Merging a quantized base dequantizes — memory spike + output drift
**Risk:** `merge_and_unload()` on a 4-bit/8-bit base can't fold bf16 LoRA deltas into
quantized weights directly; PEFT **dequantizes to fp16** (≈15 GB for 7B) — which can
OOM the merge and shifts outputs slightly vs the adapter model.
**Why:** you can't add a continuous delta to a quantized tensor without dequantizing.
**🔧 Validate by hand:**
```bash
conda run -n post-train python -m posttrain.tools.merge_and_check \
    --config configs/experiment/train_7b.yaml --set model.quantization=8bit &
nvidia-smi --query-gpu=memory.used --format=csv -l 1   # watch the merge spike (~15GB)
```
Expected: merge completes without OOM and greedy generations match the adapter model
(this is `test_merge_matches_adapter_generation`). If the merge OOMs, merge on CPU or
offload; if outputs drift, eval the **adapter** model rather than the merged one.

### 8. Grad checkpointing + k-bit can silently kill gradient flow
**Risk:** `prepare_model_for_kbit_training` + gradient checkpointing needs
`use_reentrant=False` and input-grads enabled; a mismatch makes **no gradient flow to
the adapters** → loss/reward flat, looking like a learning-rate or reward bug.
**Why:** checkpointing detaches the graph unless input grads are explicitly enabled on
a frozen quantized base.
**🔧 Validate by hand:** after one backward, confirm an adapter actually has a
non-zero grad:
```bash
conda run -n post-train python -m posttrain.tools.grad_flow_check \
    --config configs/experiment/train_7b.yaml --set model.quantization=8bit
```
Expected: prints at least one LoRA param with `grad is not None` and `‖grad‖ > 0`, and
a non-trivial `loss` that changes across 3 steps. All-`None`/zero grads → fix
`use_reentrant`/`enable_input_require_grads` before anything else.

### 9. `nvidia-smi` vs `max_memory_allocated` disagree — trust the one that OOMs
**Risk:** `torch.cuda.max_memory_allocated()` reports only tensor bytes; the **driver**
also holds the CUDA context + the caching allocator's reserved pool. The driver number
(what `nvidia-smi` shows) is what actually OOMs — it can be several GB higher.
**🔧 Validate by hand:** always report both and budget against `nvidia-smi`:
```bash
conda run -n post-train python -c "
import torch; from posttrain.train import quick_fit_probe
quick_fit_probe('configs/experiment/train_7b.yaml', quant='8bit', steps=5)
print('tensor peak GB:', torch.cuda.max_memory_allocated()/1e9)"
nvidia-smi --query-gpu=memory.used --format=csv
```
Expected: `nvidia-smi` used < 32 GB with a couple GB to spare. A tensor-peak that fits
but an `nvidia-smi` at the ceiling means you're one long sequence away from OOM.

---

## Definition of done

- With `lora.enabled=true`, **only adapter params** are trainable and the base is
  frozen; trainable-param count is <2% of total.
- `build_quant_config` produces valid **8-bit and 4-bit** configs; both are
  selectable by the `quantization` config field alone.
- **Both** 8-bit and 4-bit 7B QLoRA are fit-tested within 32GB, with peak VRAM,
  KV headroom, and step time recorded for each.
- **8-bit is selected** when its fit-check passes (usable KV cache + acceptable step
  time); **4-bit is the recorded fallback** otherwise. The chosen precision is
  pinned in the experiment config.
- Adapter weights hot-swap into vLLM so generation reflects the current policy.
- `merge_and_unload()` produces a model whose generations match the adapter model's
  (greedy sanity).
- 7B QLoRA shows **rising reward** on the easy probe subset (on the selected precision).
- Toggling `lora.enabled` / `quantization` is **config-only** (loader handles it).
- The **Considerations & failure modes** playbook (hands-on checks 1–9) has been run
  for the chosen precision — vLLM serves the quantized base, hot-swap doesn't reload,
  rollout/trainer logprobs agree, gradients flow, merge is clean — or the failing
  check's fallback (4-bit / eval-the-adapter / version pin) is **recorded**.

## Run all tests for this step

```bash
# Everything (needs a 32GB GPU for the gpu-marked tests):
conda run -n post-train pytest tests/test_lora.py tests/test_lora_gpu.py -q

# CPU / structural subset only (no GPU required):
conda run -n post-train pytest tests/test_lora.py -m "not gpu" -q
```
