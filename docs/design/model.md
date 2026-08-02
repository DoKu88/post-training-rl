# The model

What gets loaded, how it is quantised, which layers carry LoRA, and what that costs in VRAM.

Governed by [ADR-0003](../adr/0003-3b-stepping-stone-to-7b.md) (3B → 7B) and
[ADR-0002](../adr/0002-no-vllm.md) (generation backend). Memory arithmetic is from
[`rlvr-stack.md` §3](../research/rlvr-stack.md).

---

## 1. Identity

| | stepping stone | target |
| --- | --- | --- |
| Checkpoint | `Qwen/Qwen2.5-3B-Instruct` | `Qwen/Qwen2.5-7B-Instruct` |
| Layers | 36 | 28 |
| Hidden size | 2048 | 3584 |
| KV heads (GQA) | 2 | 4 |
| Head dim | 128 | 128 |
| Embeddings | **tied** (one copy) | **untied** (`lm_head` is a second copy) |
| Vocabulary | 152,064 | 152,064 |

The 3B run's job is to prove the verifier, scorer, and reward functions are correct. Its
reward curve says nothing about whether the method works.

**The tokenizer is shared between the two checkpoints.** That matters more than it looks: the
dataset drops problems whose prompts exceed the token budget (ADR-0010), so a shared tokenizer
means the filtered corpus and the evaluation set are **identical** across 3B and 7B. If that
ever stops being true, the two runs stop being comparable.

---

## 2. Precision and quantisation

| | 3B | 7B |
| --- | --- | --- |
| Base weights | **bf16**, frozen | **NF4 + double quant**, frozen |
| LoRA adapters | bf16, trainable | bf16, trainable |
| Compute dtype | bf16 | bf16 |
| Attention | `sdpa` | `sdpa` |

3B does not need quantising — bf16 base plus LoRA is roughly 6 GB and leaves ample room.
Quantising it would add a variable while proving out the verifier, which is the opposite of
what the stepping stone is for.

7B requires NF4 to fit alongside generation and optimizer state.

> **`load_in_4bit=True` alone is not QLoRA.** `BitsAndBytesConfig` defaults to
> `bnb_4bit_quant_type="fp4"`, `bnb_4bit_use_double_quant=False`, and fp32 compute. All three
> must be set explicitly, or every memory figure below is wrong.

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
```

**`attn_implementation="sdpa"`, and flash-attn is never installed.** sm_120 has no `wgmma`, no
`tcgen05`, and ~100 KB of shared memory per block against Hopper's 228 KB, so FA3/FA4 cannot
physically run. This is a hardware limit, not a packaging gap.

---

## 3. LoRA

### Initial parameters

```yaml
lora:
  r: 32
  alpha: 64                # scaling = alpha / r = 2.0
  dropout: 0.0
  bias: none
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  modules_to_save: []      # embeddings and lm_head stay frozen
```

### Which layers, and why all of them

**Attention *and* MLP — all seven linear projections in every transformer block.** Not the
`q_proj`/`v_proj`-only configuration that is the common default.

This is the QLoRA paper's headline finding on adapter placement
([Dettmers et al., NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/1feb87871436031bdc0f2beaa62a049b-Paper-Conference.pdf)):
the most critical LoRA hyperparameter is **how many adapters are used in total**, and LoRA on
*all* linear transformer block layers is required to match full finetuning performance.
Attention-only placement underfits regardless of rank.

The MLP projections (`gate_proj`, `up_proj`, `down_proj`) are also where most of the
parameters live — in Qwen2.5-7B the intermediate size is 18,944 against a hidden size of
3,584 — so excluding them omits the majority of the model's capacity from adaptation.

### Why `dropout: 0.0`

Not a default carried over from supervised finetuning. GRPO computes an importance ratio
between the current policy and the policy that generated the rollouts:

```
ρ = π_θ(token) / π_old(token)
```

Both sides come from forward passes over the same tokens. With dropout active, those passes
are **stochastic**, so ρ picks up noise that has nothing to do with the policy having changed
— and that noise lands directly in the clipped objective. Dropout is a regulariser for many
epochs over fixed data; RL fine-tuning is few steps over fresh rollouts, so it costs
correctness and buys little.

### Why `alpha = 64` with `r = 32`

Scaling is `alpha / r = 2.0`, the common convention. It is a starting point, not a derived
value — if updates prove too aggressive, lower `alpha` before touching `r`, since `alpha`
changes the effective learning rate without changing the parameter count or memory.

### What stays frozen

Embeddings and `lm_head` are **not** adapted and **not** in `modules_to_save`. On 7B the
untied `lm_head` is 545M parameters — training it would add roughly 6.5 GB of optimizer state
and defeat the point of LoRA. On 3B they are tied, so there is only one copy either way.

---

## 4. Memory

Per-layer adapter parameters are `r × Σ(in + out)` across the seven projections:

| | per-layer factor | layers | at `r=32` | + grads + AdamW |
| --- | --- | --- | --- | --- |
| 3B | `r × 51,968` | 36 | 59.9M | **≈ 0.72 GB** |
| 7B | `r × 90,112` | 28 | 80.7M | **≈ 0.97 GB** |

Rank sensitivity on 7B, if `r=32` proves wrong:

| rank | adapter params | total with optimizer |
| --- | --- | --- |
| 16 | 40.4M | ≈ 0.49 GB |
| **32** | **80.7M** | **≈ 0.97 GB** |
| 64 | 161.5M | ≈ 1.94 GB |

### Budget against 32 GB

| Component | 3B (bf16) | 7B (NF4) |
| --- | --- | --- |
| Base weights | ≈ 6.2 GB | ≈ 5.55 GB |
| LoRA + grads + optimizer (`r=32`) | ≈ 0.72 GB | ≈ 0.97 GB |
| KV cache (8 rollouts × 4096 tok) | ≈ 1.13 GB | ≈ 1.75 GB |
| Logits + activations | ≈ 3 GB | ≈ 3 GB |
| **Total** | **≈ 11 GB** | **≈ 11.3 GB** |

**The term that dominates is logits, not weights.** Qwen's 152,064-token vocabulary makes the
fp32 logits tensor ~623 MB per forward at `max_completion_length=1024`, and GRPO does two or
three forwards per step. Hence:

- `per_device_train_batch_size=1`, leaning on `logits_to_keep`.
- `beta=0.0` — no reference model is loaded at all. With PEFT a reference is available for
  free by disabling adapters, but not paying for the extra forward pass is better still.

For contrast, the configuration that does **not** fit: 7B in bf16 with a colocated bf16 vLLM
engine needs ≈ 35 GB. That arithmetic is why ADR-0002 exists.

---

## 5. Generation

```yaml
generation:
  backend: continuous_batching     # transformers, in-process
  max_memory_percent: 0.4
  use_cuda_graph: false
```

`use_transformers_continuous_batching=True`. No server, no weight synchronisation, no second
copy of the weights. Requires `transformers >= 5.8.0`.

**Unverified at 7B:** TRL casts the unwrapped model to bf16/fp16 before `generate_batch`, and
its interaction with `Linear4bit` is untested upstream. Verify when scaling; fall back to
plain `.generate()` if it breaks. This is the single largest open risk in the 3B → 7B move.

---

## 6. Config shape

```
config/model/
├── qwen2.5-3b.yaml     quantization: null       · lora.r: 32 · backend: continuous_batching
└── qwen2.5-7b.yaml     quantization: nf4 block  · lora.r: 32 · backend: continuous_batching
```

Everything above is a config value. Scaling 3B → 7B is selecting a different file, not
editing code — the loader reads `quantization` and either builds a `BitsAndBytesConfig` or
does not.

---

## 7. Environment

The stack lives in the conda env **`post-train`**:

| Package | Version | Note |
| --- | --- | --- |
| torch | 2.11.0+**cu128** | sm_120 kernels present. Never the `cu126` index |
| transformers | 5.14.1 | ≥ 5.8.0 needed for continuous batching |
| trl | 1.9.2 | |
| peft | 0.20.0 | |
| bitsandbytes | 0.50.0 | wheel built against CUDA 12.8+, ships sm_120 cubins |
| datasets | 5.0.1 | |
| accelerate | 1.14.0 | |
| flash-attn | **absent, deliberately** | cannot run on sm_120 |

**Known environment defect:** `torchvision 0.26.0` is built against CUDA 13.0 while torch is
cu128, so any import path reaching `transformers.image_utils` raises — including
`from transformers import TrainerCallback`, which the verification cache reset depends on.
This is text-only training; uninstalling torchvision is the clean fix.
