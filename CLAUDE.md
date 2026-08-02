# post-training-rl

RLVR: train Qwen2.5-Instruct to write competitive-programming code with GRPO on
DeepMind's CodeContests, rewarded by executing generated code against tests.

## Hardware constraint

A single **RTX 5090 — 32 GB VRAM, Blackwell sm_120, compute capability 12.0**.
This is the binding constraint on every design decision. `TORCH_CUDA_ARCH_LIST="12.0"`.

**Qwen2.5-7B-Instruct is the target. Qwen2.5-3B-Instruct is the stepping stone**, and
the 3B run's job is to prove the verifier, scorer, and reward ladder are correct — not
to produce a good model.

- 3B: bf16 base + LoRA, ~6 GB.
- 7B: NF4 QLoRA required to fit, ~9 GB with no vLLM.

Model, quantization, and generation backend all live in YAML, so 3B→7B is a config
change rather than a code change.

**Generation runs through `use_transformers_continuous_batching=True`. Do not add vLLM.**
On a single 32 GB card it buys nothing and costs a second copy of the weights, a
weight-sync path that is unsupported with QLoRA, and several open sm_120 bugs. Note that
continuous batching's interaction with `Linear4bit` is untested upstream — verify it when
moving to 7B, and fall back to plain `.generate()` if it breaks.

## Environment

Conda. Activate before running anything:

```bash
conda activate post-training-rl
```

Hard-won installation facts — see `docs/research/rlvr-stack.md` for sources:

- **Install PyTorch from `cu128`+ or plain PyPI. Never the `cu126` index.** It ships a
  current torch built with zero sm_120 kernels, and bitsandbytes picks its `.so` from
  `torch.version.cuda` with no arch awareness — you get a clean-looking install that dies
  at the first CUDA op with `no kernel image is available for execution on the device`.
  Verify: `python -c "import torch; print(torch.version.cuda, torch.cuda.get_device_capability(0))"`
  → want capability `(12, 0)` and toolkit `12.8`+.
- **Do not install `flash-attn`.** sm_120 has no `wgmma`, no `tcgen05`, and ~100 KB shared
  memory per block against Hopper's 228 KB, so FA3/FA4 physically cannot run. This is a
  hardware limit, not a packaging gap — a newer release will not fix it. Use
  `attn_implementation="sdpa"`.
- **Do not `pip install triton` separately** — PyTorch pins its own.
- If vLLM is ever used, it needs **CUDA toolkit ≥ 12.9** or engine init crashes on sm_120.

## Vocabulary

Use these terms exactly. `CONTEXT.md` is the canonical glossary once it exists.

- **rollout** — one sampled completion.
- **group** — the N rollouts sharing a prompt. GRPO normalizes advantage *within* a group,
  so a group whose rollouts all score identically produces zero gradient.
- **verifier** — executes a rollout's code against tests in a sandbox. Impure, does I/O,
  returns a structured `ExecutionResult`. **Never assigns a reward.**
- **scorer** — pure function `ExecutionResult -> float`. No I/O, no execution, no subprocess.
  Trivially unit-testable, and that is the point of the split.
- **ladder** — the graded reward: no code → parses → runs → fraction of tests passed.

Do not use "environment" for the verifier. In TRL, `environment_factory` means multi-turn
tool calling, which this project does not do.

## How work gets planned and executed

1. **Decisions** → `docs/adr/NNNN-title.md`, written by `/domain-modeling`.
2. **Sprint tasks** → `docs/plans/sprint-NN.md`. Each task must name the tests it requires;
   `/write-code` writes only the tests the plan specifies and no others.
3. **Implementation** → `/write-code docs/plans/sprint-NN.md, task N`. One task per
   invocation, reviewed between tasks.

Do not implement ahead of the plan. If a task cannot be built as written, stop and say so
rather than improvising a different design.

## Reference material

- `docs/research/rlvr-stack.md` — primary-source research on TRL, vLLM, bitsandbytes on
  sm_120, the CodeContests schema, sandboxing, and reward shape. **Read it before making
  stack decisions.** It documents several places where official docs are actively wrong
  (vLLM's attention-backend doc; the CodeContests dataset card's enum tables).

## Code standards

Governed by the `coding-standards` skill and its `CODING_STANDARDS.md`, which is the single
source of truth. Not restated here — a rule written twice drifts.
