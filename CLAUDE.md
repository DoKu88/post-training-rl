# post-training-rl

RLVR: train Qwen2.5-Instruct to write competitive-programming code with GRPO on
DeepMind's CodeContests, rewarded by executing generated code against tests.

## Hardware constraint

A single **RTX 5090 — 32 GB VRAM, Blackwell sm_120, compute capability 12.0**.
This is the binding constraint on every design decision. `TORCH_CUDA_ARCH_LIST="12.0"`.

**Qwen2.5-7B-Instruct is the target. Qwen2.5-3B-Instruct is the stepping stone**, and
the 3B run's job is to prove the verifier, scorer, and reward functions are correct — not
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
conda activate post-train
```

The env holds torch 2.11.0+cu128, transformers 5.14.1, trl 1.9.2, peft 0.20.0,
bitsandbytes 0.50.0, datasets 5.0.1. Full table in [`docs/design/model.md`](docs/design/model.md).

**Known defect:** `torchvision 0.26.0` is built against CUDA 13.0 while torch is cu128, so
any import reaching `transformers.image_utils` raises — including
`from transformers import TrainerCallback`. This is text-only training; uninstall torchvision.

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

**[`CONTEXT.md`](CONTEXT.md) is the canonical glossary.** Read it before naming anything, and
use its terms exactly — including its `_Avoid_` lists. Not restated here.

The one worth flagging up front: **never call the verifier an "environment."** In TRL,
`environment_factory` means multi-turn tool calling, which this project does not do.

## How work gets planned and executed

1. **Decisions** → `docs/adr/NNNN-title.md`, written by `/domain-modeling`.
2. **Sprint objectives and gates** → [`docs/plans/roadmap.md`](docs/plans/roadmap.md).
3. **Sprint tasks** → `docs/plans/sprint-NN.md`. Each task leads with the **behaviour** it
   delivers and names the tests it requires; `/write-code` writes only those tests and no
   others. Work in vertical slices — one test, one implementation, repeat — never the whole
   test list up front.
4. **Implementation** → `/write-code docs/plans/sprint-NN.md, task N`. One task per
   invocation, reviewed between tasks.

Do not implement ahead of the plan. If a task cannot be built as written, stop and say so
rather than improvising a different design.

## Reference material

- **[`docs/project-status.md`](docs/project-status.md)** — what is done, what is next, what is
  still unmeasured. **Start here**, and update it whenever a task completes or an unknown gets
  measured.
- **[`CONTEXT.md`](CONTEXT.md)** — the glossary. Terms only, no implementation detail.
- **[`docs/adr/`](docs/adr/)** — every significant decision and why it was made. **Read the
  relevant ADR before changing behaviour in that area**, and say so explicitly if your work
  contradicts one rather than silently overriding it.
- **[`docs/design/`](docs/design/)** — how the pieces fit:
  - `rl-loop.md` — every module's place in the classic RL loop, and where the analogy breaks.
  - `verifier-scorer.md` — module design, types, seams.
  - `behavior.md` — intended behaviour per module: does / guarantees / refuses.
  - `rl-reward-functions.md` — every reward function, its shared input, and its source.
  - `model.md` — checkpoint, quantisation, LoRA placement, VRAM budget.
- `docs/research/rlvr-stack.md` — primary-source research on TRL, vLLM, bitsandbytes on
  sm_120, the CodeContests schema, sandboxing, and reward shape. **Read it before making
  stack decisions.** It documents several places where official docs are actively wrong
  (vLLM's attention-backend doc; the CodeContests dataset card's enum tables).
- `docs/research/format-adherence.md` — code extraction policies across seven
  implementations, format rewards, constrained decoding, assistant prefill, and prompt
  templates.

## Code standards

Governed by the `coding-standards` skill and its `CODING_STANDARDS.md`, which is the single
source of truth. Not restated here — a rule written twice drifts.
