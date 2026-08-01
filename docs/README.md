# RLVR for Competitive Programming — Implementation Plan

Train an LLM (starting with a 3B smoke test, then **Qwen2.5-7B-Instruct**) with
**RLVR** (Reinforcement Learning with Verifiable Rewards) on the
[**DeepMind CodeContests**](https://github.com/google-deepmind/code_contests)
dataset. The "verifiable reward" is simple: *does the generated code pass the
problem's test cases?*

> **Dataset:** DeepMind **CodeContests** — loaded from HuggingFace as
> [`deepmind/code_contests`](https://huggingface.co/datasets/deepmind/code_contests)
> (splits `train` / `valid` / `test`). Competitive-programming problems with
> stdin/stdout I/O, hidden test cases, and reference solutions. It is the
> **default** dataset but sits behind a swappable interface (see the ablation
> table below); ingestion + verification is detailed in
> [`01-data-ingestion.md`](01-data-ingestion.md).

This directory is the step-by-step build plan. Read `00`→`07` in order.

| # | Doc | What it delivers |
|---|-----|------------------|
| 0 | this file | Architecture, principles, repo layout, environment setup |
| 1 | [`01-data-ingestion.md`](01-data-ingestion.md) | Load + normalize + verify code_contests |
| 2 | [`02-rl-environment.md`](02-rl-environment.md) | Sandbox, **environment (→Outcome)**, **polymorphic reward**, **loss** |
| 3 | [`03-smoke-test-3b.md`](03-smoke-test-3b.md) | End-to-end GRPO on a 3B model, no LoRA |
| 4 | [`04-model-swapping.md`](04-model-swapping.md) | Swap in Qwen2.5-7B; model-agnostic loader |
| 5 | [`05-lora-qlora.md`](05-lora-qlora.md) | LoRA / QLoRA to fit 7B on 32GB |
| 6 | [`06-rl-algorithms.md`](06-rl-algorithms.md) | Interchangeable GRPO / PPO / RLOO |
| 7 | [`07-end-to-end-testing.md`](07-end-to-end-testing.md) | Full test suite + eval harness |

---

## Guiding principles

1. **Config-driven, nothing hardcoded.** Model, algorithm, LoRA, and dataset are
   all selected from YAML. Swapping `Qwen2.5-3B` → `Qwen2.5-7B` → any HF model is
   a config edit, never a code change. (Requirement from steps 4 & 6.)
2. **The risky components are model-free.** Data, sandbox, reward, and eval need
   no GPU or model — build and unit-test them first against known-good solutions.
3. **One variable at a time.** Prove the RL *wiring* on a small model before
   fighting 7B memory. See `03`.
4. **Standard libraries, not hand-rolled RL.** TRL for GRPO/PPO/RLOO, PEFT for
   LoRA, vLLM for rollouts, HF `datasets` for data. We write the *reward* and the
   *plumbing*, not the optimizer math.

## Two separations that drive the whole design — read before doc 2

### Environment vs. reward (produce evidence vs. judge evidence)

Composability comes from splitting *running the code* from *scoring the result*:

- **Environment** (`src/posttrain/env/`) is the **central unit**. It owns the sandbox +
  dataset, takes a completion (the "action") and produces a structured
  **`Outcome`** — the evidence (extracted code, per-test pass/fail, status,
  timings, length). Expensive, impure, security-critical, deterministic. It
  contains **no scoring logic**.
- **Reward** (`src/posttrain/rewards/`) is a **pure, polymorphic** function
  `Outcome → float` — an abstract `RewardFunction` with concrete instantiations
  (`Binary`, `Fractional`, `Staged`, `Composite`). It never touches the sandbox.

This is the classic RL loop — `action → environment → outcome → reward(outcome)` —
and it's what makes **ablations** clean: run the environment once, score the same
`Outcome` with any number of reward functions, no re-execution, no confounds. The
`Outcome` dataclass and the `RewardFunction` ABC are the two stable contracts that
hold everything together. (Task is single-step/bandit, so the env exposes
`evaluate()` not gym `step()` — see doc 2.0.)

### Reward vs. loss (what we design vs. what the library computes)

- **Reward** is what *we* design: the scalar above. **Unique to our problem.**
- **Loss** is what the *RL algorithm library* computes from a batch of
  (prompt, completion, reward) tuples. GRPO/PPO define it; TRL implements it. We
  do **not** write the loss by hand — doc 2.5 spells out what it is so you know
  what the reward feeds into.

---

## Target architecture

`train.py` builds four independently-swappable pieces — **model**, **algorithm**,
**reward**, **environment** — and wires them into the RL loop:

```
  train.py ── builds ──▶  model (loader) + algorithm (factory) + reward + environment

  RL loop (per rollout batch, orchestrated by TRL — TWO engines):
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  vLLM engine ──completions──▶ ENVIRONMENT.evaluate() ──Outcome──▶ REWARD.score  │
  │  (fast rollout)               (owns sandbox + data)              (pure)    │     │
  │       ▲                                                                    │     │
  │       │ weight / LoRA-adapter sync each step                    scalar rewards   │
  │       │                                                                    ▼     │
  │  transformers + PEFT policy ◀──── GRPO/PPO/RLOO update (backprop) ◀────────┘     │
  │  (forward+backward = the trained weights)                                        │
  └──────────────────────────────────────────────────────────────────────────────┘
        (env + reward are joined only by the TrlRewardBridge — doc 2.4)

  ENVIRONMENT  ── owns/uses ─▶  sandbox/executor  +  data (Problem set)
  REWARD (ABC) ── impls ─────▶  Binary │ Fractional │ Staged │ Composite
  MODEL        ── loader ────▶  any HF model (+LoRA +quant)          [docs 4–5]
  ALGORITHM    ── factory ───▶  GRPO │ PPO │ RLOO (TRL)              [doc 6]
```

### Two engines: generation (vLLM) vs. training (transformers)

We do **not** use transformers for everything. Each RL step runs **two model
engines**, because generation and gradient updates have opposite needs:

- **vLLM = rollouts.** Sampling G completions per prompt dominates RL wall-clock;
  HF `.generate()` (no paged attention / continuous batching) is ~10–20× slower.
  vLLM is inference-only — no autograd.
- **transformers + PEFT = the update.** Gradients (the actual training) require the
  HF/PEFT model; vLLM cannot backprop.

TRL orchestrates both: **vLLM generates → env+reward score → transformers does the
policy-gradient update → the updated weights (or just the LoRA adapter) sync back
into vLLM** so the next rollout uses the current policy.

> **Single-GPU consequence:** on one 32GB card the two engines run in vLLM
> **colocate mode** (same process/GPU) and hold **two copies of the base weights**
> — this is the "double-weights trap" that forces 4-bit on both sides for 7B
> (docs 4–5). With LoRA, the KL **reference** model is the base with the adapter
> *disabled*, so it is **not** a third copy (doc 6.4).

Each arrow is a **stable contract** → each box is an **ablation axis** you can
vary while holding the others fixed:

| Axis | Swap point | Contract |
|------|-----------|----------|
| Dataset (default: **CodeContests**) | env constructor arg | produces `Problem`s |
| Environment / verifier | `env/` | emits `Outcome` |
| Sandbox / executor | `sandbox/` | per-test pass/fail |
| Reward | `RewardFunction` ABC | `Outcome → float` |
| Model | `models/loader` | HF policy |
| Algorithm | `algos/factory` | consumes `(dataset, reward_fn)` |

**Key insight about the "RL environment":** TRL's GRPO/PPO do *not* use a
gym-style `reset()/step()` loop — their abstraction is **(a prompt dataset) + (a
reward callable)**, and our task is single-step (a contextual bandit). So the
environment exposes `evaluate(completion) -> Outcome` (not `step()`), the reward
scores the `Outcome`, and a thin `TrlRewardBridge` composes the two into the
callable TRL wants. The same `env.evaluate` + `reward.score` power the eval
harness — one scoring definition everywhere. Doc 2 defines it precisely.

## Repo layout

```
post-training-rl/
├── configs/
│   ├── base.yaml                 # shared defaults
│   ├── model/{qwen3b,qwen7b}.yaml
│   ├── algo/{grpo,ppo,rloo}.yaml
│   └── experiment/{smoke_3b,train_7b}.yaml
├── src/posttrain/
│   ├── data/
│   │   ├── schema.py             # Problem dataclass (normalized)
│   │   └── ingest.py             # load + filter + format code_contests
│   ├── sandbox/                  # swappable executor the env owns
│   │   ├── executor.py           # safe subprocess runner
│   │   └── comparators.py        # stdout matching (exact / float-tol)
│   ├── rewards/                  # pure, polymorphic — Outcome → float
│   │   ├── base.py               # RewardFunction ABC
│   │   └── code_reward.py        # Binary/Fractional/Staged/Composite rewards
│   ├── env/                      # CENTRAL unit: produces evidence
│   │   ├── outcome.py            # Outcome dataclass (env↔reward contract)
│   │   ├── code_contest_env.py   # owns sandbox+data; evaluate()→Outcome
│   │   └── reward_bridge.py      # composes env+reward into TRL's callable
│   ├── models/
│   │   └── loader.py             # config → (model, tokenizer), LoRA+quant
│   ├── algos/
│   │   └── factory.py            # algo name → configured TRL trainer
│   ├── eval/
│   │   └── evaluate.py           # pass@k on held-out split
│   ├── config.py                 # dataclass config + YAML loader
│   └── train.py                  # entrypoint
├── tests/                        # pytest: sandbox, reward, data, smoke
├── scripts/
└── docs/
```

## Tech stack & environment setup

> ⚠️ **RTX 5090 = Blackwell (sm_120).** Requires **CUDA 12.8+** and
> **PyTorch ≥ 2.7**. Older wheels fail to build kernels or fall back to CPU.
> Use **Python 3.11** (not 3.13 — several RL/sandbox libs lag on 3.13).

```bash
# Use conda for the environment (pins Python 3.11; keeps the RL env isolated).
conda create -n post-train python=3.11 -y
conda activate post-train
# Install the GPU wheels with pip *inside* the conda env — the cu128 PyTorch /
# vLLM wheels are only published on PyPI, not the conda channels.
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install vllm trl peft datasets transformers accelerate bitsandbytes
pip install pytest pyyaml   # tooling
# sandbox isolation (choose one): firejail (apt) OR bubblewrap (apt) OR nsjail
sudo apt-get install -y firejail
```

> Pin the environment for reproducibility with `conda env export > environment.yml`.
> `conda activate post-train` is the first step of every session (docs 3–7 assume it).

**Smoke-check before writing any RL code:**
```python
import torch; print(torch.cuda.get_device_capability())  # expect (12, 0)
from vllm import LLM  # imports without kernel errors
```

## Build order (dependency-first)

```
data/schema + ingest ──┐
                       ├─▶ env + reward ──▶ smoke 3B ──▶ swap 7B ──▶ LoRA ──▶ algo swap ──▶ e2e
sandbox + comparators ─┘        (doc 2)     (doc 3)     (doc 4)   (doc 5)   (doc 6)     (doc 7)
      (doc 1)
```
Docs 1 and 2's non-model pieces are built and tested **without a GPU**.

## Sprints & testing methodology

Each step doc (`01`–`07`) is broken into **sprints**. A sprint is the smallest
subcomponent that can be built and *verified on its own*, and sprints are
dependency-ordered so you always build on something already tested. Every sprint
carries its own testing in two distinct forms:

- **Unit tests** — automated `pytest` cases (listed as `tests/<file>.py::<test>`),
  fast and run on every commit. The doc says exactly what each asserts.
- **✅ Verify it works (you run)** — the explicit command(s) *you* run to confirm
  the sprint works end-to-end, each with an `Expected:` pass condition. This is
  the human acceptance check, separate from the unit tests.

Each doc ends with a **Definition of done** (aggregated sprint gates) and a
**Run all tests for this step** command.

### Test conventions (defined in doc 7, used everywhere)

- All tests live under `tests/`; framework is **pytest**.
- Tests needing the GPU are marked **`@pytest.mark.gpu`**. On a machine without
  the 5090, run the model-free subset with `-m "not gpu"`:
  ```bash
  conda activate post-train
  pytest -m "not gpu"     # everything in docs 1–2 + all pure logic — no GPU
  pytest                  # full suite, requires the 5090
  ```
- The **model-free** core (data, sandbox, reward, eval estimators) is fully
  testable on a laptop — that's the payoff of the architecture. GPU only enters
  at the training/rollout sprints (docs 3–6).
