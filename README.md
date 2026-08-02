# post-training-rl

Reinforcement learning with **verifiable** rewards (RLVR) for competitive programming.

A Qwen2.5-Instruct policy is trained with GRPO to write solutions to
[DeepMind CodeContests](https://github.com/google-deepmind/code_contests) problems. The reward
is not predicted by a learned model — it is **observed by executing the generated code against
the problem's test cases**:

```
RLHF:   completion ──▶ reward model ──▶ predicted scalar   (a guess, hackable)
RLVR:   completion ──▶ execution     ──▶ observed outcome  (a fact)
```

That single substitution is what the whole repository is built around. It also means every
training step runs untrusted, model-authored code on this machine, which is why sandboxing is a
hard requirement rather than a nicety.

---

## Status

**Design complete. Implementation in progress — sprint 1 of 4.**

[`docs/project-status.md`](docs/project-status.md) is the running record: what exists, what is
decided, what is still unmeasured. Start there.

---

## How it works

One training step, end to end:

1. The **dataset builder** yields a filtered problem — prompt, graded tests, public tests.
2. **GRPOTrainer** samples *G* completions for that one prompt. The set is a **group**, and it
   is the unit that carries learning signal.
3. **Extraction** recovers Python from each completion, recording *which fence it arrived in*
   and *whether it parses* as two independent facts.
4. The **verifier** prepends a determinism preamble and runs each graded test inside a
   **sandbox**, comparing output with CodeContests token semantics — not exact string match.
5. A **reward function**, chosen by config from a registry, turns each report into a float.
6. GRPO normalises the rewards within the group into advantages and updates the LoRA weights.

Episodes have length 1 — the next prompt is the next row, drawn independently of what the model
just wrote. This is a contextual bandit, not an MDP, which is why there is no critic, no
discount factor, and no value function.

[`docs/design/rl-loop.md`](docs/design/rl-loop.md) maps every module onto the classic
agent–environment diagram and states the four places the analogy breaks.

Two design decisions worth knowing before reading any code:

- **The verifier never assigns a reward, and the scorer never executes anything**
  ([ADR-0004](docs/adr/0004-verifier-scorer-split.md)). One execution therefore feeds *every*
  reward function, so a single run logs the counterfactual curve for every reward shape in the
  registry.
- **No module is called "the environment."** That role is filled by three modules, and in TRL
  the word already means multi-turn tool calling, which this project does not do. See
  [`CONTEXT.md`](CONTEXT.md), which is the canonical glossary.

---

## Requirements

### System dependencies

**`firejail` is required.** Generated code is executed inside it, and the configured backend is
an explicit config value with **no default** — if `config/verifier.yaml` names `firejail` and
the binary is absent, startup raises naming it. It never silently downgrades to a weaker
sandbox, because a quietly degraded sandbox is indistinguishable from a working one until
something escapes.

```bash
sudo add-apt-repository ppa:deki/firejail
sudo apt update && sudo apt install firejail
firejail --version        # developed against 0.9.74
```

The wall-clock limit is enforced by **`timeout(1)`** (GNU coreutils, present on every
mainstream Linux) wrapping the firejail invocation, rather than by firejail's own `--timeout`.
That flag was measured at a flat ~2 s per execution regardless of its value, and it cannot
distinguish its own timeout from an uncaught exception — see
[ADR-0014](docs/adr/0014-external-wall-clock-timer.md). Both binaries are checked at
construction; a missing one raises naming it.

Containment tests (`pytest -m containment`) **skip loudly** without firejail rather than
passing vacuously — a containment test that passes because it did not run is false assurance
about the only thing protecting the host.

A weaker `subprocess` backend exists for CI and for development machines without firejail. It
provides resource limits but **not** network isolation or a private filesystem, and it is not
held to the containment guarantees. See
[ADR-0005](docs/adr/0005-firejail-sandbox.md).

### Hardware

A single **RTX 5090 — 32 GB VRAM, Blackwell sm_120, compute capability 12.0**. This is the
binding constraint on every design decision in the repository. Qwen2.5-3B is the stepping stone
(bf16 + LoRA, ~6 GB); Qwen2.5-7B is the target (NF4 QLoRA, ~9 GB).

### Python environment

Conda, environment name **`post-train`** (it differs from the repo name):

```bash
conda activate post-train
```

Holds torch 2.11.0+cu128, transformers 5.14.1, trl 1.9.2, peft 0.20.0, bitsandbytes 0.50.0,
datasets 5.0.1. Full table in [`docs/design/model.md`](docs/design/model.md).

### Things that must never be installed

Each of these is a hardware or packaging trap that produces a clean-looking install which dies
later. Sources in [`docs/research/rlvr-stack.md`](docs/research/rlvr-stack.md).

| Never | Why |
| --- | --- |
| PyTorch from the `cu126` index | Ships zero sm_120 kernels; bitsandbytes picks its `.so` from `torch.version.cuda` with no arch awareness, so the first CUDA op dies with `no kernel image is available for execution on the device` |
| `flash-attn` | sm_120 has no `wgmma`, no `tcgen05`, and ~100 KB shared memory per block against Hopper's 228 KB. FA3/FA4 physically cannot run. Use `attn_implementation="sdpa"` |
| `triton`, installed separately | PyTorch pins its own |
| vLLM | On a single 32 GB card it buys nothing and costs a second copy of the weights, an unsupported weight-sync path under QLoRA, and several open sm_120 bugs ([ADR-0002](docs/adr/0002-no-vllm.md)) |

Verify the install is sane:

```bash
python -c "import torch; print(torch.version.cuda, torch.cuda.get_device_capability(0))"
# want toolkit 12.8+ and capability (12, 0)
```

**Known defect:** `torchvision 0.26.0` is built against CUDA 13.0 while torch is cu128, so any
import reaching `transformers.image_utils` raises — including
`from transformers import TrainerCallback`. This is text-only training; uninstall torchvision.

---

## Running the tests

Three suites, separated by pytest marker. The default command spawns no subprocesses and stays
sub-second, so it is cheap enough to run on every red–green cycle:

```bash
conda activate post-train
pytest -q                         # unit tests only — no subprocess, sub-second
pytest -q -m subprocess_backend   # real subprocesses, seconds
pytest -q -m containment          # hostile programs, requires firejail
```

The suites arrive with sprint 1; see the task board in
[`docs/project-status.md`](docs/project-status.md) for what is green today.

---

## Repository layout

```
config/                 every tunable value — limits, caps, seeds, weights, prompts
src/post_training_rl/
  types.py              the frozen dataclasses that cross every seam
  config.py             YAML loading
  comparator.py         CodeContests output matching
  extraction.py         syntax-gated code recovery from a completion
  sandbox/              the one real seam — firejail, subprocess, and fake adapters
  verifier.py           executes a rollout against a problem's tests
  rewards.py            the reward registry
  startup.py            hostile-program self-test, run before the first step
docs/                   see below
tests/
```

---

## Documentation

Read in this order if you are new:

| Document | Answers |
| --- | --- |
| [`CLAUDE.md`](CLAUDE.md) | The hardware constraint, the conda env, what must never be installed, how work is planned |
| [`CONTEXT.md`](CONTEXT.md) | The glossary — *rollout*, *group*, *degenerate group*, *verifier*, *scorer*, *fence*. Includes `_Avoid_` lists, and they are enforced |
| [`docs/design/rl-loop.md`](docs/design/rl-loop.md) | Where each module sits in the RL loop, and where the analogy breaks |
| [`docs/plans/roadmap.md`](docs/plans/roadmap.md) | Four sprints, each with an objective and a gate |
| [`docs/project-status.md`](docs/project-status.md) | What is done, what is next, what is still unmeasured |

Reference material:

- [`docs/adr/`](docs/adr/) — 13 decision records. **Read the relevant ADR before changing
  behaviour in that area.** Work that contradicts one says so explicitly rather than silently
  overriding it.
- [`docs/design/`](docs/design/) — `verifier-scorer.md` (module shapes, types, seams),
  `behavior.md` (does / guarantees / refuses, per module — the spec tests derive from),
  `rl-reward-functions.md` (every reward function and its source), `model.md` (checkpoint,
  quantisation, LoRA placement, VRAM budget).
- [`docs/research/`](docs/research/) — primary-source research on the RLVR stack and on format
  adherence. Documents several places where official docs are actively wrong, including vLLM's
  attention-backend page and the CodeContests dataset card's enum tables.

---

## Two traps this project exists downstream of

Both are silent, and both would corrupt every number in the system without looking like a bug.

1. **Exact string matching is the wrong comparator.** It scores correct solutions as failures
   over trailing whitespace, capitalisation, and float formatting — indistinguishable from a
   genuine wrong answer. At least one widely-used open implementation judges CodeContests this
   way, and under its semantics DeepMind's own gold solutions score zero.
   [ADR-0007](docs/adr/0007-codecontests-token-comparator.md).
2. **"Cannot format output" and "cannot write valid Python" are opposite problems.** Every
   surveyed harness collapses them into one bucket. This project records fence and parse as two
   independent facts and logs both every step — no published source reports a code
   parse-failure rate for any model on any benchmark.
   [ADR-0012](docs/adr/0012-syntax-gated-extraction.md).
