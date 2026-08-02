# Roadmap

Four sprints from an empty repo to a first ablation. Each has one objective, a set of modules
it delivers, and a gate that must hold before the next begins.

This is also the entry point to the project's documentation. If you are new here, read in this
order: [`CLAUDE.md`](../../CLAUDE.md) for the hardware constraint and how work is planned →
[`CONTEXT.md`](../../CONTEXT.md) for the vocabulary → [`docs/design/rl-loop.md`](../design/rl-loop.md)
for how the pieces fit together → this file.

---

## The document set

| Document | Answers |
| --- | --- |
| [`CLAUDE.md`](../../CLAUDE.md) | What is the hardware constraint, which conda env, what must never be installed, how work is planned |
| [`CONTEXT.md`](../../CONTEXT.md) | What does a term mean — *rollout*, *group*, *degenerate group*, *verifier*, *scorer*, *fence*. Includes `_Avoid_` lists |
| [`docs/adr/`](../adr/) | Why a decision was made, and what was rejected. 13 ADRs |
| [`docs/design/rl-loop.md`](../design/rl-loop.md) | Where each module sits in the classic RL loop, and the four places the analogy breaks |
| [`docs/design/verifier-scorer.md`](../design/verifier-scorer.md) | Module shapes, types, seams, known gaps |
| [`docs/design/behavior.md`](../design/behavior.md) | What each module *does*, *guarantees*, and *refuses*. The specification tests derive from |
| [`docs/design/rl-reward-functions.md`](../design/rl-reward-functions.md) | Every reward function, its shared input, its source, and how likely each is to work |
| [`docs/design/model.md`](../design/model.md) | Checkpoint, quantisation, LoRA placement, VRAM budget, environment versions |
| [`docs/research/rlvr-stack.md`](../research/rlvr-stack.md) | Primary-source research: TRL, vLLM, bitsandbytes on sm_120, CodeContests schema, sandboxing, reward shape |
| [`docs/research/format-adherence.md`](../research/format-adherence.md) | Code extraction across seven implementations, format rewards, constrained decoding, prefill |

**Work is test-driven.** Every task in a sprint file leads with the behaviour it delivers,
traced to a numbered item in [`behavior.md`](../design/behavior.md). Tests are how that
behaviour gets pinned — they are derived from the spec one slice at a time, never transcribed
in bulk. Nothing in a sprint file invents behaviour that `behavior.md` does not already state.

---

## Sprint 1 — A verifier and scorer that can be trusted

**Objective:** grade a rollout correctly, in isolation, with no model involved.

### Where this sits in the loop

Everything inside the environment box of
[`rl-loop.md` §2a](../design/rl-loop.md#2a-the-same-loop-with-every-module) **except** the
dataset builder — extraction, the sandbox, the comparator, the verifier, and the reward
registry. In the numbered trace of one step
([`rl-loop.md` §5](../design/rl-loop.md)), this sprint builds steps 5 through 7.

### Why it comes first

Three reasons, in order of force.

1. **[ADR-0003](../adr/0003-3b-stepping-stone-to-7b.md) says the 3B run exists to prove the
   verifier, scorer, and reward functions are correct.** That is only possible if they are
   already trustworthy when it starts. A 3B run against an untrusted verifier proves nothing
   about either.
2. **This is where silent bugs live.** [ADR-0007](../adr/0007-codecontests-token-comparator.md)
   exists because exact string matching — the obvious implementation — scores correct
   solutions as failures over trailing whitespace and capitalisation, and that failure is
   indistinguishable from a wrong answer. `rlvr-stack.md` §4.7 documents that at least one
   widely-used open implementation judges CodeContests with the wrong comparator, under which
   DeepMind's own gold solutions score zero.
3. **None of it needs a GPU.** Pure functions and subprocesses only, so all of it can be
   verified before anything expensive runs.

### Delivers

| Module | Design | Behaviour spec | Governed by |
| --- | --- | --- | --- |
| `comparator` | [verifier-scorer §5](../design/verifier-scorer.md) | [behavior §1](../design/behavior.md) | [ADR-0007](../adr/0007-codecontests-token-comparator.md) |
| `extraction` | [verifier-scorer §4](../design/verifier-scorer.md) | [behavior §2](../design/behavior.md) | [ADR-0012](../adr/0012-syntax-gated-extraction.md) |
| `sandbox` — protocol + firejail, subprocess, fake | [verifier-scorer §3](../design/verifier-scorer.md) | [behavior §4](../design/behavior.md) | [ADR-0005](../adr/0005-firejail-sandbox.md), [ADR-0006](../adr/0006-flat-timeout-ignore-dataset-limit.md), [ADR-0008](../adr/0008-deterministic-execution.md) |
| `verifier` | [verifier-scorer §6](../design/verifier-scorer.md) | [behavior §5](../design/behavior.md) | [ADR-0004](../adr/0004-verifier-scorer-split.md) |
| `rewards` — six of nine registry entries | [rl-reward-functions §1, §3](../design/rl-reward-functions.md) | [behavior §3](../design/behavior.md) | [ADR-0011](../adr/0011-reward-registry.md), [ADR-0013](../adr/0013-public-tests-are-a-diagnostic.md) |
| `config`, `types`, startup self-test | [verifier-scorer §2, §10](../design/verifier-scorer.md) | [behavior §9](../design/behavior.md) | — |

### Glossary terms this makes real

`CONTEXT.md`'s **verifier** and **scorer** stop being words and become two modules with a line
between them — impure and sandboxed on one side, pure and testable on the other. **Fence** and
**extraction outcome** become a type with two independent fields. The `_Avoid_` list matters
here: nothing in this sprint may be named *environment*, because the environment role in the
loop is filled by three modules and only one of them is being built.

### Gate

- Every unit test green — **63**, sub-second, no subprocess. Plus 4 subprocess-backend and 9
  containment tests, each behind its own marker.
- Containment verified against hostile programs in CI with firejail installed.
- A rollout can be scored end to end by all six implemented reward functions **with no model
  loaded**.
- Every behaviour-governing constant lives in `config/verifier.yaml`.

### What it unblocks

Sprint 3 cannot produce a meaningful reward without this, and sprint 4's baseline is
meaningless if the verifier is wrong. It also settles a risk carried in
[verifier-scorer §12](../design/verifier-scorer.md): **sandbox throughput is unmeasured**, and
firejail startup × 15 tests × group size may make a thread pool insufficient.

---

## Sprint 2 — The corpus, and the numbers we have been guessing

**Objective:** turn 13,328 raw problems into the filtered training set, and replace four
guesses with measurements.

### Where this sits in the loop

The dataset builder — the **state distribution** that supplies `S_t`
([`rl-loop.md` §3](../design/rl-loop.md)). It is the one part of the environment box sprint 1
leaves untouched, and per [`rl-loop.md` §4.1](../design/rl-loop.md) it is also where the
classic loop's `S_t+1` arrow does *not* apply: the next prompt is the next row, drawn
independently of what the model just wrote.

### Why it comes second

The filters decide what training ever sees, and
[ADR-0010](../adr/0010-aggressive-problem-filtering.md) makes them aggressive enough to break
comparability with every published CodeContests number. That is an accepted cost, but it means
the filtered corpus **is** the experiment — and a filter bug is invisible in the reward curve.

It comes after sprint 1 rather than before because the measurements that matter most
(format-failure rate, base-model pass rate) need a working extractor and verifier.

### The four unknowns

Each currently sits in config as an estimate, and each changes a decision:

| Unknown | Decides | Recorded in |
| --- | --- | --- |
| `private_tests` count distribution | Whether the 15-cap and 5-floor are right, and whether generated filler ends up dominating the graded signal | [ADR-0009](../adr/0009-test-pool-selection.md) |
| Private/generated share per problem | The guard deliberately deferred rather than rejected — a problem with 2 private and 3 generated tests clears the floor with 60% of its signal from a pool measured at a 46% false-positive-or-slow rate | [ADR-0009](../adr/0009-test-pool-selection.md) |
| Prompt token-length distribution | `max_prompt_length`, and how many problems get dropped rather than truncated | [ADR-0010](../adr/0010-aggressive-problem-filtering.md) |
| Filter drop rates per rule | Whether the multi-output regex is too greedy — `rlvr-stack.md` §4.6 puts multi-output problems at ~25% of the validation set, so a filter removing far more than that is over-matching |

### Delivers

| Module | Design | Behaviour spec | Governed by |
| --- | --- | --- | --- |
| `dataset` builder, filters, test selection | [verifier-scorer §9](../design/verifier-scorer.md) | [behavior §7](../design/behavior.md) | [ADR-0009](../adr/0009-test-pool-selection.md), [ADR-0010](../adr/0010-aggressive-problem-filtering.md) |
| Measurement report | — | — | Feeds the deferred decision in ADR-0009 |

### Two data traps this sprint must not fall into

Both from [`rlvr-stack.md` §4.3 and §4.4](../research/rlvr-stack.md), and both silent:

- **The HuggingFace datasets-server view is truncated to 3,762 of 13,328 training rows.** The
  viewer, `/rows`, `/statistics`, and `refs/convert/parquet` all serve 28% of the training set
  while reporting success. Load from `data/` on `main`.
- **`difficulty` decodes incorrectly** via `ClassLabel.int2str()` for every value ≥ 19,
  because the stored integers are raw proto values while the label list is dense. Use the
  proto mapping. `source` has the opposite bug — the HF mapping is correct but does not match
  the proto.

### Gate

- The filtered dataset builds reproducibly from a config file.
- Drop counts are reported per filter rule, and the totals surfaced.
- The four distributions are recorded, and any config value they contradict is updated from
  measurement rather than estimate.

---

## Sprint 3 — The loop closes

**Objective:** a two-step GRPO run completes on a tiny model, with every metric emitted.

### Where this sits in the loop

The wiring — the `trl_adapter` box in
[`rl-loop.md` §2a](../design/rl-loop.md#2a-the-same-loop-with-every-module), plus the agent
side: policy, LoRA, and continuous-batching generation per
[`model.md` §3 and §5](../design/model.md).

### Why it comes third

It is the first time the four TRL assumptions get *exercised* rather than read. All four were
verified against the installed `trl 1.9.2` source and are recorded in
[verifier-scorer §12](../design/verifier-scorer.md), but source-reading is not running code.
The one that matters most:

> `mask_truncated_completions` multiplies `completion_mask` **only** — `rewards` is never
> touched. A truncated completion's reward therefore **does** feed the group mean and standard
> deviation, shaping every sibling's advantage. Truncated completions must be verified
> normally.

This is also the first time the `VerificationCache`
([behavior §6](../design/behavior.md)) runs under the trainer, and its correctness depends on
[ADR-0008](../adr/0008-deterministic-execution.md) — two rollouts with identical text must
verify identically, or sharing a cache entry is a bug.

### Blocked until fixed

**`torchvision 0.26.0` is built against CUDA 13.0 while torch is `2.11.0+cu128`**, so
`from transformers import TrainerCallback` raises — and that callback is exactly what the
cache reset hooks into. Recorded in [`CLAUDE.md` §Environment](../../CLAUDE.md) and
[`model.md` §7](../design/model.md). This is text-only training; uninstalling torchvision is
the clean fix. Sprint 1 is unaffected because nothing in it imports transformers.

### Delivers

| Module | Design | Behaviour spec | Governed by |
| --- | --- | --- | --- |
| `trl_adapter`, `VerificationCache` | [verifier-scorer §8](../design/verifier-scorer.md), [rl-reward-functions §2](../design/rl-reward-functions.md) | [behavior §6, §8](../design/behavior.md) | [ADR-0001](../adr/0001-trl-grpotrainer-single-turn.md), [ADR-0004](../adr/0004-verifier-scorer-split.md) |
| Model loading, LoRA, generation backend | [model.md §1–§6](../design/model.md) | — | [ADR-0002](../adr/0002-no-vllm.md), [ADR-0003](../adr/0003-3b-stepping-stone-to-7b.md) |
| `train.py`, end-to-end smoke test | — | [behavior §10](../design/behavior.md) | — |

### Gate

- The smoke test passes: a two-step run on a randomly-initialised tiny model, finite rewards,
  no exception.
- Fence and parse histograms appear in the logs, separately — per
  [ADR-0012](../adr/0012-syntax-gated-extraction.md) these are **the measurement nobody has
  published**, and they are what decides whether assistant prefill is worth adopting.
- `diag/public_pass_rate` appears — the ground-truth signal that does not move with the reward
  ([ADR-0013](../adr/0013-public-tests-are-a-diagnostic.md)).
- Every shadow-logged reward appears alongside the training reward at weight 0.0
  ([rl-reward-functions §4](../design/rl-reward-functions.md)).
- `frac_reward_zero_std` is being tracked.

---

## Sprint 4 — First real run, and the first thing we learn

**Objective:** a baseline, then one honest ablation.

### The baseline comes first, and is not optional

**Measure base-model pass@1 and pass@10 on the filtered evaluation set before training
anything.** [ADR-0010](../adr/0010-aggressive-problem-filtering.md) removed comparability with
every published CodeContests number by filtering the evaluation split, and accepted that cost
deliberately. The consequence is that this baseline is the *only* reference point that exists
— without it there is no way to say whether a run improved the model.

Evaluation must use **binary all-tests-pass on the full test suite**, not the training reward
and not the 15-test subset. A metric that shares the reward's shape cannot distinguish learning
from the reward optimising itself
([rl-reward-functions §6](../design/rl-reward-functions.md)).

### The first ablation

`binary` versus `pass_rate` — ranks **1** and **5** in
[rl-reward-functions §3](../design/rl-reward-functions.md). They sit at opposite ends of the
evidence, and the literature genuinely disagrees:

- **For binary:** [arXiv:2605.02944](https://arxiv.org/html/2605.02944) ran the ablation on
  Qwen2.5-7B-Instruct with GRPO at 16 rollouts per prompt — pass-rate matched at pass@1 and
  **lost 2 points at pass@16**, with 57.4% of groups containing both harmful samples with
  positive advantage and helpful samples with negative advantage.
- **For pass-rate:** SWE-RL found continuous beat discrete precisely where exact match almost
  never fires — arguably a 3B model on CodeContests.

Nobody has published the comparison this project needs. That is why the registry exists, and
why [rl-reward-functions §3.1](../design/rl-reward-functions.md) labels its ranking a prior
rather than a result.

### Delivers

Evaluation harness, recorded baseline, one completed comparison.

### Gate

- Baseline pass@1 and pass@10 recorded on the filtered evaluation set.
- One ablation completed with seed, filters, group size, and LoRA config held fixed
  ([`model.md` §3](../design/model.md)).
- `frac_reward_zero_std` reported alongside. **If it sits near 1.0 the comparison measured
  nothing** — the answer is then dynamic sampling or a difficulty curriculum, not a different
  reward shape.

---

## Deliberately out of scope

| Not doing | Why, and what would change it |
| --- | --- |
| **7B** | Everything is config-driven ([ADR-0003](../adr/0003-3b-stepping-stone-to-7b.md), [`model.md` §6](../design/model.md)) so scaling is a file rather than a rewrite. But TRL casts the unwrapped model to bf16/fp16 before `generate_batch`, and its interaction with `Linear4bit` is **untested upstream** ([`model.md` §5](../design/model.md)). That verification is its own piece of work, with plain `.generate()` as the fallback |
| **Dynamic sampling** | [ADR-0011](../adr/0011-reward-registry.md) records it as the largest single measured gain available (+9 points in DAPO's ablation, against +8 for Clip-Higher). It is **not a reward function** — it alters the loop's control flow rather than the contents of a box ([`rl-loop.md` §6](../design/rl-loop.md)), so it lives behind its own flag. Worth building once sprint 4 shows how degenerate the groups actually are |
| **Assistant prefill** | Sound but unattested — it conditions the prompt rather than the sampling distribution, so unlike constrained decoding there is no mismatch between the sampled and trained policy ([ADR-0012](../adr/0012-syntax-gated-extraction.md)). Held in reserve because it forecloses reasoning before the code. Gated on the fence and parse histograms from sprint 3 |
| **Constrained decoding** | Rejected outright, not deferred. Unavailable on the chosen generation backend, and by making format failure unreachable it would remove exactly the variance an all-wrong group needs ([ADR-0012](../adr/0012-syntax-gated-extraction.md)) |
| **SFT warmup** | Held in reserve. The trigger is the same as for a difficulty curriculum: a base pass rate flat at zero across a few hundred problems |
| **`hierarchical`, `verpo`, `overlong` rewards** | In the registry's design space, not built. Each needs machinery no other entry needs — AST structural alignment, Gaussian-KDE calibration, a length schedule ([rl-reward-functions §3](../design/rl-reward-functions.md)). They earn a behaviour spec when a run selects one |
