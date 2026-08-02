# Project status

**Last updated:** 2026-08-02 · **Phase:** sprint 1 in progress — task 2 of 8 complete

A running record of what exists, what is decided, and what happens next. Update it whenever a
sprint task completes, a decision is made, or an unknown gets measured.

---

## One-line summary

Every design decision is made and written down. **Sprint 1 is under way** — scaffold, types, config, and
the comparator are in. The next action is
`/write-code docs/plans/sprint-01.md, task 3`.

---

## Where things stand

| | Status |
| --- | --- |
| Research | ✅ Complete — 2,918 lines across two primary-source documents |
| Decisions | ✅ 13 ADRs |
| Vocabulary | ✅ `CONTEXT.md` |
| Design | ✅ 5 documents — loop, modules, behaviour, rewards, model |
| Plan | ✅ Roadmap + sprint 1 fully specified |
| **Code** | 🚧 **Sprint 1, task 2 of 8** — comparator green, 12 unit tests |
| Environment | ⚠️ Conda env `post-train`, one known defect · firejail 0.9.74 installed |

---

## What is done

### Research — `docs/research/`

| Document | Covers |
| --- | --- |
| [`rlvr-stack.md`](research/rlvr-stack.md) (1,420 lines) | TRL `GRPOTrainer` API, vLLM colocation, bitsandbytes on sm_120, CodeContests schema, sandboxing, reward shape. Version-pinned permalinks throughout |
| [`format-adherence.md`](research/format-adherence.md) (1,498 lines) | Code extraction across seven implementations, format rewards, constrained decoding, assistant prefill, prompt templates. 161 inline sources |

Both include a blunt "risks and unknowns" section listing what could **not** be verified.

### Decisions — `docs/adr/`

| ADR | Decision |
| --- | --- |
| [0001](adr/0001-trl-grpotrainer-single-turn.md) | TRL `GRPOTrainer`, single-turn episodes — not hand-rolled |
| [0002](adr/0002-no-vllm.md) | transformers continuous batching, **no vLLM** |
| [0003](adr/0003-3b-stepping-stone-to-7b.md) | Qwen2.5-3B is a stepping stone; 7B is the target; scaling is config |
| [0004](adr/0004-verifier-scorer-split.md) | Verifier ≠ scorer; no gym-style `step()` |
| [0005](adr/0005-firejail-sandbox.md) | firejail, not bare rlimits |
| [0006](adr/0006-flat-timeout-ignore-dataset-limit.md) | Flat 10 s timeout; dataset `time_limit` never read |
| [0007](adr/0007-codecontests-token-comparator.md) | CodeContests token comparator — not exact match, not LiveCodeBench's |
| [0008](adr/0008-deterministic-execution.md) | Deterministic execution of generated solutions |
| [0009](adr/0009-test-pool-selection.md) | Private tests first, generated as filler, cap 15, floor 5 |
| [0010](adr/0010-aggressive-problem-filtering.md) | Filter unwinnable problems from **both** splits — comparability knowingly sacrificed |
| [0011](adr/0011-reward-registry.md) | Reward functions are a config-selected registry; `binary` is the default |
| [0012](adr/0012-syntax-gated-extraction.md) | Syntax-gated extraction; fence and parse recorded separately |
| [0013](adr/0013-public-tests-are-a-diagnostic.md) | Public tests are a diagnostic, never a reward input |

### Design — `docs/design/`

| Document | Answers |
| --- | --- |
| [`rl-loop.md`](design/rl-loop.md) | Where each module sits in the classic RL loop, and the four places the analogy breaks |
| [`verifier-scorer.md`](design/verifier-scorer.md) | Module shapes, types, the one real seam, known gaps |
| [`behavior.md`](design/behavior.md) | Does / guarantees / refuses, per module — the spec tests derive from |
| [`rl-reward-functions.md`](design/rl-reward-functions.md) | All nine registry entries, the shared input, and a likelihood ranking |
| [`model.md`](design/model.md) | Checkpoints, quantisation, LoRA placement, VRAM budget, environment versions |

### Plan — `docs/plans/`

[`roadmap.md`](plans/roadmap.md) — four sprints, each with objective, delivers-table, and gate.
[`sprint-01.md`](plans/sprint-01.md) — 8 tasks, 63 unit + 13 integration tests, fully specified.

---

## What happens next

### Immediate

```bash
conda activate post-train
# then, one task per invocation, reviewed between each:
/write-code docs/plans/sprint-01.md, task 3
```

### Sprint 1 task board

| # | Task | Tests | Status |
| --- | --- | --- | --- |
| 1 | Scaffold, types, config loading | 0 (deliberate) | ✅ |
| 2 | Comparator | 12 unit | ✅ |
| 3 | Extraction cascade | 16 unit | ☐ **review after** |
| 4 | Sandbox seam + fake + subprocess | 4 integration | ☐ |
| 5 | Firejail + containment | 9 integration | ☐ **review after** |
| 6 | Verifier | 13 unit | ☐ |
| 7 | Reward registry | 20 unit | ☐ |
| 8 | Startup self-test | 2 unit | ☐ |

**Sprint 1 gate:** all suites green · containment verified in CI with firejail · a rollout
scorable by all six implemented reward functions **with no model loaded** · every
behaviour-governing constant in `config/verifier.yaml` · nothing imports `trl`,
`transformers`, `peft`, or `datasets`.

### Sprint order after that

| Sprint | Objective | Blocked on |
| --- | --- | --- |
| 2 | Filtered corpus + four measurements | Sprint 1 |
| 3 | Two-step GRPO run on a tiny model | Sprint 2, **and the torchvision fix** |
| 4 | Baseline pass@1, then `binary` vs `pass_rate` | Sprint 3 |

---

## Open items

### Blocking, with a known fix

**`torchvision 0.26.0` is built against CUDA 13.0 while torch is `2.11.0+cu128`.** Any import
path reaching `transformers.image_utils` raises — including
`from transformers import TrainerCallback`, which is exactly the hook the verification cache
reset uses. This is text-only training, so:

```bash
conda activate post-train && pip uninstall torchvision
```

Does not affect sprint 1 (nothing there imports transformers). **Must be fixed before
sprint 3.**

### Unmeasured — each currently an estimate in config

| Unknown | Decides | Measured in |
| --- | --- | --- |
| `private_tests` count distribution | Whether the 15-cap and 5-floor hold, and whether generated filler dominates | Sprint 2 |
| Private/generated share per problem | The guard deferred in ADR-0009 | Sprint 2 |
| Prompt token-length distribution | `max_prompt_length` and the drop rate | Sprint 2 |
| Filter drop rate per rule | Whether the multi-output regex over-matches (expect ~25%) | Sprint 2 |
| Sandbox throughput | Whether a thread pool suffices, or firejail startup forces a process pool | Sprint 1 |
| Fence and parse rates | Whether `extractability` earns its keep, and whether prefill is worth adopting | Sprint 3 |
| Base-model pass@1 | The only reference point that exists, since ADR-0010 broke comparability | Sprint 4 |

### Unverified assumptions

| Assumption | Risk |
| --- | --- |
| Arrow round-trips nested test structures | Adapter rework if not |
| Continuous batching works with `Linear4bit` | 7B only; fallback is plain `.generate()` |

The four TRL assumptions that used to sit here — reward-function signature, kwarg forwarding,
`log_metric`, and whether `mask_truncated_completions` touches group reward statistics — were
**all verified against the installed `trl 1.9.2` source**. The last one changed a design
decision: truncated completions still feed the group mean and standard deviation, so they must
be verified normally.

---

## Accepted risks

| Risk | Why accepted |
| --- | --- |
| **Contamination** — CodeContests splits end 2021; Qwen2.5 almost certainly saw these problems | Explicitly accepted. Affects interpretation of eval numbers, not the training loop |
| **No comparability with published numbers** — ADR-0010 filters the eval split | Deliberate. Makes the sprint-4 baseline mandatory rather than optional |
| **Multi-output residue** — the keyword filter misses unusually-phrased problems | No field marks them; the filter is the only lever |
| **Cosmetic grading in `extractability`** — the fence term pays for output the cascade already recovers | Bounded at weight 0.1, and the fence histogram makes it observable. First thing to drop if formatting saturates while accuracy stalls |
| **`binary` may be the wrong default** | The literature genuinely disagrees. This is why the registry exists; ADR-0011's ranking is labelled a prior, not a result |

---

## Deliberately out of scope

7B · dynamic sampling · assistant prefill · SFT warmup · the `hierarchical`, `verpo`, and
`overlong` rewards. Each with its trigger condition in
[`roadmap.md`](plans/roadmap.md#deliberately-out-of-scope). Constrained decoding is **rejected
outright**, not deferred.

---

## Environment

Conda env **`post-train`** — note the name differs from the repo.

| Package | Version |
| --- | --- |
| torch | 2.11.0+**cu128** |
| transformers | 5.14.1 |
| trl | 1.9.2 |
| peft | 0.20.0 |
| bitsandbytes | 0.50.0 |
| datasets | 5.0.1 |
| accelerate | 1.14.0 |
| flash-attn | **absent, deliberately** — cannot run on sm_120 |

Hardware: single RTX 5090, 32 GB, Blackwell sm_120. Full detail and the sm_120 traps are in
[`model.md` §7](design/model.md) and [`CLAUDE.md`](../CLAUDE.md).
