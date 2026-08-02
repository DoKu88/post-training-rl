# How the modules fit the classic RL loop

Maps every module in this project onto the standard agent–environment interaction diagram,
and states precisely where the analogy holds and where it breaks. Companion to
[`verifier-scorer.md`](./verifier-scorer.md) and [`behavior.md`](./behavior.md).

---

## 1. The canonical loop

```
                        ┌─────────────┐
                ┌──────▶│    Agent    │───────┐
                │       └─────────────┘       │
                │                             │
         state S_t                     action A_t
         reward R_t                           │
                │                             ▼
                │       ┌─────────────────────────┐
                └───────│      Environment        │
                        └─────────────────────────┘
                            R_t+1 ,  S_t+1
```

The agent observes a state, takes an action, and the environment returns a reward and the
**next state**. That last arrow — the environment producing `S_t+1` in response to `A_t` — is
the part that does not survive contact with this project.

---

## 2. Our instantiation

```
                     ┌────────────────────────────────────────┐
              ┌─────▶│  AGENT                                 │──────┐
              │      │  Qwen2.5-Instruct + LoRA   (policy π)  │      │
              │      │  GRPOTrainer               (learner)   │      │
              │      └────────────────────────────────────────┘      │
              │                                                      │
     S_t = prompt                                         A_t = completion
     R_t = float                                              (raw text)
              │                                                      │
              │      ┌────────────────────────────────────────┐      │
              └──────│  THE "ENVIRONMENT" ROLE — a composite  │◀─────┘
                     │                                        │
                     │   dataset builder   → supplies S       │
                     │   verifier          → executes A       │
                     │   scorer            → produces R       │
                     └────────────────────────────────────────┘
                         R_t+1 = float
                         S_t+1 = next prompt, drawn independently
```

### Why no single module is called "the environment"

The diagram's environment box is filled by **three** of our modules, which is exactly why
[`CONTEXT.md`](../../CONTEXT.md) lists *environment* under `_Avoid_` for the verifier. The
verifier only fills the middle third — it executes the action. It does not supply states and
it does not produce rewards.

There is a second reason, specific to our stack: in TRL, `environment_factory` means
multi-turn tool calling. Naming a module "environment" would collide with a framework concept
this project deliberately does not use (ADR-0001).

### Why the verifier is what executes the action

In the canonical loop the environment *applies* the action to the world and observes what
happens. For a game that means advancing the simulation. Here the action is a **program**, so
applying it means one thing only: running it.

That is the whole content of the **V** in RLVR. The reward is not *predicted* from the text of
the action by a learned model — it is *observed* by execution against test cases. Contrast
the two:

```
   RLHF:   completion ──▶ reward model ──▶ predicted scalar   (a guess, hackable)
   RLVR:   completion ──▶ execution     ──▶ observed outcome  (a fact)
```

So execution is not an implementation detail sitting inside the reward — it *is* the
environment's dynamics. The verifier is the module that owns it for three reasons:

1. **It is the only impure step.** Execution is where subprocesses, timeouts, filesystem
   access, and every containment concern live. Concentrating them in one module is what
   leaves the scorer a pure function of a value — which is exactly ADR-0004.
2. **Observation must be separated from valuation.** What happened (12 of 15 tests passed,
   one timed out) is a fact. What that is worth is a policy decision that seven different
   reward functions answer differently. Fusing them would mean re-executing once per reward
   shape, and would make reward logic untestable without spawning processes.
3. **The agent must not touch it.** If the policy could run its own code and report the
   result, the reward channel would be under the control of the thing being optimised — the
   textbook setup for reward hacking. Execution sits on the far side of the loop from the
   agent, deliberately.

---

## 2a. The same loop, with every module

Diagram 2 with the boxes opened up. Each component is tagged with the ADR that governs it.

```
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │ AGENT                                                                       │
 │  ┌──────────────────────────┐   ┌────────────────────────────────────────┐  │
 │  │ Policy       [ADR-0003]  │   │ GRPOTrainer                [ADR-0001]  │  │
 │  │ Qwen2.5-3B → 7B          │◀─▶│ G rollouts per prompt = the group      │  │
 │  │ LoRA bf16 │ NF4 QLoRA    │   │ A_i = (r_i − mean_group) / std_group   │  │
 │  │ attn_implementation=sdpa │   │ clipped PG loss · beta = 0.0 (no ref)  │  │
 │  │ per_device_batch_size=1  │   │ mask_truncated_completions = True      │  │
 │  └────────────┬─────────────┘   └────────────────────────────────────────┘  │
 │               │ generate                                                    │
 │  ┌────────────▼─────────────────────────────────┐                           │
 │  │ transformers continuous batching [ADR-0002]  │                           │
 │  │ use_transformers_continuous_batching = True  │                           │
 │  │ deliberately NOT vLLM                        │                           │
 │  └──────────────────────────────────────────────┘                           │
 └────────┬───────────────────────────────────────────────────────▲────────────┘
          │                                                       │
   A_t = completion ×G                              R_t+1 = float ×G
   (raw text, no code yet)                          S_t+1 = next row, independent of A_t
          │                                                       │
 ┌────────▼───────────────────────────────────────────────────────┴────────────┐
 │ "ENVIRONMENT" ROLE — a composite; no single module carries this name        │
 │                                                                             │
 │  ┌───────────────────────────────────────────────────────────────────────┐  │
 │  │ trl_adapter                                                           │  │
 │  │  · rebuilds Problem from Arrow columns (dataset stores dicts)         │  │
 │  │  · VerificationCache — one execution per rollout, keyed               │  │
 │  │      (problem_id, completion); correct ONLY because of [ADR-0008]     │  │
 │  └──────────────────────────────┬────────────────────────────────────────┘  │
 │                                 │ (completion, Problem)                     │
 │  ┌──────────────────────────────▼────────────────────────────────────────┐  │
 │  │ Verifier.verify_batch — executes the action        [ADR-0004]         │  │
 │  │ ThreadPoolExecutor fan-out (subprocess-bound, GIL released)           │  │
 │  │                                                                       │  │
 │  │ ① extract_python(completion, prefill)              [ADR-0012]         │  │
 │  │     tagged → untagged → any → bare → none, ast.parse gate             │  │
 │  │     LAST syntactically valid candidate wins                           │  │
 │  │     records (fence, parsed) as TWO facts, never one tier              │  │
 │  │     no code recovered ──▶ SHORT-CIRCUIT: zero sandbox calls           │  │
 │  │                                                                       │  │
 │  │ ② prepend determinism preamble                     [ADR-0008]         │  │
 │  │     random.seed(N) · np.random.seed(N) · record line offset           │  │
 │  │                                                                       │  │
 │  │ ③ for each graded test (≤15, private-first)        [ADR-0009]         │  │
 │  │   ┌─────────────────────────────────────────────────────────────┐     │  │
 │  │   │ Sandbox (Protocol) — the one real seam                      │     │  │
 │  │   │  ├── FirejailSandbox                       [ADR-0005]       │     │  │
 │  │   │  │     --private          fresh tmpfs home                  │     │  │
 │  │   │  │     --seccomp=socket   blocks network                    │     │  │
 │  │   │  │     --rlimit-nproc=32  fork bombs                        │     │  │
 │  │   │  │     --rlimit-nofile=32                                   │     │  │
 │  │   │  │     --rlimit-fsize=2m  file flooding                     │     │  │
 │  │   │  │     --rlimit-as=4g     memory                            │     │  │
 │  │   │  │     --timeout=00:00:10 flat, ignores dataset [ADR-0006]  │     │  │
 │  │   │  │     PYTHONHASHSEED=0                         [ADR-0008]  │     │  │
 │  │   │  │     stdout capped 10 MB IN PARENT            [ADR-0009]  │     │  │
 │  │   │  │       (rlimit-fsize does not apply to pipes)             │     │  │
 │  │   │  ├── SubprocessSandbox   CI/dev; no network block, weaker   │     │  │
 │  │   │  └── FakeSandbox         tests; scripted results            │     │  │
 │  │   └─────────────────────────┬───────────────────────────────────┘     │  │
 │  │                             │ SandboxResult                           │  │
 │  │ ④ outputs_match(actual, expected)                  [ADR-0007]         │  │
 │  │     split on any whitespace · drop empties · case-insensitive         │  │
 │  │     1e-5 ABSOLUTE float tolerance · token counts must match           │  │
 │  │     NOT exact match · NOT LiveCodeBench semantics                     │  │
 │  │                                                                       │  │
 │  │ ⑤ on TIMEOUT → abandon remaining, record SKIPPED   [ADR-0006]         │  │
 │  │     result count always == test count                                 │  │
 │  └──────────────────────────────┬────────────────────────────────────────┘  │
 │                                 │ VerificationReport                        │
 │  ┌──────────────────────────────▼────────────────────────────────────────┐  │
 │  │ + completion_token_count + completion_was_truncated                   │  │
 │  │   ══▶ RolloutOutcome   (the type every reward function consumes)      │  │
 │  └──────────────────────────────┬────────────────────────────────────────┘  │
 │                                 │                                           │
 │  ┌──────────────────────────────▼────────────────────────────────────────┐  │
 │  │ REWARD_FUNCTIONS registry — produces R            [ADR-0011]          │  │
 │  │  primary   w=1.0   binary ◀ DEFAULT                                   │  │
 │  │                    pass_rate · binary_threshold · ladder ·            │  │
 │  │                    code_r1 · hierarchical · verpo                     │  │
 │  │  auxiliary w=0.1   extractability      (overlong: deferred)           │  │
 │  │  shadow    w=0.0   every other entry · fence + parse histograms ·     │  │
 │  │                    public pass rate            [ADR-0013]             │  │
 │  │                    → the parse-failure rate nobody has published      │  │
 │  └───────────────────────────────────────────────────────────────────────┘  │
 │                                                                             │
 │  ┌───────────────────────────────────────────────────────────────────────┐  │
 │  │ build_dataset — supplies S_t          [ADR-0009] [ADR-0010]           │  │
 │  │  load from data/ on main — NOT datasets-server (28% of train)         │  │
 │  │  difficulty via PROTO mapping, never ClassLabel.int2str()             │  │
 │  │  drop: multi-output · interactive · file-IO · over-length prompt      │  │
 │  │        (from BOTH train and eval — breaks published comparability)    │  │
 │  │  tests: private-first → generated filler · cap 15 · ≥5 floor          │  │
 │  │         longest-input-first · public kept separate                    │  │
 │  └───────────────────────────────────────────────────────────────────────┘  │
 └─────────────────────────────────────────────────────────────────────────────┘

 OUTSIDE THE LOOP
  ┌──────────────────────────────────────────────────────────────────────┐
  │ startup self-test  [ADR-0005]  4 hostile programs, must be contained │
  │ config/*.yaml                  every limit, cap, seed, threshold     │
  │ dynamic sampling   [ADR-0011]  NOT BUILT — would sit between R and   │
  │                                the update, discarding degenerate     │
  │                                groups. Alters control flow, not a    │
  │                                box. Hence not a reward function.     │
  └──────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component inputs and outputs

Ordered by where each sits in the loop.

| Module | Role in the diagram | Input | Output |
| --- | --- | --- | --- |
| **Dataset builder** | The state distribution — supplies `S_t` | Raw corpus + config | `Dataset` of `prompt`, `problem_id`, `graded_tests`, `public_tests` |
| **Policy** (Qwen + LoRA) | The agent's `π(a·s)` | Prompt token ids | Completion token ids ×G |
| **GRPOTrainer** (sampling) | Agent's action selection | `Dataset` row | G completions per prompt — **the group** |
| **TRL adapter** | The wiring between the boxes | Completions + forwarded dataset columns | `list[float]` of rewards |
| **Verification cache** | *(no role — an optimisation)* | Batch of completions + problems | `list[RolloutOutcome]`, computing each once |
| **Extraction** | Environment's reading of `A_t` | `completion: str`, `prefill: str` | `Extraction(code, fence, parsed)` |
| **Verifier** | Environment dynamics — executes `A_t` | `(completion: str, problem: Problem)` | `VerificationReport` |
| **Sandbox** | Inner mechanics of execution | `(source: str, stdin_text: str, timeout_seconds: float)` | `SandboxResult` |
| **Comparator** | Success predicate on one test | `(actual: str, expected: str)` | `bool` |
| **Reward function** | `R(s, a)` | `RolloutOutcome` | `float` |
| **GRPO update** | The agent's learning algorithm | `list[float]` rewards | Updated LoRA weights |

Note the asymmetry the table makes visible: **the verifier's output is not a reward, and the
scorer's input is not an action.** `RolloutOutcome` sits between them and is what makes ADR-0004's
split real rather than nominal. Every reward function in the registry consumes exactly that
type, which is why they are interchangeable.

---

## 4. Where the analogy breaks

Four disanalogies, in descending order of how much they matter.

### 4.1 There is no state transition — this is a contextual bandit

The diagram's `S_t+1` arrow implies the environment's next state depends on the agent's
action. Ours does not. The next prompt is the next row of the dataset, sampled independently
of whatever the model just wrote.

```
   Textbook MDP:      S_t --A_t--> S_t+1        (action causes the next state)
   This project:      S_t --A_t--> R           (episode ends; S_t+1 drawn i.i.d.)
```

Every episode has length 1. That single fact explains a cluster of design decisions that
would otherwise look arbitrary:

- **No value function and no critic.** There is no future to estimate the value of.
- **No discount factor.** Nothing to discount.
- **No temporal-difference error, no bootstrapping, no eligibility traces.**
- **Advantage comes from the group, not from a critic.** GRPO's whole premise is replacing a
  learned baseline with the mean reward of sibling rollouts on the same prompt — which is
  only available *because* there is no temporal structure to model.

### 4.2 The diagram shows one action per state; we take G

The classic loop has a single action at each timestep. We sample **G rollouts for the same
prompt** and grade them against each other. The group has no notation in the diagram at all,
yet it is the unit that carries the learning signal:

```
                        ┌──▶ completion 1 ──▶ reward 1 ──┐
    prompt S_t ────────▶├──▶ completion 2 ──▶ reward 2 ──┼──▶ normalise ──▶ advantages
                        │        ...                     │    within group
                        └──▶ completion G ──▶ reward G ──┘
```

This is also where the project's central failure mode lives: if all G rewards are equal, the
normalised advantage is zero for every member and the prompt contributes **no gradient at
all**, having cost G full rollouts. `CONTEXT.md` names this a *degenerate group*, and the
reward registry (ADR-0011) exists largely to study it.

### 4.3 The reward is not emitted by the thing that executes

In the diagram, one box produces both `R_t+1` and `S_t+1`. We split that deliberately
(ADR-0004): the verifier executes and reports, the scorer decides what the report is worth.

The payoff is direct. One execution feeds **every** reward function in the registry, so a
single run logs the counterfactual reward curves for shapes that are not driving training.
A monolithic environment box could not do that without executing once per reward shape.

### 4.4 The action the agent takes is not the action the environment runs

The policy emits **text**. The verifier executes **a program**. Between them sits extraction,
which may recover nothing at all.

```
   A_t (text)  ──extraction──▶  code | none  ──sandbox──▶  outcome
```

That gap is not incidental plumbing — it is a reward-bearing step. A completion from which no
code can be recovered short-circuits the environment entirely: no execution, no test results,
and the reward is determined by the extraction outcome alone.

---

## 5. One step, traced end to end

Concretely, for a group size of 4:

1. **Dataset builder** yields a row — `prompt`, `problem_id`, `graded_tests`,
   `public_tests`. *(supplies `S_t`)*
2. **GRPOTrainer** repeats the prompt 4× and generates via transformers continuous batching.
   → 4 completions. *(4 actions from `π(a·s)`)*
3. **TRL adapter** reconstructs `Problem` objects from the forwarded columns and asks the
   **cache** for outcomes.
4. **Cache** calls `Verifier.verify_batch` once for all 4. *(This is sound only because
   execution is deterministic — ADR-0008.)*
5. For each rollout the **verifier**:
   a. runs **extraction** → `Extraction(code, fence, parsed)`;
   b. short-circuits with an empty report if no code was recovered;
   c. otherwise prepends the determinism preamble and, per graded test, calls the
      **sandbox** and the **comparator**, abandoning the rest on the first timeout.
   → `VerificationReport`
6. **Adapter** wraps each report with token count and truncation status → `RolloutOutcome`.
7. **Reward functions** map each outcome to a float. The selected primary drives training;
   shadow-logged ones are computed at weight 0.0 and emitted as metrics. *(produces `R_t+1`)*
8. **GRPOTrainer** normalises the 4 rewards within the group → advantages → clipped
   policy-gradient loss → LoRA update. *(the agent learns)*
9. The next batch draws new prompts. **No state was carried forward.**

---

## 6. What has no place in this diagram

Three modules do real work but occupy no position in the loop, and it is worth being explicit
that this is fine rather than a modelling gap:

| Module | Why it is outside |
| --- | --- |
| **Verification cache** | A performance optimisation. Removing it changes cost, not behaviour. |
| **Startup self-test** | Runs once before the loop begins. Verifies the apparatus, not the agent. |
| **Config loading** | Determines the shape of every box before any of them exist. |

And one thing that *would* appear if it were built: **dynamic sampling** (ADR-0011) sits
between steps 7 and 8, discarding degenerate groups and regenerating. It is the only
contemplated change that alters the loop's control flow rather than the contents of a box —
which is precisely why ADR-0011 insists it is not a reward function and must stay behind its
own flag.
