# Extract code with a syntax-gated cascade, recording fence and parse separately

Code is recovered from a completion by a tiered cascade: prefer python-tagged fenced blocks,
then untagged fenced blocks, then any fenced block, then bare code with no fence — taking the
**last** syntactically valid candidate at the first tier that yields one, with `ast.parse` as
the gate.

Extraction records **two independent facts**, not one collapsed tier: which `fence` the code
arrived in (`tagged` / `untagged` / `other_tag` / `none`) and whether the recovered code
`parsed`. Both are logged every step.

## Why two facts rather than one tier

An earlier draft of this decision recorded a single ordered tier, where a python-tagged block
containing a syntax error fell through the cascade and was recorded as `any_invalid`. That
fused two unrelated failures: *the model cannot format its output* and *the model cannot write
valid Python*. They call for opposite responses — the first argues for a prompt change or
assistant prefill, the second argues for more training and nothing else — and the histogram
exists precisely to tell them apart. Nothing had been built on the single-tier model when this
was revised.

Fence and parse are genuinely orthogonal: a flawless fence can wrap broken code, and correct
code can arrive with no fence at all.

Five of seven surveyed implementations take the last fenced block rather than the first,
because models routinely quote the problem's example or sketch a naive version before giving
the real solution. Notably, Qwen's own evaluation harness is a dissenter — it takes the first
— so Qwen2.5-Coder's published numbers rest on a policy we are not using.

## How this feeds the reward

The `extractability` entry in ADR-0011's registry scores both facts additively, weighted
against the execution reward. **The concrete values live in
[`docs/design/rl-reward-functions.md`](../design/rl-reward-functions.md) §3**, not here — this
ADR records the two properties they must satisfy:

1. **Fence quality is rewarded in its own right.** Well-formed output is wanted even when the
   program already runs, so a better fence scores higher at equal parse status.
2. **The parse swing must exceed the fence swing.** The worst parsing rollout has to outrank
   the best non-parsing one — a beautifully fenced broken program must never beat a bare
   working one. Any future retuning of the values has to preserve this.

The cosmetic-grading risk is accepted knowingly: the cascade already recovers code from
untagged and unfenced completions, so the fence term pays for output that was already usable,
and SimpleRL-Zoo warns that format rewards "penalize many correct explorations." At weight
0.1 the effect is bounded, and the fence histogram makes it observable — if formatting
saturates while accuracy stalls, this term is the first thing to drop.

## Consequences

Logging is the point as much as extraction is. **No published source reports a code
parse-failure rate for any model on any benchmark** — Qwen's technical reports do not mention
it, and every surveyed harness collapses "no parseable code" into the same bucket as "wrong
answer." The fence and parse distributions are therefore a measurement we make ourselves, and
they decide whether `extractability` is worth keeping: below roughly 2% failure it buys almost
no group variance and should be dropped.

Never fall back to executing the whole completion unguarded — that is a known bug in one
reference implementation, which ships prose to the interpreter. The bare tier is syntax-gated
precisely so prose fails it.

Assistant prefill (forcing the completion to open inside a code block) is **not** used by
default, though it is sound: it conditions the prompt rather than the sampling distribution,
so unlike constrained decoding it introduces no mismatch between the sampled and trained
policy, and TRL masks prompt tokens from the loss for free. It is held in reserve because it
forecloses reasoning before the code. If adopted, the prefill must be re-prepended inside the
reward function or every rollout silently scores zero.

Constrained decoding is rejected outright: it is unavailable on the chosen generation backend
(ADR-0002), and by making format failure unreachable it would remove exactly the variance an
all-wrong group needs.
