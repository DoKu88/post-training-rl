# Extract code with a syntax-gated cascade and log which tier fired

Code is recovered from a completion by a tiered cascade: prefer python-tagged fenced blocks,
then untagged fenced blocks, then any fenced block, then bare code with no fence — taking the
**last** syntactically valid candidate at the first tier that yields one, with `ast.parse` as
the gate. Which tier fired is recorded on every rollout.

Five of seven surveyed implementations take the last fenced block rather than the first,
because models routinely quote the problem's example or sketch a naive version before giving
the real solution. Notably, Qwen's own evaluation harness is a dissenter — it takes the first
— so Qwen2.5-Coder's published numbers rest on a policy we are not using.

## Consequences

Tier logging is the point as much as extraction is. **No published source reports a
code parse-failure rate for any model on any benchmark** — Qwen's technical reports do not
mention it, and every surveyed harness collapses "no parseable code" into the same bucket as
"wrong answer." The tier histogram is therefore a measurement we have to make ourselves, and
it decides whether the `extractability` auxiliary reward is worth keeping: below roughly 2%
failure it buys almost no group variance and should be dropped.

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
