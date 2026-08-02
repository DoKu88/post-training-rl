# Filter unwinnable problems from both train and eval

Problems are dropped from **both** the training and evaluation splits when they are
multiple-output (matched by description patterns such as "print any", "output any", "if there
are multiple", "in any order"), interactive, use file I/O rather than stdin/stdout, or exceed
the prompt token budget. The exact filter patterns belong in config and must not drift, since
they define the evaluation set.

About a quarter of CodeContests problems admit more than one correct answer but are stored
with a single expected output, chosen as whatever the majority of human solutions printed.
There is no field marking them. On a real judge these are graded by a checker program; here
a correct solution that emits a different valid answer scores zero. For evaluation that
merely understates the score, but for RL the false negative becomes a gradient that pushes
away from correct policies and reinforces matching the majority human output — a style
objective, not a correctness one.

## Consequences

**Our numbers are no longer comparable to any published CodeContests result**, since every
published number is computed against the unfiltered split. This is accepted deliberately.
It makes the pre-training baseline measurement mandatory rather than optional: without a
pass@1 measured on the same filtered set, there is no reference point at all.

Over-length problems are dropped rather than truncated. A truncated problem statement loses
the constraints or the I/O format and is unsolvable by construction, so keeping it buys only
guaranteed-degenerate groups at full rollout cost.

The keyword filter is imprecise in both directions and its drop rate must be reported. A
residue of multiple-output problems phrased unusually will survive it.
