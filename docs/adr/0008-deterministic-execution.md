# Force deterministic execution of generated solutions

Solutions run with `PYTHONHASHSEED=0` and a seeding preamble (`random.seed`, and numpy's
when importable) prepended before execution. The seed value lives in config.

The reason is specific to group-relative RL rather than to correctness. A randomised or
hash-order-dependent solution scores differently across runs, and when two such rollouts sit
in the same group, GRPO reads that difference as advantage — the model receives gradient for
variance it did not cause. Seeding does not break randomised algorithms; a randomised
quicksort still sorts, it simply picks the same pivots every time.

## Consequences

No surveyed harness does this — a grep across code_contests, LiveCodeBench, rLLM, open-r1,
SandboxFusion and Piston found zero of them seeding executed solutions. We are deliberately
diverging.

`PYTHONHASHSEED=0` disables hash-collision DoS protection. That is acceptable here because
the CPU rlimit and wall-clock timeout already bound the damage.

The preamble shifts traceback line numbers, so the offset must be recorded if tracebacks are
ever surfaced. A solution that explicitly reseeds from the clock defeats the preamble.
