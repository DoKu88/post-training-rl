# The verifier and the scorer are separate components

Executing a rollout's code and deciding what that execution is worth are two different jobs,
and they are split: the verifier is impure, sandboxed, and returns a structured result; the
scorer is a pure function from that result to a number. Neither does the other's work.

The split is not decoration. It puts all the I/O, sandboxing, timeouts, and subprocess
handling on one side of a line and leaves the reward logic trivially unit-testable on the
other. It is also what makes the reward registry (ADR-0011) possible — one execution can
feed every reward function at once, so alternative rewards can be logged for free alongside
the one actually training the policy.

## Consequences

The verifier must return structure, not a string. A reward that distinguishes "no code
found" from "crashed" from "passed 3 of 15" cannot be computed from stdout alone.

There is deliberately no gym-style `step()` interface. Episodes are single-turn (ADR-0001),
so a state-machine interface would be ceremony around a loop that is never called twice.
