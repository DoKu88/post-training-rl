# Public tests are a diagnostic, never a reward input

Public tests are executed on every rollout, and their pass rate is logged every step — but no
reward function reads them. They exist to answer "is the model actually getting better at the
task?" independently of whatever reward is currently driving training.

The reward is a proxy for the task, and ADR-0011 makes the choice of proxy deliberately
interchangeable because nobody knows which one trains this model best. That creates a
measurement problem: a rising reward curve is consistent with both real improvement and a
reward optimising itself. Public pass rate does not move with the reward, so it distinguishes
the two — and it does so every step, where the full-suite evaluation only arrives every few
epochs.

## Consequences

Public tests add roughly 10–20% to per-rollout execution cost (typically 1–3 tests against up
to 15 graded ones). That is the price of the diagnostic and is accepted.

**Wiring public results into a reward function requires a new decision.** They are printed in
the problem statement and therefore visible to the model, so partial credit over them is
directly hackable — DeepCoder's stated failure mode is a model that "learns to directly print
out the answers of public tests." A `public_fractional` entry remains a legitimate future
registry addition, but it is deliberately unregistered today.

A reader who finds the verifier executing tests that no reward consumes should read this file
rather than assume it is dead code.
