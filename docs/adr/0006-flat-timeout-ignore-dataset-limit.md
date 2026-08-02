# Use a flat timeout and ignore the dataset's time_limit

Every execution gets a flat wall-clock timeout. The `time_limit` field shipped with
each CodeContests problem is deliberately never read.

> **The value is provisionally 2.0s, not the 10s below — sprint 1, pending measurement.**
> The decision recorded here is unchanged: the limit is flat, and `time_limit` is never read.
> Only the number moved, and it lives in `sandbox.timeout_seconds` in
> `config/verifier.yaml`.
>
> The 10s figure followed DeepMind's evaluator on the reasoning that Python is ~10× slower
> than the C++ the 1–6s contest limits were calibrated for. Measured on CPython 3.11 during
> sprint 1, that headroom is far larger than needed: every algorithmically sound solution
> tried — O(n log n) at n=10⁶, a segment tree over 10⁵ queries, a 2000×2000 DP — finished
> within **0.57s**, while genuinely quadratic solutions took 1.9s to 11s. 2.0 keeps ~3.5×
> margin over the slowest sound solution while placing the "too slow" boundary below the
> quadratic cases, and cuts a hung rollout from 22s to 6s.
>
> **This is deliberately not settled.** The timeout *rate* is unmeasured until a real run,
> so 2.0 is a reasonable value chosen to avoid over-optimising early. Sprint 3 should ablate
> 1 / 2 / 5 / 10 against measured timeout rates and pass@k. Full measurements in
> [`sprint-01-status.md` §3.3](../plans/sprint-01-status.md).

A reader will reasonably assume the dataset's own limits should be honoured, so the reasons
are recorded here. Those limits are 1–6 seconds and were calibrated for C++ contest
submissions; Python is roughly an order of magnitude slower, so enforcing them would fail
correct Python solutions in bulk and train the model against approaches that would pass on a
real judge. DeepMind's own released evaluator makes the same choice — it never reads
`time_limit` and uses a flat 10 s.

## Consequences

The ~16% of training rows where `time_limit` is null and `memory_limit_bytes` is zero stop
being a special case: there is no fallback to choose because the fields are never read.

Timeouts are the most expensive executions in the system, so a rollout's remaining tests are
abandoned once one test times out — a solution that exceeds the limit on one input almost
always exceeds it on the rest.
