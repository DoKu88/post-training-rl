# Use a flat 10s timeout and ignore the dataset's time_limit

Every execution gets a flat 10-second wall-clock timeout. The `time_limit` field shipped with
each CodeContests problem is deliberately never read.

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
