# Enforce the wall-clock limit with timeout(1) outside firejail, not firejail's --timeout

The execution limit is applied by GNU `timeout(1)` wrapping the firejail invocation:

```
timeout --kill-after=1.0 <timeout_seconds> firejail --quiet --private --noprofile \
        --seccomp=socket --rlimit-nproc --rlimit-nofile --rlimit-fsize --rlimit-as \
        --whitelist=<tmpdir> python3 <tmpdir>/solution.py
```

**This supersedes the `--timeout` clause of [ADR-0005](0005-firejail-sandbox.md).** Every
other flag in that decision is unchanged, and [ADR-0006](0006-flat-timeout-ignore-dataset-limit.md)
is untouched — the limit is still flat, still 10 s, and the dataset's `time_limit` is still
never read. Only the mechanism that enforces it has moved.

## Why

Firejail's `--timeout` costs a flat ~2 s per execution, and the cost is independent of the
value. Measured on firejail 0.9.74, running a program that completes in 0.02 s:

| Invocation | Elapsed |
| --- | --- |
| `--timeout=00:00:01` | 1.12 s, exit 1 — **killed despite having finished** |
| `--timeout=00:00:03` | 2.03 s |
| `--timeout=00:00:05` | 2.02 s |
| `--timeout=00:00:10` | 2.02 s |
| `--timeout=00:01:00` | 2.02 s |
| no `--timeout` | 0.02 s |

It polls rather than waking on child exit, so the floor is the poll interval and no choice of
timeout value escapes it. At 17 executions per rollout, group size 8, and 12 worker threads
that is roughly 23 s of dead time per optimizer step — about three hours across a 500-step
run, spent waiting on a poll loop.

The flag was also **actively wrong at short limits**: at `--timeout=00:00:01` a program that
had already finished successfully was killed and reported as exit 1.

Second, and worse than the cost: firejail's reap overhead is billed to the program, and
firejail returns exit 1 both when its own timeout fires and when the program raises an
uncaught exception. Any timeout detection built on those signals misclassifies a
slow-but-correct solution as a timeout — and because the verifier abandons every remaining
test on a timeout ([ADR-0006](0006-flat-timeout-ignore-dataset-limit.md)), one such solution
loses its entire test suite. `timeout(1)` kills by signal, and a program that merely crashes
exits with a positive status, so the two never collapse.

## Considered Options

**A Python-side deadline alone** was rejected for being a single layer: the verifier fans
out over a thread pool, and a starved or wedged worker leaves nothing to kill the child.

**`RLIMIT_CPU`** was already rejected by ADR-0005 — a `SIGXCPU` handler can defang it, and
CPU time is not wall-clock time.

**Lowering `timeout_seconds`** does not help. The overhead is the poll interval, not the
value, and a lower limit would only fail more slow-but-correct solutions.

## Consequences

Three layers still stand between a hung solution and the training run, which is one more
than the original design had: `timeout(1)` SIGTERM, then its `--kill-after` SIGKILL, then
the sandbox's own `killpg` backstop on the child's session.

`timeout(1)` (GNU coreutils) becomes a system dependency alongside firejail. It is present
on every mainstream Linux, and its absence is checked at construction like firejail's.

Measured after the change: **0.029 s per execution at the production `timeout_seconds: 10.0`,
against 2.03 s before — 71× faster.** The containment suite dropped from 17.4 s to 2.3 s.
Twenty consecutive executions and five killed infinite loops leaked zero firejail and zero
python3 processes.

`timeout(1)` accepts fractional seconds, so the one-second granularity that forced the
containment tests up to a 3 s limit no longer applies and they run at the 1.0 s floor the
sprint plan specifies.

The residual ambiguity is narrow and documented: a solution killed by something other than
our timers — the OOM killer, say — also dies by signal and would be recorded as a timeout.
`--rlimit-as` makes the child raise `MemoryError` rather than be killed, so this requires
system-wide pressure to occur at all.
