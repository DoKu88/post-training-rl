# Sandbox generated code with firejail, not bare rlimits

Every rollout's code is executed inside `firejail` — `--private` for a fresh tmpfs home,
`--seccomp=socket` to block network syscalls, `--rlimit-nproc=32` against fork bombs,
`--rlimit-fsize=2MiB` against file flooding, `--rlimit-as=4GiB`, and a wall-clock
`--timeout` — running under a dedicated UID. Limits are set by the sandbox binary rather
than a `preexec_fn`.

> **Amended by [ADR-0014](0014-external-wall-clock-timer.md).** The `--timeout` flag named
> above is no longer used: it costs a flat ~2 s per execution regardless of its value, and
> it cannot distinguish its own timeout from an uncaught exception. The wall-clock limit is
> now enforced by `timeout(1)` wrapping the firejail invocation. **Every other flag in this
> decision stands unchanged**, as does the flat 10 s limit of ADR-0006.

## Considered Options

The initial plan was a plain subprocess with `RLIMIT_*` set in the parent. That was
rejected on two documented grounds: `RLIMIT_CPU` can be defanged by a `SIGXCPU` handler that
returns control — and a kernel bug then *raises the soft limit by one second* — and rlimits
cannot provide network isolation or a private filesystem at all.

Docker was rejected on throughput: container startup in the hundreds of milliseconds, at
thousands of executions per optimizer step, would dominate wall-clock.

## Consequences

`firejail` becomes a system dependency (from `ppa:deki/firejail` on Ubuntu) and must be
verified at startup rather than failing mid-run.

This is a blast-radius limiter, not a security boundary against a determined adversary. The
threat model is deliberate: the code being run is our own model's failed attempts, not an
attacker's payload.

Two hazards firejail does **not** cover, handled in the verifier instead: pipe capacity is
64 KiB and `--rlimit-fsize` does not apply to pipes, so captured stdout needs its own cap
(ADR-0009); and `Popen.communicate(timeout=)` does not kill the child, so `subprocess.run`
is used instead.
