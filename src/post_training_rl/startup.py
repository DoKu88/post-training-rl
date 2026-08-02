"""Proves the configured sandbox actually contains anything, before the first training step.

A misconfigured sandbox is otherwise indistinguishable from the model producing wrong
answers, which is the expensive kind of bug — it looks like a training problem and is
discovered hours in. This costs a few seconds once per run and reuses the same hostile
programs as the containment tests.
"""

from collections.abc import Callable
from dataclasses import dataclass

from post_training_rl.sandbox import Sandbox
from post_training_rl.sandbox.hostile_programs import (
    FORK_BOMB,
    INFINITE_LOOP,
    NETWORK_CONNECTION,
    NETWORK_SUCCESS_MARKER,
    OUTPUT_FLOOD,
)
from post_training_rl.types import SandboxResult



@dataclass(frozen=True)
class _ContainmentCheck:
    name: str
    source: str
    is_contained: Callable[[SandboxResult], bool]


# Each program has its OWN containment mechanism, and each predicate insists on that
# mechanism rather than accepting a timeout as proof. Only the infinite loop is *supposed*
# to be stopped by the clock. If the others were allowed to pass by timing out, a sandbox
# with nothing working except the wall-clock timer would sail through this self-test — which
# is the exact failure it exists to catch.
_CHECKS = (
    # Stopped by the timer, by definition.
    _ContainmentCheck("infinite loop", INFINITE_LOOP, lambda r: r.timed_out),
    # Stopped by --rlimit-nproc: fork() raises, Python exits non-zero, and it does so
    # promptly. A fork bomb that merely runs out the clock was NOT capped.
    _ContainmentCheck(
        "fork bomb", FORK_BOMB, lambda r: not r.timed_out and r.exit_code != 0
    ),
    # Stopped by --seccomp=socket: the connect raises immediately. A run that times out
    # proves nothing — the connection may simply have been blocking.
    _ContainmentCheck(
        "network connection",
        NETWORK_CONNECTION,
        lambda r: not r.timed_out and NETWORK_SUCCESS_MARKER not in r.stdout,
    ),
    # Stopped by the reader's cap. Truncation is the guarantee; timing out is not.
    _ContainmentCheck("output flood", OUTPUT_FLOOD, lambda r: r.stdout_was_truncated),
)


def verify_sandbox_or_raise(sandbox: Sandbox, timeout_seconds: float) -> None:
    """Run each known-hostile program through `sandbox` and raise if one is not contained.

    `timeout_seconds` comes from `startup.self_test_timeout_seconds` in config. It is kept
    short on purpose: the self-test proves containment, not throughput, and the infinite
    loop has to run the clock out. It is passed as a value rather than a config object
    because that is the only field this function would read.

    Raises `RuntimeError` naming the check that failed, so the message says which
    containment property is missing rather than that something went wrong.
    """
    for check in _CHECKS:
        result = sandbox.run(check.source, "", timeout_seconds)
        if not check.is_contained(result):
            raise RuntimeError(
                f"sandbox self-test failed: the {check.name!r} program was not contained "
                f"(exit_code={result.exit_code}, timed_out={result.timed_out}, "
                f"stdout_was_truncated={result.stdout_was_truncated}). Refusing to start — "
                f"an uncontained sandbox looks exactly like a model producing wrong answers."
            )
