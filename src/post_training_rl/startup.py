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
    OUTPUT_FLOOD,
)
from post_training_rl.types import SandboxResult

# Short on purpose: the self-test proves containment, not throughput, and the infinite loop
# has to actually run out the clock. The production timeout would make startup wait ten
# seconds to learn something two proves.
_SELF_TEST_TIMEOUT_SECONDS = 2.0

_NETWORK_SUCCESS_MARKER = "connected"


@dataclass(frozen=True)
class _ContainmentCheck:
    name: str
    source: str
    is_contained: Callable[[SandboxResult], bool]


def _survived(result: SandboxResult) -> bool:
    """Whether the program ran to a clean finish — which for a hostile program is a failure."""
    return not result.timed_out and result.exit_code == 0


_CHECKS = (
    _ContainmentCheck("infinite loop", INFINITE_LOOP, lambda r: r.timed_out),
    _ContainmentCheck("fork bomb", FORK_BOMB, lambda r: not _survived(r)),
    _ContainmentCheck(
        "network connection",
        NETWORK_CONNECTION,
        lambda r: _NETWORK_SUCCESS_MARKER not in r.stdout,
    ),
    _ContainmentCheck(
        "output flood", OUTPUT_FLOOD, lambda r: r.stdout_was_truncated or r.timed_out
    ),
)


def verify_sandbox_or_raise(sandbox: Sandbox) -> None:
    """Run each known-hostile program through `sandbox` and raise if one is not contained.

    Raises `RuntimeError` naming the check that failed, so the message says which
    containment property is missing rather than that something went wrong.
    """
    for check in _CHECKS:
        result = sandbox.run(check.source, "", _SELF_TEST_TIMEOUT_SECONDS)
        if not check.is_contained(result):
            raise RuntimeError(
                f"sandbox self-test failed: the {check.name!r} program was not contained "
                f"(exit_code={result.exit_code}, timed_out={result.timed_out}, "
                f"stdout_was_truncated={result.stdout_was_truncated}). Refusing to start — "
                f"an uncontained sandbox looks exactly like a model producing wrong answers."
            )
