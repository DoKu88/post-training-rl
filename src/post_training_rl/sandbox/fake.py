"""A sandbox that returns scripted results and records every call.

This is a test fixture, and it lives in `src/` rather than `tests/` because it has two
consumers: the verifier's unit tests and the startup self-test. It is the reason the
`Sandbox` seam pays for itself — verifier tests need no firejail, no temp files, and no
wall-clock waiting.

It records every call so a test can assert the sandbox was *not* invoked, which is how the
"no extractable code costs nothing" guarantee gets pinned.

**Scripted results are validated against what a real adapter can actually produce.** This is
not defensiveness — it is the fix for a defect that survived an entire sprint. A test scripted
`exit_code=0` together with `stdout_was_truncated=True`, which `FirejailSandbox` can never
return (truncation means the reader *killed* the program, so the status is `None`), and the
test stayed green while the production path auto-failed every truncated rollout. See
`sprint-01-status.md` §5.6. A fake that accepts impossible states tests a system that does not
exist.
"""

import threading
from collections.abc import Sequence
from dataclasses import dataclass

from post_training_rl.types import SandboxResult


@dataclass(frozen=True)
class SandboxCall:
    source: str
    stdin_text: str
    timeout_seconds: float


class FakeSandbox:
    def __init__(self, results: Sequence[SandboxResult] = ()) -> None:
        for index, result in enumerate(results):
            _reject_unproducible(result, index)
        self._results = list(results)
        self.calls: list[SandboxCall] = []
        # `verify_batch` fans out over a thread pool, so recording a call and taking the
        # next scripted result must happen together or the two lists drift apart under
        # concurrency — and the check-then-pop below is otherwise a race.
        self._lock = threading.Lock()

    def run(self, source: str, stdin_text: str, timeout_seconds: float) -> SandboxResult:
        with self._lock:
            self.calls.append(SandboxCall(source, stdin_text, timeout_seconds))
            if not self._results:
                # A test that scripted too few results has a bug in the test, not in the
                # code under test, and it should say so rather than invent a passing result.
                raise AssertionError(
                    f"FakeSandbox ran out of scripted results on call {len(self.calls)}"
                )
            return self._results.pop(0)


def _reject_unproducible(result: SandboxResult, index: int) -> None:
    """Refuse a scripted result that no real adapter could return.

    The three invariants below were derived from `FirejailSandbox` and confirmed by running
    it: a clean exit, a raise, an explicit exit code, an infinite loop, and an output flood.
    They hold for `SubprocessSandbox` too. Anything violating them describes a sandbox that
    does not exist, and a test written against it proves nothing about the one that does.
    """
    violations = []
    if result.stdout_was_truncated and result.exit_code is not None:
        violations.append(
            f"stdout_was_truncated=True with exit_code={result.exit_code!r}: hitting the cap "
            f"kills the program, so the status is the sandbox's kill, never an exit code"
        )
    if result.stdout_was_truncated and result.timed_out:
        violations.append(
            "stdout_was_truncated=True with timed_out=True: the reader stops at the cap or "
            "at the deadline, never both"
        )
    if result.timed_out and result.exit_code is not None:
        violations.append(
            f"timed_out=True with exit_code={result.exit_code!r}: a timed-out child is killed "
            f"by signal and has no exit code of its own"
        )
    if violations:
        raise AssertionError(
            f"FakeSandbox result {index} describes a state no real sandbox produces — "
            + "; ".join(violations)
        )
