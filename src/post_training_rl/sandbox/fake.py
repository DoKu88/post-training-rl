"""A sandbox that returns scripted results and records every call.

This is a test fixture, and it lives in `src/` rather than `tests/` because it has two
consumers: the verifier's unit tests and the startup self-test. It is the reason the
`Sandbox` seam pays for itself — verifier tests need no firejail, no temp files, and no
wall-clock waiting.

It records every call so a test can assert the sandbox was *not* invoked, which is how the
"no extractable code costs nothing" guarantee gets pinned.
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
