"""The one real seam in the verifier.

A sandbox accepts program **source text** — not a path — delivers stdin, captures stdout and
stderr, and reports what happened. Callers never learn that temporary files are involved;
that is the depth this seam buys.

Three adapters implement it: `FirejailSandbox` for training, `SubprocessSandbox` for CI and
machines without firejail, and `FakeSandbox` for tests. Two of those are genuine production
alternatives, which is what earns a seam here when extraction and comparison — each with one
implementation — are plain functions instead.
"""

from typing import Protocol

from post_training_rl.types import SandboxResult


class Sandbox(Protocol):
    def run(self, source: str, stdin_text: str, timeout_seconds: float) -> SandboxResult:
        """Execute `source` as a Python program with `stdin_text` on stdin.

        Raises only on infrastructure failure (sandbox binary missing, temp dir
        unwritable). A program that crashes, hangs, or floods output is a normal result,
        not an exception.

        Implementations **must not modify the source they are given.** Seeding the program
        is the verifier's half of ADR-0008; the sandbox's contract is "run exactly this",
        and the startup self-test depends on it — it runs hostile programs through this same
        seam and must get them unmodified.
        """
