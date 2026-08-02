"""Startup self-test behaviour, per `docs/design/behavior.md` §9 item 3.

It exists because a misconfigured sandbox is otherwise indistinguishable from the model
producing wrong answers — the expensive kind of bug, discovered hours into a run.
"""

import pytest

from post_training_rl.sandbox.fake import FakeSandbox
from post_training_rl.startup import verify_sandbox_or_raise
from post_training_rl.types import SandboxResult


def _contained(
    stdout: str = "",
    exit_code: int | None = None,
    timed_out: bool = True,
    truncated: bool = True,
) -> SandboxResult:
    return SandboxResult(
        stdout=stdout,
        stderr="",
        exit_code=exit_code,
        duration_seconds=1.0,
        timed_out=timed_out,
        stdout_was_truncated=truncated,
    )


def _escaped() -> SandboxResult:
    """What a sandbox that contained nothing would return: the program ran to completion."""
    return SandboxResult(
        stdout="connected",
        stderr="",
        exit_code=0,
        duration_seconds=0.01,
        timed_out=False,
        stdout_was_truncated=False,
    )


def test_self_test_passes_when_all_programs_contained():
    sandbox = FakeSandbox([_contained() for _ in range(4)])

    verify_sandbox_or_raise(sandbox)  # returns, and does not raise

    assert len(sandbox.calls) == 4


def test_self_test_raises_naming_the_uncontained_program():
    # The fork bomb runs to completion — a sandbox with no process cap.
    sandbox = FakeSandbox([_contained(), _escaped(), _contained(), _contained()])

    with pytest.raises(RuntimeError) as raised:
        verify_sandbox_or_raise(sandbox)

    assert "fork bomb" in str(raised.value)
