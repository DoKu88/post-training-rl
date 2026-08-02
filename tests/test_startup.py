"""Startup self-test behaviour, per `docs/design/behavior.md` §9 item 3.

It exists because a misconfigured sandbox is otherwise indistinguishable from the model
producing wrong answers — the expensive kind of bug, discovered hours into a run.
"""

from pathlib import Path

import pytest

from post_training_rl.config import load_verifier_config
from post_training_rl.sandbox.fake import FakeSandbox
from post_training_rl.startup import verify_sandbox_or_raise
from post_training_rl.types import SandboxResult

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "verifier.yaml"
SELF_TEST_TIMEOUT = load_verifier_config(_CONFIG_PATH).startup.self_test_timeout_seconds


def _result(
    stdout: str = "",
    exit_code: int | None = 1,
    timed_out: bool = False,
    truncated: bool = False,
) -> SandboxResult:
    return SandboxResult(
        stdout=stdout,
        stderr="",
        exit_code=exit_code,
        duration_seconds=0.05,
        timed_out=timed_out,
        stdout_was_truncated=truncated,
    )


# One per check, in `_CHECKS` order, each showing the mechanism that program is supposed to
# be stopped by. Only the infinite loop is stopped by the clock — scripting a timeout for
# the other three would pass a sandbox whose only working component is the wall-clock timer.
_CONTAINED = [
    _result(timed_out=True, exit_code=None),  # infinite loop: killed by the timer
    _result(exit_code=1),  # fork bomb: --rlimit-nproc makes fork() raise
    _result(exit_code=1),  # network: --seccomp=socket makes connect() raise
    _result(truncated=True, exit_code=None),  # flood: capped by the reader
]


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
    sandbox = FakeSandbox(_CONTAINED)

    verify_sandbox_or_raise(sandbox, SELF_TEST_TIMEOUT)  # returns, and does not raise

    assert len(sandbox.calls) == 4
    # The self-test must use its own short timeout, not the production one. Getting this
    # wrong would make startup wait the full execution limit on the infinite loop.
    assert all(call.timeout_seconds == SELF_TEST_TIMEOUT for call in sandbox.calls)


def test_self_test_raises_naming_the_uncontained_program():
    # The fork bomb runs to completion — a sandbox with no process cap. The message must say
    # which containment property is missing, not that something went wrong.
    scripted = list(_CONTAINED)
    scripted[1] = _escaped()
    sandbox = FakeSandbox(scripted)

    with pytest.raises(RuntimeError) as raised:
        verify_sandbox_or_raise(sandbox, SELF_TEST_TIMEOUT)

    assert "fork bomb" in str(raised.value)
