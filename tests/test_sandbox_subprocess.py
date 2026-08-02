"""SubprocessSandbox behaviour, per `docs/design/behavior.md` §4 items 1-3, 9, 11.

These four names are reused verbatim in `test_sandbox_firejail.py`. They live in separate
modules because two same-named tests in one module means the second silently shadows the
first, and a test that quietly stops existing is worse than one that fails.

**No hostile-containment tests here.** `SubprocessSandbox` is documented as the weaker
adapter (ADR-0005); testing it as a security boundary would imply a guarantee it does not
make.
"""

from pathlib import Path

import pytest

from post_training_rl.config import load_verifier_config
from post_training_rl.sandbox.subprocess_ import SubprocessSandbox

# Tests pass a short timeout explicitly. One test at the production 10.0 would dominate the
# suite it lives in. 1.0 s is the floor, because firejail's --timeout has one-second
# granularity and the firejail module reuses these same test names.
TEST_TIMEOUT_SECONDS = 1.0

# Generous, because this asserts "the timeout fired" rather than "the timeout was prompt".
TIMEOUT_SLACK_SECONDS = 5.0

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "verifier.yaml"


def _sandbox() -> SubprocessSandbox:
    config = load_verifier_config(_CONFIG_PATH)
    return SubprocessSandbox(config.sandbox, hash_seed=0)


@pytest.mark.subprocess_backend
def test_program_output_is_captured():
    result = _sandbox().run("print('hello')", "", TEST_TIMEOUT_SECONDS)
    assert result.stdout.strip() == "hello"
    assert result.exit_code == 0
    assert result.timed_out is False


@pytest.mark.subprocess_backend
def test_stdin_is_delivered():
    program = "import sys\nprint(sys.stdin.read().strip())"
    result = _sandbox().run(program, "ping", TEST_TIMEOUT_SECONDS)
    assert result.stdout.strip() == "ping"


@pytest.mark.subprocess_backend
def test_infinite_loop_times_out():
    result = _sandbox().run("while True:\n    pass", "", TEST_TIMEOUT_SECONDS)
    assert result.timed_out is True
    assert result.duration_seconds < TEST_TIMEOUT_SECONDS + TIMEOUT_SLACK_SECONDS


@pytest.mark.subprocess_backend
def test_hash_seed_is_deterministic():
    # Two separate runs, so this pins the child environment rather than one process's luck.
    # Without a fixed PYTHONHASHSEED, str hashing is salted per process and set/dict
    # iteration order moves — two identical rollouts in one group would then score
    # differently and GRPO would read that difference as advantage (ADR-0008).
    sandbox = _sandbox()
    program = "print(hash('x'))"
    first = sandbox.run(program, "", TEST_TIMEOUT_SECONDS)
    second = sandbox.run(program, "", TEST_TIMEOUT_SECONDS)
    assert first.stdout.strip() == second.stdout.strip()
    assert first.stdout.strip() != ""
