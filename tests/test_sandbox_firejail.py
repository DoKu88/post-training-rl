"""Containment guarantees, per `docs/design/behavior.md` §4 items 4-8, 10, 12.

This is the only thing protecting the host from a training run's own failed attempts, so
when firejail is absent these **skip loudly** with an explicit message rather than passing.
A containment test that passes because it did not run is false assurance about the one
component whose failure is silent.

The first four names are shared verbatim with `test_sandbox_subprocess.py`. Separate modules,
because two same-named tests in one module means the second silently shadows the first.
"""

import shutil
from pathlib import Path

import pytest

from post_training_rl.config import load_verifier_config
from post_training_rl.sandbox import hostile_programs
from post_training_rl.sandbox.hostile_programs import NETWORK_SUCCESS_MARKER
from post_training_rl.sandbox.firejail import FIREJAIL_BINARY, FirejailSandbox

# The sprint's stated floor. It holds again under ADR-0014: the wall-clock limit is enforced
# by timeout(1), which takes fractional seconds and fires only at the deadline, so a program
# finishing in 0.02s costs 0.02s. Under firejail's own --timeout this had to be raised to
# 3.0, because that flag polls at ~2s intervals and killed a program that had already
# finished.
TEST_TIMEOUT_SECONDS = 1.0
# Tight on purpose. With timeout_seconds=1.0, kill_after=1.0 and backstop_slack=5.0 the
# parent-side killpg fires at 7s, so a slack of 10 would let this test pass even if
# timeout(1) never worked and only the backstop did. 3.0 asserts the OUTER timer fired.
TIMEOUT_SLACK_SECONDS = 3.0

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "verifier.yaml"

requires_firejail = pytest.mark.skipif(
    shutil.which(FIREJAIL_BINARY) is None,
    reason=(
        f"{FIREJAIL_BINARY!r} is not installed, so containment is UNVERIFIED on this "
        "machine. Install it with: sudo add-apt-repository ppa:deki/firejail && "
        "sudo apt install firejail"
    ),
)

pytestmark = pytest.mark.containment


def _sandbox() -> FirejailSandbox:
    config = load_verifier_config(_CONFIG_PATH)
    return FirejailSandbox(config.sandbox, hash_seed=0)


@requires_firejail
def test_program_output_is_captured():
    result = _sandbox().run("print('hello')", "", TEST_TIMEOUT_SECONDS)
    assert result.stdout.strip() == "hello"
    assert result.exit_code == 0
    assert result.timed_out is False


@requires_firejail
def test_stdin_is_delivered():
    program = "import sys\nprint(sys.stdin.read().strip())"
    result = _sandbox().run(program, "ping", TEST_TIMEOUT_SECONDS)
    assert result.stdout.strip() == "ping"


@requires_firejail
def test_infinite_loop_times_out():
    result = _sandbox().run(hostile_programs.INFINITE_LOOP, "", TEST_TIMEOUT_SECONDS)
    assert result.timed_out is True
    assert result.duration_seconds < TEST_TIMEOUT_SECONDS + TIMEOUT_SLACK_SECONDS


@requires_firejail
def test_fork_bomb_is_contained():
    # Contained means: this returns at all, within the timeout budget, and the host is still
    # usable afterwards. The following assertion is itself part of the test — if the fork
    # bomb had escaped, the test process would not get this far.
    result = _sandbox().run(hostile_programs.FORK_BOMB, "", TEST_TIMEOUT_SECONDS)
    assert result.duration_seconds < TEST_TIMEOUT_SECONDS + TIMEOUT_SLACK_SECONDS
    assert result.exit_code != 0 or result.timed_out


@requires_firejail
def test_network_access_is_blocked():
    result = _sandbox().run(hostile_programs.NETWORK_CONNECTION, "", TEST_TIMEOUT_SECONDS)
    assert NETWORK_SUCCESS_MARKER not in result.stdout


@requires_firejail
def test_output_flood_is_truncated():
    config = load_verifier_config(_CONFIG_PATH)
    result = _sandbox().run(hostile_programs.OUTPUT_FLOOD, "", TEST_TIMEOUT_SECONDS)
    assert result.stdout_was_truncated is True
    # The observable form of "the truncation never exhausts memory in the calling process":
    # the parent holds at most the cap, however much the child wrote.
    assert len(result.stdout) <= config.sandbox.stdout_cap_bytes


@requires_firejail
def test_file_write_beyond_limit_is_contained():
    result = _sandbox().run(hostile_programs.FILE_WRITE_FLOOD, "", TEST_TIMEOUT_SECONDS)
    assert result.exit_code != 0 or result.timed_out


@requires_firejail
def test_hash_seed_is_deterministic():
    sandbox = _sandbox()
    program = "print(hash('x'))"
    first = sandbox.run(program, "", TEST_TIMEOUT_SECONDS)
    second = sandbox.run(program, "", TEST_TIMEOUT_SECONDS)
    assert first.stdout.strip() == second.stdout.strip()
    assert first.stdout.strip() != ""


def test_missing_firejail_binary_raises_naming_it():
    # Deliberately not skipped when firejail is absent: this is the guard against a silent
    # downgrade, and it is exactly the machine without firejail that needs it to hold.
    config = load_verifier_config(_CONFIG_PATH)
    missing = "firejail-not-installed"
    with pytest.raises(FileNotFoundError) as raised:
        FirejailSandbox(config.sandbox, hash_seed=0, binary=missing)
    assert missing in str(raised.value)
