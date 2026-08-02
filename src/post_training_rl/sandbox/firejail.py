"""The training sandbox: firejail behind an external wall-clock timer, per ADR-0005 as
amended by ADR-0014.

A blast-radius limiter, not a security boundary against a determined adversary. The threat
model is deliberate — the code being run is our own model's failed attempts, not an
attacker's payload.

The wall-clock limit is enforced by `timeout(1)` **outside** firejail rather than by
firejail's own `--timeout`, which was measured at a flat ~2s per execution regardless of its
value. ADR-0014 records the measurement and the reasoning. Three layers still stand between
a hung solution and the training run, which is more than the original design had:

    timeout(1) SIGTERM  ->  timeout(1) SIGKILL after the grace  ->  our killpg backstop

Two hazards firejail does **not** cover are handled here rather than by flags:

- `--rlimit-fsize` does not apply to pipes and Linux pipe capacity is 64 KiB, so a runaway
  print loop would exhaust memory in the *training process*. Stdout is therefore read
  through a bounded reader that stops at `stdout_cap_bytes` and kills the child (ADR-0009).
- `Popen.communicate(timeout=)` does not kill the child on expiry, so the final backstop is
  an explicit `killpg` on the child's own session.

**Recorded deviation from sprint-01.md task 5.** That section says "Use
`subprocess.run(..., timeout=)`", and this module uses `Popen`. The two constraints in that
same section cannot both be met by `subprocess.run`: it buffers the child's output without
bound, so it cannot "read at most `stdout_cap_bytes`, then kill the child", and behavior.md
§4.7 requires that the truncation "never exhausts memory in the calling process". The stated
*reason* for the `subprocess.run` rule is that `Popen.communicate(timeout=)` "does not kill
the child on expiry" — the explicit `killpg` below honours that reason and is stricter.
"""

import os
import selectors
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from post_training_rl.config import SandboxConfig
from post_training_rl.types import SandboxResult

FIREJAIL_BINARY = "firejail"

# GNU coreutils. Enforces the wall-clock limit from outside the sandbox (ADR-0014).
TIMEOUT_BINARY = "timeout"

_BYTES_PER_MIB = 1024**2
_PROGRAM_FILENAME = "solution.py"
_STDIN_FILENAME = "stdin.txt"
_STDERR_FILENAME = "stderr.txt"

# The interpreter is the *system* python3, not `sys.executable`. Under `--private` the home
# directory is a fresh tmpfs, so a conda interpreter living under $HOME is unreachable
# inside the sandbox.
_PYTHON_INTERPRETER = "python3"

# The child gets a minimal environment rather than the parent's, so a solution cannot reach
# the training process's PYTHONPATH, credentials, or caches.
_CHILD_PATH = "/usr/bin:/bin"

_READ_CHUNK_BYTES = 65536

# What GNU timeout(1) returns when the command exceeded its limit and SIGTERM was enough.
# When --kill-after fires it re-raises SIGKILL on itself instead, so the parent observes
# signal death rather than an exit code — measured as returncode -9, not 137.
_TIMEOUT_EXIT_CODE = 124
_TIMEOUT_SIGNALS = frozenset({-int(signal.SIGKILL), -int(signal.SIGTERM)})


class FirejailSandbox:
    """Runs one program against one input inside firejail.

    Raises on infrastructure failure only. A program that crashes, hangs, or floods output
    is a normal result.
    """

    def __init__(
        self, config: SandboxConfig, hash_seed: int, binary: str = FIREJAIL_BINARY
    ) -> None:
        for required in (TIMEOUT_BINARY, binary):
            if shutil.which(required) is None:
                # Never silently downgrade to a weaker backend. The backend is an explicit
                # config value with no default, and a quietly degraded sandbox is
                # indistinguishable from a working one until something escapes (ADR-0005).
                raise FileNotFoundError(
                    f"sandbox backend requires the {required!r} binary, which is not on "
                    f"PATH. Install it with: sudo add-apt-repository ppa:deki/firejail && "
                    f"sudo apt install {required}"
                )
        self._config = config
        self._hash_seed = hash_seed
        self._binary = binary

    def run(self, source: str, stdin_text: str, timeout_seconds: float) -> SandboxResult:
        """Execute `source` as a Python program with `stdin_text` on stdin."""
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / _PROGRAM_FILENAME).write_text(source)
            (workspace / _STDIN_FILENAME).write_text(stdin_text)
            return self._execute(workspace, timeout_seconds)

    def _execute(self, workspace: Path, timeout_seconds: float) -> SandboxResult:
        deadline_seconds = (
            timeout_seconds
            + self._config.kill_after_seconds
            + self._config.backstop_slack_seconds
        )
        stderr_path = workspace / _STDERR_FILENAME

        started_at = time.monotonic()
        with (
            (workspace / _STDIN_FILENAME).open("rb") as stdin_file,
            stderr_path.open("wb") as stderr_file,
        ):
            process = subprocess.Popen(
                self._command(workspace, timeout_seconds),
                stdin=stdin_file,
                stdout=subprocess.PIPE,
                stderr=stderr_file,
                env=self._child_environment(),
                cwd=workspace,
                # Its own session, so the whole tree can be killed rather than just the
                # timeout process that started it.
                start_new_session=True,
            )
            try:
                stdout, was_truncated, deadline_fired = self._read_capped(
                    process, started_at, deadline_seconds
                )
            except BaseException:
                # Never leave a sandboxed process running because the parent failed. This
                # covers os.read errors and KeyboardInterrupt alike.
                self._reap(process, force=True)
                raise
            self._reap(process, force=was_truncated or deadline_fired)

        killed_by_parent = was_truncated or deadline_fired
        return SandboxResult(
            stdout=stdout.decode(errors="replace"),
            stderr=self._stderr_excerpt(stderr_path),
            exit_code=self._exit_code(process.returncode, killed_by_parent),
            duration_seconds=time.monotonic() - started_at,
            timed_out=self._timed_out(
                process.returncode, deadline_fired, killed_by_parent
            ),
            stdout_was_truncated=was_truncated,
        )

    def _timed_out(
        self, returncode: int, deadline_fired: bool, killed_by_parent: bool
    ) -> bool:
        """Whether the program was killed for running too long, by either timer.

        Signal death is the discriminator, not elapsed time. An earlier version inferred
        this from the clock and was wrong: firejail's reap overhead is billed to the
        program, so a solution that finished successfully well inside its limit still shows
        elapsed time at the limit. Reporting that as a timeout collapses "hung" into
        "crashed" — and worse, the verifier abandons every remaining test on a timeout, so
        one slow-but-correct solution would lose its whole suite.

        A program that merely crashes exits with a positive status, never a signal.
        """
        if deadline_fired:
            return True
        if killed_by_parent:
            return False  # we killed it for flooding output, which is not a timeout
        return returncode == _TIMEOUT_EXIT_CODE or returncode in _TIMEOUT_SIGNALS

    def _exit_code(self, returncode: int, killed_by_parent: bool) -> int | None:
        """None when the program was killed by a signal rather than exiting on its own.

        Kept distinct from `timed_out` so the verifier can map a signal death to
        RUNTIME_ERROR — "hung" and "crashed" must never become one outcome.
        """
        if killed_by_parent or returncode < 0:
            return None
        return returncode

    def _read_capped(
        self, process: subprocess.Popen, started_at: float, deadline_seconds: float
    ) -> tuple[bytes, bool, bool]:
        """Read stdout up to the cap, stopping early when the cap or deadline is hit.

        Returns the bytes read, whether the cap truncated them, and whether the deadline
        fired. Only stdout is a pipe — stdin and stderr are files — so there is no second
        stream to drain and this loop cannot deadlock on a full pipe.
        """
        cap = self._config.stdout_cap_bytes
        chunks: list[bytes] = []
        total = 0
        deadline_fired = False

        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        try:
            # One byte past the cap, so output landing exactly on the cap is reported as
            # complete rather than truncated.
            while total <= cap:
                remaining = deadline_seconds - (time.monotonic() - started_at)
                if remaining <= 0:
                    deadline_fired = True
                    break
                if not selector.select(timeout=remaining):
                    deadline_fired = True
                    break
                wanted = min(_READ_CHUNK_BYTES, cap + 1 - total)
                chunk = os.read(process.stdout.fileno(), wanted)
                if not chunk:
                    break  # the child closed stdout
                chunks.append(chunk)
                total += len(chunk)
        finally:
            selector.close()

        return b"".join(chunks)[:cap], total > cap, deadline_fired

    def _reap(self, process: subprocess.Popen, force: bool) -> None:
        """Ensure the process tree is dead and collected, without blocking indefinitely."""
        if force:
            self._kill_group(process)
        if process.stdout is not None:
            process.stdout.close()
        try:
            process.wait(timeout=self._config.reap_timeout_seconds)
        except subprocess.TimeoutExpired:
            # The child closed stdout but the tree has not exited — the wedge case the
            # backstop exists for. An unbounded wait() here would hang the training run.
            self._kill_group(process)
            process.wait()

    def _kill_group(self, process: subprocess.Popen) -> None:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass  # already gone, or not ours to kill — either way, nothing left to do

    def _command(self, workspace: Path, timeout_seconds: float) -> list[str]:
        config = self._config
        return [
            TIMEOUT_BINARY,
            f"--kill-after={self._config.kill_after_seconds}",
            str(timeout_seconds),
            self._binary,
            "--quiet",
            "--private",  # fresh tmpfs home
            "--noprofile",
            "--seccomp=socket",  # blocks network syscalls
            f"--rlimit-nproc={config.max_processes}",
            f"--rlimit-nofile={config.max_open_files}",
            f"--rlimit-fsize={config.max_file_size_mib * _BYTES_PER_MIB}",
            f"--rlimit-as={config.memory_limit_gib}g",
            f"--whitelist={workspace}",
            _PYTHON_INTERPRETER,
            str(workspace / _PROGRAM_FILENAME),
        ]

    def _child_environment(self) -> dict[str, str]:
        # Firejail preserves the environment it is given. Fixing PYTHONHASHSEED keeps `set`
        # and `dict` iteration order stable across runs, so two identical rollouts in one
        # group cannot score differently and be read as advantage (ADR-0008).
        return {"PYTHONHASHSEED": str(self._hash_seed), "PATH": _CHILD_PATH}

    def _stderr_excerpt(self, path: Path) -> str:
        with path.open("rb") as handle:
            return handle.read(self._config.stderr_excerpt_bytes).decode(errors="replace")
