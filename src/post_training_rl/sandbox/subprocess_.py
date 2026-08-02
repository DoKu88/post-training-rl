"""The weaker sandbox adapter: a plain subprocess with resource limits.

Exists for CI and for developing on a machine without firejail. It provides the functional
behaviour and resource limits, but **not** network isolation and **not** a private
filesystem, so `behavior.md` §4 explicitly does not hold it to the containment guarantees
(items 5-8). ADR-0005 rejected this as the training backend for exactly that reason.

Two consequences of being the weaker adapter, both deliberate:

- Output is truncated to the configured cap *after* the read rather than during it, so a
  runaway print loop still costs the parent that much memory transiently. `FirejailSandbox`
  caps during the read, which is what ADR-0009 requires of the training path.
- Limits are applied with `preexec_fn`, which runs between fork and exec and is documented
  as unsafe in a multi-threaded process. The verifier fans out over a thread pool, so this
  adapter carries a real hazard the firejail adapter does not — firejail sets its limits in
  the sandbox binary itself.
"""

import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from post_training_rl.config import SandboxConfig
from post_training_rl.types import SandboxResult

_BYTES_PER_GIB = 1024**3
_BYTES_PER_MIB = 1024**2

_PROGRAM_FILENAME = "solution.py"

# The child gets a minimal environment rather than the parent's, so a solution cannot reach
# the training process's PYTHONPATH, credentials, or caches. Not a containment guarantee —
# that is firejail's job — but there is no reason to hand them over either.
_CHILD_PATH = "/usr/bin:/bin"


class SubprocessSandbox:
    def __init__(self, config: SandboxConfig, hash_seed: int) -> None:
        self._config = config
        self._hash_seed = hash_seed

    def run(self, source: str, stdin_text: str, timeout_seconds: float) -> SandboxResult:
        """Execute `source` as a Python program with `stdin_text` on stdin."""
        with tempfile.TemporaryDirectory() as directory:
            program = Path(directory) / _PROGRAM_FILENAME
            program.write_text(source)
            return self._execute(program, directory, stdin_text, timeout_seconds)

    def _execute(
        self, program: Path, directory: str, stdin_text: str, timeout_seconds: float
    ) -> SandboxResult:
        started_at = time.monotonic()
        try:
            completed = subprocess.run(
                [sys.executable, str(program)],
                input=stdin_text,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout_seconds,
                env=self._child_environment(),
                cwd=directory,
                preexec_fn=self._apply_limits,  # noqa: PLW1509 — see module docstring
                check=False,
            )
        except subprocess.TimeoutExpired as expired:
            # subprocess.run kills the child on expiry; Popen.communicate(timeout=) does
            # not, which is why it is never used here (ADR-0005).
            return self._result(
                stdout=_as_text(expired.stdout),
                stderr=_as_text(expired.stderr),
                exit_code=None,
                started_at=started_at,
                timed_out=True,
            )
        return self._result(
            stdout=completed.stdout,
            stderr=completed.stderr,
            exit_code=completed.returncode,
            started_at=started_at,
            timed_out=False,
        )

    def _result(
        self,
        stdout: str,
        stderr: str,
        exit_code: int | None,
        started_at: float,
        timed_out: bool,
    ) -> SandboxResult:
        # Character count against a byte cap. The two agree for the ASCII that contest
        # output almost always is, and this adapter is not held to the truncation guarantee
        # anyway — FirejailSandbox caps real bytes during the read, which is the path that
        # matters (ADR-0009).
        cap = self._config.stdout_cap_bytes
        return SandboxResult(
            stdout=stdout[:cap],
            stderr=stderr[: self._config.stderr_excerpt_bytes],
            exit_code=exit_code,
            duration_seconds=time.monotonic() - started_at,
            timed_out=timed_out,
            stdout_was_truncated=len(stdout) > cap,
        )

    def _child_environment(self) -> dict[str, str]:
        # Fixed so that `set` and `dict` iteration order is stable across runs. Without it,
        # a hash-order-dependent solution scores differently between two rollouts in the
        # same group and GRPO reads that difference as advantage (ADR-0008).
        return {"PYTHONHASHSEED": str(self._hash_seed), "PATH": _CHILD_PATH}

    def _apply_limits(self) -> None:
        limits = (
            (resource.RLIMIT_AS, self._config.memory_limit_gib * _BYTES_PER_GIB),
            (resource.RLIMIT_NPROC, self._config.max_processes),
            (resource.RLIMIT_NOFILE, self._config.max_open_files),
            (resource.RLIMIT_FSIZE, self._config.max_file_size_mib * _BYTES_PER_MIB),
        )
        for limit, value in limits:
            resource.setrlimit(limit, (value, value))


def _as_text(captured: str | bytes | None) -> str:
    """Normalise what `TimeoutExpired` carries, which may be bytes or absent entirely."""
    if captured is None:
        return ""
    if isinstance(captured, bytes):
        return captured.decode(errors="replace")
    return captured
