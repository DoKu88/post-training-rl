"""Known-hostile programs used to prove the sandbox actually contains anything.

Two call sites, which is why these live in `src/` rather than `tests/`: the containment
tests, and the startup self-test that runs before the first training step. Same fixtures,
one definition.

**Deviation from the sprint plan, recorded deliberately.** `docs/plans/sprint-01.md` task 5
lists this file at `tests/fixtures/hostile_programs.py`. It is here instead, because task 8
puts `verify_sandbox_or_raise` in `src/` and production code importing from `tests/` breaks
the moment the package is installed anywhere. The sprint makes exactly this argument itself
for `FakeSandbox` — "lives in `src/`, not `tests/`, because the verifier's tests and the
startup self-test both use it" — and the same reasoning applies here unchanged.
"""

# Must be killed by the wall-clock timeout rather than running forever.
INFINITE_LOOP = "while True:\n    pass\n"

# Must be capped by --rlimit-nproc, and must not affect the host.
FORK_BOMB = "import os\n\nwhile True:\n    os.fork()\n"

# Must fail under --seccomp=socket. Uses a routable address so that a *successful* connect
# is unambiguous — if this returns cleanly, the network was genuinely reachable.
NETWORK_CONNECTION = (
    "import socket\n\n"
    "socket.create_connection(('1.1.1.1', 80), timeout=3)\n"
    "print('connected')\n"
)

# Must be truncated by the reader in the parent. Large writes rather than many small ones,
# so the cap is reached quickly enough to be observed before the timeout fires. Pipe
# capacity is 64 KiB and --rlimit-fsize does not apply to pipes, so nothing but the reader
# stands between this and the training process's memory (ADR-0009).
OUTPUT_FLOOD = "import sys\n\nwhile True:\n    sys.stdout.write('x' * 65536)\n"

# Must be capped by --rlimit-fsize.
FILE_WRITE_FLOOD = (
    "with open('flood.bin', 'wb') as handle:\n"
    "    while True:\n"
    "        handle.write(b'x' * 65536)\n"
)

# The four the startup self-test runs, per behavior.md §9 item 3 and verifier-scorer.md §3.
# Named so a failure says which containment property is missing rather than which index.
SELF_TEST_PROGRAMS = {
    "infinite loop": INFINITE_LOOP,
    "fork bomb": FORK_BOMB,
    "network connection": NETWORK_CONNECTION,
    "output flood": OUTPUT_FLOOD,
}
