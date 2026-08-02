"""Verifier behaviour, per `docs/design/behavior.md` §5 and ADR-0004/0006/0008.

Every test here drives `verify_batch` through a `FakeSandbox`. No subprocess is spawned by
this file, because the interface is the test surface and the seam exists precisely so this
suite stays sub-second and deterministic.
"""

from pathlib import Path

from post_training_rl.config import load_verifier_config
from post_training_rl.sandbox.fake import FakeSandbox
from post_training_rl.types import Problem, SandboxResult, TestCase, TestOutcome, TestPool
from post_training_rl.verifier import Verifier, build_preamble

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "verifier.yaml"
_CONFIG = load_verifier_config(_CONFIG_PATH)

SOLUTION = "```python\nprint(1)\n```"


def _case(expected: str = "1", pool: TestPool = TestPool.PRIVATE) -> TestCase:
    return TestCase(input_text="in", expected_output=expected, pool=pool)


def _problem(
    graded: int = 1, public: int = 0, expected: str = "1", problem_id: str = "p1"
) -> Problem:
    return Problem(
        problem_id=problem_id,
        description="d",
        graded_tests=tuple(_case(expected) for _ in range(graded)),
        public_tests=tuple(_case(expected, TestPool.PUBLIC) for _ in range(public)),
    )


def _ran(stdout: str = "1", exit_code: int = 0, truncated: bool = False) -> SandboxResult:
    return SandboxResult(
        stdout=stdout,
        stderr="",
        exit_code=exit_code,
        duration_seconds=0.01,
        timed_out=False,
        stdout_was_truncated=truncated,
    )


def _hung() -> SandboxResult:
    return SandboxResult(
        stdout="",
        stderr="",
        exit_code=None,
        duration_seconds=1.0,
        timed_out=True,
        stdout_was_truncated=False,
    )


def _outcomes(results) -> list[TestOutcome]:
    return [result.outcome for result in results]


def test_no_extractable_code_skips_execution_entirely():
    # The most common outcome early in training, and it must cost nothing.
    sandbox = FakeSandbox()
    verifier = Verifier(sandbox, _CONFIG)

    [report] = verifier.verify_batch([("I would use a greedy algorithm.", _problem(5))])

    assert sandbox.calls == []
    assert report.graded_results == ()
    assert report.extraction.code is None


def test_all_passing_tests_report_passed():
    sandbox = FakeSandbox([_ran("1"), _ran("1"), _ran("1")])
    verifier = Verifier(sandbox, _CONFIG)

    [report] = verifier.verify_batch([(SOLUTION, _problem(graded=3))])

    assert _outcomes(report.graded_results) == [TestOutcome.PASSED] * 3


def test_mismatched_output_reports_wrong_output():
    sandbox = FakeSandbox([_ran("2")])
    verifier = Verifier(sandbox, _CONFIG)

    [report] = verifier.verify_batch([(SOLUTION, _problem(graded=1, expected="1"))])

    assert _outcomes(report.graded_results) == [TestOutcome.WRONG_OUTPUT]


def test_nonzero_exit_reports_runtime_error():
    sandbox = FakeSandbox([_ran("", exit_code=1)])
    verifier = Verifier(sandbox, _CONFIG)

    [report] = verifier.verify_batch([(SOLUTION, _problem(graded=1))])

    assert _outcomes(report.graded_results) == [TestOutcome.RUNTIME_ERROR]


def test_timeout_aborts_remaining_tests():
    # A solution that exceeds the limit on one input almost always exceeds it on the rest,
    # and timeouts are the most expensive executions in the system (ADR-0006).
    sandbox = FakeSandbox([_ran("1"), _hung()])
    verifier = Verifier(sandbox, _CONFIG)

    [report] = verifier.verify_batch([(SOLUTION, _problem(graded=5))])

    assert _outcomes(report.graded_results) == [
        TestOutcome.PASSED,
        TestOutcome.TIMEOUT,
        TestOutcome.SKIPPED_AFTER_TIMEOUT,
        TestOutcome.SKIPPED_AFTER_TIMEOUT,
        TestOutcome.SKIPPED_AFTER_TIMEOUT,
    ]
    assert len(sandbox.calls) == 2  # tests 3-5 were never executed


def test_result_count_always_matches_test_count():
    # Recorded rather than omitted, so no reward function ever has to reason about why a
    # rollout has fewer results than the problem has tests.
    sandbox = FakeSandbox([_hung()])
    verifier = Verifier(sandbox, _CONFIG)
    problem = _problem(graded=5)

    [report] = verifier.verify_batch([(SOLUTION, problem)])

    assert len(report.graded_results) == len(problem.graded_tests)


def test_truncated_stdout_is_still_compared():
    # Compared, not auto-failed. Because the comparator requires matching token counts,
    # truncation almost always fails anyway — but through the normal path.
    #
    # `exit_code=None` is the point. When the reader hits the cap it kills the program, so
    # the status that comes back is the sandbox's kill, not the program's own exit. Scripting
    # exit_code=0 here would test a state FirejailSandbox can never produce, and the auto-fail
    # bug this pins would go unnoticed.
    sandbox = FakeSandbox([_ran("1", exit_code=None, truncated=True)])
    verifier = Verifier(sandbox, _CONFIG)

    [report] = verifier.verify_batch([(SOLUTION, _problem(graded=1, expected="1"))])

    [result] = report.graded_results
    assert result.outcome is TestOutcome.PASSED
    assert result.stdout_was_truncated is True


def test_public_tests_are_reported_separately():
    sandbox = FakeSandbox([_ran("1"), _ran("1"), _ran("1")])
    verifier = Verifier(sandbox, _CONFIG)

    [report] = verifier.verify_batch([(SOLUTION, _problem(graded=2, public=1))])

    assert len(report.graded_results) == 2
    assert len(report.public_results) == 1
    assert report.public_results[0].pool is TestPool.PUBLIC
    assert all(r.pool is TestPool.PRIVATE for r in report.graded_results)


def test_determinism_preamble_is_prepended():
    sandbox = FakeSandbox([_ran("1")])
    verifier = Verifier(sandbox, _CONFIG)

    verifier.verify_batch([(SOLUTION, _problem(graded=1))])

    [call] = sandbox.calls
    assert call.source.startswith(build_preamble(_CONFIG.determinism.seed).source)
    assert call.source.endswith("print(1)")


def test_preamble_seeds_random_and_hash():
    # The verifier's half of ADR-0008. The sandbox owns the other half — PYTHONHASHSEED in
    # the child environment — and neither may assume the other does it: if both seed, the
    # preamble appears twice; if neither does, determinism vanishes with no symptom except
    # reward noise inside groups.
    preamble = build_preamble(_CONFIG.determinism.seed)

    assert f"random.seed({_CONFIG.determinism.seed})" in preamble.source
    assert f"numpy.random.seed({_CONFIG.determinism.seed})" in preamble.source
    assert "except ImportError" in preamble.source  # numpy only when importable


def test_preamble_line_offset_is_recorded():
    # The offset that maps a traceback line number back to the model's own source. Without
    # it, every reported line number is wrong by the length of the preamble.
    preamble = build_preamble(seed=0)
    combined = preamble.source + "raise ValueError('boom')\n"

    # The assertion that can actually fail: the model's first line must land exactly
    # line_offset lines down. Restating the formula the implementation uses would pin
    # nothing.
    assert combined.splitlines()[preamble.line_offset] == "raise ValueError('boom')"
    assert preamble.line_offset > 0


def test_batch_preserves_input_order():
    sandbox = FakeSandbox([_ran("1") for _ in range(3)])
    verifier = Verifier(sandbox, _CONFIG)
    items = [(SOLUTION, _problem(problem_id=f"p{i}")) for i in range(3)]

    reports = verifier.verify_batch(items)

    assert [report.problem_id for report in reports] == ["p0", "p1", "p2"]


def test_infrastructure_error_propagates():
    # A missing sandbox binary or an unwritable temp directory is systematic: every
    # subsequent reward would be meaningless, so it must not be swallowed into a TestOutcome.
    class RaisingSandbox:
        def run(self, source: str, stdin_text: str, timeout_seconds: float):
            raise FileNotFoundError("firejail binary is missing")

    verifier = Verifier(RaisingSandbox(), _CONFIG)

    try:
        verifier.verify_batch([(SOLUTION, _problem(graded=1))])
    except FileNotFoundError as raised:
        assert "firejail" in str(raised)
    else:
        raise AssertionError("infrastructure failure was swallowed into a TestOutcome")
