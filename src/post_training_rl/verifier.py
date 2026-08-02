"""Executes a rollout's code against a problem's tests and reports what happened.

**It assigns no rewards and holds no opinion about what an outcome is worth.** That is
ADR-0004's split, and it is what lets one execution feed every reward function in the
registry instead of one execution per reward shape.

**Raises only on infrastructure failure.** A solution that crashes, hangs, floods output, or
produces no parseable code is *data* — it becomes a `TestOutcome` or an `Extraction` result,
never an exception. A missing sandbox binary or an unwritable temp directory raises, because
those are systematic and every subsequent reward would be meaningless.

This module also owns the seeding preamble, which is the verifier's half of ADR-0008: the
sandbox fixes the environment (`PYTHONHASHSEED`), the verifier seeds the program. Neither may
assume the other does it — if both do, the preamble appears twice; if neither does,
determinism vanishes with no symptom except reward noise inside groups.
"""

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from post_training_rl.comparator import outputs_match
from post_training_rl.config import ComparatorConfig, VerifierConfig
from post_training_rl.extraction import extract_python
from post_training_rl.sandbox import Sandbox
from post_training_rl.types import (
    Problem,
    SandboxResult,
    TestCase,
    TestOutcome,
    TestResult,
    VerificationReport,
)


@dataclass(frozen=True)
class Preamble:
    """The seeding prologue prepended to a solution, and how many lines it added.

    `line_offset` is what maps a traceback line number back to the model's own source. It is
    carried here rather than on `VerificationReport` because it is a property of the
    preamble, identical for every rollout in a run.
    """

    source: str
    line_offset: int


def build_preamble(seed: int | None) -> Preamble:
    """Build the determinism preamble for `seed`, or an empty one when seeding is off.

    Seeding does not break randomised algorithms — a randomised quicksort still sorts, it
    simply picks the same pivots every time. What it prevents is two rollouts with identical
    text scoring differently, which GRPO would read as advantage the model did not earn.
    """
    if seed is None:
        return Preamble(source="", line_offset=0)
    source = (
        "import random\n"
        f"random.seed({seed})\n"
        "try:\n"
        "    import numpy\n"
        f"    numpy.random.seed({seed})\n"
        "except ImportError:\n"
        "    pass\n"
    )
    return Preamble(source=source, line_offset=source.count("\n"))


class Verifier:
    def __init__(self, sandbox: Sandbox, config: VerifierConfig) -> None:
        self._sandbox = sandbox
        self._config = config
        self._preamble = build_preamble(config.determinism.seed)

    def verify_batch(
        self, items: Sequence[tuple[str, Problem]]
    ) -> list[VerificationReport]:
        """Verify each (completion, problem) pair, returning reports in input order.

        Threads rather than processes: the work is `subprocess.run`, which releases the GIL
        while waiting, so there is nothing to gain from pickling across processes.
        """
        if not items:
            return []
        with ThreadPoolExecutor(max_workers=self._config.sandbox.worker_threads) as pool:
            # Executor.map preserves input order and re-raises the first exception, which is
            # what makes an infrastructure failure propagate rather than become an outcome.
            return list(pool.map(lambda item: self._verify_one(*item), items))

    def _verify_one(self, completion: str, problem: Problem) -> VerificationReport:
        extraction = extract_python(completion, self._config.extraction.prefill)
        if extraction.code is None:
            # No sandbox invocation at all — the most common outcome early in training, and
            # it must cost nothing.
            #
            # The gate is `code is None`, deliberately not `not parsed`. Code that was
            # recovered but does not parse IS executed, and produces RUNTIME_ERROR on every
            # test. That distinction is load-bearing for `code_r1`, which scores "no code"
            # at -1.1 and "wrong" at +0.1: collapsing unparseable code into the no-code rung
            # would move it a full 1.2 in reward. behavior.md §2.8 keeps the two apart at
            # extraction for exactly this reason, and §5.5 scopes this short-circuit to
            # "when no code can be extracted".
            return VerificationReport(
                problem_id=problem.problem_id,
                extraction=extraction,
                graded_results=(),
                public_results=(),
            )
        source = self._preamble.source + extraction.code
        return VerificationReport(
            problem_id=problem.problem_id,
            extraction=extraction,
            graded_results=self._run_tests(source, problem.graded_tests),
            public_results=self._run_tests(source, problem.public_tests),
        )

    def _run_tests(
        self, source: str, tests: Sequence[TestCase]
    ) -> tuple[TestResult, ...]:
        """Run each test in order, abandoning the rest once one times out.

        Abandoned tests are still *recorded*, so the number of results always equals the
        number of tests and no reward function has to reason about why results are missing.
        """
        results: list[TestResult] = []
        for index, test in enumerate(tests):
            if _timed_out(results):
                results.append(_skipped(index, test))
                continue
            result = self._sandbox.run(
                source, test.input_text, self._config.sandbox.timeout_seconds
            )
            results.append(
                _classify(index, test, result, self._config.comparator)
            )
        return tuple(results)


def _classify(
    index: int, test: TestCase, result: SandboxResult, comparator: ComparatorConfig
) -> TestResult:
    return TestResult(
        test_index=index,
        pool=test.pool,
        outcome=_outcome(test, result, comparator),
        duration_seconds=result.duration_seconds,
        stdout_was_truncated=result.stdout_was_truncated,
        stderr_excerpt=result.stderr,
    )


def _outcome(
    test: TestCase, result: SandboxResult, comparator: ComparatorConfig
) -> TestOutcome:
    # Timeout is checked before exit status because a timed-out child also has no exit code,
    # and "hung" must not collapse into "crashed".
    if result.timed_out:
        return TestOutcome.TIMEOUT
    # Truncation is checked before exit status too, and for the same kind of reason. When the
    # reader hits the cap it *kills* the program, so the exit status that comes back is the
    # sandbox's, not the program's — treating it as a crash would auto-fail every truncated
    # rollout. behavior.md §5.8 and ADR-0009 both require the truncated output be compared
    # anyway: because the comparator demands matching token counts it will almost always
    # fail, but through the normal path, with no special case.
    if not result.stdout_was_truncated and (
        result.exit_code is None or result.exit_code != 0
    ):
        return TestOutcome.RUNTIME_ERROR
    if outputs_match(
        result.stdout, test.expected_output, comparator.absolute_float_tolerance
    ):
        return TestOutcome.PASSED
    return TestOutcome.WRONG_OUTPUT


def _skipped(index: int, test: TestCase) -> TestResult:
    return TestResult(
        test_index=index,
        pool=test.pool,
        outcome=TestOutcome.SKIPPED_AFTER_TIMEOUT,
        duration_seconds=0.0,
        stdout_was_truncated=False,
        stderr_excerpt="",
    )


def _timed_out(results: Sequence[TestResult]) -> bool:
    return any(result.outcome is TestOutcome.TIMEOUT for result in results)
