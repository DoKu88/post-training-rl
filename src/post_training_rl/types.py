"""The types that cross every seam between the verifier and the scorer.

All frozen, all data with no behaviour. Field names carry their units.

Specified in `docs/design/verifier-scorer.md` §2 and §3. The split these types make real
is ADR-0004's: the verifier produces a `VerificationReport` and holds no opinion about what
it is worth; every reward function consumes a `RolloutOutcome` and executes nothing.
"""

from dataclasses import dataclass
from enum import StrEnum


class TestPool(StrEnum):
    """Which pool a test case came from, and therefore how much it can be trusted.

    Recorded per result because the pools are not equally trustworthy: generated tests were
    produced by mutating existing inputs and validated only by consensus among human
    solutions, and AlphaCode's own measurement puts the false-positive-or-slow rate for the
    shipped suites at 46% (ADR-0009).
    """

    PUBLIC = "public"
    PRIVATE = "private"
    GENERATED = "generated"


class TestOutcome(StrEnum):
    """What happened when one test ran."""

    PASSED = "passed"
    WRONG_OUTPUT = "wrong_output"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    # Recorded rather than omitted so the result count always equals the test count — a
    # reward function must never have to reason about why results are missing (ADR-0006).
    SKIPPED_AFTER_TIMEOUT = "skipped_after_timeout"


class Fence(StrEnum):
    """The Markdown delimiter the recovered code arrived in — packaging, not content."""

    TAGGED = "tagged"  # ```python / ```py / ```python3 / ```py3
    UNTAGGED = "untagged"  # bare ``` with no language tag
    OTHER_TAG = "other_tag"  # ```cpp — fenced, but not tagged as Python
    NONE = "none"  # no fence anywhere in the completion


@dataclass(frozen=True)
class TestCase:
    input_text: str
    expected_output: str
    pool: TestPool


@dataclass(frozen=True)
class Problem:
    problem_id: str
    description: str
    graded_tests: tuple[TestCase, ...]  # private-first + generated filler, capped (ADR-0009)
    public_tests: tuple[TestCase, ...]  # never feeds any reward — diagnostic only (ADR-0013)


@dataclass(frozen=True)
class TestResult:
    test_index: int
    pool: TestPool
    outcome: TestOutcome
    duration_seconds: float
    stdout_was_truncated: bool
    stderr_excerpt: str  # capped; for debugging a failing rollout only


@dataclass(frozen=True)
class Extraction:
    """What recovering code from a completion yielded.

    `fence` and `parsed` are two independent facts and are never collapsed into one value.
    A flawless fence can wrap broken code and correct code can arrive unfenced, and the two
    failures call for opposite responses — a prompt change versus more training. Every
    surveyed harness fuses them, which is why no published source reports a parse-failure
    rate for any model on any benchmark (ADR-0012).
    """

    code: str | None
    fence: Fence  # packaging — how the code was wrapped
    parsed: bool  # content   — whether `ast.parse` accepted it


@dataclass(frozen=True)
class VerificationReport:
    problem_id: str
    extraction: Extraction
    graded_results: tuple[TestResult, ...]
    public_results: tuple[TestResult, ...]


@dataclass(frozen=True)
class RolloutOutcome:
    """A `VerificationReport` plus the facts the verifier cannot know.

    The verifier receives a string and has no tokenizer, so token count and truncation
    status belong to the TRL adapter, which holds both. Wrapping rather than extending keeps
    each type owned by the layer that can actually populate it.

    This is the single argument every reward function takes. That uniformity is the
    interchangeability contract (ADR-0011).
    """

    report: VerificationReport
    completion_token_count: int
    completion_was_truncated: bool  # hit max_completion_length


@dataclass(frozen=True)
class SandboxResult:
    """What running one program against one input produced.

    `exit_code` is None when the child was killed by a signal. That is deliberately distinct
    from `timed_out`, so that "hung" and "crashed" never collapse into one outcome — a child
    killed for any reason other than the timeout reports `exit_code=None, timed_out=False`
    and the verifier maps it to `RUNTIME_ERROR`.
    """

    stdout: str
    stderr: str
    exit_code: int | None
    duration_seconds: float
    timed_out: bool
    stdout_was_truncated: bool
