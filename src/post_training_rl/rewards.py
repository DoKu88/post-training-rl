"""The reward registry, per ADR-0011.

Every entry has the identical signature `RolloutOutcome -> float` and is pure. That
uniformity is the interchangeability contract: a function needing anything else does not
belong here. It is also what lets one execution feed every entry, so a single run logs the
counterfactual curve for every reward shape without executing once per shape.

Three conventions are decided here, once, so no individual function re-decides them:

- `SKIPPED_AFTER_TIMEOUT` counts as **not passed**, everywhere.
- `public_results` is **never read** by any reward. Public tests are printed in the problem
  statement, so partial credit over them is directly hackable — DeepCoder's stated failure
  mode is a model that learns to print the answers of public tests (ADR-0013).
- A rollout with recovered code but no graded results is an **apparatus failure** and
  raises. The dataset builder guarantees at least five graded tests per problem
  (ADR-0009), so that state cannot arise from a model's behaviour. A rollout with no
  recovered code legitimately has no results, and is scored on its own rung.

The numeric constants below are the *identity* of each function, not tunables: they come
from the published implementations each entry reproduces, and changing one makes a different
reward rather than a differently-configured one. `config/reward.yaml` selects which entries
run and at what weight.
"""

from collections.abc import Callable, Sequence

from post_training_rl.types import Fence, RolloutOutcome, TestOutcome, TestResult

RewardFn = Callable[[RolloutOutcome], float]

# open-r1 `binary_code_reward`. A float-comparison guard, not a real tolerance.
_PASS_RATE_THRESHOLD = 0.99

# code-r1 `coder1/__init__.py`, which trains on CodeContests, Python, stdin/stdout.
_CODE_R1_NO_CODE = -1.1
_CODE_R1_WRONG = 0.1
_CODE_R1_CORRECT = 1.1

# This project's own construction, shaped like DHRCL's hierarchy. Ships unannealed.
_LADDER_PARSES = 0.05
_LADDER_RUNS = 0.10
_LADDER_PASS_WEIGHT = 0.90

# ADR-0012. The parse swing (1.2) deliberately exceeds the fence swing (0.8), so the worst
# parsing rollout still outranks the best non-parsing one — a well-fenced broken program
# must never beat a bare working one.
_PARSE_TERM = 0.6
_FENCE_TERMS = {
    Fence.TAGGED: 0.4,
    Fence.UNTAGGED: 0.2,
    Fence.OTHER_TAG: 0.0,
    Fence.NONE: -0.4,
}


def binary_reward(outcome: RolloutOutcome) -> float:
    """1.0 when every graded test passed, else 0.0.

    DeepCoder/rLLM `check_correctness` -> `all(passed)`; DeepSeek-R1's rule-based rewards
    (arXiv:2501.12948). The default, and the baseline every other entry is compared against.
    """
    results = _graded_results(outcome)
    if not results:
        return 0.0
    return 1.0 if all(_passed(result) for result in results) else 0.0


def pass_rate_reward(outcome: RolloutOutcome) -> float:
    """The fraction of graded tests that passed. open-r1 `code_reward`."""
    return _pass_rate(_graded_results(outcome))


def binary_threshold_reward(outcome: RolloutOutcome) -> float:
    """1.0 when the pass rate exceeds 0.99, else 0.0. open-r1 `binary_code_reward`."""
    return 1.0 if _pass_rate(_graded_results(outcome)) > _PASS_RATE_THRESHOLD else 0.0


def ladder_reward(outcome: RolloutOutcome) -> float:
    """Graded rungs: no code, code that parses, code that runs, then pass rate.

    This project's own construction, shaped like DHRCL's hierarchical decomposition
    (arXiv:2607.26457). Ships unannealed — an annealed variant needs no extra machinery,
    since the trainer passes `trainer_state` and its `global_step` to every reward function,
    but it is not part of the first run.
    """
    results = _graded_results(outcome)
    if not _has_code(outcome) or not outcome.report.extraction.parsed:
        return 0.0
    if not any(_ran(result) for result in results):
        return _LADDER_PARSES
    return _LADDER_RUNS + _LADDER_PASS_WEIGHT * _pass_rate(results)


def code_r1_reward(outcome: RolloutOutcome) -> float:
    """-1.1 for no code, +0.1 for a wrong answer, +1.1 when every test passes.

    code-r1 `coder1/__init__.py`. The only surveyed design where failing to produce code is
    distinguishable from producing wrong code.
    """
    if not _has_code(outcome):
        return _CODE_R1_NO_CODE
    results = _graded_results(outcome)
    if results and all(_passed(result) for result in results):
        return _CODE_R1_CORRECT
    return _CODE_R1_WRONG


def extractability_reward(outcome: RolloutOutcome) -> float:
    """A parse term plus a fence term, so both dimensions are rewarded independently.

    ADR-0012, values from rl-reward-functions.md §3. Carries weight 0.1 against the primary
    reward and exists to give an otherwise-degenerate group some variance: when every
    rollout fails every test, the primary is identical across the group, the standard
    deviation is zero, and the prompt yields no gradient at all having cost a full group.
    """
    extraction = outcome.report.extraction
    parse_term = _PARSE_TERM if extraction.parsed else -_PARSE_TERM
    return parse_term + _FENCE_TERMS[extraction.fence]


REWARD_FUNCTIONS: dict[str, RewardFn] = {
    "binary": binary_reward,
    "pass_rate": pass_rate_reward,
    "binary_threshold": binary_threshold_reward,
    "ladder": ladder_reward,
    "code_r1": code_r1_reward,
    "extractability": extractability_reward,
}


def _graded_results(outcome: RolloutOutcome) -> Sequence[TestResult]:
    """The graded results, refusing the one state that can only be an apparatus failure."""
    report = outcome.report
    if not report.graded_results and _has_code(outcome):
        raise ValueError(
            f"no graded results for a rollout with recovered code, problem "
            f"{report.problem_id!r} — the dataset builder drops problems under "
            f"tests.min_tests_required, so this is an apparatus failure rather than a "
            f"model outcome"
        )
    return report.graded_results


def _has_code(outcome: RolloutOutcome) -> bool:
    return outcome.report.extraction.code is not None


def _passed(result: TestResult) -> bool:
    return result.outcome is TestOutcome.PASSED


def _ran(result: TestResult) -> bool:
    """Whether the program executed far enough to produce output worth comparing."""
    return result.outcome in (TestOutcome.PASSED, TestOutcome.WRONG_OUTPUT)


def _pass_rate(results: Sequence[TestResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for result in results if _passed(result)) / len(results)
