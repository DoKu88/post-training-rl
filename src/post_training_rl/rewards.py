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
  raises. The dataset builder drops problems under `tests.min_tests_required` (ADR-0009), so
  that state cannot arise from a model's behaviour. A rollout with no recovered code
  legitimately has no results, and is scored on its own rung.

The numbers defining each shape come from `config/reward.yaml`, not from literals here, so
that two runs which graded differently can be diffed as files. `build_reward_functions`
closes over them — the mechanism `verifier-scorer.md` §7 already sanctions for a
parameterised entry, and it keeps every registry entry a plain one-argument function.

**Treat a change to a shape value as defining a new reward, not tuning an existing one.**
Each reproduces a published implementation, and a run using different values is not
comparable to one that did not.
"""

from collections.abc import Callable, Sequence

from post_training_rl.config import RewardShapes
from post_training_rl.types import Fence, RolloutOutcome, TestOutcome, TestResult

RewardFn = Callable[[RolloutOutcome], float]


def build_reward_functions(shapes: RewardShapes) -> dict[str, RewardFn]:
    """Build the registry, with each entry closing over the configured shape values.

    Returns exactly the six entries implemented for the first run. `hierarchical`, `verpo`
    and `overlong` are in ADR-0011's design space but are not built — each needs machinery
    no other entry needs, and none is in the first run.
    """
    fence_terms = _fence_terms(shapes)

    def binary(outcome: RolloutOutcome) -> float:
        """1.0 when every graded test passed, else 0.0.

        DeepCoder/rLLM `check_correctness` -> `all(passed)`; DeepSeek-R1's rule-based
        rewards (arXiv:2501.12948). The default, and the baseline every other entry is
        compared against.
        """
        results = _graded_results(outcome)
        if not results:
            return 0.0
        return 1.0 if all(_passed(result) for result in results) else 0.0

    def pass_rate(outcome: RolloutOutcome) -> float:
        """The fraction of graded tests that passed. open-r1 `code_reward`."""
        return _pass_rate(_graded_results(outcome))

    def binary_threshold(outcome: RolloutOutcome) -> float:
        """1.0 when the pass rate exceeds the threshold. open-r1 `binary_code_reward`."""
        rate = _pass_rate(_graded_results(outcome))
        return 1.0 if rate > shapes.pass_rate_threshold else 0.0

    def ladder(outcome: RolloutOutcome) -> float:
        """Graded rungs: no code, code that parses, code that runs, then pass rate.

        This project's own construction, shaped like DHRCL's hierarchical decomposition
        (arXiv:2607.26457). Ships unannealed — an annealed variant needs no extra machinery,
        since the trainer passes `trainer_state` and its `global_step` to every reward
        function, but it is not part of the first run.
        """
        results = _graded_results(outcome)
        if not _has_code(outcome) or not outcome.report.extraction.parsed:
            return 0.0
        if not any(_ran(result) for result in results):
            return shapes.ladder_parses
        return shapes.ladder_runs + shapes.ladder_pass_weight * _pass_rate(results)

    def code_r1(outcome: RolloutOutcome) -> float:
        """No code scores below a wrong answer, which scores below a correct one.

        code-r1 `coder1/__init__.py`. The only surveyed design where failing to produce code
        is distinguishable from producing wrong code.
        """
        if not _has_code(outcome):
            return shapes.code_r1_no_code
        results = _graded_results(outcome)
        if results and all(_passed(result) for result in results):
            return shapes.code_r1_correct
        return shapes.code_r1_wrong

    def extractability(outcome: RolloutOutcome) -> float:
        """A parse term plus a fence term, so both dimensions are rewarded independently.

        ADR-0012. Carries weight 0.1 against the primary reward and exists to give an
        otherwise-degenerate group some variance: when every rollout fails every test the
        primary is identical across the group, the standard deviation is zero, and the
        prompt yields no gradient at all having cost a full group of rollouts.
        """
        extraction = outcome.report.extraction
        parse_term = shapes.parse_term if extraction.parsed else -shapes.parse_term
        return parse_term + fence_terms[extraction.fence]

    return {
        "binary": binary,
        "pass_rate": pass_rate,
        "binary_threshold": binary_threshold,
        "ladder": ladder,
        "code_r1": code_r1,
        "extractability": extractability,
    }


def _fence_terms(shapes: RewardShapes) -> dict[Fence, float]:
    """Resolve the config's string keys to `Fence` members, naming any that do not map."""
    try:
        return {Fence(name): value for name, value in shapes.fence_terms.items()}
    except ValueError as unknown:
        raise ValueError(
            f"shapes.extractability.fence in config/reward.yaml names a fence that does "
            f"not exist: {unknown}. Valid names: {sorted(f.value for f in Fence)}"
        ) from unknown


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
