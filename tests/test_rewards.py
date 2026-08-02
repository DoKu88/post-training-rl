"""Reward registry behaviour, per `docs/design/rl-reward-functions.md` and behavior.md §3.

Every reward is a pure function of one `RolloutOutcome`, so every test here builds a literal
and asserts a number. No sandbox, no subprocess, no model.

The registry is built from `config/reward.yaml`, so these tests assert the shipped
configuration, not a second copy of the numbers living in the test file.
"""

from pathlib import Path

import pytest

from post_training_rl.config import load_reward_config
from post_training_rl.rewards import build_reward_functions
from post_training_rl.types import (
    Extraction,
    Fence,
    RolloutOutcome,
    TestOutcome,
    TestPool,
    TestResult,
    VerificationReport,
)

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "reward.yaml"
REWARD_FUNCTIONS = build_reward_functions(load_reward_config(_CONFIG_PATH).shapes)

binary_reward = REWARD_FUNCTIONS["binary"]
pass_rate_reward = REWARD_FUNCTIONS["pass_rate"]
binary_threshold_reward = REWARD_FUNCTIONS["binary_threshold"]
ladder_reward = REWARD_FUNCTIONS["ladder"]
code_r1_reward = REWARD_FUNCTIONS["code_r1"]
extractability_reward = REWARD_FUNCTIONS["extractability"]


def _result(outcome: TestOutcome, index: int = 0) -> TestResult:
    return TestResult(
        test_index=index,
        pool=TestPool.PRIVATE,
        outcome=outcome,
        duration_seconds=0.01,
        stdout_was_truncated=False,
        stderr_excerpt="",
    )


def _outcome(
    graded: list[TestOutcome] | None = None,
    code: str | None = "print(1)",
    fence: Fence = Fence.TAGGED,
    parsed: bool = True,
    public: list[TestOutcome] | None = None,
) -> RolloutOutcome:
    graded = [] if graded is None else graded
    public = [] if public is None else public
    return RolloutOutcome(
        report=VerificationReport(
            problem_id="p1",
            extraction=Extraction(code=code, fence=fence, parsed=parsed),
            graded_results=tuple(_result(o, i) for i, o in enumerate(graded)),
            public_results=tuple(_result(o, i) for i, o in enumerate(public)),
        ),
        completion_token_count=128,
        completion_was_truncated=False,
    )


_PASSED_5 = [TestOutcome.PASSED] * 5


def test_binary_rewards_all_passing():
    assert binary_reward(_outcome(_PASSED_5)) == 1.0


def test_binary_rejects_single_failure():
    # A single failure collapses it. Maximum sparsity is binary's known weakness and the
    # reason it ships composed with extractability (ADR-0011).
    graded = [TestOutcome.PASSED] * 4 + [TestOutcome.WRONG_OUTPUT]
    assert binary_reward(_outcome(graded)) == 0.0


def test_binary_rejects_timeout():
    graded = [TestOutcome.PASSED] * 4 + [TestOutcome.TIMEOUT]
    assert binary_reward(_outcome(graded)) == 0.0


def test_pass_rate_is_fraction_passed():
    graded = [TestOutcome.PASSED] * 3 + [TestOutcome.WRONG_OUTPUT] * 2
    assert pass_rate_reward(_outcome(graded)) == 0.6


def test_pass_rate_is_one_when_all_pass():
    assert pass_rate_reward(_outcome(_PASSED_5)) == 1.0


def test_pass_rate_is_zero_when_none_pass():
    assert pass_rate_reward(_outcome([TestOutcome.WRONG_OUTPUT] * 5)) == 0.0


def test_threshold_accepts_full_pass():
    assert binary_threshold_reward(_outcome(_PASSED_5)) == 1.0


def test_threshold_rejects_partial_pass():
    graded = [TestOutcome.PASSED] * 4 + [TestOutcome.WRONG_OUTPUT]
    assert binary_threshold_reward(_outcome(graded)) == 0.0


def test_ladder_rung_no_code():
    assert ladder_reward(_outcome(code=None, fence=Fence.NONE, parsed=False)) == 0.0


def test_ladder_rung_parses():
    # Code that parses but never runs cleanly on any test.
    graded = [TestOutcome.RUNTIME_ERROR] * 5
    assert ladder_reward(_outcome(graded)) == 0.05


def test_ladder_rung_runs():
    # It ran on every test and got every one wrong: the rung's floor, pass rate 0.
    graded = [TestOutcome.WRONG_OUTPUT] * 5
    assert ladder_reward(_outcome(graded)) == pytest.approx(0.10)


def test_ladder_rung_partial_pass():
    graded = [TestOutcome.PASSED] * 3 + [TestOutcome.WRONG_OUTPUT] * 2
    assert ladder_reward(_outcome(graded)) == pytest.approx(0.10 + 0.90 * 0.6)


def test_code_r1_penalises_missing_code():
    outcome = _outcome(code=None, fence=Fence.NONE, parsed=False)
    assert code_r1_reward(outcome) == pytest.approx(-1.1)


def test_code_r1_scores_wrong_answer():
    graded = [TestOutcome.PASSED] * 4 + [TestOutcome.WRONG_OUTPUT]
    assert code_r1_reward(_outcome(graded)) == pytest.approx(0.1)


def test_code_r1_scores_correct():
    assert code_r1_reward(_outcome(_PASSED_5)) == pytest.approx(1.1)


def test_extractability_scores_each_fence_at_equal_parse():
    # Fence quality is rewarded in its own right: well-formed output is wanted even when the
    # program already runs, so a better fence scores higher at equal parse status.
    scores = {
        fence: extractability_reward(_outcome(_PASSED_5, fence=fence, parsed=True))
        for fence in Fence
    }
    assert scores[Fence.TAGGED] == pytest.approx(1.0)
    assert scores[Fence.UNTAGGED] == pytest.approx(0.8)
    assert scores[Fence.OTHER_TAG] == pytest.approx(0.6)
    assert scores[Fence.NONE] == pytest.approx(0.2)


def test_extractability_parsing_always_outranks_non_parsing():
    # The invariant ADR-0012 requires, over all sixteen (parsing, non-parsing) pairs rather
    # than a sampled few: a beautifully fenced broken program must never beat a bare working
    # one. Any future retuning of the values has to preserve this.
    for parsing_fence in Fence:
        parsing = extractability_reward(
            _outcome(_PASSED_5, fence=parsing_fence, parsed=True)
        )
        for broken_fence in Fence:
            broken = extractability_reward(
                _outcome(_PASSED_5, code="x", fence=broken_fence, parsed=False)
            )
            assert parsing > broken, f"{parsing_fence} parsing lost to {broken_fence} broken"


def test_skipped_after_timeout_counts_as_not_passed():
    # Decided once in the registry so no individual function re-decides it. Asserted over
    # every registered entry: a rollout whose tail was skipped must score identically to one
    # whose tail simply failed.
    timed_out = [TestOutcome.PASSED] * 2 + [TestOutcome.TIMEOUT]
    timed_out += [TestOutcome.SKIPPED_AFTER_TIMEOUT] * 2
    treated_as_failed = [TestOutcome.PASSED] * 2 + [TestOutcome.TIMEOUT]
    treated_as_failed += [TestOutcome.WRONG_OUTPUT] * 2

    for name, reward in REWARD_FUNCTIONS.items():
        assert reward(_outcome(timed_out)) == pytest.approx(
            reward(_outcome(treated_as_failed))
        ), name


def test_every_registered_function_accepts_rollout_outcome():
    # The interchangeability contract, as an executable assertion.
    for name, reward in REWARD_FUNCTIONS.items():
        value = reward(_outcome(_PASSED_5))
        assert isinstance(value, float), name


def test_public_results_never_affect_any_reward():
    # Public tests are a diagnostic, never a reward input (ADR-0013). A reward computed over
    # tests the model can read is directly hackable — DeepCoder's stated failure mode is a
    # model that learns to print the answers of public tests.
    graded = [TestOutcome.PASSED] * 3 + [TestOutcome.WRONG_OUTPUT] * 2

    for name, reward in REWARD_FUNCTIONS.items():
        without_public = reward(_outcome(graded))
        assert without_public == pytest.approx(
            reward(_outcome(graded, public=[TestOutcome.PASSED] * 3))
        ), name
        assert without_public == pytest.approx(
            reward(_outcome(graded, public=[TestOutcome.WRONG_OUTPUT] * 3))
        ), name
