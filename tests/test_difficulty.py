"""Empirical difficulty tests — all pure functions of a pass rate, no GPU."""

from __future__ import annotations

import pytest

from posttrain.data.difficulty import (
    DifficultyProfile,
    aggregate_rollouts,
    apply_profiles,
    bucket_from_pass_rate,
    has_learning_signal,
    load_profiles,
    partition_by_signal,
    profile_from_rollouts,
    save_profiles,
    signal_report,
)
from posttrain.data.schema import Problem, TestCase


def mk(pid: str, bucket: str = "unknown") -> Problem:
    return Problem(
        id=pid,
        statement="s",
        tests=[TestCase("1", "2", "public")],
        difficulty=0,
        reference_solutions=["print(1)"],
        bucket=bucket,
    )


def test_bucket_from_pass_rate_boundaries():
    # a HIGH pass rate means the policy finds it easy — inverted vs a rating
    assert bucket_from_pass_rate(1.0) == "easy"
    assert bucket_from_pass_rate(0.7) == "easy"
    assert bucket_from_pass_rate(0.69) == "medium"
    assert bucket_from_pass_rate(0.3) == "medium"
    assert bucket_from_pass_rate(0.29) == "hard"
    assert bucket_from_pass_rate(0.0) == "hard"

    for bad in (-0.01, 1.01):
        with pytest.raises(ValueError):
            bucket_from_pass_rate(bad)


def test_has_learning_signal_flags_zero_gradient_groups():
    # GRPO advantage is (r - mean(r))/std(r): identical rewards -> zero gradient
    assert not has_learning_signal(DifficultyProfile("a", 0.0, 8))
    assert not has_learning_signal(DifficultyProfile("a", 1.0, 8))
    assert has_learning_signal(DifficultyProfile("a", 0.125, 8))
    assert has_learning_signal(DifficultyProfile("a", 0.875, 8))


def test_profile_from_rollouts_counts_only_full_passes():
    # a partially-passing program is not solved
    p = profile_from_rollouts("x", [1.0, 0.0, 0.9, 1.0], model="qwen3b")
    assert p.pass_rate == 0.5 and p.n_samples == 4 and p.model == "qwen3b"

    assert profile_from_rollouts("x", []).n_samples == 0
    assert profile_from_rollouts("x", [0.6, 0.7], solved_threshold=0.5).pass_rate == 1.0


def test_aggregate_rollouts_accepts_passed_or_reward():
    records = [
        {"problem_id": "a", "passed": True},
        {"problem_id": "a", "passed": False},
        {"problem_id": "b", "reward": 1.0},
        {"problem_id": "b", "reward": 1.0},
    ]
    profiles = aggregate_rollouts(records, model="m")
    assert profiles["a"].pass_rate == 0.5 and profiles["a"].n_samples == 2
    assert profiles["b"].pass_rate == 1.0
    assert not has_learning_signal(profiles["b"])


def test_apply_profiles_rebuckets_and_tags_source():
    problems = [mk("a", "unknown"), mk("b", "hard"), mk("c", "easy")]
    profiles = {
        "a": DifficultyProfile("a", 0.9, 8),  # policy finds it easy
        "b": DifficultyProfile("b", 0.1, 8),  # stays hard
        # "c" has no profile
    }
    out = {p.id: p for p in apply_profiles(problems, profiles)}

    assert out["a"].bucket == "easy" and out["a"].bucket_source == "empirical"
    assert out["a"].pass_rate == 0.9
    assert out["b"].bucket == "hard" and out["b"].bucket_source == "empirical"
    # unprofiled problems degrade gracefully: keep the static bucket
    assert out["c"].bucket == "easy" and out["c"].bucket_source == "static"
    assert out["c"].pass_rate == -1.0


def test_apply_profiles_ignores_undersampled():
    problems = [mk("a", "unknown")]
    profiles = {"a": DifficultyProfile("a", 1.0, 2)}
    out = apply_profiles(problems, profiles, min_samples=4)
    assert out[0].bucket == "unknown" and out[0].bucket_source == "static"

    out = apply_profiles(problems, profiles, min_samples=2)
    assert out[0].bucket_source == "empirical"


def test_partition_by_signal():
    problems = [
        Problem("learn", "s", [], 0, [], pass_rate=0.5),
        Problem("sat", "s", [], 0, [], pass_rate=1.0),
        Problem("uns", "s", [], 0, [], pass_rate=0.0),
        Problem("unmeasured", "s", [], 0, [], pass_rate=-1.0),
    ]
    learnable, saturated, unsolved = partition_by_signal(problems)

    assert {p.id for p in learnable} == {"learn", "unmeasured"}
    assert [p.id for p in saturated] == ["sat"]
    assert [p.id for p in unsolved] == ["uns"]
    assert len(learnable) + len(saturated) + len(unsolved) == len(problems)


def test_profiles_roundtrip(tmp_path):
    profiles = [
        DifficultyProfile("a", 0.25, 8, "qwen3b"),
        DifficultyProfile("b", 1.0, 8, "qwen3b"),
    ]
    loaded = load_profiles(save_profiles(profiles, tmp_path / "train"))
    assert loaded == {p.problem_id: p for p in profiles}

    with pytest.raises(FileNotFoundError):
        load_profiles(tmp_path / "missing")


def test_signal_report_mentions_wasted_groups():
    problems = [
        Problem("a", "s", [], 0, [], pass_rate=0.5),
        Problem("b", "s", [], 0, [], pass_rate=1.0),
    ]
    report = signal_report(problems)
    assert "learnable" in report and "saturated" in report and "no GRPO gradient" in report


def test_rebucketed_problems_survive_the_ingest_cache(tmp_path):
    """Re-bucketed sets must round-trip through save/load like any other split."""
    from posttrain.data.ingest import load_cached, save

    problems = apply_profiles([mk("a")], {"a": DifficultyProfile("a", 0.9, 8, "qwen3b")})
    restored = load_cached(save(problems, tmp_path / "empirical"))

    assert restored == problems
    assert restored[0].bucket_source == "empirical" and restored[0].pass_rate == 0.9
