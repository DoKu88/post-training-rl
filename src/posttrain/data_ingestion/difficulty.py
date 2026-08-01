"""Empirical difficulty — bucket problems by *measured* policy pass rate.

`cf_rating` measures how hard a problem was for human competitors, which is only
loosely related to how hard it is for a 7B model, and 36% of our train split has
no rating at all. This module replaces that proxy with the real thing: sample the
policy G times per problem, and use the fraction that fully passed.

**The sharper reason this matters is GRPO itself.** GRPO normalizes advantage
*within* a group of G completions for the same prompt:

    A_i = (r_i - mean(r)) / std(r)

If every completion in the group earns the same reward — all fail, or all pass —
then `r_i - mean(r) == 0` for every i, the advantage vanishes, and the problem
contributes **no gradient**. Under a binary reward that is exactly the
`pass_rate ∈ {0, 1}` case. Those rollouts still cost full generation + sandbox
time, so an unfiltered corpus can burn a large share of each step producing
nothing. Measuring pass rate identifies them.

Split of responsibilities (docs/README.md: "the risky components are model-free"):

  * generating rollouts       -> needs the env (doc 2) + a policy (doc 3)
  * everything in this file   -> pure functions of a pass rate; no GPU, no model

So this module is fully testable today; only the thing that *produces* the
per-problem pass rates has to wait.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping

from posttrain.data_ingestion.schema import Problem

__all__ = [
    "DifficultyProfile",
    "EMPIRICAL_EASY_MIN",
    "EMPIRICAL_MEDIUM_MIN",
    "bucket_from_pass_rate",
    "has_learning_signal",
    "profile_from_rollouts",
    "aggregate_rollouts",
    "save_profiles",
    "load_profiles",
    "apply_profiles",
    "partition_by_signal",
    "signal_report",
]

#: Pass-rate → bucket. Inverted relative to a rating: a *high* pass rate means
#: the policy finds it easy. Boundaries are half-open: [0.7, 1.0] easy,
#: [0.3, 0.7) medium, [0.0, 0.3) hard.
EMPIRICAL_EASY_MIN = 0.7
EMPIRICAL_MEDIUM_MIN = 0.3

#: Below this many samples a pass rate is too noisy to re-bucket on. With G=8 a
#: single lucky completion moves the estimate by 0.125.
DEFAULT_MIN_SAMPLES = 4


@dataclass(frozen=True)
class DifficultyProfile:
    """What one policy measured on one problem.

    Model-tagged on purpose: a profile from Qwen2.5-3B does not describe
    Qwen2.5-7B, and silently reusing it across a model swap (doc 4) would
    invalidate the curriculum.
    """

    problem_id: str
    pass_rate: float  # fraction of sampled completions that fully passed
    n_samples: int
    model: str = ""

    def to_dict(self) -> dict:
        return {
            "problem_id": self.problem_id,
            "pass_rate": self.pass_rate,
            "n_samples": self.n_samples,
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "DifficultyProfile":
        return cls(
            problem_id=d["problem_id"],
            pass_rate=float(d["pass_rate"]),
            n_samples=int(d["n_samples"]),
            model=d.get("model", ""),
        )


def bucket_from_pass_rate(pass_rate: float) -> str:
    """Measured pass rate → `easy` / `medium` / `hard`."""
    if not 0.0 <= pass_rate <= 1.0:
        raise ValueError(f"pass_rate must be in [0, 1], got {pass_rate}")
    if pass_rate >= EMPIRICAL_EASY_MIN:
        return "easy"
    if pass_rate >= EMPIRICAL_MEDIUM_MIN:
        return "medium"
    return "hard"


def has_learning_signal(profile: DifficultyProfile) -> bool:
    """True when a GRPO group on this problem can produce a non-zero advantage.

    Requires at least one pass *and* one failure among the samples. Note this is
    an estimate at the sampled G: `pass_rate == 0.0` at G=8 means "no signal at
    G=8", not "unsolvable" — raising G or improving the policy can revive it.
    """
    return 0.0 < profile.pass_rate < 1.0


# --------------------------------------------------------------------------
# building profiles from rollout records
# --------------------------------------------------------------------------


def profile_from_rollouts(
    problem_id: str,
    rewards: Iterable[float],
    *,
    solved_threshold: float = 1.0,
    model: str = "",
) -> DifficultyProfile:
    """Aggregate one problem's rollout rewards into a profile.

    `solved_threshold` is what counts as "fully passed" — 1.0 under a Binary or
    Fractional reward (doc 2), since a partially-passing program is not solved.
    """
    rewards = list(rewards)
    if not rewards:
        return DifficultyProfile(problem_id, 0.0, 0, model)
    solved = sum(1 for r in rewards if r >= solved_threshold)
    return DifficultyProfile(problem_id, solved / len(rewards), len(rewards), model)


def aggregate_rollouts(
    records: Iterable[Mapping],
    *,
    solved_threshold: float = 1.0,
    model: str = "",
) -> dict[str, DifficultyProfile]:
    """Turn a stream of per-rollout records into per-problem profiles.

    Each record needs a `problem_id` plus either `passed` (bool) or `reward`
    (float). This is the seam doc 3 writes to: dump one record per rollout, then
    aggregate here — no coupling to how the rollouts were produced.
    """
    by_problem: dict[str, list[float]] = {}
    for rec in records:
        pid = rec["problem_id"]
        if "passed" in rec:
            reward = 1.0 if rec["passed"] else 0.0
        else:
            reward = float(rec["reward"])
        by_problem.setdefault(pid, []).append(reward)
    return {
        pid: profile_from_rollouts(pid, rs, solved_threshold=solved_threshold, model=model)
        for pid, rs in by_problem.items()
    }


# --------------------------------------------------------------------------
# sidecar i/o — kept beside the split, never mixed into the ingest cache
# --------------------------------------------------------------------------


def _sidecar_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.suffix == ".jsonl" else p / "difficulty.jsonl"


def save_profiles(profiles: Iterable[DifficultyProfile], path: str | Path) -> Path:
    """Write profiles beside a cached split; returns the file path."""
    out = _sidecar_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for prof in profiles:
            fh.write(json.dumps(prof.to_dict()) + "\n")
    return out


def load_profiles(path: str | Path) -> dict[str, DifficultyProfile]:
    src = _sidecar_path(path)
    if not src.is_file():
        raise FileNotFoundError(f"no difficulty profiles at {src}")
    with src.open("r", encoding="utf-8") as fh:
        profiles = [DifficultyProfile.from_dict(json.loads(l)) for l in fh if l.strip()]
    return {p.problem_id: p for p in profiles}


# --------------------------------------------------------------------------
# applying profiles
# --------------------------------------------------------------------------


def apply_profiles(
    problems: Iterable[Problem],
    profiles: Mapping[str, DifficultyProfile],
    *,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> list[Problem]:
    """Re-bucket problems by measured pass rate where a profile is available.

    Problems with no profile — or too few samples to trust — keep their static
    bucket and `bucket_source == "static"`, so a partial profiling run degrades
    gracefully instead of corrupting the corpus.
    """
    out = []
    for p in problems:
        prof = profiles.get(p.id)
        if prof is None or prof.n_samples < min_samples:
            out.append(p)
            continue
        out.append(
            replace(
                p,
                bucket=bucket_from_pass_rate(prof.pass_rate),
                bucket_source="empirical",
                pass_rate=prof.pass_rate,
            )
        )
    return out


def partition_by_signal(
    problems: Iterable[Problem],
) -> tuple[list[Problem], list[Problem], list[Problem]]:
    """Split into `(learnable, saturated, unsolved)` by measured pass rate.

    * **learnable**  — 0 < pass_rate < 1: GRPO gets a non-zero advantage here.
    * **saturated**  — pass_rate == 1: solved every time, no gradient.
    * **unsolved**   — pass_rate == 0: failed every time, no gradient.

    Unmeasured problems (`pass_rate < 0`) count as learnable: we have no evidence
    against them, and dropping unmeasured data would silently shrink the corpus.
    """
    learnable, saturated, unsolved = [], [], []
    for p in problems:
        if p.pass_rate < 0:
            learnable.append(p)
        elif p.pass_rate >= 1.0:
            saturated.append(p)
        elif p.pass_rate <= 0.0:
            unsolved.append(p)
        else:
            learnable.append(p)
    return learnable, saturated, unsolved


def signal_report(problems: Iterable[Problem]) -> str:
    """Human-readable summary of where the rollout budget would actually go."""
    problems = list(problems)
    learnable, saturated, unsolved = partition_by_signal(problems)
    total = len(problems) or 1
    buckets = Counter(p.bucket for p in problems)
    sources = Counter(p.bucket_source for p in problems)
    lines = [
        f"problems={len(problems)}  bucketed_by={dict(sources)}",
        f"  learnable (0<pr<1) : {len(learnable):>6}  ({100*len(learnable)/total:5.1f}%)",
        f"  saturated (pr==1)  : {len(saturated):>6}  ({100*len(saturated)/total:5.1f}%)"
        "   <- no GRPO gradient",
        f"  unsolved  (pr==0)  : {len(unsolved):>6}  ({100*len(unsolved)/total:5.1f}%)"
        "   <- no GRPO gradient",
        f"  buckets: {dict(buckets)}",
    ]
    return "\n".join(lines)
