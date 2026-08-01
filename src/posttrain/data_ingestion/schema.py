"""Normalized problem schema — our own contract, not raw HuggingFace rows.

Sprint 1.1. Everything downstream (sandbox, env, reward, eval) depends *only*
on `Problem`, which is what makes "any dataset / any model" tractable: a new
dataset only has to emit `Problem` records.

Both dataclasses are frozen so a `Problem` can be handed to the environment,
the reward and the eval harness without any of them being able to mutate it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

__all__ = ["TestCase", "Problem"]

#: Coarse difficulty tags, ordered easiest → hardest. `"unknown"` is used when
#: the source dataset gives us no usable difficulty signal.
BUCKETS: tuple[str, ...] = ("easy", "medium", "hard", "unknown")


@dataclass(frozen=True)
class TestCase:
    """One stdin → stdout example.

    CodeContests problems are *stdin/stdout* programs (not function-call based
    like HumanEval/MBPP): the sandbox writes `input` to the process' stdin and
    compares captured stdout against `output`.
    """

    __test__ = False  # not a pytest test class, despite the name

    input: str
    output: str
    kind: str  # "public" | "private" | "generated"


@dataclass(frozen=True)
class Problem:
    """A single trainable problem, normalized away from its source dataset."""

    id: str  # slugified `name`; stable across re-ingestion
    statement: str  # cleaned description
    tests: list[TestCase]
    difficulty: int  # raw source difficulty code (see data.ingest)
    reference_solutions: list[str]  # Python 3 only — ground truth for doc 2
    source: str = "code_contests"  # which *dataset* this came from
    bucket: str = "unknown"  # coarse easy/medium/hard tag (doc 3 curriculum)
    split: str = ""  # "train" | "valid" | "test" — leak checks in doc 7
    cf_rating: int = 0  # Codeforces rating, 0 when unknown
    time_limit_s: float = 0.0  # source per-test time limit; 0 = unspecified
    bucket_source: str = "static"  # "static" (cf_rating/enum) | "empirical"
    pass_rate: float = -1.0  # measured policy pass rate; -1.0 = unmeasured


    # -- test partitions ---------------------------------------------------
    def public(self) -> list[TestCase]:
        """Tests the solver is allowed to see (they appear in the statement)."""
        return [t for t in self.tests if t.kind == "public"]

    def hidden(self) -> list[TestCase]:
        """Tests the solver never sees — private + generated."""
        return [t for t in self.tests if t.kind != "public"]

    # -- serialization -----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON representation; round-trips exactly via `from_dict`."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Problem":
        payload = dict(d)
        payload["tests"] = [TestCase(**t) for t in payload.get("tests", [])]
        return cls(**payload)
