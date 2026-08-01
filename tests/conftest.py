"""Shared fixtures for the model-free data tests.

Nothing here touches the network, a GPU, or a real tokenizer. The doc-1.4
"cached fixture split" fixtures prefer a *real* cache under `data/processed/`
when one exists, and fall back to a synthetic one so the suite is green on a
fresh checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

# Allow `pytest` from a fresh checkout without `pip install -e .`
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from posttrain.data.ingest import DEFAULT_CACHE_ROOT, cache_exists, ingest, load_cached  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# fake tokenizer
# --------------------------------------------------------------------------


class FakeTokenizer:
    """Stand-in for a HF tokenizer's `apply_chat_template`.

    Records every call so tests can assert on the roles and kwargs the prompt
    builder passed, and renders a ChatML-ish string so the output is inspectable.
    """

    def __init__(self, name: str = "fake/chatml") -> None:
        self.name_or_path = name
        self.calls: list[dict[str, Any]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        /,
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        **kwargs: Any,
    ) -> str:
        self.calls.append(
            {
                "messages": [dict(m) for m in messages],
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                **kwargs,
            }
        )
        rendered = "\n".join(
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in messages
        )
        if add_generation_prompt:
            rendered += "\n<|im_start|>assistant\n"
        return rendered


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


# --------------------------------------------------------------------------
# synthetic raw HF rows
# --------------------------------------------------------------------------


def make_raw_row(
    name: str = "1000_A. Sample Problem",
    *,
    n_public: int = 2,
    n_private: int = 1,
    n_generated: int = 3,
    languages: tuple[int, ...] = (3, 2, 3),
    description: str = "Read one integer n from stdin and print n + 1 to stdout.",
    difficulty: int = 7,
    cf_rating: int = 800,
    cf_tags: tuple[str, ...] = ("implementation", "math"),
    input_file: str = "",
    output_file: str = "",
    time_limit_seconds: int | None = 2,
) -> dict[str, Any]:
    """Build a raw-row dict shaped like a `deepmind/code_contests` record."""

    def block(n: int, tag: str) -> dict[str, list[str]]:
        return {
            "input": [f"{tag}-in-{i}\n" for i in range(n)],
            "output": [f"{tag}-out-{i}\n" for i in range(n)],
        }

    return {
        "name": name,
        "description": description,
        "public_tests": block(n_public, "pub"),
        "private_tests": block(n_private, "priv"),
        "generated_tests": block(n_generated, "gen"),
        "solutions": {
            "language": list(languages),
            "solution": [f"# lang={lang}\nprint(int(input()) + 1)" for lang in languages],
        },
        "difficulty": difficulty,
        "cf_rating": cf_rating,
        "cf_tags": list(cf_tags),
        "input_file": input_file,
        "output_file": output_file,
        "time_limit": (
            None if time_limit_seconds is None else {"seconds": time_limit_seconds, "nanos": 0}
        ),
    }


@pytest.fixture
def raw_row() -> dict[str, Any]:
    return make_raw_row()


def _synthetic_rows(prefix: str, n: int) -> list[dict[str, Any]]:
    ratings = (800, 1200, 1500, 1800, 2400)
    return [
        make_raw_row(
            name=f"{prefix} {i}. Synthetic Problem",
            n_public=2,
            n_private=2,
            n_generated=25,
            cf_rating=ratings[i % len(ratings)],
        )
        for i in range(n)
    ]


# --------------------------------------------------------------------------
# "cached fixture split" — real cache if present, else synthetic
# --------------------------------------------------------------------------


def _split_problems(split: str, prefix: str, n: int = 12):
    cache = REPO_ROOT / DEFAULT_CACHE_ROOT / split
    if cache_exists(cache):
        return load_cached(cache), "real"
    return ingest(split, rows=_synthetic_rows(prefix, n)), "synthetic"


@pytest.fixture(scope="session")
def train_problems():
    problems, _ = _split_problems("train", "TRAIN")
    return problems


@pytest.fixture(scope="session")
def test_problems():
    problems, _ = _split_problems("test", "TEST")
    return problems


@pytest.fixture(scope="session")
def cached_split(train_problems):
    """The split the doc-1.4 invariant checks run over."""
    return train_problems
