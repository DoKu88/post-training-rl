"""Step 1 tests — schema, ingestion, prompts, and data-quality invariants.

All model-free and offline: run with `pytest tests/test_data.py -m "not gpu"`.
"""

from __future__ import annotations

import dataclasses

import pytest

from posttrain.data.ingest import (
    cap_generated,
    difficulty_bucket,
    estimate_tokens,
    extract_python_solutions,
    extract_time_limit,
    flatten_tests,
    ingest,
    is_scorable,
    ingest_with_stats,
    load_cached,
    row_to_problem,
    save,
    slugify,
    unsupported_reason,
)
from posttrain.data.prompts import SYSTEM_PROMPT, to_chat_prompt
from posttrain.data.schema import Problem, TestCase

from conftest import make_raw_row

# ==========================================================================
# Sprint 1.1 — normalized schema
# ==========================================================================


def test_testcase_is_frozen():
    t = TestCase(input="1\n", output="2\n", kind="public")
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.input = "99\n"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.kind = "private"  # type: ignore[misc]
    assert t.input == "1\n"


def test_problem_public_hidden_partition():
    tests = [
        TestCase("a", "1", "public"),
        TestCase("b", "2", "private"),
        TestCase("c", "3", "generated"),
        TestCase("d", "4", "public"),
    ]
    p = Problem(id="x", statement="s", tests=tests, difficulty=1, reference_solutions=["print(1)"])

    assert all(t.kind == "public" for t in p.public())
    assert all(t.kind != "public" for t in p.hidden())
    # partition: covers everything, no overlap, no duplicates
    assert len(p.public()) + len(p.hidden()) == len(tests)
    assert p.public() + p.hidden() != []
    assert {id(t) for t in p.public()} & {id(t) for t in p.hidden()} == set()
    assert {id(t) for t in p.public()} | {id(t) for t in p.hidden()} == {id(t) for t in tests}


def test_problem_defaults():
    p = Problem(id="x", statement="s", tests=[], difficulty=1, reference_solutions=[])
    assert p.source == "code_contests"
    assert p.bucket == "unknown"
    assert p.split == ""
    assert p.cf_rating == 0
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.source = "other"  # type: ignore[misc]


# ==========================================================================
# Sprint 1.2 — ingestion pipeline
# ==========================================================================


def test_flatten_tests_tags_and_zips(raw_row):
    tests = flatten_tests(raw_row)

    kinds = [t.kind for t in tests]
    assert kinds.count("public") == 2
    assert kinds.count("private") == 1
    assert kinds.count("generated") == 3
    assert len(tests) == 6

    # every pair is zipped 1:1 from the parallel input/output lists
    for t in tests:
        tag = {"public": "pub", "private": "priv", "generated": "gen"}[t.kind]
        assert t.input.startswith(f"{tag}-in-")
        assert t.output.startswith(f"{tag}-out-")
        assert t.input.split("-in-")[1] == t.output.split("-out-")[1]


def test_flatten_tests_truncates_misaligned_block():
    row = make_raw_row(n_public=0, n_private=0, n_generated=0)
    row["public_tests"] = {"input": ["a", "b", "c"], "output": ["1"]}
    tests = flatten_tests(row)
    # a dangling input with no expected output is unscorable → dropped
    assert len(tests) == 1
    assert tests[0].input == "a" and tests[0].output == "1"


def test_flatten_tests_drops_unscorable_pairs():
    """A test with no expected output would pay full reward for printing nothing."""
    row = make_raw_row(n_public=0, n_private=0, n_generated=0)
    row["public_tests"] = {
        "input": ["1\n", "", "2\n", "3\n"],
        "output": ["ok\n", "", "", "   \n"],
    }
    tests = flatten_tests(row)

    # kept: real pair, and the empty-INPUT pair (problems that take no stdin)
    assert len(tests) == 1
    assert tests[0].input == "1\n" and tests[0].output == "ok\n"

    assert is_scorable(TestCase("1\n", "ok\n", "public"))
    assert is_scorable(TestCase("", "answer\n", "public"))  # no-stdin problem
    assert not is_scorable(TestCase("1\n", "", "public"))  # scraper lost output
    assert not is_scorable(TestCase("", "", "public"))
    assert not is_scorable(TestCase("1\n", "  \n", "public"))  # whitespace only


def test_problem_with_only_unscorable_tests_is_dropped():
    """Regression: p02197_Twins survived `no_tests` with one empty-pair test."""
    row = make_raw_row(n_public=0, n_private=0, n_generated=0)
    row["public_tests"] = {"input": [""], "output": [""]}
    problem, reason = row_to_problem(row, "train")
    assert problem is None and reason == "no_tests"


def test_extract_python_solutions_filters_language():
    row = make_raw_row(languages=(1, 2, 3, 4, 3))  # PY2, CPP, PY3, JAVA, PY3
    refs = extract_python_solutions(row)
    assert len(refs) == 2
    assert all("lang=3" in r for r in refs)

    assert extract_python_solutions(make_raw_row(languages=(1, 2, 4))) == []
    assert extract_python_solutions({}) == []


def test_filter_drops_untrainable():
    # a valid row survives
    problem, reason = row_to_problem(make_raw_row(), "train")
    assert reason is None and problem is not None
    assert problem.tests and problem.reference_solutions

    # zero tests
    _, reason = row_to_problem(make_raw_row(n_public=0, n_private=0, n_generated=0), "train")
    assert reason == "no_tests"

    # zero Python references
    _, reason = row_to_problem(make_raw_row(languages=(1, 2, 4)), "train")
    assert reason == "no_python_reference"

    # statement over the token budget
    long_row = make_raw_row(description="word " * 5000)
    _, reason = row_to_problem(long_row, "train", max_stmt_tokens=64)
    assert reason == "statement_too_long"

    # interactive / special-judge / file-io are not string-comparable
    _, reason = row_to_problem(make_raw_row(cf_tags=("interactive",)), "train")
    assert reason == "interactive"
    _, reason = row_to_problem(
        make_raw_row(description="This is an interactive problem. Flush the output."), "train"
    )
    assert reason == "interactive"
    _, reason = row_to_problem(
        make_raw_row(description="Find a path. If there are multiple answers, print any of them."),
        "train",
    )
    assert reason == "special_judge"
    _, reason = row_to_problem(make_raw_row(input_file="input.txt"), "train")
    assert reason == "file_io"


def test_max_refs_caps_reference_solutions():
    row = make_raw_row(languages=(3,) * 40)
    problem, _ = row_to_problem(row, "train")
    assert problem is not None and len(problem.reference_solutions) == 40  # 0 = keep all

    problem, _ = row_to_problem(row, "train", max_refs=5)
    assert problem is not None and len(problem.reference_solutions) == 5

    # capping must never turn a trainable row into a dropped one
    problem, reason = row_to_problem(make_raw_row(languages=(3,)), "train", max_refs=5)
    assert reason is None and problem is not None and len(problem.reference_solutions) == 1


def test_extract_time_limit():
    # carried through so doc 2's sandbox can set a per-problem timeout
    assert extract_time_limit({"time_limit": {"seconds": 2, "nanos": 500_000_000}}) == 2.5
    assert extract_time_limit({"time_limit": None}) == 0.0
    assert extract_time_limit({}) == 0.0

    problem, _ = row_to_problem(make_raw_row(time_limit_seconds=3), "train")
    assert problem is not None and problem.time_limit_s == 3.0


def test_unsupported_reason_keeps_ordinary_problems():
    assert unsupported_reason(make_raw_row()) is None
    # float-tolerance problems are kept — doc 2's comparators handle them
    assert unsupported_reason(make_raw_row(description="Answer with absolute error 1e-6.")) is None


def test_cap_generated_is_deterministic():
    tests = (
        [TestCase(f"pi{i}", f"po{i}", "public") for i in range(3)]
        + [TestCase(f"ri{i}", f"ro{i}", "private") for i in range(4)]
        + [TestCase(f"gi{i}", f"go{i}", "generated") for i in range(200)]
    )

    a = cap_generated(tests, max_generated=15, seed=0, key="p1")
    b = cap_generated(tests, max_generated=15, seed=0, key="p1")

    # capped at K, and public + private always survive in full
    assert len([t for t in a if t.kind == "generated"]) == 15
    assert len([t for t in a if t.kind == "public"]) == 3
    assert len([t for t in a if t.kind == "private"]) == 4
    assert len(a) == 22

    # same seed → identical subsample
    assert a == b
    # different seed (or different problem) → a different subsample
    assert cap_generated(tests, max_generated=15, seed=1, key="p1") != a
    assert cap_generated(tests, max_generated=15, seed=0, key="p2") != a

    # under the cap, nothing is dropped
    few = tests[:7] + [TestCase("g", "g", "generated")]
    assert len(cap_generated(few, max_generated=15, seed=0, key="p1")) == len(few)


def test_difficulty_bucketing():
    # cf_rating is the authoritative signal; check both sides of each boundary
    assert difficulty_bucket(0, cf_rating=800) == "easy"
    assert difficulty_bucket(0, cf_rating=1199) == "easy"
    assert difficulty_bucket(0, cf_rating=1200) == "medium"
    assert difficulty_bucket(0, cf_rating=1799) == "medium"
    assert difficulty_bucket(0, cf_rating=1800) == "hard"
    assert difficulty_bucket(0, cf_rating=3500) == "hard"

    # fall back to the difficulty enum when cf_rating is missing
    assert difficulty_bucket(1) == "easy"  # EASY
    assert difficulty_bucket(2) == "medium"  # MEDIUM
    assert difficulty_bucket(3) == "hard"  # HARD
    assert difficulty_bucket(5) == "hard"  # HARDEST

    # 7+ encodes the Codeforces letter: 7=A, 8=B, 9=C, 10=D, 11=E
    assert difficulty_bucket(7) == "easy"  # A
    assert difficulty_bucket(8) == "easy"  # B
    assert difficulty_bucket(9) == "medium"  # C
    assert difficulty_bucket(10) == "medium"  # D
    assert difficulty_bucket(11) == "hard"  # E

    # no signal at all
    assert difficulty_bucket(0) == "unknown"
    assert difficulty_bucket(6) == "unknown"  # EXTERNAL

    # cf_rating overrides the enum
    assert difficulty_bucket(1, cf_rating=2500) == "hard"


def test_ingest_reports_kept_and_dropped(caplog):
    rows = [
        make_raw_row(name=f"ok {i}", cf_rating=900) for i in range(3)
    ] + [
        make_raw_row(name="bad-tests", n_public=0, n_private=0, n_generated=0),
        make_raw_row(name="bad-lang", languages=(2, 4)),
        make_raw_row(name="bad-interactive", cf_tags=("interactive",)),
        make_raw_row(name="ok 0"),  # duplicate id
    ]
    problems, stats = ingest_with_stats("train", rows=rows)

    assert len(problems) == 3
    assert stats.total_rows == 7
    assert stats.kept == 3
    assert stats.dropped["no_tests"] == 1
    assert stats.dropped["no_python_reference"] == 1
    assert stats.dropped["interactive"] == 1
    assert stats.dropped["duplicate_id"] == 1
    assert stats.buckets["easy"] == 3
    assert all(p.split == "train" for p in problems)
    assert "dropped by reason" in stats.render()


def test_slugify_is_stable_and_filesystem_safe():
    assert slugify("1575_A. Another Sorting Problem") == "1575_a-another-sorting-problem"
    assert slugify("1575_A. Another Sorting Problem") == slugify("1575_A.  Another  Sorting Problem")
    assert slugify("") == "unnamed"


def test_estimate_tokens_is_monotonic_and_tokenizer_aware():
    assert estimate_tokens("") == 0
    assert estimate_tokens("a" * 400) == 100
    assert estimate_tokens("x" * 100) < estimate_tokens("x" * 1000)

    class Tok:
        def encode(self, text):
            return text.split()

    assert estimate_tokens("one two three", Tok()) == 3


def test_save_load_roundtrip(tmp_path):
    problems = ingest("train", rows=[make_raw_row(name=f"p {i}") for i in range(5)])
    assert problems

    loaded = load_cached(save(problems, tmp_path / "train"))

    assert loaded == problems
    for original, restored in zip(problems, loaded):
        assert restored.to_dict() == original.to_dict()
        assert all(isinstance(t, TestCase) for t in restored.tests)
        assert isinstance(restored.difficulty, int)


def test_load_cached_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no cache at"):
        load_cached(tmp_path / "nope")


# ==========================================================================
# Sprint 1.3 — prompt formatting
# ==========================================================================


@pytest.fixture
def one_problem():
    return Problem(
        id="p1",
        statement="Read n from stdin and print n+1.",
        tests=[TestCase("1\n", "2\n", "public")],
        difficulty=7,
        reference_solutions=["print(int(input()) + 1)"],
    )


def test_prompt_includes_statement(one_problem, fake_tokenizer):
    prompt = to_chat_prompt(one_problem, fake_tokenizer)
    assert one_problem.statement in prompt


def test_prompt_passes_system_and_user_roles(one_problem, fake_tokenizer):
    to_chat_prompt(one_problem, fake_tokenizer)

    assert len(fake_tokenizer.calls) == 1
    messages = fake_tokenizer.calls[0]["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert messages[1]["content"] == one_problem.statement
    # the stdin/stdout contract must be stated — models default to functions
    assert "stdin" in messages[0]["content"] and "stdout" in messages[0]["content"]


def test_prompt_requests_add_generation_prompt(one_problem, fake_tokenizer):
    to_chat_prompt(one_problem, fake_tokenizer)
    call = fake_tokenizer.calls[0]
    assert call["add_generation_prompt"] is True
    assert call["tokenize"] is False


# ==========================================================================
# Sprint 1.4 — data-quality invariants over a cached split
# ==========================================================================


def test_every_problem_has_tests_and_ref(cached_split):
    assert cached_split, "cached split is empty"
    for p in cached_split:
        assert len(p.tests) >= 1, f"{p.id} has no tests"
        assert len(p.reference_solutions) >= 1, f"{p.id} has no Python reference"
        assert all(isinstance(s, str) and s.strip() for s in p.reference_solutions)


def test_test_io_lengths_aligned(cached_split):
    for p in cached_split:
        for i, t in enumerate(p.tests):
            assert isinstance(t.input, str), f"{p.id}[{i}] input is not a str"
            assert isinstance(t.output, str), f"{p.id}[{i}] output is not a str"
            # Every test must be able to fail a wrong program. An empty input is
            # fine (no-stdin problems); an empty *output* is not — it would pay
            # full reward for printing nothing.
            assert t.output.strip() != "", f"{p.id}[{i}] has no expected output"
            assert t.kind in ("public", "private", "generated")


def test_no_train_test_leak(train_problems, test_problems):
    train_ids = {p.id for p in train_problems}
    test_ids = {p.id for p in test_problems}
    overlap = train_ids & test_ids
    assert not overlap, f"{len(overlap)} test ids leaked into train: {sorted(overlap)[:5]}"


def test_statements_within_token_budget(cached_split):
    from posttrain.data.ingest import DEFAULT_MAX_STMT_TOKENS

    for p in cached_split:
        assert p.statement.strip(), f"{p.id} has an empty statement"
        assert estimate_tokens(p.statement) <= DEFAULT_MAX_STMT_TOKENS, f"{p.id} statement too long"


def test_generated_tests_respect_cap(cached_split):
    from posttrain.data.ingest import DEFAULT_MAX_GENERATED

    for p in cached_split:
        generated = [t for t in p.tests if t.kind == "generated"]
        assert len(generated) <= DEFAULT_MAX_GENERATED, f"{p.id} exceeds the generated cap"


def test_buckets_are_valid(cached_split):
    from posttrain.data.schema import BUCKETS

    assert {p.bucket for p in cached_split} <= set(BUCKETS)


def test_prompt_roundtrips_through_tokenizer(cached_split, fake_tokenizer):
    for p in cached_split[:25]:
        prompt = to_chat_prompt(p, fake_tokenizer)
        assert isinstance(prompt, str) and prompt
        assert p.statement in prompt
        assert prompt.endswith("<|im_start|>assistant\n")
