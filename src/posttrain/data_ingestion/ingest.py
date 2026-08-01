"""CodeContests ingestion — raw HuggingFace rows → cached `Problem` records.

Sprint 1.2. No GPU, no model, no tokenizer required.

Pipeline, in order (see docs/01-data-ingestion.md):

  1. flatten public/private/generated tests into one tagged list
  2. keep only PYTHON3 reference solutions (ground truth for doc 2's sandbox)
  3. drop untrainable rows (no tests / no refs / too long / not stdin-stdout)
  4. cap generated tests at K with a fixed, per-problem-deterministic seed
  5. attach a coarse easy/medium/hard bucket
  6. keep HF's train/valid/test boundaries — the eval harness only ever sees test
  7. cache to jsonl so re-ingestion is a load, not a re-download

Every per-step helper is importable on its own so it can be unit-tested against
synthetic rows with no network access.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol

from posttrain.data_ingestion.schema import Problem, TestCase

LOG = logging.getLogger("posttrain.data_ingestion.ingest")

__all__ = [
    "ingest",
    "ingest_with_stats",
    "save",
    "load_cached",
    "cache_exists",
    "slugify",
    "clean_statement",
    "estimate_tokens",
    "flatten_tests",
    "is_scorable",
    "extract_python_solutions",
    "extract_time_limit",
    "unsupported_reason",
    "cap_generated",
    "difficulty_bucket",
    "row_to_problem",
    "IngestStats",
]

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------

HF_DATASET = "deepmind/code_contests"

#: `solutions.language` enum from the CodeContests proto. We keep PYTHON3 only:
#: PYTHON (1) is Python 2 and will not run under our sandbox's interpreter.
LANG_PYTHON3 = 3

VALID_SPLITS = ("train", "valid", "test")
_SPLIT_ALIASES = {"validation": "valid", "val": "valid", "dev": "valid"}

DEFAULT_CACHE_ROOT = Path("data/processed")
DEFAULT_MAX_GENERATED = 15
DEFAULT_MAX_STMT_TOKENS = 2048
#: 0 = keep every Python reference. Popular problems carry 100+ of them, which
#: dominates the cache size; doc 2 only needs a handful to verify the sandbox.
DEFAULT_MAX_REFS = 0

#: Codeforces rating → bucket. cf_rating is the only *real* numeric difficulty
#: signal in this dataset (see `difficulty_bucket` for why `difficulty` is not).
CF_EASY_MAX = 1199  # div2 A/B
CF_MEDIUM_MAX = 1799  # div2 C/D

#: `difficulty` enum from the proto. 1-5 are the human bands; 7+ are the
#: Codeforces problem *index* (7=A, 8=B, 9=C, ...), which is ordinal within a
#: contest but is emphatically not a rating.
_DIFF_UNKNOWN = 0
_DIFF_EASY, _DIFF_MEDIUM, _DIFF_HARD, _DIFF_HARDER, _DIFF_HARDEST = 1, 2, 3, 4, 5
_DIFF_EXTERNAL = 6
_DIFF_INDEX_A = 7  # 7=A .. 28=V

#: Problems whose correctness cannot be decided by comparing one stdout string
#: to one expected string. These poison a simple exact-match reward, so they are
#: dropped rather than silently scored as failures.
_INTERACTIVE_TAGS = {"interactive"}
_INTERACTIVE_PATTERNS = re.compile(
    r"this is an interactive problem"
    r"|interaction protocol"
    r"|\binteractor\b"
    r"|flush (?:the )?output"
    r"|fflush\(stdout\)"
    r"|sys\.stdout\.flush",
    re.IGNORECASE,
)
#: "any valid answer accepted" → a checker, not string equality.
_SPECIAL_JUDGE_PATTERNS = re.compile(
    r"if (?:there are |there exist |)(?:multiple|several|many) (?:possible |valid |correct |)"
    r"(?:answers|solutions|ways|outputs)"
    r"|print any(?:one)? of them"
    r"|output any(?:one)? of them"
    r"|print any (?:one |valid |such |)"
    r"|output any (?:one |valid |such |)"
    r"|you may print any"
    r"|any of them (?:will be |may be |)(?:accepted|considered correct)",
    re.IGNORECASE,
)

_WS_RUN = re.compile(r"[ \t]+$", re.MULTILINE)
_BLANK_RUN = re.compile(r"\n{3,}")
# Underscores are kept: CodeContests names encode the contest as "1575_A. ...".
_SLUG_STRIP = re.compile(r"[^a-z0-9_]+")


class _Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...


# --------------------------------------------------------------------------
# stats
# --------------------------------------------------------------------------


@dataclass
class IngestStats:
    """Kept-vs-dropped accounting, logged and written next to the cache."""

    split: str = ""
    total_rows: int = 0
    kept: int = 0
    dropped: Counter = field(default_factory=Counter)  # reason -> count
    buckets: Counter = field(default_factory=Counter)  # bucket -> count
    tests_kept: int = 0
    generated_dropped: int = 0  # generated tests removed by the cap
    io_misaligned: int = 0  # input/output list length mismatches seen
    unscorable_tests: int = 0  # tests dropped for having no expected output

    @property
    def dropped_total(self) -> int:
        return sum(self.dropped.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "total_rows": self.total_rows,
            "kept": self.kept,
            "dropped_total": self.dropped_total,
            "dropped": dict(self.dropped),
            "buckets": dict(self.buckets),
            "tests_kept": self.tests_kept,
            "generated_dropped": self.generated_dropped,
            "io_misaligned": self.io_misaligned,
            "unscorable_tests": self.unscorable_tests,
        }

    def render(self) -> str:
        pct = (100.0 * self.kept / self.total_rows) if self.total_rows else 0.0
        lines = [
            f"split={self.split!r}  rows={self.total_rows}  "
            f"kept={self.kept} ({pct:.1f}%)  dropped={self.dropped_total}",
            "  dropped by reason:",
        ]
        if self.dropped:
            for reason, n in self.dropped.most_common():
                lines.append(f"    {reason:<24} {n:>7}")
        else:
            lines.append("    (none)")
        lines.append("  kept by difficulty bucket:")
        for bucket in ("easy", "medium", "hard", "unknown"):
            lines.append(f"    {bucket:<24} {self.buckets.get(bucket, 0):>7}")
        avg = (self.tests_kept / self.kept) if self.kept else 0.0
        lines.append(
            f"  tests kept={self.tests_kept} (avg {avg:.1f}/problem); "
            f"generated dropped by cap={self.generated_dropped}; "
            f"misaligned io pairs={self.io_misaligned}; "
            f"unscorable (no expected output)={self.unscorable_tests}"
        )
        return "\n".join(lines)


# --------------------------------------------------------------------------
# step helpers (all pure, all unit-testable against synthetic rows)
# --------------------------------------------------------------------------


def slugify(name: str) -> str:
    """`"1575_A. Another Sorting Problem"` → `"1575_a-another-sorting-problem"`.

    Stable across runs so cached ids can be compared between splits (leak check)
    and across re-ingestion.
    """
    norm = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    slug = _SLUG_STRIP.sub("-", norm.lower()).strip("-_")
    return slug or "unnamed"


def clean_statement(description: str) -> str:
    """Light normalization only — never reword or truncate the statement."""
    text = (description or "").replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RUN.sub("", text)
    text = _BLANK_RUN.sub("\n\n", text)
    return text.strip()


def estimate_tokens(text: str, tokenizer: _Tokenizer | None = None) -> int:
    """Token count for the statement-length filter.

    Model-free by default (~4 chars/token, the usual BPE rule of thumb) so
    ingestion never needs a tokenizer download. Pass a real `tokenizer` when an
    exact count for a specific model matters.
    """
    if tokenizer is not None:
        return len(tokenizer.encode(text))
    return (len(text) + 3) // 4


def _pairs(block: Any, kind: str, stats: IngestStats | None = None) -> list[TestCase]:
    """Zip one `{input: [...], output: [...]}` block into tagged `TestCase`s.

    Drops unscorable pairs — see `is_scorable`.
    """
    if not block:
        return []
    inputs = block.get("input") or []
    outputs = block.get("output") or []
    if len(inputs) != len(outputs):
        # Truncate to the aligned prefix rather than emitting dangling sides —
        # a test with no expected output is unscorable.
        if stats is not None:
            stats.io_misaligned += abs(len(inputs) - len(outputs))
    n = min(len(inputs), len(outputs))
    kept = []
    for i in range(n):
        test = TestCase(input=inputs[i], output=outputs[i], kind=kind)
        if not is_scorable(test):
            if stats is not None:
                stats.unscorable_tests += 1
            continue
        kept.append(test)
    return kept


def is_scorable(test: TestCase) -> bool:
    """Can this test distinguish a correct program from a broken one?

    An **empty expected output** cannot: a handful of rows in CodeContests lost
    their output during scraping (e.g. `p02197_Twins`, whose statement clearly
    specifies an answer), and keeping them would hand full reward to any program
    that prints nothing — including one that crashes or emits no code at all.
    That is a reward-hacking hole, so these are dropped.

    An **empty input** is fine and is kept: plenty of problems state "No input
    is given" and are scored purely on what the program prints.
    """
    return test.output.strip() != ""


def flatten_tests(row: dict[str, Any], stats: IngestStats | None = None) -> list[TestCase]:
    """Merge the three test blocks into one list, tagging each with its `kind`."""
    return (
        _pairs(row.get("public_tests"), "public", stats)
        + _pairs(row.get("private_tests"), "private", stats)
        + _pairs(row.get("generated_tests"), "generated", stats)
    )


def extract_python_solutions(row: dict[str, Any]) -> list[str]:
    """Keep only PYTHON3 (`language == 3`) reference solutions.

    These are the ground truth for doc 2: a correct reference must score 1.0
    against our own sandbox, or the sandbox is wrong.
    """
    sols = row.get("solutions") or {}
    languages = sols.get("language") or []
    sources = sols.get("solution") or []
    return [
        src
        for lang, src in zip(languages, sources)
        if lang == LANG_PYTHON3 and src and src.strip()
    ]


def extract_time_limit(row: dict[str, Any]) -> float:
    """Per-test time limit in seconds; 0.0 when the source doesn't give one.

    Carried through ingestion (rather than looked up later) so doc 2's sandbox
    can set a per-problem timeout without a re-ingest.
    """
    tl = row.get("time_limit")
    if not tl:
        return 0.0
    return float(tl.get("seconds") or 0) + float(tl.get("nanos") or 0) / 1e9


def unsupported_reason(row: dict[str, Any]) -> str | None:
    """Why this problem can't be scored by exact stdout comparison, or None.

    Returns one of `"interactive"`, `"special_judge"`, `"file_io"`.

    Note: float-tolerance problems are *not* dropped — doc 2's `comparators.py`
    handles those with a float-tolerant match.
    """
    tags = row.get("cf_tags") or []
    if any(str(t).lower() in _INTERACTIVE_TAGS for t in tags):
        return "interactive"

    description = row.get("description") or ""
    if _INTERACTIVE_PATTERNS.search(description):
        return "interactive"

    # Non-empty input_file/output_file means the problem reads/writes a named
    # file instead of stdin/stdout, which our sandbox contract does not model.
    if (row.get("input_file") or "").strip() or (row.get("output_file") or "").strip():
        return "file_io"

    if _SPECIAL_JUDGE_PATTERNS.search(description):
        return "special_judge"

    return None


def cap_generated(
    tests: Iterable[TestCase],
    *,
    max_generated: int = DEFAULT_MAX_GENERATED,
    seed: int = 0,
    key: str = "",
) -> list[TestCase]:
    """Keep every public + private test; subsample generated tests to ≤ K.

    `generated_tests` can run to hundreds of cases per problem, and the reward
    pays for every one of them on every rollout. The RNG is seeded from
    `(seed, key)` — not from iteration order — so a problem's subsample is
    identical no matter how or when the split is ingested.
    """
    tests = list(tests)
    generated = [t for t in tests if t.kind == "generated"]
    kept = [t for t in tests if t.kind != "generated"]
    if len(generated) <= max_generated:
        return kept + generated
    rng = random.Random(f"{seed}:{key}")
    picked = sorted(rng.sample(range(len(generated)), max_generated))
    return kept + [generated[i] for i in picked]


def difficulty_bucket(difficulty: int, cf_rating: int = 0) -> str:
    """Coarse `easy` / `medium` / `hard` / `unknown` tag (doc 3 curriculum).

    `cf_rating` wins when present because CodeContests' `difficulty` field is an
    enum where 7+ encodes the Codeforces problem letter (7=A, 8=B, ...), not a
    magnitude. Falls back to the enum, then to `"unknown"`.
    """
    if cf_rating and cf_rating > 0:
        if cf_rating <= CF_EASY_MAX:
            return "easy"
        if cf_rating <= CF_MEDIUM_MAX:
            return "medium"
        return "hard"

    if difficulty in (_DIFF_EASY,):
        return "easy"
    if difficulty in (_DIFF_MEDIUM,):
        return "medium"
    if difficulty in (_DIFF_HARD, _DIFF_HARDER, _DIFF_HARDEST):
        return "hard"
    if difficulty >= _DIFF_INDEX_A:
        # A/B → easy, C/D → medium, E and beyond → hard.
        offset = difficulty - _DIFF_INDEX_A
        if offset <= 1:
            return "easy"
        if offset <= 3:
            return "medium"
        return "hard"
    # 0 (UNKNOWN) and 6 (EXTERNAL) carry no usable signal.
    return "unknown"


def row_to_problem(
    row: dict[str, Any],
    split: str = "",
    *,
    max_generated: int = DEFAULT_MAX_GENERATED,
    max_stmt_tokens: int = DEFAULT_MAX_STMT_TOKENS,
    max_refs: int = DEFAULT_MAX_REFS,
    seed: int = 0,
    tokenizer: _Tokenizer | None = None,
    stats: IngestStats | None = None,
) -> tuple[Problem | None, str | None]:
    """Normalize one raw row.

    Returns `(problem, None)` when the row is trainable, else `(None, reason)`.
    """
    reason = unsupported_reason(row)
    if reason is not None:
        return None, reason

    tests = flatten_tests(row, stats)
    if not tests:
        return None, "no_tests"

    refs = extract_python_solutions(row)
    if not refs:
        return None, "no_python_reference"
    if max_refs > 0:
        refs = refs[:max_refs]  # dataset order — no correctness assumption

    statement = clean_statement(row.get("description", ""))
    if not statement:
        return None, "empty_statement"
    if estimate_tokens(statement, tokenizer) > max_stmt_tokens:
        return None, "statement_too_long"

    pid = slugify(row.get("name") or "")
    capped = cap_generated(tests, max_generated=max_generated, seed=seed, key=pid)
    if stats is not None:
        stats.generated_dropped += len(tests) - len(capped)
        stats.tests_kept += len(capped)

    difficulty = int(row.get("difficulty") or 0)
    cf_rating = int(row.get("cf_rating") or 0)
    problem = Problem(
        id=pid,
        statement=statement,
        tests=capped,
        difficulty=difficulty,
        reference_solutions=refs,
        source="code_contests",
        bucket=difficulty_bucket(difficulty, cf_rating),
        split=split,
        cf_rating=cf_rating,
        time_limit_s=extract_time_limit(row),
    )
    return problem, None


# --------------------------------------------------------------------------
# ingestion
# --------------------------------------------------------------------------


def normalize_split(split: str) -> str:
    s = _SPLIT_ALIASES.get(split.lower(), split.lower())
    if s not in VALID_SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {VALID_SPLITS}")
    return s


def ingest_with_stats(
    split: str,
    *,
    max_generated: int = DEFAULT_MAX_GENERATED,
    max_stmt_tokens: int = DEFAULT_MAX_STMT_TOKENS,
    max_refs: int = DEFAULT_MAX_REFS,
    seed: int = 0,
    limit: int | None = None,
    rows: Iterable[dict[str, Any]] | None = None,
    tokenizer: _Tokenizer | None = None,
) -> tuple[list[Problem], IngestStats]:
    """`ingest`, but also returns the kept/dropped accounting.

    Pass `rows` to run the pipeline over an in-memory iterable instead of
    downloading from HuggingFace (used by the tests).
    """
    split = normalize_split(split)
    stats = IngestStats(split=split)

    if rows is None:
        from datasets import load_dataset  # imported lazily: keeps tests offline

        LOG.info("loading %s split=%s from HuggingFace ...", HF_DATASET, split)
        rows = load_dataset(HF_DATASET, split=split)

    problems: list[Problem] = []
    seen_ids: set[str] = set()
    for i, row in enumerate(rows):
        if limit is not None and i >= limit:
            break
        stats.total_rows += 1
        problem, reason = row_to_problem(
            row,
            split,
            max_generated=max_generated,
            max_stmt_tokens=max_stmt_tokens,
            seed=seed,
            tokenizer=tokenizer,
            stats=stats,
        )
        if problem is None:
            stats.dropped[reason or "unknown"] += 1
            continue
        if problem.id in seen_ids:
            stats.dropped["duplicate_id"] += 1
            continue
        seen_ids.add(problem.id)
        problems.append(problem)
        stats.kept += 1
        stats.buckets[problem.bucket] += 1

    LOG.info("ingest complete\n%s", stats.render())
    return problems, stats


def ingest(
    split: str,
    *,
    max_generated: int = DEFAULT_MAX_GENERATED,
    max_stmt_tokens: int = DEFAULT_MAX_STMT_TOKENS,
    max_refs: int = DEFAULT_MAX_REFS,
    seed: int = 0,
    limit: int | None = None,
    rows: Iterable[dict[str, Any]] | None = None,
    tokenizer: _Tokenizer | None = None,
) -> list[Problem]:
    """Raw CodeContests `split` → normalized, filtered, capped `Problem`s."""
    problems, _ = ingest_with_stats(
        split,
        max_generated=max_generated,
        max_stmt_tokens=max_stmt_tokens,
        max_refs=max_refs,
        seed=seed,
        limit=limit,
        rows=rows,
        tokenizer=tokenizer,
    )
    return problems


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------

_PROBLEMS_FILE = "problems.jsonl"
_META_FILE = "meta.json"


def _resolve_cache(path: str | Path) -> tuple[Path, Path]:
    """Accept either the cache directory or the jsonl file itself."""
    p = Path(path)
    if p.suffix == ".jsonl":
        return p.parent, p
    return p, p / _PROBLEMS_FILE


def cache_exists(path: str | Path) -> bool:
    _, jsonl = _resolve_cache(path)
    return jsonl.is_file()


def save(problems: Iterable[Problem], path: str | Path, stats: IngestStats | None = None) -> Path:
    """Write `problems` to `path` as jsonl; returns the path for round-tripping.

    jsonl (rather than arrow) keeps the cache greppable and diffable, and
    round-trips our dataclasses exactly with no feature-schema coercion.
    """
    directory, jsonl = _resolve_cache(path)
    directory.mkdir(parents=True, exist_ok=True)
    n = 0
    with jsonl.open("w", encoding="utf-8") as fh:
        for problem in problems:
            fh.write(json.dumps(problem.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    meta = {"count": n, "format": "jsonl", "schema_version": 1}
    if stats is not None:
        meta["stats"] = stats.to_dict()
    (directory / _META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    LOG.info("wrote %d problems → %s", n, jsonl)
    return directory


def load_cached(path: str | Path) -> list[Problem]:
    """Load a cached split. One file read — never a re-download."""
    _, jsonl = _resolve_cache(path)
    if not jsonl.is_file():
        raise FileNotFoundError(
            f"no cache at {jsonl}; run `python -m posttrain.data_ingestion.ingest --split <split> "
            f"--cache {Path(path)}` first"
        )
    with jsonl.open("r", encoding="utf-8") as fh:
        return [Problem.from_dict(json.loads(line)) for line in fh if line.strip()]


def load_meta(path: str | Path) -> dict[str, Any]:
    directory, _ = _resolve_cache(path)
    meta = directory / _META_FILE
    return json.loads(meta.read_text(encoding="utf-8")) if meta.is_file() else {}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m posttrain.data_ingestion.ingest",
        description="Ingest a CodeContests split into a cached set of Problem records.",
    )
    p.add_argument("--split", default="train", help="train | valid | test")
    p.add_argument(
        "--cache",
        default=None,
        help="cache directory (default: data/processed/<split>)",
    )
    p.add_argument("--max-generated", type=int, default=DEFAULT_MAX_GENERATED)
    p.add_argument("--max-stmt-tokens", type=int, default=DEFAULT_MAX_STMT_TOKENS)
    p.add_argument(
        "--max-refs",
        type=int,
        default=DEFAULT_MAX_REFS,
        help="cap Python reference solutions per problem (0 = keep all). "
        "Popular problems carry 100+; capping shrinks the cache a lot.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="only read the first N raw rows")
    p.add_argument("--force", action="store_true", help="re-ingest even if a cache exists")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )
    split = normalize_split(args.split)
    cache = Path(args.cache) if args.cache else DEFAULT_CACHE_ROOT / split

    if cache_exists(cache) and not args.force:
        problems = load_cached(cache)
        LOG.info("cache hit: loaded %d problems from %s (no download)", len(problems), cache)
        meta = load_meta(cache)
        if "stats" in meta:
            LOG.info("cached ingest stats: %s", json.dumps(meta["stats"], indent=2))
        return 0

    problems, stats = ingest_with_stats(
        split,
        max_generated=args.max_generated,
        max_stmt_tokens=args.max_stmt_tokens,
        max_refs=args.max_refs,
        seed=args.seed,
        limit=args.limit,
    )
    save(problems, cache, stats)
    LOG.info("done: %d problems cached at %s", len(problems), cache)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
