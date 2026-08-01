#!/usr/bin/env python
"""Human-readable inspection of a cached split — the manual gate for doc 1.

Sprint 1.4. Prints N random problems (statement + a reference solution + first
test) plus a per-difficulty-bucket count table.

**Do not proceed to doc 2 until the reference solutions here look real** — doc
2's sandbox test asserts that a correct reference scores exactly 1.0, so a
split full of junk references would silently invalidate that check.

    python scripts/inspect_data.py --split train --n 5
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

# Allow running straight from a checkout without `pip install -e .`
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from posttrain.data.ingest import (  # noqa: E402
    DEFAULT_CACHE_ROOT,
    load_cached,
    load_meta,
    normalize_split,
)
from posttrain.data.schema import Problem  # noqa: E402

RULE = "=" * 78
THIN = "-" * 78


def _clip(text: str, limit: int) -> str:
    text = text.rstrip()
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"\n... [{len(text) - limit} more chars]"


def _show_problem(p: Problem, *, stmt_chars: int, sol_chars: int) -> None:
    print(RULE)
    print(f"id         : {p.id}")
    print(
        f"split      : {p.split or '?'}   bucket: {p.bucket}   "
        f"difficulty: {p.difficulty}   cf_rating: {p.cf_rating or '-'}"
    )
    print(
        f"tests      : {len(p.tests)} total  "
        f"({len(p.public())} public / {len(p.hidden())} hidden)   "
        f"python refs: {len(p.reference_solutions)}"
    )
    print(THIN)
    print("STATEMENT")
    print(_clip(p.statement, stmt_chars))
    print(THIN)
    print("REFERENCE SOLUTION [0]")
    print(_clip(p.reference_solutions[0], sol_chars) if p.reference_solutions else "(none!)")
    print(THIN)
    if p.tests:
        t = p.tests[0]
        print(f"FIRST TEST (kind={t.kind})")
        print("  stdin  : " + repr(_clip(t.input, 300)))
        print("  stdout : " + repr(_clip(t.output, 300)))
    else:
        print("FIRST TEST: (none!)")
    print()


def _bucket_table(problems: list[Problem]) -> None:
    buckets = Counter(p.bucket for p in problems)
    tests = Counter()
    refs = Counter()
    for p in problems:
        tests[p.bucket] += len(p.tests)
        refs[p.bucket] += len(p.reference_solutions)

    print(RULE)
    print("TRAINABLE PROBLEMS PER DIFFICULTY BUCKET")
    print(THIN)
    print(f"{'bucket':<10}{'problems':>10}{'%':>8}{'avg tests':>12}{'avg refs':>11}")
    total = len(problems)
    for bucket in ("easy", "medium", "hard", "unknown"):
        n = buckets.get(bucket, 0)
        if not n:
            print(f"{bucket:<10}{0:>10}{0.0:>7.1f}%{'-':>12}{'-':>11}")
            continue
        print(
            f"{bucket:<10}{n:>10}{100.0 * n / total:>7.1f}%"
            f"{tests[bucket] / n:>12.1f}{refs[bucket] / n:>11.1f}"
        )
    print(THIN)
    print(
        f"{'TOTAL':<10}{total:>10}{100.0:>7.1f}%"
        f"{sum(tests.values()) / total if total else 0:>12.1f}"
        f"{sum(refs.values()) / total if total else 0:>11.1f}"
    )
    print(RULE)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train", help="train | valid | test")
    ap.add_argument("--n", type=int, default=5, help="how many random problems to print")
    ap.add_argument("--cache", default=None, help="cache dir (default: data/processed/<split>)")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed (0 = reproducible)")
    ap.add_argument("--stmt-chars", type=int, default=1500, help="statement clip length")
    ap.add_argument("--sol-chars", type=int, default=1200, help="solution clip length")
    ap.add_argument("--bucket", default=None, help="only sample from this bucket")
    args = ap.parse_args(argv)

    split = normalize_split(args.split)
    cache = Path(args.cache) if args.cache else DEFAULT_CACHE_ROOT / split

    try:
        problems = load_cached(cache)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not problems:
        print(f"error: cache at {cache} is empty", file=sys.stderr)
        return 1

    meta = load_meta(cache)
    print(f"loaded {len(problems)} problems from {cache}")
    if "stats" in meta:
        s = meta["stats"]
        print(
            f"ingest: {s.get('total_rows')} raw rows → {s.get('kept')} kept, "
            f"{s.get('dropped_total')} dropped {s.get('dropped')}"
        )
    print()

    pool = [p for p in problems if p.bucket == args.bucket] if args.bucket else problems
    if not pool:
        print(f"error: no problems in bucket {args.bucket!r}", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    for p in rng.sample(pool, min(args.n, len(pool))):
        _show_problem(p, stmt_chars=args.stmt_chars, sol_chars=args.sol_chars)

    _bucket_table(problems)
    print("\nManual gate: do the reference solutions above read as real, working")
    print("Python? If yes, doc 1 is done and doc 2's sandbox check is meaningful.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
