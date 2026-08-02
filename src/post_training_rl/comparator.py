"""Decides whether a program's output counts as matching the expected output.

Reimplements `OutputsMatch` from DeepMind's own CodeContests evaluator (ADR-0007). Exact
string comparison is the obvious implementation and is wrong: it scores correct solutions as
failures over trailing whitespace, line breaks, capitalisation, and float formatting. Those
failures are indistinguishable from genuine wrong answers, so the error is silent and would
corrupt every reward in the system.

Do not substitute LiveCodeBench's comparator, which is case-sensitive, line-strict, and uses
exact `Decimal` comparison. Under its semantics DeepMind's own gold solutions score zero.
"""

def outputs_match(actual: str, expected: str, absolute_tolerance: float) -> bool:
    """Compare two program outputs under CodeContests semantics.

    `absolute_tolerance` is absolute, never relative — a difference of that size is accepted
    however large the operands are. It comes from `comparator.absolute_float_tolerance` in
    config rather than a literal here, because it decides what counts as a correct answer
    and two runs that graded differently must be diffable as files.

    Two sharp edges ride on it, both inherited from the reference implementation and
    preserved deliberately. They are separate mechanisms and behavior.md §1.8-1.9 lists
    them together, which is easy to misread:

    1. The tolerance being absolute makes it *lax at small magnitudes* — at operands near
       1e-9 it accepts a relative difference of ten thousand fold. At large magnitudes an
       absolute tolerance is, if anything, strict.
    2. The *large*-magnitude danger is a different thing: the reference parses integers as
       32-bit, so values beyond that range reach the float path, and float64 cannot
       represent them exactly — two distinct very large integers compare equal. That is
       precision loss, not tolerance.
    """
    actual_tokens = actual.split()
    expected_tokens = expected.split()
    if len(actual_tokens) != len(expected_tokens):
        return False
    return all(
        _tokens_match(a, e, absolute_tolerance)
        for a, e in zip(actual_tokens, expected_tokens)
    )


def _tokens_match(actual: str, expected: str, absolute_tolerance: float) -> bool:
    if actual.lower() == expected.lower():
        return True
    # behavior.md §1.3 phrases the numeric path as "when *either* side parses as a float".
    # Requiring both is the same rule: if only one side is a number the pair cannot be
    # numerically equal, and the case-insensitive comparison above has already rejected it.
    actual_value = _as_float(actual)
    expected_value = _as_float(expected)
    if actual_value is None or expected_value is None:
        return False
    return abs(actual_value - expected_value) <= absolute_tolerance


def _as_float(token: str) -> float | None:
    """Parse `token` as a float, or None when it is not numeric.

    A non-numeric token is ordinary data — most program output is not a number — so this
    falls back to the caller's string comparison rather than propagating.
    """
    try:
        return float(token)
    except ValueError:
        return None
