"""Decides whether a program's output counts as matching the expected output.

Reimplements `OutputsMatch` from DeepMind's own CodeContests evaluator (ADR-0007). Exact
string comparison is the obvious implementation and is wrong: it scores correct solutions as
failures over trailing whitespace, line breaks, capitalisation, and float formatting. Those
failures are indistinguishable from genuine wrong answers, so the error is silent and would
corrupt every reward in the system.

Do not substitute LiveCodeBench's comparator, which is case-sensitive, line-strict, and uses
exact `Decimal` comparison. Under its semantics DeepMind's own gold solutions score zero.
"""

# Absolute, not relative — a difference of 1e-5 is accepted however large the operands are.
# Inherited from the reference implementation and preserved deliberately. It is lax at small
# magnitudes, and it combines with the second sharp edge below: the reference parses integers
# as 32-bit, so values beyond that range reach the float path and lose precision, which makes
# two distinct very large integers compare equal.
FLOAT_TOLERANCE = 1e-5


def outputs_match(actual: str, expected: str) -> bool:
    """Compare two program outputs under CodeContests semantics."""
    actual_tokens = actual.split()
    expected_tokens = expected.split()
    if len(actual_tokens) != len(expected_tokens):
        return False
    return all(_tokens_match(a, e) for a, e in zip(actual_tokens, expected_tokens))


def _tokens_match(actual: str, expected: str) -> bool:
    if actual.lower() == expected.lower():
        return True
    actual_value = _as_float(actual)
    expected_value = _as_float(expected)
    if actual_value is None or expected_value is None:
        return False
    return abs(actual_value - expected_value) <= FLOAT_TOLERANCE


def _as_float(token: str) -> float | None:
    try:
        return float(token)
    except ValueError:
        return None
