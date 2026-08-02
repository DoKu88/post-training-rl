"""Comparator behaviour, per `docs/design/behavior.md` §1 and ADR-0007.

Expected values come from CodeContests semantics, never from re-deriving what the
implementation happens to do.
"""

from pathlib import Path

from post_training_rl.comparator import outputs_match
from post_training_rl.config import load_verifier_config

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "verifier.yaml"
TOLERANCE = load_verifier_config(_CONFIG_PATH).comparator.absolute_float_tolerance


def test_identical_output_matches():
    assert outputs_match("3", "3", TOLERANCE)


def test_trailing_newline_ignored():
    assert outputs_match("3\n", "3", TOLERANCE)


def test_line_structure_ignored():
    # Tokens, not lines — the comparator splits on any whitespace, so how the program chose
    # to break its output across lines cannot make a correct answer wrong.
    assert outputs_match("1 2\n3", "1\n2 3", TOLERANCE)


def test_comparison_is_case_insensitive():
    assert outputs_match("YES", "yes", TOLERANCE)


def test_extra_token_fails():
    assert not outputs_match("1 2 3", "1 2", TOLERANCE)


def test_short_output_does_not_match_long_expected():
    # Pins the zip-truncation bug other implementations have: zipping the two token lists
    # without a length check silently compares only the shorter prefix, so a program that
    # prints the first answer and stops scores as correct.
    assert not outputs_match("1", "1 2 3", TOLERANCE)


def test_float_within_tolerance_matches():
    assert outputs_match("1.000001", "1.0", TOLERANCE)


def test_float_outside_tolerance_fails():
    assert not outputs_match("1.001", "1.0", TOLERANCE)


def test_integer_and_float_forms_match():
    assert outputs_match("1", "1.0", TOLERANCE)


def test_large_magnitude_uses_absolute_tolerance():
    # Pins the sharp edge inherited from the reference implementation: the tolerance is
    # absolute, so a difference of 1.0 is rejected however large the operands are. A
    # relative tolerance — math.isclose's default rel_tol=1e-9, say — would accept this
    # pair, so swapping one in makes this test fail rather than silently changing grading.
    assert not outputs_match("1000000000.0", "1000000001.0", TOLERANCE)


def test_empty_matches_empty():
    assert outputs_match("", "", TOLERANCE)


def test_empty_does_not_match_nonempty():
    assert not outputs_match("", "1", TOLERANCE)
