"""Extraction behaviour, per `docs/design/behavior.md` §2 and ADR-0012.

The distinction under test throughout is that `fence` and `parsed` are two independent
facts. A flawless fence can wrap broken code and correct code can arrive unfenced.
"""

from post_training_rl.extraction import extract_python
from post_training_rl.types import Fence


def test_python_tagged_fence_extracted():
    completion = "Here is my solution:\n```python\nprint(1)\n```\nDone."
    result = extract_python(completion)
    assert result.code == "print(1)"
    assert result.fence is Fence.TAGGED
    assert result.parsed is True


def test_last_tagged_fence_wins():
    # Models routinely quote the problem's example or sketch a naive version before giving
    # the real solution, so the last block is the answer. Five of seven surveyed
    # implementations agree; Qwen's own harness is the dissenter.
    completion = "First attempt:\n```python\nprint(1)\n```\nBetter:\n```python\nprint(2)\n```"
    result = extract_python(completion)
    assert result.code == "print(2)"


def test_untagged_fence_extracted_when_no_tagged():
    completion = "```\nprint(1)\n```"
    result = extract_python(completion)
    assert result.code == "print(1)"
    assert result.fence is Fence.UNTAGGED


def test_tagged_fence_preferred_over_untagged():
    # Tier order beats document order: the tagged block wins even though it comes first.
    completion = "```python\nprint(1)\n```\nand\n```\nprint(2)\n```"
    result = extract_python(completion)
    assert result.code == "print(1)"
    assert result.fence is Fence.TAGGED


def test_language_tag_aliases_accepted():
    for tag in ("py", "python3", "py3"):
        result = extract_python(f"```{tag}\nprint(1)\n```")
        assert result.fence is Fence.TAGGED, tag
        assert result.code == "print(1)", tag


def test_unparseable_code_reports_parsed_false_with_its_fence():
    # The conflation this design exists to prevent. An earlier draft let a python-tagged
    # block containing a syntax error fall through the cascade and be recorded as
    # "any_invalid", fusing two unrelated failures: the model cannot format its output, and
    # the model cannot write valid Python. They call for opposite responses — a prompt
    # change versus more training — so the fence must survive the parse failure.
    completion = "```python\ndef broken(:\n```"
    result = extract_python(completion)
    assert result.fence is Fence.TAGGED
    assert result.parsed is False
    assert result.code == "def broken(:"


def test_syntax_gate_prefers_earlier_valid_block():
    # Last-block-wins yields to the syntax gate: the last candidate does not parse, so the
    # earlier one that does is returned instead.
    completion = "```python\nprint(1)\n```\nOops:\n```python\ndef broken(:\n```"
    result = extract_python(completion)
    assert result.code == "print(1)"
    assert result.parsed is True


def test_non_python_tag_reports_other_tag():
    # Fenced, but not tagged as Python. The cascade still reaches it when no better-tagged
    # block exists, and the fence it arrived in is recorded rather than discarded.
    completion = "```cpp\nprint(1)\n```"
    result = extract_python(completion)
    assert result.fence is Fence.OTHER_TAG
    assert result.code == "print(1)"


def test_unterminated_fence_extracts_to_end():
    # What a truncated generation looks like: the model hit the token limit mid-block.
    completion = "```python\nprint(1)\nprint(2)"
    result = extract_python(completion)
    assert result.code == "print(1)\nprint(2)"
    assert result.fence is Fence.TAGGED


def test_trailing_space_after_fence_marker_tolerated():
    result = extract_python("```python   \nprint(1)\n```")
    assert result.code == "print(1)"
    assert result.fence is Fence.TAGGED


def test_indented_closing_fence_tolerated():
    result = extract_python("```python\nprint(1)\n    ```")
    assert result.code == "print(1)"
    assert result.fence is Fence.TAGGED


def test_bare_valid_python_without_fence():
    result = extract_python("print(1)")
    assert result.code == "print(1)"
    assert result.fence is Fence.NONE
    assert result.parsed is True


def test_prose_without_fence_reports_no_code():
    # Prose must fail the syntax gate. Returning the whole completion unguarded is a known
    # bug in one reference implementation, which ships prose to the interpreter.
    result = extract_python("I would solve this with a greedy algorithm.")
    assert result.code is None
    assert result.fence is Fence.NONE


def test_empty_completion_reports_no_code():
    result = extract_python("")
    assert result.code is None
    assert result.fence is Fence.NONE
    assert result.parsed is False


def test_prefill_is_reprepended():
    # The completion opens mid-block because the prompt ended inside one.
    result = extract_python("print(1)\n```", prefill="```python\n")
    assert result.code == "print(1)"
    assert result.fence is Fence.TAGGED
    assert result.parsed is True


def test_missing_prefill_reports_no_code():
    # Pins the documented silent-zero trap: the same completion with the prefill forgotten
    # recovers nothing at all, rather than returning plausible-looking partial code. The
    # prefill value has two consumers — the dataset builder that renders the prompt and the
    # reward path that re-prepends before extraction — and they must share one config key.
    result = extract_python("print(1)\n```")
    assert result.code is None
    assert result.fence is Fence.NONE
