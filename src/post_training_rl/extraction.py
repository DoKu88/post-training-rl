"""Recovers executable Python from a completion, per ADR-0012.

Records **two independent facts**: which fence the code arrived in, and whether that code
parses. They are never collapsed. A flawless fence can wrap broken code and correct code can
arrive unfenced, and the two failures call for opposite responses — the first argues for a
prompt change or assistant prefill, the second argues for more training and nothing else.

No published source reports a code parse-failure rate for any model on any benchmark,
because every surveyed harness fuses these two into one bucket. Keeping them apart is what
makes that measurement possible.
"""

import ast
import re

from post_training_rl.types import Extraction, Fence

_PYTHON_TAGS = frozenset({"python", "py", "python3", "py3"})

# Tolerates trailing whitespace after the opening marker and an indented closing fence. The
# `\Z` alternative on the closing group treats an unterminated fence as running to the end
# of the completion, which is what a truncated generation looks like.
#
# The opening marker must start at column 0, which behavior.md §2.4 asks for by asking only
# for the *closing* fence to be indent-tolerant. Matching an indented opening would capture
# the block's indentation along with it, and `ast.parse` rejects a leading indent — so a
# list-nested block would be recorded as (TAGGED, parsed=False), reporting "cannot write
# valid Python" about a model that wrote valid Python. That is the exact false signal
# ADR-0012 exists to prevent, so the tolerance is not extended without dedenting to match.
_FENCE_PATTERN = re.compile(
    r"^```[ \t]*(?P<tag>[A-Za-z0-9_+#.-]*)[ \t]*\r?\n"
    r"(?P<code>.*?)"
    r"(?:\r?\n[ \t]*```|\Z)",
    re.DOTALL | re.MULTILINE,
)

_FENCE_TIERS = (Fence.TAGGED, Fence.UNTAGGED, Fence.OTHER_TAG)


def extract_python(completion: str, prefill: str = "") -> Extraction:
    """Recover executable Python from a completion via a syntax-gated cascade.

    `prefill` is prepended before matching. It defaults to "" and ADR-0012 keeps prefill off,
    but the parameter exists because forgetting to re-prepend it is documented as the single
    most likely way to get this wrong and silently score zero on everything. Making it a
    parameter of the only function that could need it puts the mistake at the call site.
    """
    text = prefill + completion
    blocks = _fenced_blocks(text)

    # First pass: the first tier holding a candidate that parses wins, and within it the
    # last such candidate is the answer.
    for fence in _FENCE_TIERS:
        parsing = [code for code in blocks.get(fence, []) if _parses(code)]
        if parsing:
            return Extraction(code=parsing[-1], fence=fence, parsed=True)

    # Bare code with no fence at all. Syntax-gated precisely so that prose fails it — never
    # fall back to returning the whole completion unguarded, which is a known bug in one
    # reference implementation that ships prose to the interpreter.
    if _is_code(text):
        return Extraction(code=text, fence=Fence.NONE, parsed=True)

    # Second pass: nothing anywhere parses. Return the last candidate from the
    # best-fenced tier that had one, so "malformed code" stays distinguishable from "no code
    # at all". The bare tier is deliberately absent here — it already failed the gate above,
    # and admitting it would return prose as code.
    for fence in _FENCE_TIERS:
        candidates = blocks.get(fence, [])
        if candidates:
            return Extraction(code=candidates[-1], fence=fence, parsed=False)

    return Extraction(code=None, fence=Fence.NONE, parsed=False)


def _fenced_blocks(text: str) -> dict[Fence, list[str]]:
    """Group every fenced block in `text` by the fence it arrived in, in document order."""
    blocks: dict[Fence, list[str]] = {}
    for match in _FENCE_PATTERN.finditer(text):
        code = match.group("code")
        # A whitespace-only block is not a candidate. `ast.parse("")` succeeds, so without
        # this an empty fence would be recorded as recovered, parsing code.
        if code.strip():
            blocks.setdefault(_classify(match.group("tag")), []).append(code)
    return blocks


def _classify(tag: str) -> Fence:
    if not tag:
        return Fence.UNTAGGED
    if tag.lower() in _PYTHON_TAGS:
        return Fence.TAGGED
    return Fence.OTHER_TAG


def _is_code(text: str) -> bool:
    return bool(text.strip()) and _parses(text)


def _parses(code: str) -> bool:
    try:
        ast.parse(code)
    except (SyntaxError, ValueError):
        # ValueError covers source containing null bytes. A completion that does not parse
        # is data, not an apparatus failure — it becomes an extraction result, never an
        # exception.
        return False
    return True
