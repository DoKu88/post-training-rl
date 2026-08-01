"""Prompt formatting — deliberately separate from ingestion.

Sprint 1.3. Kept out of `ingest.py` so that tweaking the prompt never forces a
re-ingest of the dataset.

Model-agnostic by construction: we hand the tokenizer a plain message list and
let *its own* chat template decide the wire format. Qwen, Llama, Mistral, etc.
each render their own special tokens, so swapping the model in config (doc 4)
needs no change here.
"""

from __future__ import annotations

from typing import Any, Protocol

from posttrain.data.schema import Problem

__all__ = ["SYSTEM_PROMPT", "build_messages", "to_chat_prompt"]

#: CodeContests is stdin → stdout, so the system prompt has to say so
#: explicitly; models default to writing a function otherwise. The fenced-block
#: instruction is what doc 2's code extractor keys off.
SYSTEM_PROMPT = (
    "You are an expert competitive programmer. Read input from stdin, "
    "write the answer to stdout. Put your final solution in a single "
    "```python code block."
)


class _ChatTokenizer(Protocol):
    def apply_chat_template(
        self, messages: list[dict[str, str]], /, *, tokenize: bool, add_generation_prompt: bool
    ) -> Any: ...


def build_messages(p: Problem, system: str = SYSTEM_PROMPT) -> list[dict[str, str]]:
    """The model-independent half: a plain system + user message list."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": p.statement},
    ]


def to_chat_prompt(p: Problem, tokenizer: _ChatTokenizer, system: str = SYSTEM_PROMPT) -> str:
    """Render `p` into the target model's chat format, ready for generation."""
    return tokenizer.apply_chat_template(
        build_messages(p, system),
        tokenize=False,
        add_generation_prompt=True,
    )
