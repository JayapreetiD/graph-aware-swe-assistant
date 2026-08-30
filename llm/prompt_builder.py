

"""
llm/prompt_builder.py

Single responsibility: assemble retrieved chunks and the user's
question into a single, fixed-token-budget prompt string for the LLM.

This is the ONE place where "context budget" is enforced. Both
semantic-only and hybrid retrieval feed chunks through this same
function, so the prompt-assembly logic is retrieval-mode-agnostic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import tiktoken

logger = logging.getLogger(__name__)

CONTEXT_TOKEN_BUDGET = 4000
# Reserve some of the budget for the question, instructions, and
# formatting overhead — not just raw code. Without this, a large
# question or verbose instructions could push the total over budget
# even after chunks are trimmed to fit.
RESERVED_TOKENS_FOR_OVERHEAD = 500
CHUNK_TOKEN_BUDGET = CONTEXT_TOKEN_BUDGET - RESERVED_TOKENS_FOR_OVERHEAD

_ENCODING = tiktoken.get_encoding("cl100k_base")

SYSTEM_INSTRUCTIONS = """You are a software engineering assistant answering questions about a Python codebase.

Rules you MUST follow:
1. Answer using ONLY the code context provided below. Do not use outside knowledge of the library.
2. Every factual claim about the code must be followed by a citation in the exact format (file:start_line-end_line).
3. Only cite chunks that appear in the context below. Never invent a file path or line number.
4. If the provided context does not contain enough information to answer, say so explicitly instead of guessing.
5. Be concise and technical. Do not repeat the full code back verbatim; explain it.
"""


@dataclass
class PromptResult:
    prompt: str
    chunks_used: list[dict]
    chunks_dropped: int
    total_tokens: int


def _count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def _format_chunk(chunk: dict) -> str:
    """
    Render one chunk as a labeled block. The label is deliberately
    explicit and repeated (node_id AND file:line) so the LLM has an
    unambiguous, copy-pasteable citation target — this directly
    supports the citation-verification step later in Phase 4.
    """
    header = (
        f"### {chunk['node_id']}\n"
        f"File: {chunk['filepath']} (lines {chunk['start_line']}-{chunk['end_line']})\n"
    )
    if chunk.get("signature"):
        header += f"Signature: {chunk['signature']}\n"
    if chunk.get("docstring"):
        header += f'Docstring: "{chunk["docstring"]}"\n'
    return f"{header}\n```python\n{chunk['code']}\n```\n"


def build_prompt(question: str, chunks: list[dict]) -> PromptResult:
    """
    Build a fixed-budget prompt from ranked chunks (highest-relevance
    first — caller's ranking order is trusted and preserved).

    Truncation policy: chunks are added in the given order. If a chunk
    fits in the remaining budget, it's added whole. If a chunk does NOT
    fit, it is TRUNCATED to fit the remaining space (header + signature +
    docstring + as much of the code body as fits, marked with an
    explicit "...truncated..." notice) rather than dropped entirely.

    FIX (see project notes): the previous policy skipped any chunk whose
    full size exceeded the *entire* CHUNK_TOKEN_BUDGET, even before any
    other chunk had used any space. This made large but highly relevant
    classes (e.g. a 7934-token Field class that is real ground truth for
    several benchmark queries) structurally unreachable regardless of
    retrieval ranking. Truncating instead of skipping means an oversized,
    genuinely relevant chunk still contributes its citation-critical
    header info and as much real code as space allows, rather than
    contributing nothing.

    A chunk is only fully skipped now if there is negligible space left
    (less than MIN_TRUNCATED_CHUNK_TOKENS) to make truncation worthwhile.
    """
    context_blocks: list[str] = []
    chunks_used: list[dict] = []
    running_tokens = 0

    MIN_TRUNCATED_CHUNK_TOKENS = 150  # below this, truncation isn't useful

    for chunk in chunks:
        remaining = CHUNK_TOKEN_BUDGET - running_tokens
        if remaining < MIN_TRUNCATED_CHUNK_TOKENS:
            logger.info(
                "Skipping chunk %s -- negligible budget remaining (%d tokens left)",
                chunk.get("node_id", "?"), remaining,
            )
            continue

        block = _format_chunk(chunk)
        block_tokens = _count_tokens(block)

        if block_tokens <= remaining:
            context_blocks.append(block)
            chunks_used.append(chunk)
            running_tokens += block_tokens
            continue

        header = (
            f"### {chunk['node_id']}\n"
            f"File: {chunk['filepath']} (lines {chunk['start_line']}-{chunk['end_line']})\n"
        )
        if chunk.get("signature"):
            header += f"Signature: {chunk['signature']}\n"
        if chunk.get("docstring"):
            header += f'Docstring: "{chunk["docstring"]}"\n'
        header_tokens = _count_tokens(header)

        code_budget_tokens = remaining - header_tokens - 30
        if code_budget_tokens < MIN_TRUNCATED_CHUNK_TOKENS:
            logger.info(
                "Skipping chunk %s -- not enough space even for a truncated body",
                chunk.get("node_id", "?"),
            )
            continue

        code_tokens = _ENCODING.encode(chunk["code"])
        truncated_code = _ENCODING.decode(code_tokens[:code_budget_tokens])

        block = (
            f"{header}\n```python\n{truncated_code}\n"
            f"# ... [TRUNCATED: {len(code_tokens) - code_budget_tokens} more tokens omitted "
            f"to fit context budget -- full code is at {chunk['filepath']}:{chunk['start_line']}-{chunk['end_line']}]\n"
            f"```\n"
        )
        block_tokens = _count_tokens(block)

        logger.info(
            "Truncated oversized chunk %s: %d tokens -> %d tokens (kept header + %d/%d code tokens)",
            chunk.get("node_id", "?"), header_tokens + len(code_tokens), block_tokens,
            code_budget_tokens, len(code_tokens),
        )

        context_blocks.append(block)
        chunks_used.append(chunk)
        running_tokens += block_tokens

    context_text = "\n".join(context_blocks) if context_blocks else "(No relevant code context found.)"
    prompt = (
        f"{SYSTEM_INSTRUCTIONS}\n"
        f"## Code Context\n\n{context_text}\n"
        f"## Question\n{question}\n"
    )
    total_tokens = _count_tokens(prompt)
    if total_tokens > CONTEXT_TOKEN_BUDGET:
        logger.warning(
            "Prompt exceeded CONTEXT_TOKEN_BUDGET: %d > %d tokens",
            total_tokens, CONTEXT_TOKEN_BUDGET,
        )

    return PromptResult(
        prompt=prompt,
        chunks_used=chunks_used,
        chunks_dropped=len(chunks) - len(chunks_used),
        total_tokens=total_tokens,
    )