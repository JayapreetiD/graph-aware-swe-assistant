

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
RESERVED_TOKENS_FOR_OVERHEAD = 500
CHUNK_TOKEN_BUDGET = CONTEXT_TOKEN_BUDGET - RESERVED_TOKENS_FOR_OVERHEAD

# CAP on how much space a single TRUNCATED chunk may consume, even if
# more space is technically available. Without this, one huge class
# (e.g. Field at 7929 tokens) can crowd out several smaller, equally
# relevant chunks within the same fixed budget -- confirmed empirically:
# an end-to-end Django rerun after the first truncation fix showed
# recall going DOWN in places, likely because one truncated giant chunk
# consumed space that would otherwise have fit 2-3 smaller relevant
# chunks. This cap gives an oversized chunk a meaningful preview without
# letting it dominate the whole prompt.
MAX_TRUNCATED_CHUNK_TOKENS = 900

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
    first). Chunks that fit whole are added whole. Chunks that don't fit
    are truncated to fit, capped at MAX_TRUNCATED_CHUNK_TOKENS so one
    oversized chunk can't consume space that would otherwise fit several
    smaller relevant chunks.
    """
    context_blocks: list[str] = []
    chunks_used: list[dict] = []
    running_tokens = 0

    MIN_TRUNCATED_CHUNK_TOKENS = 150

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

        # NEW: cap the truncated body at MAX_TRUNCATED_CHUNK_TOKENS,
        # not just whatever happens to be "remaining" -- this is the fix.
        code_budget_tokens = min(
            remaining - header_tokens - 30,
            MAX_TRUNCATED_CHUNK_TOKENS,
        )
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
            "Truncated oversized chunk %s: %d tokens -> %d tokens (capped at %d code tokens)",
            chunk.get("node_id", "?"), header_tokens + len(code_tokens), block_tokens,
            code_budget_tokens,
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