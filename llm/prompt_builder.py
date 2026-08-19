

"""
llm/prompt_builder.py

Responsibility: convert a ranked list of retrieved code chunks + a user
question into a single, fixed-token-budget prompt string for the LLM.

This is the ONE place where "context budget" is enforced. Both
SemanticRetriever and HybridRetriever produce the same chunk shape
(list of dicts with node_id, filepath, start_line, end_line, code, etc.),
so this file is retrieval-mode-agnostic by design — Phase 5 depends on
that: semantic-only and hybrid runs MUST go through the exact same
prompt construction logic, or a difference in results could be caused
by prompt differences instead of retrieval differences.
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

    Truncation policy: chunks are added in the given order until the
    NEXT chunk would exceed CHUNK_TOKEN_BUDGET, then stop. This means
    lowest-ranked chunks are dropped first, which is what you want —
    the retriever already did the ranking work; the prompt builder's
    only job is to respect the budget, not re-rank.
    """
    context_blocks: list[str] = []
    chunks_used: list[dict] = []
    running_tokens = 0

    for chunk in chunks:
        block = _format_chunk(chunk)
        block_tokens = _count_tokens(block)

        if running_tokens + block_tokens > CHUNK_TOKEN_BUDGET:
            logger.info(
                "Context budget reached: stopping at %d/%d chunks (%d tokens used)",
                len(chunks_used), len(chunks), running_tokens,
            )
            break

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
        # Should be rare given the reserved overhead margin, but log
        # loudly if it happens — it means RESERVED_TOKENS_FOR_OVERHEAD
        # needs to be raised.
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


if __name__ == "__main__":
    # Minimal smoke test using fake chunks — no dependency on the
    # retriever, graph, or Qdrant. Just verifies budget math and
    # formatting work correctly in isolation.
    fake_chunks = [
        {
            "node_id": f"click/core.py::fake_func_{i}",
            "filepath": "click/core.py",
            "start_line": i * 10,
            "end_line": i * 10 + 8,
            "signature": f"def fake_func_{i}(x: int) -> int:",
            "docstring": "A fake function for testing.",
            "code": f"def fake_func_{i}(x: int) -> int:\n    return x + {i}",
        }
        for i in range(50)  # deliberately more than will fit, to test truncation
    ]

    result = build_prompt("How does fake_func_3 work?", fake_chunks)

    print(f"Chunks used: {len(result.chunks_used)} / {len(fake_chunks)}")
    print(f"Chunks dropped: {result.chunks_dropped}")
    print(f"Total tokens: {result.total_tokens} (budget: {CONTEXT_TOKEN_BUDGET})")
    print("\n--- First 500 chars of prompt ---")
    print(result.prompt[:500])