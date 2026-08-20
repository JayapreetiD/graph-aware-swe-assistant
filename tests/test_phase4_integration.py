

"""
tests/test_phase4_integration.py

End-to-end smoke test for Phase 4: real retriever output -> prompt
builder -> LLM -> citation verifier, on a real click codebase question.

Runs BOTH SemanticRetriever (baseline) and HybridRetriever through the
IDENTICAL downstream pipeline (same prompt_builder, same llm_service,
same citation_verifier) on the SAME question. This directly tests the
load-bearing assumption behind Phase 5: that prompt_builder.py and
citation_verifier.py are retrieval-mode-agnostic, i.e. both retrievers'
chunk dicts are compatible with the same downstream code.

Run with: python -m tests.test_phase4_integration
"""

from __future__ import annotations

from pathlib import Path

from graph.graph_builder import load_graph
from llm.citation_verifier import verify_citations
from llm.llm_service import generate_answer
from llm.prompt_builder import build_prompt
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.retriever import SemanticRetriever

GRAPH_PATH = "data/graph/graph.gpickle"

TEST_QUESTION = "How does click resolve which command to invoke in a group?"


def run_pipeline(label: str, chunks: list[dict]) -> None:
    print(f"\n{'='*70}")
    print(f"PIPELINE RUN: {label}")
    print(f"{'='*70}")

    print(f"Retrieved {len(chunks)} chunks.")
    if not chunks:
        print(f"FAILED [{label}]: retriever returned zero chunks.")
        return

    print("First chunk keys:", list(chunks[0].keys()))
    print("First chunk node_id:", chunks[0].get("node_id"))

    prompt_result = build_prompt(TEST_QUESTION, chunks)
    print(f"Chunks used: {len(prompt_result.chunks_used)} / {len(chunks)}")
    print(f"Chunks dropped: {prompt_result.chunks_dropped}")
    print(f"Total prompt tokens: {prompt_result.total_tokens}")

    llm_result = generate_answer(prompt_result.prompt)
    if not llm_result.success:
        print(f"FAILED [{label}]: LLM call failed - {llm_result.error}")
        return

    print(f"Model: {llm_result.model}")
    print(f"Truncated: {llm_result.truncated}")
    print("\n--- Answer ---")
    print(llm_result.answer)
    print("--- End answer ---")

    report = verify_citations(llm_result.answer, prompt_result.chunks_used)
    print(f"\nTotal citations: {report.total_citations}")
    print(f"Valid: {report.valid_citations}")
    print(f"Invalid: {report.invalid_citations}")
    print(f"Citation correctness rate: {report.citation_correctness_rate:.2f}")

    if report.verdicts:
        for v in report.verdicts:
            print(f"  {v.citation.raw_text} -> valid={v.is_valid} ({v.reason})")
    else:
        print("WARNING: zero parseable citations.")


def run_integration_test() -> None:
    print("=" * 70)
    print("PHASE 4 INTEGRATION TEST - SEMANTIC vs HYBRID")
    print("=" * 70)

    if not Path(GRAPH_PATH).exists():
        print(f"FAILED: graph file not found at {GRAPH_PATH}")
        return

    graph = load_graph(GRAPH_PATH)
    semantic_retriever = SemanticRetriever()

    try:
        semantic_chunks = semantic_retriever.retrieve(TEST_QUESTION, top_k=10)
        run_pipeline("SEMANTIC-ONLY", semantic_chunks)

        hybrid_retriever = HybridRetriever(semantic_retriever, graph, decay=0.6)
        hybrid_chunks = hybrid_retriever.retrieve(TEST_QUESTION)
        run_pipeline("HYBRID", hybrid_chunks)

        print("\n" + "=" * 70)
        print("INTEGRATION TEST COMPLETE")
        print("=" * 70)

    finally:
        semantic_retriever.close()


if __name__ == "__main__":
    run_integration_test()