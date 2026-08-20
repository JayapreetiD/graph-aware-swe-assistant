

"""
tests/test_phase4_integration.py

End-to-end smoke test for Phase 4: real retriever output -> prompt
builder -> LLM -> citation verifier, on a real click codebase question.

This is NOT a unit test with mocked data - it deliberately uses real
retrieval output to catch data-shape mismatches between what the
retriever returns and what prompt_builder.py / citation_verifier.py
expect (dict key names, types, etc.) before those mismatches show up
buried inside a FastAPI request cycle.

Construction pattern for SemanticRetriever / graph / HybridRetriever
is copied from tests/test_hybrid_retriever.py's fixtures, since that
is the real, working pattern already used elsewhere in this project.

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

# A real question about the click codebase - deliberately something
# with a concrete, checkable answer rather than an open-ended one,
# so it's easy to eyeball whether the answer is grounded or not.
TEST_QUESTION = "How does click resolve which command to invoke in a group?"


def run_integration_test() -> None:
    print("=" * 70)
    print("PHASE 4 INTEGRATION TEST")
    print("=" * 70)

    # --- Step 1: Retrieval setup ---
    print("\n[1/4] Setting up retrievers...")
    if not Path(GRAPH_PATH).exists():
        print(f"FAILED: graph file not found at {GRAPH_PATH}")
        return

    graph = load_graph(GRAPH_PATH)
    semantic_retriever = SemanticRetriever()

    try:
        hybrid_retriever = HybridRetriever(semantic_retriever, graph, decay=0.6)

        print("Running hybrid retrieval...")
        chunks = hybrid_retriever.retrieve(TEST_QUESTION)

        print(f"Retrieved {len(chunks)} chunks.")
        if not chunks:
            print("FAILED: retriever returned zero chunks. Stopping.")
            return

        # Print the shape of the first chunk so any key mismatch with
        # prompt_builder.py / citation_verifier.py is immediately visible.
        print("First chunk keys:", list(chunks[0].keys()))
        print("First chunk node_id:", chunks[0].get("node_id"))

        # --- Step 2: Prompt building ---
        print("\n[2/4] Building prompt...")
        prompt_result = build_prompt(TEST_QUESTION, chunks)
        print(f"Chunks used: {len(prompt_result.chunks_used)} / {len(chunks)}")
        print(f"Chunks dropped: {prompt_result.chunks_dropped}")
        print(f"Total prompt tokens: {prompt_result.total_tokens}")

        # --- Step 3: LLM call ---
        print("\n[3/4] Calling LLM...")
        llm_result = generate_answer(prompt_result.prompt)

        if not llm_result.success:
            print(f"FAILED: LLM call failed - {llm_result.error}")
            return

        print(f"Model: {llm_result.model}")
        print("\n--- Answer ---")
        print(llm_result.answer)
        print("--- End answer ---")

        # --- Step 4: Citation verification ---
        print("\n[4/4] Verifying citations...")
        report = verify_citations(llm_result.answer, prompt_result.chunks_used)

        print(f"Total citations found: {report.total_citations}")
        print(f"Valid: {report.valid_citations}")
        print(f"Invalid: {report.invalid_citations}")
        print(f"Citation correctness rate: {report.citation_correctness_rate:.2f}")

        if report.verdicts:
            print("\nPer-citation breakdown:")
            for v in report.verdicts:
                print(f"  {v.citation.raw_text} -> valid={v.is_valid} ({v.reason})")
        else:
            print(
                "\nWARNING: LLM produced zero parseable citations. This is a "
                "real failure worth investigating - either the model ignored "
                "the citation instruction, or its citation format doesn't "
                "match CITATION_PATTERN in citation_verifier.py."
            )

        print("\n" + "=" * 70)
        print("INTEGRATION TEST COMPLETE")
        print("=" * 70)

    finally:
        # SemanticRetriever holds a Qdrant connection open - always
        # close it, even if something above failed or returned early.
        semantic_retriever.close()


if __name__ == "__main__":
    run_integration_test()