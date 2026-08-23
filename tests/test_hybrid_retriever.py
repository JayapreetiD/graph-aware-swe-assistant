"""
tests/test_hybrid_retriever.py

Assertion-based tests for HybridRetriever, replacing the manual
print()-based smoke tests used during Phase 3 development.

These run against the REAL graph.gpickle and REAL Qdrant collection
built from pallets/click — same data your manual tests used. This
project has no mocking infrastructure (Phase 1's validation followed
the same "test against real data" approach), so these are integration
tests, not unit tests: they require data/graph/graph.gpickle and the
Qdrant collection to already exist (run graph_builder.py, chunker.py,
embedder.py, vector_store.py first if starting fresh).

Run: pytest tests/test_hybrid_retriever.py -v
"""
from __future__ import annotations

from pathlib import Path

import pytest

from graph.graph_builder import load_graph
from graph.graph_service import get_neighbors
from retrieval.retriever import SemanticRetriever
from retrieval.hybrid_retriever import HybridRetriever

from config.settings import GRAPH_PATH

# A node known (from manual Phase 3 testing) to have neighbors at hop 1, 2, and 3.
# If this node ever gets removed/renamed by a repo re-parse, update this constant.

KNOWN_MULTI_HOP_NODE = "core.py::Command"
TEST_QUESTIONS = [
    "how does click parse command line arguments",
    "how is the default value of an option determined",
    "how are subcommands registered in a group",
]


@pytest.fixture(scope="module")
def graph():
    if not Path(GRAPH_PATH).exists():
        pytest.skip(f"Graph file not found at {GRAPH_PATH} — run graph_builder.py first.")
    return load_graph(GRAPH_PATH)


@pytest.fixture(scope="module")
def semantic_retriever():
    retriever = SemanticRetriever()
    yield retriever
    retriever.close()


@pytest.fixture(scope="module")
def hybrid_retriever(semantic_retriever, graph):
    return HybridRetriever(semantic_retriever, graph, decay=0.6)


class TestHopZeroInvariant:
    """hop_depth=0 must degrade exactly to semantic-only behavior.
    This is the core correctness invariant: disabling graph expansion
    should never change relevance ordering vs the Phase 2 baseline."""

    def test_hop_zero_matches_semantic_baseline(self, hybrid_retriever, semantic_retriever):
        question = TEST_QUESTIONS[0]
        baseline = semantic_retriever.retrieve(question, top_k=5)
        hybrid = hybrid_retriever.retrieve(question, k_semantic=5, top_n=5, hop_depth=0)

        assert len(hybrid) == len(baseline)
        for h_res, b_res in zip(hybrid, baseline):
            assert h_res["node_id"] == b_res["node_id"]
            assert h_res["score"] == pytest.approx(b_res["score"])

    def test_hop_zero_all_results_are_hop_zero(self, hybrid_retriever):
        results = hybrid_retriever.retrieve(TEST_QUESTIONS[0], k_semantic=5, top_n=5, hop_depth=0)
        assert all(r["hop_distance"] == 0 for r in results)


class TestSeedLabeling:
    """A node that is itself a semantic seed must always be labeled
    hop_distance=0, never overwritten by a lower-scoring path found
    via graph expansion from a different seed."""

    def test_no_seed_is_mislabeled(self, hybrid_retriever, semantic_retriever):
        question = TEST_QUESTIONS[0]
        seeds = semantic_retriever.retrieve(question, top_k=5)
        seed_ids = {s["node_id"] for s in seeds}

        results = hybrid_retriever.retrieve(question, k_semantic=5, top_n=20, hop_depth=2)
        mislabeled = [r for r in results if r["node_id"] in seed_ids and r["hop_distance"] != 0]

        assert mislabeled == [], f"Seeds incorrectly labeled with hop>0: {mislabeled}"


class TestGraphTraversalReachesAllHops:
    """Confirms get_neighbors (the traversal engine hybrid_retriever
    depends on) actually reaches hop 1/2/3, independent of scoring —
    this isolates traversal correctness from decay/ranking behavior."""

    def test_raw_traversal_reaches_hop_three(self, graph):
        assert KNOWN_MULTI_HOP_NODE in graph, (
            f"{KNOWN_MULTI_HOP_NODE} not found in graph — repo may have "
            f"been re-parsed; update KNOWN_MULTI_HOP_NODE."
        )
        neighbors = get_neighbors(graph, KNOWN_MULTI_HOP_NODE, hops=3)
        hop_distances = set(neighbors.values())

        assert 1 in hop_distances, "No hop=1 neighbors found"
        assert 2 in hop_distances, "No hop=2 neighbors found"
        assert 3 in hop_distances, "No hop=3 neighbors found"


class TestHybridCandidatePoolReachesDeepHops:
    """Confirms hop=2/3 candidates exist in HybridRetriever's SCORED
    pool (not just the raw graph) when truncation is removed. Without
    this, a small top_n could mask a genuine traversal bug behind
    'ranking just cut them' — this proves the candidates are really
    there before any cutoff is applied."""

    def test_deep_hops_present_when_untruncated(self, hybrid_retriever):
        results = hybrid_retriever.retrieve(
            TEST_QUESTIONS[1], k_semantic=5, top_n=1000, hop_depth=3
        )
        hop_distances = {r["hop_distance"] for r in results}

        assert 2 in hop_distances, "No hop=2 candidates in unranked pool"
        assert 3 in hop_distances, "No hop=3 candidates in unranked pool"


class TestModuleFiltering:
    """Module-type graph nodes must never appear in results — they
    aren't chunked/embedded, so they have no retrievable chunk payload."""

    def test_no_module_nodes_in_results(self, hybrid_retriever, graph):
        results = hybrid_retriever.retrieve(
            TEST_QUESTIONS[0], k_semantic=5, top_n=1000, hop_depth=2
        )
        for r in results:
            node_type = graph.nodes.get(r["node_id"], {}).get("type")
            assert node_type != "module", f"Module node leaked into results: {r['node_id']}"


class TestNoCrashOnEdgeCases:
    """Robustness checks: hybrid retrieval must not crash on inputs
    that are valid but unusual."""

    def test_nonsense_query_does_not_crash(self, hybrid_retriever):
        results = hybrid_retriever.retrieve(
            "asdkjfh qwoeiur zzxcv nonsense query", k_semantic=5, top_n=10, hop_depth=2
        )
        assert isinstance(results, list)
        # Qdrant is nearest-neighbor, not threshold-based — it will
        # always return *something*, so we only assert non-crash + shape.

    def test_zero_hop_depth_does_not_crash(self, hybrid_retriever):
        results = hybrid_retriever.retrieve(TEST_QUESTIONS[0], k_semantic=5, top_n=5, hop_depth=0)
        assert isinstance(results, list)

    @pytest.mark.parametrize("question", TEST_QUESTIONS)
    def test_all_known_questions_return_results(self, hybrid_retriever, question):
        results = hybrid_retriever.retrieve(question, k_semantic=10, top_n=15, hop_depth=2)
        assert len(results) > 0
        for r in results:
            assert "node_id" in r
            assert "score" in r
            assert "hop_distance" in r
            assert r["hop_distance"] >= 0