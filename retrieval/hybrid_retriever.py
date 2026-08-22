"""
retrieval/hybrid_retriever.py

Single responsibility: graph-aware hybrid retrieval. Takes semantic
search results as seeds, expands each into its graph neighborhood,
scores the combined candidate pool with hop-distance decay, dedupes,
reranks, and truncates to top_n.

This is the Phase 3 deliverable and the actual research variable of
the whole project — semantic_retriever.py (Phase 2) is the baseline
this must be compared against. Everything here is built ON TOP of
retriever.py and graph_service.py, and modifies neither.
"""

from __future__ import annotations

import logging

import networkx as nx

from graph.graph_service import get_neighbors
from retrieval.retriever import SemanticRetriever
from retrieval.vector_store import get_chunks_by_node_ids

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Wraps a SemanticRetriever + a loaded graph into a single
    question-in, graph-expanded-chunks-out interface."""

    def __init__(
        self,
        semantic_retriever: SemanticRetriever,
        graph: nx.DiGraph,
        decay: float = 0.6,
    ):
        self.semantic_retriever = semantic_retriever
        self.graph = graph
        self.decay = decay

    def retrieve(
        self,
        question: str,
        k_semantic: int = 10,
        hop_depth: int = 2,
        top_n: int = 10,
        edge_types: set[str] | None = None,
        direction: str = "both",
    ) -> list[dict]:
        """question -> semantic seeds -> graph-expanded, reranked chunks.

        Scoring: combined_score = seed_semantic_score * (decay ** hop_distance).
        Seeds themselves get hop_distance=0, so combined_score == semantic
        score there (decay**0 == 1) — a seed's own relevance is never
        discounted, only what's reached FROM it.

        If a node is reachable from multiple seeds at different hop
        distances, it keeps the MAX combined score across all paths, not
        the sum — this treats the score as "best evidence found," and
        avoids letting many weak paths outrank one strong path. This is
        a design choice, not a law of nature; sum is a legitimate
        alternative worth ablating if time allows.

        LIMITATION (document this in your report): a graph-expanded
        node's score can never exceed the semantic score of the seed it
        came from. So hybrid retrieval can't outrank semantic-only on
        nodes semantic-only already finds — its entire possible benefit
        is surfacing relevant nodes semantic search MISSED. That's
        exactly what Phase 5 needs to measure.
        """
        seeds = self.semantic_retriever.retrieve(question, top_k=k_semantic)

        # candidates: node_id -> (combined_score, hop_distance)
        candidates: dict[str, tuple[float, int]] = {}
        seed_chunks: dict[str, dict] = {}

        for seed in seeds:
            node_id = seed["node_id"]
            sem_score = seed["score"]
            seed_chunks[node_id] = seed
            candidates[node_id] = (sem_score, 0)

            if node_id not in self.graph:
                logger.warning(
                    "Seed node_id not found in graph (stale graph vs vector "
                    "index?): %s — skipping expansion for this seed", node_id
                )
                continue

            neighbors = get_neighbors(
                self.graph, node_id, hops=hop_depth,
                edge_types=edge_types, direction=direction,
            )
            for neighbor_id, hop_dist in neighbors.items():
                if self.graph.nodes[neighbor_id].get("type") == "module":
                    continue  # modules aren't chunked/embedded — not a retrievable unit
                combined = sem_score * (self.decay ** hop_dist)
                existing = candidates.get(neighbor_id)
                if existing is None or combined > existing[0]:
                    candidates[neighbor_id] = (combined, hop_dist)
        ranked = sorted(candidates.items(), key=lambda kv: kv[1][0], reverse=True)[:top_n]

        # Only fetch chunk payloads for candidates we're actually returning,
        # and only for ones not already carrying full data from the seed search.
        missing_ids = [nid for nid, _ in ranked if nid not in seed_chunks]
        fetched = get_chunks_by_node_ids(
            self.semantic_retriever.client, missing_ids,
            self.semantic_retriever.collection_name,
        )

        results = []
        for node_id, (score, hop_dist) in ranked:
            if node_id in seed_chunks:
                chunk = dict(seed_chunks[node_id])
            else:
                payload = fetched.get(node_id)
                if payload is None:
                    logger.warning(
                        "node_id %s reachable via graph but has no Qdrant "
                        "payload (never chunked/embedded?) — skipping", node_id
                    )
                    continue
                chunk = dict(payload)

            chunk["score"] = score
            chunk["hop_distance"] = hop_dist
            results.append(chunk)

        logger.info(
            "Query %r -> %d seeds -> %d candidates -> %d final results",
            question, len(seeds), len(candidates), len(results),
        )
        return results


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from graph.graph_builder import load_graph

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from config.settings import GRAPH_PATH

    if not Path(GRAPH_PATH).exists():
        print(f"Graph file not found at {GRAPH_PATH} — update GRAPH_PATH to match your Phase 1 output.")
        sys.exit(1)

    graph = load_graph(GRAPH_PATH)
    retriever = SemanticRetriever()
    hybrid = HybridRetriever(retriever, graph, decay=0.6)

    test_questions = [
        "how does click parse command line arguments",
        "how is the default value of an option determined",
        "how are subcommands registered in a group",
    ]

    for q in test_questions:
        print(f"\nQuery: {q}")
        results = hybrid.retrieve(q, k_semantic=10, hop_depth=2, top_n=10)
        for r in results:
            print(f"  score={r['score']:.4f}  hop={r['hop_distance']}  {r['node_id']}")

    retriever.close()