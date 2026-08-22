"""
retrieval/retriever.py

Single responsibility: the semantic-only retrieval baseline.
Takes a plain-English question, embeds it with the same model used
for chunks, and returns top-K matching code chunks from Qdrant.

This IS the "Semantic Retrieval Baseline" required by the project
scope (Phase 2) — the thing graph-aware hybrid retrieval (Phase 3)
must be compared against. It must stay simple and unmodified by any
graph logic; hybrid_retriever.py builds ON TOP of this, it doesn't
change this file's behavior.

Design decision: the embedding model can be passed in pre-loaded, or
left None to lazy-load on first use. Reasoning — Phase 5's evaluation
will run 30-50 queries in a loop; reloading the model per-query would
add seconds of dead time per query for no benefit. Passing a shared
model instance once amortizes that load cost across the whole eval run.
"""

from __future__ import annotations

import logging
from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from retrieval.vector_store import COLLECTION_NAME, get_client, search
from retrieval.embedder import MODEL_NAME
from config.settings import VECTOR_DB_PATH

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """Wraps an embedding model + Qdrant client into a single
    question-in, chunks-out interface."""

    def __init__(
        self,
        db_path: str | Path = VECTOR_DB_PATH,
        collection_name: str = COLLECTION_NAME,
        model: SentenceTransformer | None = None,
        model_name: str = MODEL_NAME,
    ):
        self.client: QdrantClient = get_client(db_path)
        self.collection_name = collection_name
        self._model = model  # may be None -> lazy-loaded on first query
        self._model_name = model_name

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Lazy-loading embedding model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def retrieve(self, question: str, top_k: int = 5) -> list[dict]:
        """question -> query embedding -> top_k code chunks (semantic-only).
        Returns list of dicts: {score, node_id, type, name, qualified_name,
        filepath, start_line, end_line, signature, docstring, code}."""
        model = self._get_model()
        query_vector = model.encode(
            question,
            normalize_embeddings=True,  # must match embedder.py's normalization
            convert_to_numpy=True,
        ).tolist()

        results = search(self.client, query_vector, top_k=top_k, collection_name=self.collection_name)
        logger.info("Query %r -> %d results (top score=%.4f)",
                    question, len(results), results[0]["score"] if results else 0.0)
        return results

    def close(self) -> None:
        self.client.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    retriever = SemanticRetriever()

    # Real semantic test — an actual English developer question, not a
    # self-match. This is the first genuine test of whether the whole
    # pipeline (chunk -> embed -> store -> retrieve) does the real job.
    test_questions = [
        "how does click parse command line arguments",
        "how is the default value of an option determined",
        "how are subcommands registered in a group",
    ]

    for q in test_questions:
        print(f"\nQuery: {q}")
        results = retriever.retrieve(q, top_k=3)
        for r in results:
            print(f"  score={r['score']:.4f}  {r['node_id']}")

    retriever.close()