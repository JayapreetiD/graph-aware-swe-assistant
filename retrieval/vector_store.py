

"""
retrieval/vector_store.py

Single responsibility: load chunk metadata + embeddings into a local
Qdrant collection, and provide a query function for semantic search.

Design decisions:
  1. Embedded/local Qdrant (QdrantClient(path=...)), not a server —
     solo dev, single laptop, offline-reproducible research pipeline.
     No Docker/infra to manage. Switching to a real server later is a
     one-line change (path= -> host=/port=), not a redesign.
  2. Distance metric: COSINE. Embeddings are already L2-normalized
     (verified in embedder.py), so cosine and dot-product are
     mathematically identical here — COSINE is used for clarity.
  3. Qdrant point IDs must be unsigned int or UUID, not arbitrary
     strings — so points get a sequential integer ID, and the real
     join key (node_id) is stored as a payload field instead. This
     is the same node_id used in the graph, enabling graph<->vector
     linkage in Phase 3's hop expansion.
  4. Full chunk metadata (code, signature, docstring, filepath, line
     range) is stored in the payload, not just node_id — so a single
     query returns everything needed for LLM context without a
     second lookup into chunks.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)

COLLECTION_NAME = "code_chunks"
VECTOR_DIM = 768  # output dim of st-codesearch-distilroberta-base, verified in embedder.py


def load_chunks(chunks_path: str | Path) -> dict[str, dict]:
    """Returns node_id -> chunk metadata dict, for O(1) lookup during join."""
    chunks_path = Path(chunks_path)
    with chunks_path.open("r", encoding="utf-8") as f:
        chunks = json.load(f)
    return {c["node_id"]: c for c in chunks}


def load_embeddings(embeddings_path: str | Path) -> tuple[list[str], np.ndarray]:
    data = np.load(embeddings_path, allow_pickle=True)
    return list(data["node_ids"]), data["embeddings"]


def get_client(db_path: str | Path) -> QdrantClient:
    db_path = Path(db_path)
    db_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(db_path))


def build_collection(
    client: QdrantClient,
    chunks_by_id: dict[str, dict],
    node_ids: list[str],
    embeddings: np.ndarray,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """Create (or recreate) the collection and upsert all points.
    Recreates from scratch each run — this pipeline is meant to be
    rerun end-to-end when source data changes, not incrementally
    updated (matches project scope: no live/incremental re-indexing)."""
    if client.collection_exists(collection_name):
        logger.info("Collection '%s' exists, deleting to rebuild fresh", collection_name)
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=VECTOR_DIM, distance=models.Distance.COSINE),
    )

    points = []
    missing_metadata = 0
    for idx, (node_id, vector) in enumerate(zip(node_ids, embeddings)):
        chunk = chunks_by_id.get(node_id)
        if chunk is None:
            missing_metadata += 1
            continue

        points.append(
            models.PointStruct(
                id=idx,
                vector=vector.tolist(),
                payload={
                    "node_id": node_id,
                    "type": chunk["type"],
                    "name": chunk["name"],
                    "qualified_name": chunk["qualified_name"],
                    "filepath": chunk["filepath"],
                    "start_line": chunk["start_line"],
                    "end_line": chunk["end_line"],
                    "signature": chunk["signature"],
                    "docstring": chunk["docstring"],
                    "code": chunk["code"],
                },
            )
        )

    if missing_metadata:
        logger.warning(
            "%d embeddings had no matching chunk metadata (node_id mismatch) — skipped",
            missing_metadata,
        )

    client.upsert(collection_name=collection_name, points=points, wait=True)
    logger.info("Upserted %d points into collection '%s'", len(points), collection_name)


def search(
    client: QdrantClient,
    query_vector: list[float],
    top_k: int = 5,
    collection_name: str = COLLECTION_NAME,
) -> list[dict]:
    """Semantic-only search: query_vector -> top_k chunks by cosine similarity.
    Returns list of dicts with 'score' plus full payload — this IS the
    semantic-only baseline the project scope requires (Phase 2, step
    'Semantic Retrieval Baseline')."""
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
    ).points

    return [{"score": r.score, **r.payload} for r in results]
def get_chunks_by_node_ids(
    client: QdrantClient,
    node_ids: list[str],
    collection_name: str = COLLECTION_NAME,
) -> dict[str, dict]:
    """Fetch full chunk payloads for a specific set of node_ids.

    Needed by hybrid_retriever.py: graph expansion (graph_service.get_neighbors)
    returns node_ids and hop distances only, never chunk content — that's a
    deliberate separation (graph = structure, vector store = content). This
    function is the join point: given node_ids discovered via the graph,
    pull their code/signature/filepath/etc. back out of Qdrant.

    Uses a payload filter scroll, not a vector search — there's no query
    vector here, just "give me these specific IDs." Point IDs in this
    collection are sequential ints (see build_collection's design note),
    so node_id -> point lookup has to go through the payload field, not
    the point ID directly.

    Returns node_id -> payload dict. node_ids with no match (e.g. graph
    has nodes that were filtered out during chunking, such as trivially
    short functions) are silently omitted — callers must handle missing
    keys, not assume every requested node_id comes back.
    """
    if not node_ids:
        return {}

    results, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="node_id", match=models.MatchAny(any=node_ids))]
        ),
        limit=len(node_ids),
        with_payload=True,
        with_vectors=False,
    )
    return {point.payload["node_id"]: point.payload for point in results}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    CHUNKS_PATH = "data/chunks/chunks.json"
    EMBEDDINGS_PATH = "data/vectors/embeddings.npz"
    DB_PATH = "data/vectors/qdrant_db"

    chunks_by_id = load_chunks(CHUNKS_PATH)
    node_ids, embeddings = load_embeddings(EMBEDDINGS_PATH)

    client = get_client(DB_PATH)
    build_collection(client, chunks_by_id, node_ids, embeddings)

    # Smoke test: use one of the actual stored embeddings as a query,
    # to sanity-check that search returns itself as the top hit with
    # score ~1.0 (cosine similarity of a vector with itself).
    test_vector = embeddings[0].tolist()
    results = search(client, test_vector, top_k=3)
    logger.info("Smoke test — top match for embeddings[0]:")
    for r in results:
        logger.info("  score=%.4f  node_id=%s", r["score"], r["node_id"])
    client.close()
