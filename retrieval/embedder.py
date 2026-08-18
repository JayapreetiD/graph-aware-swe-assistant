

"""
retrieval/embedder.py

Single responsibility: convert CodeChunks into dense vector embeddings
using a code-aware embedding model, and persist them for the vector
store step.

Design decisions:
  1. Model: flax-sentence-embeddings/st-codesearch-distilroberta-base
     - trained on CodeSearchNet (code<->docstring pairs), matching our
       use case (developer questions <-> code chunks)
     - small enough for CPU-only local inference, no API/cost/network
       dependency in the reproducible core pipeline
  2. Embedding text = signature + docstring + code, in that order,
     with code truncated to ~1500 chars before tokenization. The model
     truncates from the end at its max sequence length; putting the
     most information-dense fields first means truncation (if any)
     eats into the code body, not the signature/docstring.
  3. Embeddings are L2-normalized so downstream retrieval can use dot
     product as a cosine-similarity proxy (Qdrant supports this natively
     and it's cheaper than computing cosine at query time).
  4. Output is kept SEPARATE from chunks.json (an .npz array + aligned
     node_id list), not merged into the chunk JSON. Embedding vectors
     are 768 floats each; jamming them into JSON bloats the file and
     mixes two different concerns (readable metadata vs numeric
     vectors). vector_store.py will join them by node_id.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME = "flax-sentence-embeddings/st-codesearch-distilroberta-base"
MAX_CODE_CHARS = 1500  # truncate code body before tokenization; see module docstring
BATCH_SIZE = 32


def load_chunks(chunks_path: str | Path) -> list[dict]:
    chunks_path = Path(chunks_path)
    with chunks_path.open("r", encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info("Loaded %d chunks from %s", len(chunks), chunks_path)
    return chunks


def build_embedding_text(chunk: dict) -> str:
    """Combine signature + docstring + (truncated) code into the text
    that gets embedded. Order matters — see module docstring point 2."""
    signature = chunk.get("signature", "") or ""
    docstring = chunk.get("docstring", "") or ""
    code = chunk.get("code", "") or ""
    code_truncated = code[:MAX_CODE_CHARS]

    parts = [signature, docstring, code_truncated]
    return "\n".join(p for p in parts if p.strip())


def embed_chunks(
    chunks: list[dict],
    model_name: str = MODEL_NAME,
    batch_size: int = BATCH_SIZE,
) -> tuple[list[str], np.ndarray]:
    """Returns (node_ids, embeddings) where embeddings[i] corresponds
    to node_ids[i]. Embeddings are L2-normalized float32."""
    logger.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)

    node_ids = [c["node_id"] for c in chunks]
    texts = [build_embedding_text(c) for c in chunks]

    logger.info("Encoding %d chunks (batch_size=%d)...", len(texts), batch_size)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2 normalize -> dot product == cosine similarity
        convert_to_numpy=True,
    ).astype(np.float32)

    logger.info(
        "Encoded embeddings: shape=%s (dim=%d)",
        embeddings.shape, embeddings.shape[1],
    )
    return node_ids, embeddings


def save_embeddings(node_ids: list[str], embeddings: np.ndarray, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        node_ids=np.array(node_ids, dtype=object),
        embeddings=embeddings,
    )
    logger.info("Saved embeddings to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    CHUNKS_PATH = "data/chunks/chunks.json"
    OUTPUT_PATH = "data/vectors/embeddings.npz"

    chunks = load_chunks(CHUNKS_PATH)
    node_ids, embeddings = embed_chunks(chunks)
    save_embeddings(node_ids, embeddings, OUTPUT_PATH)