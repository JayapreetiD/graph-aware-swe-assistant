

"""
api/main.py

FastAPI backend exposing a single /query endpoint that wires together
the full Phase 2-4 pipeline: retrieval (semantic-only OR hybrid,
selectable per-request) -> prompt building -> LLM generation ->
citation verification.

Retriever and graph objects are expensive to construct (Qdrant
connection, graph.gpickle load, embedding model load) so they are
built ONCE at server startup via FastAPI's lifespan context, not
per-request.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from graph.graph_builder import load_graph
from llm.citation_verifier import verify_citations
from llm.llm_service import generate_answer
from llm.prompt_builder import build_prompt
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.retriever import SemanticRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GRAPH_PATH = "data/graph/graph.gpickle"

# Populated at startup, used by the /query endpoint. Not module-level
# constants because they require I/O (file load, DB connection) that
# must not happen at import time (e.g. when pytest imports this file).
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading graph from %s ...", GRAPH_PATH)
    if not Path(GRAPH_PATH).exists():
        raise RuntimeError(
            f"Graph file not found at {GRAPH_PATH}. Run graph_builder.py first."
        )
    _state["graph"] = load_graph(GRAPH_PATH)

    logger.info("Initializing SemanticRetriever ...")
    _state["semantic_retriever"] = SemanticRetriever()

    _state["hybrid_retriever"] = HybridRetriever(
        _state["semantic_retriever"], _state["graph"], decay=0.6
    )

    logger.info("Startup complete. Ready to serve /query.")
    yield

    logger.info("Shutting down: closing SemanticRetriever connection.")
    _state["semantic_retriever"].close()


app = FastAPI(title="Graph-Aware SWE Knowledge Assistant", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str
    mode: Literal["semantic", "hybrid"] = "hybrid"
    top_k: int = 10  # used only for semantic mode


class ChunkOut(BaseModel):
    node_id: str
    filepath: str
    start_line: int
    end_line: int
    score: float
    hop_distance: int | None = None  # None for semantic-only chunks


class CitationOut(BaseModel):
    raw_text: str
    filepath: str
    start_line: int
    end_line: int
    is_valid: bool
    reason: str


class QueryResponse(BaseModel):
    question: str
    mode: str
    answer: str
    truncated: bool
    chunks_used: list[ChunkOut]
    citations: list[CitationOut]
    citation_correctness_rate: float


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    # --- Retrieval: mode-selectable, per roadmap requirement that the
    # UI can run semantic-only OR graph-aware retrieval on demand. ---
    if req.mode == "semantic":
        chunks = _state["semantic_retriever"].retrieve(req.question, top_k=req.top_k)
    else:
        chunks = _state["hybrid_retriever"].retrieve(req.question)

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No relevant code chunks found for this question.",
        )

    # --- Prompt building (identical code path for both modes - this
    # is the retrieval-mode-agnostic guarantee verified in Phase 4's
    # integration test). ---
    prompt_result = build_prompt(req.question, chunks)

    # --- LLM generation ---
    llm_result = generate_answer(prompt_result.prompt)
    if not llm_result.success:
        raise HTTPException(
            status_code=502,
            detail=f"LLM generation failed: {llm_result.error}",
        )

    # --- Citation verification ---
    report = verify_citations(llm_result.answer, prompt_result.chunks_used)

    return QueryResponse(
        question=req.question,
        mode=req.mode,
        answer=llm_result.answer,
        truncated=llm_result.truncated,
        chunks_used=[
            ChunkOut(
                node_id=c["node_id"],
                filepath=c["filepath"],
                start_line=c["start_line"],
                end_line=c["end_line"],
                score=c.get("score", 0.0),
                hop_distance=c.get("hop_distance"),
            )
            for c in prompt_result.chunks_used
        ],
        citations=[
            CitationOut(
                raw_text=v.citation.raw_text,
                filepath=v.citation.filepath,
                start_line=v.citation.start_line,
                end_line=v.citation.end_line,
                is_valid=v.is_valid,
                reason=v.reason,
            )
            for v in report.verdicts
        ],
        citation_correctness_rate=report.citation_correctness_rate,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}