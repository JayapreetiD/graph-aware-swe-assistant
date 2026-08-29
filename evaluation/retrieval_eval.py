

"""
evaluation/retrieval_eval.py

Single responsibility: score Precision@K, Recall@K, MRR, and retrieval
latency DIRECTLY against SemanticRetriever / HybridRetriever, with ZERO
LLM API calls. This exists because those four metrics never actually
needed an LLM in the first place -- they only depend on which chunks got
retrieved, not on what the LLM said about them. Only grounding_accuracy
and citation_correctness genuinely require an LLM call (they measure the
generated answer's citations).

This means sample size for the retrieval-quality metrics is no longer
bottlenecked by the 20-requests/day free-tier quota that blocked this
project for days. You can run this against all 27+17 existing
ground-truthed queries, or write many more, entirely for free.

Usage
-----
    export ACTIVE_REPO=click
    python -m evaluation.retrieval_eval

    export ACTIVE_REPO=django
    python -m evaluation.retrieval_eval

Requires ONLY a local Qdrant collection already built (chunker/embedder/
vector_store already run for the active repo) -- no server, no API key.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from config.settings import ACTIVE_REPO
from graph.graph_builder import load_graph
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.retriever import SemanticRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_BENCHMARKS_DIR = Path("data/benchmarks")
DEFAULT_OUTPUT_PATH = Path(f"data/results/{ACTIVE_REPO}/retrieval_only_metrics.json")


@dataclass
class RetrievalScore:
    query_id: str
    mode: str  # "semantic" or "hybrid"
    hop_depth: int | None  # None for semantic
    category: str
    k_retrieved: int
    ground_truth_count: int
    precision_at_k: float
    recall_at_k: float | None
    reciprocal_rank: float
    latency_seconds: float
    matched_ground_truth_ids: list[str]


def load_all_queries(benchmarks_dir: Path) -> list[dict[str, Any]]:
    """Load every query from every {ACTIVE_REPO}_queries_batch*.json file,
    same discovery pattern as benchmark.py, so this stays in sync with
    whatever ground truth already exists -- no duplication of query data."""
    pattern = str(benchmarks_dir / f"{ACTIVE_REPO}_queries_batch*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern}")

    queries: list[dict[str, Any]] = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        queries.extend(data.get("queries", []))
    logger.info("Loaded %d queries from %d file(s)", len(queries), len(files))
    return queries


def _node_ids(ground_truth: list[dict[str, Any]]) -> set[str]:
    return {n["node_id"] for n in ground_truth if "node_id" in n}


def score_one(
    retrieved_ids: list[str],
    ground_truth_ids: set[str],
) -> tuple[float, float | None, float, list[str]]:
    """Same math as metrics.py's _score_retrieval, kept minimal here
    since this script only ever needs precision/recall/MRR, not the full
    grounding-accuracy machinery (which requires citations, i.e. an LLM
    call, and is out of scope for this API-free script by design)."""
    retrieved_set = set(retrieved_ids)
    matched = retrieved_set & ground_truth_ids
    k = len(retrieved_ids)

    precision = len(matched) / k if k > 0 else 0.0
    recall = len(matched) / len(ground_truth_ids) if ground_truth_ids else None

    rr = 0.0
    for rank, node_id in enumerate(retrieved_ids, start=1):
        if node_id in ground_truth_ids:
            rr = 1.0 / rank
            break

    return precision, recall, rr, sorted(matched)


def run_retrieval_eval(
    queries: list[dict[str, Any]],
    hop_depths: tuple[int, ...] = (2,),
    k_semantic: int = 10,
    top_n: int = 15,
) -> list[RetrievalScore]:
    """Run every query through semantic AND hybrid (at each hop_depth in
    hop_depths) directly against the retriever objects. No FastAPI, no
    uvicorn, no LLM -- just the retrieval layer, which is all these four
    metrics ever needed."""
    from config.settings import GRAPH_PATH

    graph = load_graph(GRAPH_PATH)
    semantic_retriever = SemanticRetriever()
    hybrid_retriever = HybridRetriever(semantic_retriever, graph, decay=0.6)

    results: list[RetrievalScore] = []

    for q in queries:
        query_id = q["query_id"]
        question = q["question"]
        category = q.get("category", "unknown")
        ground_truth_ids = _node_ids(q.get("ground_truth_nodes", []))

        # --- semantic ---
        start = time.monotonic()
        sem_chunks = semantic_retriever.retrieve(question, top_k=k_semantic)
        sem_latency = time.monotonic() - start
        sem_ids = [c["node_id"] for c in sem_chunks]
        precision, recall, rr, matched = score_one(sem_ids, ground_truth_ids)
        results.append(RetrievalScore(
            query_id=query_id, mode="semantic", hop_depth=None, category=category,
            k_retrieved=len(sem_ids), ground_truth_count=len(ground_truth_ids),
            precision_at_k=round(precision, 4),
            recall_at_k=round(recall, 4) if recall is not None else None,
            reciprocal_rank=round(rr, 4), latency_seconds=round(sem_latency, 4),
            matched_ground_truth_ids=matched,
        ))

        # --- hybrid, at each requested hop depth ---
        for hop_depth in hop_depths:
            start = time.monotonic()
            hyb_chunks = hybrid_retriever.retrieve(
                question, k_semantic=k_semantic, hop_depth=hop_depth, top_n=top_n
            )
            hyb_latency = time.monotonic() - start
            hyb_ids = [c["node_id"] for c in hyb_chunks]
            precision, recall, rr, matched = score_one(hyb_ids, ground_truth_ids)
            results.append(RetrievalScore(
                query_id=query_id, mode="hybrid", hop_depth=hop_depth, category=category,
                k_retrieved=len(hyb_ids), ground_truth_count=len(ground_truth_ids),
                precision_at_k=round(precision, 4),
                recall_at_k=round(recall, 4) if recall is not None else None,
                reciprocal_rank=round(rr, 4), latency_seconds=round(hyb_latency, 4),
                matched_ground_truth_ids=matched,
            ))

        logger.info("Scored %s (%s)", query_id, category)

    semantic_retriever.close()
    return results


def _mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def summarize(results: list[RetrievalScore]) -> dict[str, Any]:
    by_key: dict[tuple[str, int | None], list[RetrievalScore]] = {}
    for r in results:
        by_key.setdefault((r.mode, r.hop_depth), []).append(r)

    summary = {}
    for (mode, hop_depth), items in sorted(by_key.items(), key=lambda kv: (kv[0][0], kv[0][1] or -1)):
        key = mode if hop_depth is None else f"hybrid_hop{hop_depth}"
        summary[key] = {
            "n": len(items),
            "mean_precision_at_k": _mean([r.precision_at_k for r in items]),
            "mean_recall_at_k": _mean([r.recall_at_k for r in items]),
            "mrr": _mean([r.reciprocal_rank for r in items]),
            "mean_latency_seconds": _mean([r.latency_seconds for r in items]),
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-API retrieval-only evaluation (Precision@K, Recall@K, MRR, latency)."
    )
    parser.add_argument("--benchmarks-dir", type=Path, default=DEFAULT_BENCHMARKS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--hop-depths", type=str, default="0,1,2,3",
        help="Comma-separated hop depths to test for hybrid mode (default: 0,1,2,3)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hop_depths = tuple(int(x) for x in args.hop_depths.split(","))

    queries = load_all_queries(args.benchmarks_dir)
    logger.info(
        "Running retrieval-only eval: %d queries x (1 semantic + %d hybrid hop depths) = %d retrieval calls, 0 API calls",
        len(queries), len(hop_depths), len(queries) * (1 + len(hop_depths)),
    )

    results = run_retrieval_eval(queries, hop_depths=hop_depths)
    summary = summarize(results)

    print()
    print("=" * 72)
    print(f"  RETRIEVAL-ONLY EVAL -- {ACTIVE_REPO} -- {len(queries)} queries, 0 API calls used")
    print("=" * 72)
    for key, stats in summary.items():
        print(
            f"  {key:<14} n={stats['n']:<3} "
            f"precision@k={stats['mean_precision_at_k']}  "
            f"recall@k={stats['mean_recall_at_k']}  "
            f"MRR={stats['mrr']}  "
            f"latency={stats['mean_latency_seconds']}s"
        )
    print("=" * 72)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "per_query": [asdict(r) for r in results],
        }, f, indent=2)
    logger.info("Wrote retrieval-only metrics to %s", args.output)


if __name__ == "__main__":
    main()