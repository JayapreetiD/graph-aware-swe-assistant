"""
evaluation/metrics.py

Single responsibility: read a benchmark_results.jsonl file (produced by
evaluation/benchmark.py) and compute the Phase 5 metrics -- Precision@K,
Recall@K, MRR, grounding accuracy -- plus aggregate the citation
correctness and latency figures the API already returns per call.

This module does NOT call the API and does NOT produce the final written
comparison report (that's evaluation/evaluation.py's job). It reads what
benchmark.py already saved and scores it against ground_truth_nodes.

Works correctly on a PARTIAL results file (e.g. 19 of 54 calls done) --
every summary this script prints or saves is explicitly labeled with the
sample size it's based on, so a partial run can never be mistaken for a
final result.

Usage
-----
    python -m evaluation.metrics
    python -m evaluation.metrics --results data/results/click/benchmark_results.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_RESULTS_PATH = Path("data/results/click/benchmark_results.jsonl")
DEFAULT_ABLATION_RESULTS_PATH = Path("data/results/click/ablation_results.jsonl")
DEFAULT_SUMMARY_PATH = Path("data/results/click/metrics_summary.json")


@dataclass
class QueryMetrics:
    """Metrics computed for one (query_id, mode) result."""

    query_id: str
    mode: str
    category: str
    expected_hybrid_advantage: bool
    success: bool
    k_retrieved: int
    ground_truth_count: int
    precision_at_k: float | None
    recall_at_k: float | None
    reciprocal_rank: float | None
    grounding_accuracy: float | None
    citation_correctness_rate: float | None
    latency_seconds: float | None
    matched_ground_truth_ids: list[str]
    unmatched_ground_truth_ids: list[str]


def load_results(results_path: Path) -> list[dict[str, Any]]:
    """Load every line from the JSONL results file. Includes both
    successful and failed records -- failed ones are filtered out where
    scoring requires a real response, but their presence is still used to
    report an honest success/failure count."""
    if not results_path.exists():
        raise FileNotFoundError(
            f"{results_path} not found. Run evaluation/benchmark.py first."
        )

    records: list[dict[str, Any]] = []
    with results_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    return records


def deduplicate_latest_per_pair(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    benchmark.py appends a new line every time a (query_id, mode) pair is
    attempted -- including retries across separate runs -- so the raw
    file can contain multiple lines for the same pair (e.g. several
    failed attempts before an eventual success, or several failed
    attempts with none succeeding yet). Keep only the LAST record per
    pair, preferring the most recent successful one if any exists, so
    scoring never double-counts a pair or scores a stale failed attempt
    when a later success is available.
    """
    best: dict[tuple[str, str], dict[str, Any]] = {}

    for record in records:
        key = (record["query_id"], record["mode"])
        existing = best.get(key)

        if existing is None:
            best[key] = record
            continue

        # Prefer a success over a failure regardless of order. Between
        # two successes or two failures, prefer the later one (more
        # recent attempt, by file order).
        if record["success"] and not existing["success"]:
            best[key] = record
        elif record["success"] == existing["success"]:
            best[key] = record  # later record wins on a tie

    return list(best.values())


def _node_ids(nodes: list[dict[str, Any]]) -> set[str]:
    return {n["node_id"] for n in nodes if "node_id" in n}


def _ranges_overlap(
    a_file: str, a_start: int, a_end: int,
    b_file: str, b_start: int, b_end: int,
) -> bool:
    """True if two (filepath, start_line, end_line) ranges refer to the
    same file and their line ranges overlap at all."""
    if a_file != b_file:
        return False
    return a_start <= b_end and b_start <= a_end


def _score_retrieval(
    chunks_used: list[dict[str, Any]],
    ground_truth_nodes: list[dict[str, Any]],
    citations: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Shared scoring core used by BOTH compute_query_metrics (main
    benchmark, keyed by mode) and compute_ablation_query_metrics
    (hop-depth ablation, keyed by hop_depth). Extracted so there is only
    ONE place this math can be wrong -- duplicating it would have meant
    the grounding_accuracy bug fixed earlier could silently reappear in
    a second, unsynced copy for ablation scoring.

    See compute_query_metrics's docstring for the precision@k / recall@k
    / MRR / grounding_accuracy definitions -- unchanged here, just
    factored out.
    """
    ground_truth_ids = _node_ids(ground_truth_nodes)
    ground_truth_count = len(ground_truth_ids)

    retrieved_ids = [c["node_id"] for c in chunks_used if "node_id" in c]
    retrieved_id_set = set(retrieved_ids)
    k_retrieved = len(retrieved_ids)

    matched = retrieved_id_set & ground_truth_ids
    unmatched = ground_truth_ids - retrieved_id_set

    precision_at_k = (len(matched) / k_retrieved) if k_retrieved > 0 else 0.0
    recall_at_k = (
        (len(matched) / ground_truth_count) if ground_truth_count > 0 else None
    )

    reciprocal_rank = 0.0
    for rank, node_id in enumerate(retrieved_ids, start=1):
        if node_id in ground_truth_ids:
            reciprocal_rank = 1.0 / rank
            break

    valid_citations = [c for c in citations if c.get("is_valid")]
    grounding_accuracy = None
    if ground_truth_count > 0:
        grounded_count = 0
        for gt_node in ground_truth_nodes:
            gt_file = gt_node.get("filepath")
            gt_start = gt_node.get("start_line")
            gt_end = gt_node.get("end_line")
            if gt_file is None or gt_start is None or gt_end is None:
                continue
            is_grounded = any(
                _ranges_overlap(
                    gt_file, gt_start, gt_end,
                    c.get("filepath", ""), c.get("start_line", -1), c.get("end_line", -1),
                )
                for c in valid_citations
            )
            if is_grounded:
                grounded_count += 1
        grounding_accuracy = grounded_count / ground_truth_count

    return {
        "k_retrieved": k_retrieved,
        "ground_truth_count": ground_truth_count,
        "precision_at_k": round(precision_at_k, 4),
        "recall_at_k": round(recall_at_k, 4) if recall_at_k is not None else None,
        "reciprocal_rank": round(reciprocal_rank, 4),
        "grounding_accuracy": round(grounding_accuracy, 4) if grounding_accuracy is not None else None,
        "matched_ground_truth_ids": sorted(matched),
        "unmatched_ground_truth_ids": sorted(unmatched),
    }


def compute_query_metrics(record: dict[str, Any]) -> QueryMetrics:
    """
    Score a single MAIN BENCHMARK result (keyed by mode) against its
    ground_truth_nodes. See _score_retrieval for the actual math.
    """
    ground_truth = record.get("ground_truth_nodes", [])

    if not record.get("success"):
        return QueryMetrics(
            query_id=record["query_id"],
            mode=record["mode"],
            category=record.get("category", "unknown"),
            expected_hybrid_advantage=record.get("expected_hybrid_advantage", False),
            success=False,
            k_retrieved=0,
            ground_truth_count=len(_node_ids(ground_truth)),
            precision_at_k=None,
            recall_at_k=None,
            reciprocal_rank=None,
            grounding_accuracy=None,
            citation_correctness_rate=None,
            latency_seconds=record.get("latency_seconds"),
            matched_ground_truth_ids=[],
            unmatched_ground_truth_ids=sorted(_node_ids(ground_truth)),
        )

    scored = _score_retrieval(
        record.get("chunks_used", []), ground_truth, record.get("citations", [])
    )

    return QueryMetrics(
        query_id=record["query_id"],
        mode=record["mode"],
        category=record.get("category", "unknown"),
        expected_hybrid_advantage=record.get("expected_hybrid_advantage", False),
        success=True,
        citation_correctness_rate=record.get("citation_correctness_rate"),
        latency_seconds=record.get("latency_seconds"),
        **scored,
    )


@dataclass
class AblationQueryMetrics:
    """Metrics computed for one (query_id, hop_depth) ablation result.
    Mirrors QueryMetrics but keyed by hop_depth instead of mode, since
    every ablation call is hybrid mode by construction."""

    query_id: str
    hop_depth: int
    category: str
    success: bool
    k_retrieved: int
    ground_truth_count: int
    precision_at_k: float | None
    recall_at_k: float | None
    reciprocal_rank: float | None
    grounding_accuracy: float | None
    citation_correctness_rate: float | None
    latency_seconds: float | None
    reused_from_main_benchmark: bool
    reuse_confidence: str
    matched_ground_truth_ids: list[str]
    unmatched_ground_truth_ids: list[str]


def compute_ablation_query_metrics(record: dict[str, Any]) -> AblationQueryMetrics:
    """Score a single ABLATION result (keyed by hop_depth). Uses the
    exact same _score_retrieval core as the main benchmark path -- a
    reused hop=2 record is scored identically to a fresh one, since its
    chunks_used/citations were copied verbatim from the original
    successful call."""
    ground_truth = record.get("ground_truth_nodes", [])

    if not record.get("success"):
        return AblationQueryMetrics(
            query_id=record["query_id"],
            hop_depth=record["hop_depth"],
            category=record.get("category", "unknown"),
            success=False,
            k_retrieved=0,
            ground_truth_count=len(_node_ids(ground_truth)),
            precision_at_k=None,
            recall_at_k=None,
            reciprocal_rank=None,
            grounding_accuracy=None,
            citation_correctness_rate=None,
            latency_seconds=record.get("latency_seconds"),
            reused_from_main_benchmark=record.get("reused_from_main_benchmark", False),
            reuse_confidence=record.get("reuse_confidence", ""),
            matched_ground_truth_ids=[],
            unmatched_ground_truth_ids=sorted(_node_ids(ground_truth)),
        )

    scored = _score_retrieval(
        record.get("chunks_used", []), ground_truth, record.get("citations", [])
    )

    return AblationQueryMetrics(
        query_id=record["query_id"],
        hop_depth=record["hop_depth"],
        category=record.get("category", "unknown"),
        success=True,
        citation_correctness_rate=record.get("citation_correctness_rate"),
        latency_seconds=record.get("latency_seconds"),
        reused_from_main_benchmark=record.get("reused_from_main_benchmark", False),
        reuse_confidence=record.get("reuse_confidence", ""),
        **scored,
    )


def load_ablation_results(path: Path) -> list[dict[str, Any]]:
    """Load and deduplicate ablation_results.jsonl the same way
    load_results + deduplicate_latest_per_pair handle the main benchmark
    -- keyed on (query_id, hop_depth) instead of (query_id, mode),
    preferring a success over a failure, latest wins on ties."""
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    best: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        key = (record["query_id"], record["hop_depth"])
        existing = best.get(key)
        if existing is None:
            best[key] = record
        elif record["success"] and not existing["success"]:
            best[key] = record
        elif record["success"] == existing["success"]:
            best[key] = record
    return list(best.values())


def aggregate_ablation_by_hop_depth(
    ablation_metrics: list[AblationQueryMetrics],
) -> dict[int, dict[str, Any]]:
    """Mean metrics per hop_depth across all ablated queries -- shows
    whether precision/recall/grounding shift as hop_depth increases."""
    by_depth: dict[int, list[AblationQueryMetrics]] = defaultdict(list)
    for am in ablation_metrics:
        by_depth[am.hop_depth].append(am)

    summary: dict[int, dict[str, Any]] = {}
    for depth, items in sorted(by_depth.items()):
        successful = [a for a in items if a.success]
        summary[depth] = {
            "total_queries": len(items),
            "successful_queries": len(successful),
            "mean_precision_at_k": _mean([a.precision_at_k for a in successful]),
            "mean_recall_at_k": _mean([a.recall_at_k for a in successful]),
            "mrr": _mean([a.reciprocal_rank for a in successful]),
            "mean_grounding_accuracy": _mean([a.grounding_accuracy for a in successful]),
            "mean_latency_seconds": _mean([a.latency_seconds for a in successful]),
        }
    return summary


def aggregate_ablation_by_query(
    ablation_metrics: list[AblationQueryMetrics],
) -> dict[str, dict[int, dict[str, Any]]]:
    """Per-query view across all 4 hop depths side by side -- this IS
    the hop-depth ablation table: for query q05, what did hop=0/1/2/3
    each retrieve and score?"""
    by_query: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for am in ablation_metrics:
        by_query[am.query_id][am.hop_depth] = {
            "success": am.success,
            "precision_at_k": am.precision_at_k,
            "recall_at_k": am.recall_at_k,
            "mrr": am.reciprocal_rank,
            "grounding_accuracy": am.grounding_accuracy,
            "matched_ground_truth_ids": am.matched_ground_truth_ids,
            "reused_from_main_benchmark": am.reused_from_main_benchmark,
            "reuse_confidence": am.reuse_confidence,
        }
    return dict(by_query)


def _mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else None


def aggregate_by_mode(query_metrics: list[QueryMetrics]) -> dict[str, dict[str, Any]]:
    """Aggregate mean metrics per mode (semantic vs hybrid) -- this is
    the headline comparison the project's research question is about."""
    by_mode: dict[str, list[QueryMetrics]] = defaultdict(list)
    for qm in query_metrics:
        by_mode[qm.mode].append(qm)

    summary: dict[str, dict[str, Any]] = {}
    for mode, items in by_mode.items():
        successful = [q for q in items if q.success]
        summary[mode] = {
            "total_queries": len(items),
            "successful_queries": len(successful),
            "failed_queries": len(items) - len(successful),
            "mean_precision_at_k": _mean([q.precision_at_k for q in successful]),
            "mean_recall_at_k": _mean([q.recall_at_k for q in successful]),
            "mrr": _mean([q.reciprocal_rank for q in successful]),
            "mean_grounding_accuracy": _mean([q.grounding_accuracy for q in successful]),
            "mean_citation_correctness_rate": _mean(
                [q.citation_correctness_rate for q in successful]
            ),
            "mean_latency_seconds": _mean([q.latency_seconds for q in successful]),
        }
    return summary


def aggregate_by_category_and_mode(
    query_metrics: list[QueryMetrics],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Aggregate per (category, mode) -- lets you see, e.g., whether
    hybrid's advantage (if any) is concentrated in cross_file_2hop
    queries as predicted, versus showing no difference on local_0hop
    control queries as also predicted."""
    by_cat_mode: dict[str, dict[str, list[QueryMetrics]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for qm in query_metrics:
        by_cat_mode[qm.category][qm.mode].append(qm)

    summary: dict[str, dict[str, dict[str, Any]]] = {}
    for category, by_mode in by_cat_mode.items():
        summary[category] = {}
        for mode, items in by_mode.items():
            successful = [q for q in items if q.success]
            summary[category][mode] = {
                "total_queries": len(items),
                "successful_queries": len(successful),
                "mean_precision_at_k": _mean([q.precision_at_k for q in successful]),
                "mean_recall_at_k": _mean([q.recall_at_k for q in successful]),
                "mrr": _mean([q.reciprocal_rank for q in successful]),
                "mean_grounding_accuracy": _mean([q.grounding_accuracy for q in successful]),
            }
    return summary


def print_summary(
    by_mode: dict[str, dict[str, Any]],
    by_category: dict[str, dict[str, dict[str, Any]]],
    total_unique_pairs: int,
    total_successful_pairs: int,
    expected_total_pairs: int,
) -> None:
    # Partial/full status must be based on how many pairs actually
    # SUCCEEDED, not how many pairs merely have a recorded attempt --
    # a pair that failed every time it was tried still shows up once
    # deduplicated, so counting deduped rows alone would mislabel a
    # heavily-incomplete run as "full".
    is_partial = total_successful_pairs < expected_total_pairs

    print()
    print("=" * 72)
    if is_partial:
        print(
            f"  PARTIAL RESULTS -- {total_successful_pairs}/{expected_total_pairs} "
            f"(query_id, mode) pairs SUCCEEDED so far "
            f"({total_unique_pairs} pairs have at least one recorded attempt)."
        )
        print("  These numbers WILL change as the remaining pairs are collected.")
        print("  Do not treat this as the final Phase 5 result.")
    else:
        print(f"  FULL RESULTS -- all {expected_total_pairs} (query_id, mode) pairs succeeded.")
    print("=" * 72)
    print()

    print("--- Overall: semantic vs hybrid ---")
    print(f"{'metric':<32} {'semantic':>15} {'hybrid':>15}")
    metric_keys = [
        ("successful_queries", "successful queries"),
        ("mean_precision_at_k", "mean precision@k"),
        ("mean_recall_at_k", "mean recall@k"),
        ("mrr", "MRR"),
        ("mean_grounding_accuracy", "mean grounding accuracy"),
        ("mean_citation_correctness_rate", "mean citation correctness"),
        ("mean_latency_seconds", "mean latency (s)"),
    ]
    sem = by_mode.get("semantic", {})
    hyb = by_mode.get("hybrid", {})
    for key, label in metric_keys:
        print(f"{label:<32} {str(sem.get(key, '-')):>15} {str(hyb.get(key, '-')):>15}")

    print()
    print("--- By category ---")
    for category in sorted(by_category.keys()):
        print(f"\n  {category}:")
        modes = by_category[category]
        for mode in sorted(modes.keys()):
            m = modes[mode]
            print(
                f"    {mode:<10} n={m['successful_queries']:<3} "
                f"precision@k={m['mean_precision_at_k']}  "
                f"recall@k={m['mean_recall_at_k']}  "
                f"MRR={m['mrr']}  "
                f"grounding={m['mean_grounding_accuracy']}"
            )
    print()
    print("=" * 72)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Precision@K, Recall@K, MRR, and grounding accuracy from saved benchmark results."
    )
    parser.add_argument(
        "--results", type=Path, default=DEFAULT_RESULTS_PATH,
        help=f"Path to benchmark_results.jsonl (default: {DEFAULT_RESULTS_PATH})",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_SUMMARY_PATH,
        help=f"Path to write the metrics summary JSON (default: {DEFAULT_SUMMARY_PATH})",
    )
    parser.add_argument(
        "--ablation-results", type=Path, default=DEFAULT_ABLATION_RESULTS_PATH,
        help=f"Path to ablation_results.jsonl (default: {DEFAULT_ABLATION_RESULTS_PATH}). If missing, ablation scoring is skipped, not an error.",
    )
    parser.add_argument(
        "--expected-total-pairs", type=int, default=54,
        help="Expected number of (query_id, mode) pairs for a full main-benchmark run, used only to label output as partial/full (default: 54)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_records = load_results(args.results)
    logger.info("Loaded %d raw result lines from %s", len(raw_records), args.results)

    deduped = deduplicate_latest_per_pair(raw_records)
    logger.info("Deduplicated to %d unique (query_id, mode) pairs", len(deduped))

    query_metrics = [compute_query_metrics(r) for r in deduped]

    by_mode = aggregate_by_mode(query_metrics)
    by_category = aggregate_by_category_and_mode(query_metrics)

    total_successful_pairs = sum(1 for qm in query_metrics if qm.success)

    print_summary(
        by_mode, by_category,
        total_unique_pairs=len(deduped),
        total_successful_pairs=total_successful_pairs,
        expected_total_pairs=args.expected_total_pairs,
    )

    # --- Ablation scoring (optional -- skipped cleanly if the file
    # doesn't exist yet, so this script works before AND after the
    # ablation run without any code change). ---
    ablation_output: dict[str, Any] = {}
    ablation_records = load_ablation_results(args.ablation_results)
    if ablation_records:
        ablation_metrics = [compute_ablation_query_metrics(r) for r in ablation_records]
        ablation_by_depth = aggregate_ablation_by_hop_depth(ablation_metrics)
        ablation_by_query = aggregate_ablation_by_query(ablation_metrics)

        print()
        print("=" * 72)
        print(f"  ABLATION: {len(ablation_metrics)} (query_id, hop_depth) pairs scored")
        print("=" * 72)
        for depth, stats in sorted(ablation_by_depth.items()):
            print(
                f"  hop={depth}  n={stats['successful_queries']:<3} "
                f"precision@k={stats['mean_precision_at_k']}  "
                f"recall@k={stats['mean_recall_at_k']}  "
                f"MRR={stats['mrr']}  "
                f"grounding={stats['mean_grounding_accuracy']}"
            )
        print("=" * 72)

        ablation_output = {
            "sample_size": {
                "pairs_scored": len(ablation_metrics),
                "expected_pairs": 24,  # 6 queries x 4 hop depths, per the ablation design
                "is_partial": len(ablation_metrics) < 24,
            },
            "by_hop_depth": ablation_by_depth,
            "by_query": ablation_by_query,
        }
        logger.info("Scored %d ablation pairs", len(ablation_metrics))
    else:
        logger.info(
            "No ablation results found at %s -- skipping ablation scoring "
            "(this is expected if the ablation run hasn't happened yet).",
            args.ablation_results,
        )

    output_data = {
        "sample_size": {
            "unique_pairs_with_attempt": len(deduped),
            "successful_pairs": total_successful_pairs,
            "expected_total_pairs": args.expected_total_pairs,
            "is_partial": total_successful_pairs < args.expected_total_pairs,
        },
        "by_mode": by_mode,
        "by_category_and_mode": by_category,
        "per_query": [asdict(qm) for qm in query_metrics],
        "ablation": ablation_output,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    logger.info("Wrote metrics summary to %s", args.output)


if __name__ == "__main__":
    main()