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


def compute_query_metrics(record: dict[str, Any]) -> QueryMetrics:
    """
    Score a single benchmark result against its ground_truth_nodes.

    Precision@K / Recall@K are computed over ALL chunks actually
    retrieved (K = len(chunks_used)) rather than an arbitrary fixed
    cutoff, because that set is exactly what was sent to the LLM as
    context -- it's the retrieval quality that actually mattered for the
    answer that was generated, not a hypothetical top-5.

    MRR uses the rank (1-indexed, by score order in chunks_used) of the
    first ground-truth node found; 0.0 if none were retrieved at all.

    Grounding accuracy is answer-level, not retrieval-level: of the
    ground-truth nodes, what fraction were actually CITED in the final
    answer (citation line range overlaps the node's line range)? This is
    stricter than recall -- a chunk can be retrieved but never used in
    the answer, and this metric would correctly not credit that.
    """
    ground_truth = record.get("ground_truth_nodes", [])
    ground_truth_ids = _node_ids(ground_truth)
    ground_truth_count = len(ground_truth_ids)

    if not record.get("success"):
        return QueryMetrics(
            query_id=record["query_id"],
            mode=record["mode"],
            category=record.get("category", "unknown"),
            expected_hybrid_advantage=record.get("expected_hybrid_advantage", False),
            success=False,
            k_retrieved=0,
            ground_truth_count=ground_truth_count,
            precision_at_k=None,
            recall_at_k=None,
            reciprocal_rank=None,
            grounding_accuracy=None,
            citation_correctness_rate=None,
            latency_seconds=record.get("latency_seconds"),
            matched_ground_truth_ids=[],
            unmatched_ground_truth_ids=sorted(ground_truth_ids),
        )

    chunks_used = record.get("chunks_used", [])
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

    # Grounding accuracy: does the answer's citations actually cover the
    # ground-truth nodes, by file + overlapping line range? Only citations
    # the API already validated (is_valid=True) count -- an invalid
    # citation (e.g. a malformed or oversized line range that doesn't
    # correspond to any real retrieved chunk) must not be allowed to
    # spuriously "cover" a ground-truth node just because its range is
    # wide. Confirmed empirically: q14/semantic had a citation
    # "(exceptions.py:35-359)" flagged is_valid=False for
    # line_range_mismatch, and without this filter its huge range
    # accidentally overlapped a real ground-truth node, inflating that
    # query's grounding score from 0.0 to 0.5.
    citations = [c for c in record.get("citations", []) if c.get("is_valid")]
    grounding_accuracy = None
    if ground_truth_count > 0:
        grounded_count = 0
        for gt_node in ground_truth:
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
                for c in citations
            )
            if is_grounded:
                grounded_count += 1
        grounding_accuracy = grounded_count / ground_truth_count

    return QueryMetrics(
        query_id=record["query_id"],
        mode=record["mode"],
        category=record.get("category", "unknown"),
        expected_hybrid_advantage=record.get("expected_hybrid_advantage", False),
        success=True,
        k_retrieved=k_retrieved,
        ground_truth_count=ground_truth_count,
        precision_at_k=round(precision_at_k, 4),
        recall_at_k=round(recall_at_k, 4) if recall_at_k is not None else None,
        reciprocal_rank=round(reciprocal_rank, 4),
        grounding_accuracy=round(grounding_accuracy, 4) if grounding_accuracy is not None else None,
        citation_correctness_rate=record.get("citation_correctness_rate"),
        latency_seconds=record.get("latency_seconds"),
        matched_ground_truth_ids=sorted(matched),
        unmatched_ground_truth_ids=sorted(unmatched),
    )


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
        "--expected-total-pairs", type=int, default=54,
        help="Expected number of (query_id, mode) pairs for a full run, used only to label output as partial/full (default: 54)",
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
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    logger.info("Wrote metrics summary to %s", args.output)


if __name__ == "__main__":
    main()