"""
evaluation/evaluation.py

Single responsibility: consume metrics_summary.json (produced by
metrics.py, which itself consumes benchmark_results.jsonl and
ablation_results.jsonl) and organize it into a structured, machine-
readable evaluation_report.json covering:

  1. Semantic vs hybrid aggregate comparison
  2. Per-query comparison
  3. Hop-depth 0/1/2/3 ablation findings
  4. Queries where hybrid wins
  5. Queries where semantic wins/equal
  6. Interesting failure/limitation cases
  7. Citation correctness/truncation summary
  8. Machine-readable evaluation output (this file's entire output)

WHAT THIS FILE DELIBERATELY DOES NOT DO
----------------------------------------
This module computes NOTHING -- all precision/recall/MRR/grounding math
lives in metrics.py exactly once, in _score_retrieval(). This file only
reads already-computed numbers and organizes/categorizes them.

It never:
  - declares an overall winner ("hybrid is better than semantic")
  - writes narrative/prose sentences interpreting the results
  - blends multiple metrics into a single composite score
  - hides whether the underlying data is partial or complete

Every "wins" categorization uses an explicit, disclosed, deterministic
rule (see WIN_RULE below) applied per-query-per-metric. Where metrics
disagree with each other, that disagreement is reported as a fact, not
resolved into a verdict. The is_partial flag from metrics_summary.json
is propagated into every section of this file's output. Turning any of
this into a human-readable narrative report is a SEPARATE, later step
performed by a person, not by this script.

Usage
-----
    python -m evaluation.evaluation
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path
from typing import Any

# ACTIVE_REPO-driven paths, matching benchmark.py and metrics.py's fix.
from config.settings import ACTIVE_REPO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_METRICS_SUMMARY_PATH = Path(f"data/results/{ACTIVE_REPO}/metrics_summary.json")
DEFAULT_BENCHMARK_RESULTS_PATH = Path(f"data/results/{ACTIVE_REPO}/benchmark_results.jsonl")
DEFAULT_ABLATION_RESULTS_PATH = Path(f"data/results/{ACTIVE_REPO}/ablation_results.jsonl")
DEFAULT_BENCHMARKS_DIR = Path("data/benchmarks")
DEFAULT_OUTPUT_PATH = Path(f"data/results/{ACTIVE_REPO}/evaluation_report.json")

# The win/loss/tie rule is stated here, once, in plain sight -- not
# buried in a comparison function -- because it is the single most
# consequential judgment call in this whole script and must be easy for
# a human reviewer to find, question, and change.
#
# Primary metric: recall_at_k. Rationale: the project's core research
# question is about RETRIEVAL improvement (does graph expansion surface
# ground-truth code that semantic-only misses?), and recall_at_k directly
# measures "was the necessary code actually retrieved" -- which is a more
# direct test of that question than precision_at_k (which conflates
# retrieval quality with how large K happened to be) or grounding_accuracy
# (which measures whether the LLM cited what it was given, a downstream
# concern about answer generation, not retrieval).
#
# A query is "hybrid_win" only if hybrid's recall_at_k is STRICTLY higher
# AND grounding_accuracy is not strictly lower (hybrid must not win on
# retrieval while losing on whether the answer actually used what it
# retrieved). Symmetric rule for "semantic_win". Anything else --
# including exact ties, or cases where recall favors one mode but
# grounding favors the other -- is "tied_or_mixed" and reported as such,
# not forced into a bucket.
WIN_RULE_DESCRIPTION = (
    "hybrid_win: hybrid.recall_at_k > semantic.recall_at_k AND "
    "hybrid.grounding_accuracy >= semantic.grounding_accuracy "
    "(both queries must have succeeded). semantic_win is the symmetric "
    "condition. Anything else (ties, or recall/grounding disagreeing "
    "with each other) is tied_or_mixed."
)


def load_metrics_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run evaluation/metrics.py first."
        )
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _dedupe_latest_per_key(
    records: list[dict[str, Any]], key_fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Generic version of the dedup-latest-wins rule already used by
    metrics.py (deduplicate_latest_per_pair / load_ablation_results) --
    reused here rather than reimplemented, because build_failure_cases
    and build_citation_truncation_summary must never count a
    (query_id, mode) or (query_id, hop_depth) pair's SUPERSEDED failed
    retry attempts as if they were still-standing failures. Confirmed
    empirically: without this, a run with 152 raw lines for only 54
    unique pairs reported '104 API failures' -- counting historical
    retry attempts that later succeeded, not real unresolved failures."""
    best: dict[tuple, dict[str, Any]] = {}
    for record in records:
        key = tuple(record[f] for f in key_fields)
        existing = best.get(key)
        if existing is None:
            best[key] = record
        elif record.get("success") and not existing.get("success"):
            best[key] = record
        elif record.get("success") == existing.get("success"):
            best[key] = record  # later record wins on a tie
    return list(best.values())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_deduped_benchmark_records(path: Path) -> list[dict[str, Any]]:
    """Main benchmark results, deduplicated to one record per
    (query_id, mode) -- latest success wins over any earlier failed
    attempt, matching metrics.py's deduplicate_latest_per_pair exactly."""
    return _dedupe_latest_per_key(load_jsonl(path), ("query_id", "mode"))


def load_deduped_ablation_records(path: Path) -> list[dict[str, Any]]:
    """Ablation results, deduplicated to one record per
    (query_id, hop_depth), same rule as metrics.py's
    load_ablation_results."""
    return _dedupe_latest_per_key(load_jsonl(path), ("query_id", "hop_depth"))


def load_query_notes(benchmarks_dir: Path) -> dict[str, str]:
    """Pull each query's human-written 'notes' field forward from the
    original batch files, keyed by query_id. This is NOT this script
    inventing commentary -- it's propagating context a human (the person
    who designed the ground truth) already wrote, so later human
    interpretation has it without re-deriving it."""
    notes: dict[str, str] = {}
    pattern = str(benchmarks_dir / f"{ACTIVE_REPO}_queries_batch*.json")
    for path_str in sorted(glob.glob(pattern)):
        with open(path_str, encoding="utf-8") as f:
            data = json.load(f)
        for q in data.get("queries", []):
            notes[q["query_id"]] = q.get("notes", "")
    return notes


# --- Section 1: Semantic vs hybrid aggregate comparison -----------------

def build_aggregate_comparison(summary: dict[str, Any]) -> dict[str, Any]:
    """Directly re-exposes metrics.py's by_mode and by_category_and_mode
    sections. No new computation -- this section exists so the report is
    self-contained without requiring a second file open."""
    return {
        "by_mode": summary.get("by_mode", {}),
        "by_category_and_mode": summary.get("by_category_and_mode", {}),
    }


# --- Section 2: Per-query comparison ------------------------------------

def build_per_query_comparison(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per query_id with semantic and hybrid side by side, plus
    the raw delta on each metric (hybrid - semantic). Deltas are signed
    numbers only -- no adjective ("better"/"worse") is attached here;
    that judgment is applied separately and explicitly in build_win_loss_categorization."""
    per_query = summary.get("per_query", [])
    by_query: dict[str, dict[str, Any]] = {}
    for qm in per_query:
        by_query.setdefault(qm["query_id"], {})[qm["mode"]] = qm

    rows = []
    for query_id, modes in sorted(by_query.items()):
        sem = modes.get("semantic")
        hyb = modes.get("hybrid")
        row: dict[str, Any] = {
            "query_id": query_id,
            "category": (sem or hyb or {}).get("category", "unknown"),
            "semantic": sem,
            "hybrid": hyb,
            "delta": None,
        }
        if sem and hyb and sem["success"] and hyb["success"]:
            def _delta(field: str) -> float | None:
                a, b = sem.get(field), hyb.get(field)
                return round(b - a, 4) if a is not None and b is not None else None

            row["delta"] = {
                "precision_at_k": _delta("precision_at_k"),
                "recall_at_k": _delta("recall_at_k"),
                "reciprocal_rank": _delta("reciprocal_rank"),
                "grounding_accuracy": _delta("grounding_accuracy"),
                "citation_correctness_rate": _delta("citation_correctness_rate"),
                "latency_seconds": _delta("latency_seconds"),
            }
        rows.append(row)
    return rows


# --- Section 3: Hop-depth ablation findings -----------------------------

def build_ablation_findings(summary: dict[str, Any]) -> dict[str, Any]:
    """Directly re-exposes metrics.py's ablation section (by_hop_depth,
    by_query) plus one additional, purely descriptive computation: for
    each ablated query, does the SET of retrieved node_ids actually
    change between consecutive hop depths? This answers "does increasing
    hop_depth change what gets retrieved at all" without asserting
    whether that change was good or bad."""
    ablation = summary.get("ablation", {})
    if not ablation:
        return {"available": False, "reason": "No ablation data in metrics_summary.json yet."}

    by_query = ablation.get("by_query", {})
    retrieval_set_changes: dict[str, dict[str, Any]] = {}
    for query_id, by_depth in by_query.items():
        depths_present = sorted(d for d in by_depth.keys())
        changes = {}
        for i in range(len(depths_present) - 1):
            d1, d2 = depths_present[i], depths_present[i + 1]
            ids1 = set(by_depth[d1].get("matched_ground_truth_ids", []))
            ids2 = set(by_depth[d2].get("matched_ground_truth_ids", []))
            changes[f"hop{d1}_to_hop{d2}"] = {
                "matched_ground_truth_ids_changed": ids1 != ids2,
                "newly_matched": sorted(ids2 - ids1),
                "no_longer_matched": sorted(ids1 - ids2),
            }
        retrieval_set_changes[query_id] = changes

    return {
        "available": True,
        "sample_size": ablation.get("sample_size", {}),
        "by_hop_depth": ablation.get("by_hop_depth", {}),
        "by_query": by_query,
        "ground_truth_match_changes_between_depths": retrieval_set_changes,
    }


# --- Sections 4 & 5: Win/loss/tied categorization ------------------------

def build_win_loss_categorization(
    per_query_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Applies WIN_RULE_DESCRIPTION (stated once, at module level) to
    every query with both modes successful. Queries missing one or both
    modes (e.g. still-pending API calls) are listed separately as
    'incomplete', never silently dropped or guessed at."""
    hybrid_win: list[str] = []
    semantic_win: list[str] = []
    tied_or_mixed: list[str] = []
    incomplete: list[str] = []

    for row in per_query_rows:
        sem, hyb = row["semantic"], row["hybrid"]
        if not (sem and hyb and sem["success"] and hyb["success"]):
            incomplete.append(row["query_id"])
            continue

        sem_recall = sem.get("recall_at_k")
        hyb_recall = hyb.get("recall_at_k")
        sem_ground = sem.get("grounding_accuracy") or 0.0
        hyb_ground = hyb.get("grounding_accuracy") or 0.0

        if sem_recall is None or hyb_recall is None:
            tied_or_mixed.append(row["query_id"])
            continue

        if hyb_recall > sem_recall and hyb_ground >= sem_ground:
            hybrid_win.append(row["query_id"])
        elif sem_recall > hyb_recall and sem_ground >= hyb_ground:
            semantic_win.append(row["query_id"])
        else:
            tied_or_mixed.append(row["query_id"])

    return {
        "rule": WIN_RULE_DESCRIPTION,
        "hybrid_win": sorted(hybrid_win),
        "semantic_win": sorted(semantic_win),
        "tied_or_mixed": sorted(tied_or_mixed),
        "incomplete_missing_a_mode": sorted(incomplete),
    }


# --- Section 6: Failure / limitation cases -------------------------------

def build_failure_cases(
    benchmark_records: list[dict[str, Any]],
    ablation_records: list[dict[str, Any]],
    notes_by_query: dict[str, str],
) -> dict[str, Any]:
    """Flags records matching objective criteria -- success=False,
    truncated=True, citation_correctness_rate < 1.0, or
    grounding_accuracy == 0 on a successful call. Each flagged record is
    reported with its raw evidence (error text, answer excerpt) and the
    query's pre-existing notes field, if any -- no new commentary is
    added here."""

    def _excerpt(text: str, n: int = 300) -> str:
        return text[:n] + ("..." if len(text) > n else "")

    api_failures = [
        {
            "query_id": r["query_id"],
            "mode": r.get("mode", r.get("hop_depth")),
            "error": r.get("error", ""),
            "http_status": r.get("http_status"),
        }
        for r in benchmark_records + ablation_records
        if not r.get("success")
    ]

    truncated = [
        {
            "query_id": r["query_id"],
            "mode": r.get("mode", r.get("hop_depth")),
            "answer_excerpt": _excerpt(r.get("answer", "")),
            "notes": notes_by_query.get(r["query_id"], ""),
        }
        for r in benchmark_records + ablation_records
        if r.get("success") and r.get("truncated") is True
    ]

    imperfect_citations = [
        {
            "query_id": r["query_id"],
            "mode": r.get("mode", r.get("hop_depth")),
            "citation_correctness_rate": r.get("citation_correctness_rate"),
            "invalid_citations": [
                c for c in r.get("citations", []) if not c.get("is_valid")
            ],
        }
        for r in benchmark_records + ablation_records
        if r.get("success") and (r.get("citation_correctness_rate") or 1.0) < 1.0
    ]

    return {
        "api_failures": api_failures,
        "truncated_answers": truncated,
        "imperfect_citation_correctness": imperfect_citations,
    }


# --- Section 7: Citation correctness / truncation summary ---------------

def build_citation_truncation_summary(
    benchmark_records: list[dict[str, Any]],
    ablation_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Raw counts pulled directly from the result files -- deliberately
    separate from metrics.py's mean_citation_correctness_rate (which is
    an average), since a mean can hide, e.g., '95% correct on average'
    meaning 'one query was badly wrong' vs 'every query was slightly
    off'. Both views matter and neither substitutes for the other."""

    def _summarize(records: list[dict[str, Any]], label: str) -> dict[str, Any]:
        successful = [r for r in records if r.get("success")]
        return {
            "total_successful_calls": len(successful),
            "truncated_count": sum(1 for r in successful if r.get("truncated") is True),
            "perfect_citation_correctness_count": sum(
                1 for r in successful if (r.get("citation_correctness_rate") or 0) == 1.0
            ),
            "imperfect_citation_correctness_count": sum(
                1 for r in successful if (r.get("citation_correctness_rate") or 1.0) < 1.0
            ),
            "source": label,
        }

    return {
        "main_benchmark": _summarize(benchmark_records, "benchmark_results.jsonl"),
        "ablation": _summarize(ablation_records, "ablation_results.jsonl"),
    }


def build_evaluation_report(
    metrics_summary_path: Path,
    benchmark_results_path: Path,
    ablation_results_path: Path,
    benchmarks_dir: Path,
) -> dict[str, Any]:
    summary = load_metrics_summary(metrics_summary_path)
    benchmark_records = load_deduped_benchmark_records(benchmark_results_path)
    ablation_records = load_deduped_ablation_records(ablation_results_path)
    notes_by_query = load_query_notes(benchmarks_dir)

    per_query_rows = build_per_query_comparison(summary)

    report = {
        "meta": {
            "generated_from": {
                "metrics_summary": str(metrics_summary_path),
                "benchmark_results": str(benchmark_results_path),
                "ablation_results": str(ablation_results_path),
            },
            "main_benchmark_sample_size": summary.get("sample_size", {}),
            "ablation_sample_size": summary.get("ablation", {}).get("sample_size", {}),
            "note": (
                "This report is measurement and categorization ONLY. "
                "It contains no narrative conclusions and does not "
                "declare an overall winner. See each section's "
                "is_partial / sample_size fields before drawing any "
                "conclusion from partial data. Human interpretation is "
                "a separate, later step."
            ),
        },
        "1_aggregate_comparison": build_aggregate_comparison(summary),
        "2_per_query_comparison": per_query_rows,
        "3_hop_depth_ablation": build_ablation_findings(summary),
        "4_5_win_loss_categorization": build_win_loss_categorization(per_query_rows),
        "6_failure_and_limitation_cases": build_failure_cases(
            benchmark_records, ablation_records, notes_by_query
        ),
        "7_citation_and_truncation_summary": build_citation_truncation_summary(
            benchmark_records, ablation_records
        ),
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Organize metrics_summary.json into a structured, machine-readable evaluation report. Computes nothing new -- pure categorization of already-scored data."
    )
    parser.add_argument("--metrics-summary", type=Path, default=DEFAULT_METRICS_SUMMARY_PATH)
    parser.add_argument("--benchmark-results", type=Path, default=DEFAULT_BENCHMARK_RESULTS_PATH)
    parser.add_argument("--ablation-results", type=Path, default=DEFAULT_ABLATION_RESULTS_PATH)
    parser.add_argument("--benchmarks-dir", type=Path, default=DEFAULT_BENCHMARKS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    report = build_evaluation_report(
        args.metrics_summary, args.benchmark_results,
        args.ablation_results, args.benchmarks_dir,
    )

    is_partial = report["meta"]["main_benchmark_sample_size"].get("is_partial", True)
    print()
    print("=" * 72)
    if is_partial:
        n = report["meta"]["main_benchmark_sample_size"]
        print(
            f"  PARTIAL EVALUATION REPORT -- "
            f"{n.get('successful_pairs', '?')}/{n.get('expected_total_pairs', '?')} "
            f"main benchmark pairs complete."
        )
        print("  Sections below reflect ONLY the data collected so far.")
    else:
        print("  FULL EVALUATION REPORT -- main benchmark complete.")
    print("=" * 72)

    wl = report["4_5_win_loss_categorization"]
    print(f"\nhybrid_win: {len(wl['hybrid_win'])} queries -> {wl['hybrid_win']}")
    print(f"semantic_win: {len(wl['semantic_win'])} queries -> {wl['semantic_win']}")
    print(f"tied_or_mixed: {len(wl['tied_or_mixed'])} queries -> {wl['tied_or_mixed']}")
    print(f"incomplete (missing a mode): {len(wl['incomplete_missing_a_mode'])} queries -> {wl['incomplete_missing_a_mode']}")
    print(f"\nRule used: {wl['rule']}")

    ablation = report["3_hop_depth_ablation"]
    print(f"\nAblation data available: {ablation.get('available')}")

    failures = report["6_failure_and_limitation_cases"]
    print(f"\nAPI failures logged: {len(failures['api_failures'])}")
    print(f"Truncated answers: {len(failures['truncated_answers'])}")
    print(f"Imperfect citation correctness: {len(failures['imperfect_citation_correctness'])}")
    print("=" * 72)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote evaluation report to %s", args.output)


if __name__ == "__main__":
    main()