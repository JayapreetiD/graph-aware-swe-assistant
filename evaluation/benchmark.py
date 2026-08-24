"""
evaluation/benchmark.py

Single responsibility: run the ground-truthed benchmark queries (from
data/benchmarks/*.json) against the live /query API in both semantic and
hybrid modes, and save the raw responses to disk.

This module does NOT compute Precision@K / Recall@K / MRR / grounding
accuracy -- that is evaluation/metrics.py's job. It does NOT produce a
final comparison report -- that is evaluation/evaluation.py's job. Keeping
this file scoped to "call the API, save what comes back" means a failure
here (bad network call, API 500, truncated answer) is easy to isolate from
a failure in scoring logic later.

The FastAPI server is assumed to already be running (e.g. via
`python -m uvicorn api.main:app --reload`) at --base-url. This script only
sends HTTP requests to it; it never starts or manages the server process.

Usage
-----
Dry run against the first 3 queries, both modes (6 calls total):

    python -m evaluation.benchmark --limit 3

Full run against all discovered queries, both modes:

    python -m evaluation.benchmark

Only one mode:

    python -m evaluation.benchmark --modes hybrid

Re-running after an interruption skips (query_id, mode) pairs that already
have a successful result in the output file, so a partial 54-call run can
be safely resumed rather than restarted.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_BENCHMARKS_DIR = Path("data/benchmarks")
DEFAULT_OUTPUT_DIR = Path("data/results/click")
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_MODES = ("semantic", "hybrid")

# Generous client-side timeout: Gemini calls have been observed taking
# 15-30s in smoke testing, and MAX_OUTPUT_TOKENS was raised to 8192, which
# can push generation time higher. 90s leaves real headroom before we'd
# rather see a timeout error than kill a call that was about to succeed.
REQUEST_TIMEOUT_SECONDS = 90

# gemini-3.6-flash free tier is rate-limited to 20 requests/minute
# (confirmed empirically: a run with no delay hit 429 RESOURCE_EXHAUSTED
# after ~19 successful calls). A fixed delay between calls keeps a clean
# run comfortably under that limit rather than relying on retries alone.
SECONDS_BETWEEN_CALLS = 3.5

# On a 429, Gemini's error message includes "Please retry in X.Ys" -- we
# parse that and wait the stated time plus a small buffer, rather than
# guessing a fixed backoff that might be too short (retry too soon, get
# another 429) or too long (waste time when the quota already reset).
RETRY_WAIT_BUFFER_SECONDS = 2.0
MAX_RETRIES_PER_CALL = 3
FALLBACK_RETRY_WAIT_SECONDS = 30.0  # used if the retry time can't be parsed

_RETRY_SECONDS_PATTERN = re.compile(r"retry in ([\d.]+)")


def _is_rate_limit_error(http_status: int | None, error_text: str) -> bool:
    """True if this failure looks like a Gemini quota/rate-limit error
    (HTTP 429, or 502 wrapping a 429 as seen from this FastAPI backend --
    api/main.py currently surfaces LLM errors as 502 with the original
    Gemini error text embedded in the detail field)."""
    if http_status == 429:
        return True
    if "RESOURCE_EXHAUSTED" in error_text or "429" in error_text:
        return True
    return False


def _is_daily_quota_exhausted(error_text: str) -> bool:
    """
    True if the error is specifically the PerDay free-tier quota, as
    opposed to a short-lived per-minute throttle. Confirmed empirically:
    the Gemini free tier error includes
    'GenerateRequestsPerDayPerProjectPerModel-FreeTier' in its quotaId
    when the daily cap (20 requests/day) is hit. Retrying this with a
    30-60s backoff is pointless -- it will not reset until the next
    day's quota window, so this should fail fast rather than retry.
    """
    return "PerDay" in error_text


def _extract_retry_wait_seconds(error_text: str) -> float:
    """Pull the 'Please retry in X.Ys' hint out of Gemini's error message,
    if present. Falls back to a fixed wait if the pattern isn't found --
    the message format is Google's, not ours, and could change."""
    match = _RETRY_SECONDS_PATTERN.search(error_text)
    if match:
        try:
            return float(match.group(1)) + RETRY_WAIT_BUFFER_SECONDS
        except ValueError:
            pass
    return FALLBACK_RETRY_WAIT_SECONDS


@dataclass
class BenchmarkQuery:
    """One query loaded from a data/benchmarks/*.json file."""

    query_id: str
    question: str
    category: str
    expected_hybrid_advantage: bool
    ground_truth_nodes: list[dict[str, Any]]
    notes: str = ""
    source_file: str = ""


@dataclass
class BenchmarkResult:
    """
    Raw outcome of running one query against /query in one mode.

    success=False on any HTTP/network failure -- the caller (metrics.py,
    later) decides how to handle failed calls (exclude, retry, flag),
    same pattern already used for LLMResult.success in llm_service.py.
    """

    query_id: str
    mode: str
    question: str
    category: str
    expected_hybrid_advantage: bool
    ground_truth_nodes: list[dict[str, Any]]
    source_file: str
    success: bool
    http_status: int | None = None
    latency_seconds: float | None = None
    answer: str = ""
    truncated: bool | None = None
    chunks_used: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    citation_correctness_rate: float | None = None
    error: str = ""
    timestamp: str = ""


def load_queries(benchmarks_dir: Path) -> list[BenchmarkQuery]:
    """
    Load every query from every click_queries_batch*.json file found in
    benchmarks_dir. Deliberately glob-based rather than a hard-coded file
    list, so adding a batch4 file later requires no code change here.
    """
    pattern = str(benchmarks_dir / "click_queries_batch*.json")
    batch_files = sorted(glob.glob(pattern))

    if not batch_files:
        raise FileNotFoundError(
            f"No files matching {pattern} found. Expected "
            f"click_queries_batch1.json etc. under {benchmarks_dir}."
        )

    queries: list[BenchmarkQuery] = []
    for path_str in batch_files:
        path = Path(path_str)
        with path.open(encoding="utf-8") as f:
            data = json.load(f)

        batch_queries = data.get("queries", [])
        logger.info("Loaded %d queries from %s", len(batch_queries), path.name)

        for q in batch_queries:
            queries.append(
                BenchmarkQuery(
                    query_id=q["query_id"],
                    question=q["question"],
                    category=q.get("category", "unknown"),
                    expected_hybrid_advantage=q.get("expected_hybrid_advantage", False),
                    ground_truth_nodes=q.get("ground_truth_nodes", []),
                    notes=q.get("notes", ""),
                    source_file=path.name,
                )
            )

    # Fail loudly on duplicate query_ids across batches rather than
    # silently letting one shadow another -- this would otherwise corrupt
    # per-query metrics later without any visible error.
    seen_ids: dict[str, str] = {}
    for q in queries:
        if q.query_id in seen_ids:
            raise ValueError(
                f"Duplicate query_id '{q.query_id}' found in both "
                f"{seen_ids[q.query_id]} and {q.source_file}."
            )
        seen_ids[q.query_id] = q.source_file

    logger.info("Total queries loaded: %d", len(queries))
    return queries


def load_completed_pairs(output_path: Path) -> set[tuple[str, str]]:
    """
    Scan an existing JSONL results file (if any) and return the set of
    (query_id, mode) pairs that already have a successful result, so a
    resumed run can skip them instead of re-calling the API and paying for
    an LLM call that already succeeded.
    """
    if not output_path.exists():
        return set()

    completed: set[tuple[str, str]] = set()
    with output_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping unparseable line in %s", output_path)
                continue
            if record.get("success"):
                completed.add((record["query_id"], record["mode"]))

    if completed:
        logger.info(
            "Found %d already-completed (query_id, mode) pairs in %s -- these will be skipped.",
            len(completed),
            output_path,
        )
    return completed


def _post_query_once(
    base_url: str,
    question: str,
    mode: str,
) -> tuple[dict[str, Any], int | None, float, str]:
    """Single HTTP attempt, no retry logic. Returns (response_json,
    http_status, latency_seconds, error_message)."""
    url = f"{base_url.rstrip('/')}/query"
    payload = {"question": question, "mode": mode}

    start = time.monotonic()
    try:
        resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        latency = time.monotonic() - start
    except requests.exceptions.RequestException as e:
        latency = time.monotonic() - start
        return {}, None, latency, f"Request failed: {e}"

    if resp.status_code != 200:
        return {}, resp.status_code, latency, f"HTTP {resp.status_code}: {resp.text[:500]}"

    try:
        return resp.json(), resp.status_code, latency, ""
    except json.JSONDecodeError as e:
        return {}, resp.status_code, latency, f"Response was not valid JSON: {e}"


class DailyQuotaExhausted(Exception):
    """Raised to abort the entire benchmark run immediately once the
    free-tier daily quota is confirmed exhausted -- every remaining call
    would fail identically, so continuing just wastes time without
    producing any usable results."""


def run_single_query(
    base_url: str,
    question: str,
    mode: str,
) -> tuple[dict[str, Any], int | None, float, str]:
    """
    POST one query to /query, retrying on short-lived rate-limit errors
    up to MAX_RETRIES_PER_CALL times. Waits the time Gemini's own error
    message reports (plus a buffer) before each retry.

    Daily quota exhaustion is NOT retried -- see _is_daily_quota_exhausted
    -- and instead raises DailyQuotaExhausted so the caller can abort the
    whole run rather than grinding through every remaining call for
    nothing.

    Non-rate-limit failures (bad request, network error, server bug) are
    also NOT retried -- retrying those would just waste time on an error
    that a wait won't fix.

    Returns (response_json, http_status, total_latency_seconds,
    error_message). total_latency_seconds includes any time spent waiting
    on retries, so it's an honest "wall clock cost of this call", not
    just the final successful attempt's duration.
    """
    total_wait = 0.0

    for attempt in range(1, MAX_RETRIES_PER_CALL + 1):
        response_json, http_status, latency, error = _post_query_once(
            base_url, question, mode
        )
        total_latency = latency + total_wait

        if not error:
            return response_json, http_status, total_latency, ""

        if _is_daily_quota_exhausted(error):
            raise DailyQuotaExhausted(error)

        if not _is_rate_limit_error(http_status, error):
            # Not a rate-limit issue -- don't retry, fail immediately.
            return response_json, http_status, total_latency, error

        if attempt == MAX_RETRIES_PER_CALL:
            # Out of retries -- return the failure as-is.
            return response_json, http_status, total_latency, (
                f"{error} (gave up after {MAX_RETRIES_PER_CALL} attempts)"
            )

        wait_seconds = _extract_retry_wait_seconds(error)
        logger.warning(
            "  Rate limited (attempt %d/%d). Waiting %.1fs before retry...",
            attempt, MAX_RETRIES_PER_CALL, wait_seconds,
        )
        time.sleep(wait_seconds)
        total_wait += wait_seconds

    # Unreachable, but keeps type checkers happy.
    return {}, None, total_wait, "Unexpected retry loop exit"


def run_benchmark(
    queries: list[BenchmarkQuery],
    modes: tuple[str, ...],
    base_url: str,
    output_path: Path,
    limit: int | None = None,
) -> None:
    """
    Run queries x modes against the live API, appending each result to
    output_path as it completes (JSON Lines format) rather than holding
    everything in memory and writing once at the end -- a crash on call 40
    of 54 should not lose calls 1-39.
    """
    if limit is not None:
        queries = queries[:limit]
        logger.info("Dry run: limiting to first %d queries.", limit)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed_pairs = load_completed_pairs(output_path)

    total_calls = len(queries) * len(modes)
    logger.info(
        "Starting benchmark run: %d queries x %d modes = %d calls (base_url=%s)",
        len(queries),
        len(modes),
        total_calls,
        base_url,
    )

    call_num = 0
    for query in queries:
        for mode in modes:
            call_num += 1

            if (query.query_id, mode) in completed_pairs:
                logger.info(
                    "[%d/%d] SKIP %s / %s (already completed)",
                    call_num, total_calls, query.query_id, mode,
                )
                continue

            logger.info(
                "[%d/%d] Running %s / %s: %r",
                call_num, total_calls, query.query_id, mode, query.question,
            )

            try:
                response_json, http_status, latency, error = run_single_query(
                    base_url, query.question, mode
                )
            except DailyQuotaExhausted as e:
                logger.error(
                    "Free-tier DAILY quota exhausted (query %s / %s). "
                    "Stopping run here rather than burning time on %d "
                    "more calls that would all fail identically. "
                    "Progress so far is saved in %s -- re-run this same "
                    "command tomorrow (or once quota resets) to continue "
                    "from exactly this point.",
                    query.query_id, mode, total_calls - call_num + 1, output_path,
                )
                logger.error("Underlying error: %s", e)
                return

            success = not error and http_status == 200

            result = BenchmarkResult(
                query_id=query.query_id,
                mode=mode,
                question=query.question,
                category=query.category,
                expected_hybrid_advantage=query.expected_hybrid_advantage,
                ground_truth_nodes=query.ground_truth_nodes,
                source_file=query.source_file,
                success=success,
                http_status=http_status,
                latency_seconds=round(latency, 3),
                answer=response_json.get("answer", ""),
                truncated=response_json.get("truncated"),
                chunks_used=response_json.get("chunks_used", []),
                citations=response_json.get("citations", []),
                citation_correctness_rate=response_json.get("citation_correctness_rate"),
                error=error,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            )

            # Append immediately -- this is the incremental-save behavior.
            with output_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(result)) + "\n")

            if success:
                logger.info(
                    "  -> OK (%.1fs, truncated=%s, citation_correctness=%s)",
                    latency, result.truncated, result.citation_correctness_rate,
                )
            else:
                logger.error("  -> FAILED: %s", error)

            # Fixed pacing delay between calls (not applied after retry
            # waits, which already consumed real time) -- keeps a clean
            # run under the 20 req/min free-tier limit instead of hitting
            # it and relying on retries to recover.
            if call_num < total_calls:
                time.sleep(SECONDS_BETWEEN_CALLS)

    logger.info("Benchmark run complete. Results appended to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 5 benchmark queries against the live /query API."
    )
    parser.add_argument(
        "--benchmarks-dir",
        type=Path,
        default=DEFAULT_BENCHMARKS_DIR,
        help=f"Directory containing click_queries_batch*.json (default: {DEFAULT_BENCHMARKS_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "benchmark_results.jsonl",
        help=f"JSONL output file (default: {DEFAULT_OUTPUT_DIR / 'benchmark_results.jsonl'})",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_BASE_URL,
        help=f"Base URL of the running API (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default=",".join(DEFAULT_MODES),
        help=f"Comma-separated list of modes to run (default: {','.join(DEFAULT_MODES)})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="If set, only run the first N queries (dry run). Omit for full run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())

    queries = load_queries(args.benchmarks_dir)
    run_benchmark(
        queries=queries,
        modes=modes,
        base_url=args.base_url,
        output_path=args.output,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()