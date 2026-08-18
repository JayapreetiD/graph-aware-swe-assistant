

"""
retrieval/chunker.py

Single responsibility: convert graph nodes (functions, methods, classes)
into self-contained, embeddable code chunks.

A "chunk" is the atomic unit that gets embedded and stored in Qdrant.
Each chunk bundles the source code text (sliced from disk), a literal
signature, docstring, filepath, line range, and node_id (the same ID
used in the graph, for graph<->vector linkage).

Design decisions (confirmed against real Phase 1 graph output):
  1. The graph stores line coordinates but not source text itself.
     This module reads the file from disk and slices the exact range.
  2. Signature is extracted literally from source (the def/class
     line, possibly spanning multiple lines), not synthesized from
     the `args` list — preserves type hints and defaults exactly.
  3. Only function/method/class nodes are chunked; module nodes are
     skipped (matches scope: function/class-level granularity).
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)

CHUNKABLE_TYPES = {"function", "method", "class"}


@dataclass
class CodeChunk:
    node_id: str
    type: str
    name: str
    qualified_name: str
    filepath: str
    start_line: int
    end_line: int
    signature: str
    docstring: str
    code: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_graph(graph_path: str | Path) -> nx.DiGraph:
    graph_path = Path(graph_path)
    with graph_path.open("rb") as f:
        graph = pickle.load(f)
    logger.info(
        "Loaded graph: %d nodes, %d edges",
        graph.number_of_nodes(), graph.number_of_edges(),
    )
    return graph


class _FileCache:
    """Caches file contents (as line lists) to avoid re-reading the same
    file once per node. click has ~600 nodes across ~15 files — without
    caching that's 600 disk reads instead of 15."""

    def __init__(self, repo_root: Path):
        self._repo_root = repo_root
        self._cache: dict[str, list[str]] = {}

    def get_lines(self, filepath: str) -> list[str]:
        if filepath not in self._cache:
            full_path = self._repo_root / filepath
            try:
                with full_path.open("r", encoding="utf-8") as f:
                    self._cache[filepath] = f.readlines()
            except OSError as e:
                logger.error("Failed to read %s: %s", full_path, e)
                self._cache[filepath] = []
        return self._cache[filepath]


def _extract_signature(lines: list[str], start_line: int, end_line: int) -> str:
    """Extract the literal def/class signature, scanning forward from
    start_line until a line ending in ':' (handles multi-line param
    lists). Bounded by end_line so a malformed file can't runaway-scan."""
    sig_lines = []
    idx = start_line - 1  # start_line is 1-indexed, lines is 0-indexed
    limit = min(end_line - 1, len(lines) - 1)
    while idx <= limit:
        line = lines[idx]
        sig_lines.append(line.rstrip("\n"))
        if line.rstrip().endswith(":"):
            break
        idx += 1
    return "\n".join(sig_lines)


def _extract_code(lines: list[str], start_line: int, end_line: int) -> str:
    """Slice the full source text for this node (inclusive line range)."""
    start_idx = start_line - 1
    end_idx = end_line
    if start_idx < 0 or end_idx > len(lines):
        logger.warning(
            "Line range [%d, %d] out of bounds (file has %d lines)",
            start_line, end_line, len(lines),
        )
    return "".join(lines[max(0, start_idx):min(end_idx, len(lines))])


def chunk_graph(graph: nx.DiGraph, repo_root: str | Path) -> list[CodeChunk]:
    repo_root = Path(repo_root)
    file_cache = _FileCache(repo_root)

    chunks: list[CodeChunk] = []
    skipped_type = 0
    skipped_missing_fields = 0
    skipped_read_error = 0

    for node_id, data in graph.nodes(data=True):
        node_type = data.get("type")
        if node_type not in CHUNKABLE_TYPES:
            skipped_type += 1
            continue

        filepath = data.get("filepath")
        start_line = data.get("start_line")
        end_line = data.get("end_line")

        if not filepath or start_line is None or end_line is None:
            logger.warning("Node %s missing filepath/line range, skipping", node_id)
            skipped_missing_fields += 1
            continue

        lines = file_cache.get_lines(filepath)
        if not lines:
            skipped_read_error += 1
            continue

        chunk = CodeChunk(
            node_id=node_id,
            type=node_type,
            name=data.get("name", ""),
            qualified_name=data.get("qualified_name", data.get("name", "")),
            filepath=filepath,
            start_line=start_line,
            end_line=end_line,
            signature=_extract_signature(lines, start_line, end_line),
            docstring=data.get("docstring", "") or "",
            code=_extract_code(lines, start_line, end_line),
        )
        chunks.append(chunk)

    logger.info(
        "Chunked %d nodes | skipped: %d non-chunkable type, %d missing fields, %d read errors",
        len(chunks), skipped_type, skipped_missing_fields, skipped_read_error,
    )
    return chunks


def save_chunks(chunks: list[CodeChunk], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in chunks], f, indent=2)
    logger.info("Saved %d chunks to %s", len(chunks), output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    GRAPH_PATH = "data/graph/graph.gpickle"
    REPO_ROOT = Path.home() / "repos" / "target-small"/ "src"
    OUTPUT_PATH = "data/chunks/chunks.json"

    graph = load_graph(GRAPH_PATH)
    chunks = chunk_graph(graph, REPO_ROOT)
    save_chunks(chunks, OUTPUT_PATH)