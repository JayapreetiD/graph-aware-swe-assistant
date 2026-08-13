"""
graph/graph_service.py

Query interface over an ALREADY-BUILT graph. graph_builder.py's job
was "construct the graph once." This module's job is "answer
questions about it" - specifically the operation Phase 3 (hybrid
retrieval) needs: given a starting node (from semantic search),
what's reachable within N hops?

NOT in scope here (deliberately): any scoring, ranking, or decay by
hop distance - that's hybrid_retriever.py's job, built ON TOP of this
module's raw neighbor data, not inside it. This module only answers
"what's reachable," never "what's more relevant."
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Literal

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from graph.graph_builder import load_graph  # noqa: E402

logger = logging.getLogger(__name__)

Direction = Literal["out", "in", "both"]


def get_node(graph: nx.DiGraph, node_id: str) -> dict | None:
    """
    Returns the node's attribute dict, or None if it doesn't exist.
    None instead of raising: callers (retrieval code, given a
    node_id from Qdrant) need to handle "this ID isn't in the graph"
    as a normal case - e.g. the graph was built from an older repo
    snapshot than the vector index - not as an exceptional one.
    """
    if node_id not in graph:
        return None
    return dict(graph.nodes[node_id])


def get_neighbors(
    graph: nx.DiGraph,
    node_id: str,
    hops: int = 1,
    edge_types: set[str] | None = None,
    direction: Direction = "both",
) -> dict[str, int]:
    """
    Breadth-first expansion from node_id, up to `hops` hops away.

    Returns {neighbor_node_id: hop_distance}, NOT including node_id
    itself - this function reports what's ADDITIONALLY reachable via
    the graph. The seed node itself is retrieval's responsibility to
    include (it came from semantic search, not from here).

    direction:
        "out"  - follow edges as directed (e.g. A calls B -> from A,
                 B is reachable; from B, A is not).
        "in"   - follow edges reversed (from B, A IS reachable).
        "both" - follow edges either way (default). Chosen as the
                 default because code relationships are useful
                 context in EITHER direction for a Q&A assistant -
                 if someone asks about function A, seeing what A
                 calls AND what calls A are both plausibly relevant.
                 Retrieval code can narrow to "out" or "in" if a
                 specific experiment (per the roadmap's hop-depth
                 ablation) calls for it.

    edge_types: if given, only edges whose `types` list intersects
        this set are followed (e.g. {"calls"} to expand along call
        graph only, ignoring imports/inherits/defines). None means
        follow all edge types.

    ALGORITHM: standard multi-source BFS, expanded one hop-layer at a
    time (not a single flood-fill), so `hops` acts as a hard ceiling
    on how far the search goes - this is deliberate, not incidental:
    it's what makes the roadmap's 0/1/2/3-hop ablation experiment
    possible to run cleanly (same function, just a different `hops`
    argument, with a precise and predictable stopping point).

    COMPLEXITY: O(nodes_within_hops + edges_within_hops) - NOT O(V+E)
    of the whole graph. BFS naturally only visits what's within the
    hop ceiling, so on a graph with hundreds of thousands of nodes,
    a 2-hop query from one node is fast regardless of total graph
    size, as long as that node's local neighborhood is small (true
    for reasonably-structured code, not guaranteed for a single
    node with pathologically many direct connections - e.g. a
    "god module" imported by everything - which is itself a useful
    thing an evaluation might surface).

    Raises KeyError if node_id isn't in the graph - unlike get_node,
    this is treated as a caller error here (you should check
    existence first, or handle the exception) since silently
    returning {} could be mistaken for "genuinely no neighbors"
    rather than "you queried a node that doesn't exist."
    """
    if node_id not in graph:
        raise KeyError(f"node_id not found in graph: {node_id}")
    if hops < 0:
        raise ValueError(f"hops must be >= 0, got {hops}")

    visited: dict[str, int] = {node_id: 0}
    frontier = [node_id]

    for depth in range(1, hops + 1):
        next_frontier: list[str] = []
        for current in frontier:
            candidate_edges = []
            if direction in ("out", "both"):
                candidate_edges.extend((v, data) for _, v, data in graph.out_edges(current, data=True))
            if direction in ("in", "both"):
                candidate_edges.extend((u, data) for u, _, data in graph.in_edges(current, data=True))

            for neighbor, data in candidate_edges:
                if edge_types is not None:
                    edge_type_list = data.get("types", [data["type"]])
                    if not (set(edge_type_list) & edge_types):
                        continue
                if neighbor not in visited:
                    visited[neighbor] = depth
                    next_frontier.append(neighbor)

        frontier = next_frontier
        if not frontier:
            break  # no more reachable nodes - stop early rather than looping to no effect

    del visited[node_id]
    return visited


# ---------------------------------------------------------------------------
# CLI - manual inspection tool
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="graph_service.py",
        description="Query a saved graph: look up a node or expand its neighbors by hop distance.",
    )
    p.add_argument("graph_path", type=str, help="Path to a saved graph (.gpickle, from graph_builder.py -o).")
    p.add_argument("node_id", type=str, help="Node ID to query, e.g. 'pkg/utils.py::Greeter.greet'.")
    p.add_argument("--hops", type=int, default=1, help="Max hop distance to expand (default 1).")
    p.add_argument(
        "--edge-types", type=str, default=None,
        help="Comma-separated edge types to follow, e.g. 'calls,inherits'. Default: all types.",
    )
    p.add_argument("--direction", choices=["out", "in", "both"], default="both")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    graph = load_graph(args.graph_path)
    logger.info("Loaded graph: %d nodes, %d edges", graph.number_of_nodes(), graph.number_of_edges())

    node_data = get_node(graph, args.node_id)
    if node_data is None:
        logger.error("Node not found: %s", args.node_id)
        return 1

    print(f"\nNode: {args.node_id}")
    print(f"  type: {node_data.get('type')}")

    edge_types = set(args.edge_types.split(",")) if args.edge_types else None
    neighbors = get_neighbors(graph, args.node_id, hops=args.hops, edge_types=edge_types, direction=args.direction)

    print(f"\nNeighbors within {args.hops} hop(s) (direction={args.direction}, edge_types={edge_types or 'all'}):")
    for neighbor_id, dist in sorted(neighbors.items(), key=lambda kv: kv[1]):
        neighbor_type = graph.nodes[neighbor_id].get("type")
        print(f"  [{dist}] {neighbor_id} ({neighbor_type})")
    print(f"\nTotal neighbors found: {len(neighbors)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())