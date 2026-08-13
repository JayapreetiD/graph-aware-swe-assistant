

from __future__ import annotations
 
import argparse
import logging
import sys
from pathlib import Path
 
import matplotlib
matplotlib.use("Agg")  # non-interactive backend - this tool only ever
# saves a file, never opens a window. Setting this BEFORE importing
# pyplot avoids matplotlib trying (and possibly failing, on a
# terminal with no display) to pick a GUI backend automatically.
import matplotlib.pyplot as plt
import networkx as nx
 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from graph.graph_builder import load_graph  # noqa: E402
from graph.graph_service import get_neighbors  # noqa: E402
 
logger = logging.getLogger(__name__)
 
# Fixed color per node type - consistent across runs so a reader
# comparing two exported images doesn't have to re-learn the legend.
_NODE_COLORS = {
    "module": "#8ecae6",
    "class": "#ffb703",
    "function": "#8bc34a",
    "method": "#c8e6c9",
}
_DEFAULT_NODE_COLOR = "#cccccc"
 
# Fixed color per edge type, same rationale.
_EDGE_COLORS = {
    "defines": "#999999",
    "inherits": "#e63946",
    "imports": "#457b9d",
    "calls": "#2a9d8f",
}
_DEFAULT_EDGE_COLOR = "#333333"
 
 
def build_subgraph(graph: nx.DiGraph, node_id: str, hops: int) -> nx.DiGraph:
    """
    Extracts the induced subgraph of node_id + everything within
    `hops` hops (via get_neighbors), including edges BETWEEN those
    neighbor nodes (not just edges directly touching node_id) - using
    nx.subgraph, which is what makes cross-neighbor edges (e.g. two
    sibling methods that call each other) show up correctly, rather
    than only star-shaped edges radiating from the center node.
    """
    neighbor_ids = get_neighbors(graph, node_id, hops=hops)
    node_ids = {node_id, *neighbor_ids.keys()}
    return graph.subgraph(node_ids).copy()
 
 
def render_subgraph(subgraph: nx.DiGraph, center_node_id: str, output_path: str | Path) -> None:
    """
    Draws `subgraph` to a PNG file.
 
    LAYOUT CHOICE: spring_layout (force-directed). Chosen over e.g. a
    strict hierarchical layout because the graph mixes multiple edge
    types with different natural directions (defines flows downward,
    calls can flow sideways or backward, inherits flows upward) - no
    single tree-like layout represents all of them well. Force-
    directed is the standard general-purpose choice when there's no
    single dominant hierarchy to lay out along.
 
    LIMITATION: force-directed layout is somewhat non-deterministic
    in exact node placement between runs unless a fixed `seed` is
    given (which we do, for reproducible output images) - but layout
    quality still degrades on graphs with more than roughly 40-60
    nodes, becoming visually cluttered. This module is intended for
    SMALL subgraphs (matching the roadmap's own phrasing: "visualize
    a small subgraph") - for anything larger, reduce --hops rather
    than expecting this to scale to whole-repo visualization.
    """
    if subgraph.number_of_nodes() == 0:
        raise ValueError("Subgraph is empty - nothing to render.")
 
    pos = nx.spring_layout(subgraph, seed=42, k=0.9)
 
    fig, ax = plt.subplots(figsize=(12, 9))
 
    node_colors = [_NODE_COLORS.get(d["type"], _DEFAULT_NODE_COLOR) for _, d in subgraph.nodes(data=True)]
    node_sizes = [1600 if n == center_node_id else 900 for n in subgraph.nodes()]
    nx.draw_networkx_nodes(subgraph, pos, ax=ax, node_color=node_colors, node_size=node_sizes, edgecolors="black")
 
    # Highlight the center node with a thicker border rather than a
    # different color, so its TYPE color is still visible/consistent
    # with the legend.
    nx.draw_networkx_nodes(
        subgraph, pos, ax=ax, nodelist=[center_node_id],
        node_color=[_NODE_COLORS.get(subgraph.nodes[center_node_id]["type"], _DEFAULT_NODE_COLOR)],
        node_size=1600, edgecolors="black", linewidths=3,
    )
 
    # Edges: group by primary type for distinct coloring. An edge
    # with multiple types (see graph_builder.py's _add_typed_edge)
    # is drawn once using its first-listed type's color - a minor
    # simplification, noted rather than hidden, since drawing
    # multiple overlapping colored lines for one edge would be more
    # visually confusing than informative at this scale.
    #
    # LEGEND NOTE: nx.draw_networkx_edges with arrows=True returns
    # FancyArrowPatch objects, which do NOT register with
    # ax.legend() via the `label=` kwarg the way plain Line2D
    # objects do (confirmed empirically - legend() reported "no
    # artists with labels" despite edges being drawn correctly).
    # Rather than leave a silently-broken legend, we build it
    # manually below from proxy Line2D handles - the standard
    # workaround for this networkx/matplotlib interaction.
    edges_drawn_by_type: dict[str, bool] = {}
    for edge_type, color in _EDGE_COLORS.items():
        edges_of_type = [
            (u, v) for u, v, d in subgraph.edges(data=True)
            if edge_type in d.get("types", [d["type"]])
        ]
        if edges_of_type:
            nx.draw_networkx_edges(
                subgraph, pos, ax=ax, edgelist=edges_of_type, edge_color=color,
                arrows=True, arrowsize=15, connectionstyle="arc3,rad=0.1",
            )
            edges_drawn_by_type[edge_type] = True
 
    labels = {n: subgraph.nodes[n].get("qualified_name", n.split("::")[-1]) for n in subgraph.nodes()}
    nx.draw_networkx_labels(subgraph, pos, labels=labels, ax=ax, font_size=8)
 
    ax.set_title(f"Subgraph: {center_node_id}\n({subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges)")
 
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=color, lw=2, label=edge_type)
        for edge_type, color in _EDGE_COLORS.items() if edges_drawn_by_type.get(edge_type)
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper left", fontsize=8)
    ax.axis("off")
 
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved visualization to %s", output_path)
 
 
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
 
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="graph_visualizer.py",
        description="Render a small subgraph (a node + its N-hop neighbors) as a PNG image.",
    )
    p.add_argument("graph_path", type=str, help="Path to a saved graph (.gpickle).")
    p.add_argument("node_id", type=str, help="Center node ID, e.g. 'pkg/utils.py::Greeter.greet'.")
    p.add_argument("--hops", type=int, default=1, help="Hop radius around the center node (default 1).")
    p.add_argument("-o", "--output", type=str, default="data/graph/subgraph.png", help="Output PNG path.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p
 
 
def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    # matplotlib's own logger is EXTREMELY verbose at DEBUG level
    # (font-matching internals, hundreds of lines) - keep it at INFO
    # regardless of our own -v flag, since that noise never helps
    # debug THIS module's behavior.
    logging.getLogger("matplotlib").setLevel(logging.INFO)
 
    graph = load_graph(args.graph_path)
    if args.node_id not in graph:
        logger.error("Node not found: %s", args.node_id)
        return 1
 
    subgraph = build_subgraph(graph, args.node_id, args.hops)
    render_subgraph(subgraph, args.node_id, args.output)
    return 0
 
 
if __name__ == "__main__":
    sys.exit(main())