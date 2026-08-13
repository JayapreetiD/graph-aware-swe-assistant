

"""
graph/graph_builder.py
 
First cut of graph construction: nodes + `defines` edges only.
 
WHY defines-only for v1: `defines` needs zero resolution (a class
textually contains its methods - that's it). `calls`, `imports`, and
`inherits` all require resolving a string like "self.format" or
"Greeter" back to a real node, which is genuine logic that can be
wrong in non-obvious ways. Building and validating the zero-ambiguity
80% first means when call/import/inherit resolution is added next and
something breaks, you know the bug is in the new resolution code, not
buried somewhere in graph plumbing you haven't verified yet.
 
node_id SCHEME (the identifier every other module will key off of):
    Module nodes:              "<filepath>"
    Class/function/method nodes: "<filepath>::<qualified_name>"
 
Example: "pkg/utils.py::Greeter.greet"
 
This depends directly on parser.py's `qualified_name` field already
being correct (dotted, nesting-aware) - that's why the parser was
validated first.
"""
 
from __future__ import annotations
 
import argparse
import logging
import pickle
import sys
from pathlib import Path
 
import networkx as nx
 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from parser.parser import FileParseResult, parse_repository  # noqa: E402
 
logger = logging.getLogger(__name__)
 
 
def build_graph(parse_results: list[FileParseResult]) -> nx.DiGraph:
    """
    Build a DiGraph with one node per module/class/function/method and
    `defines` edges connecting each entity to its parent (class or
    module).
 
    DESIGN CHOICE: DiGraph, not MultiDiGraph. A MultiDiGraph would let
    us store every individual relationship instance as a separate
    edge. We don't need that for `defines` (an entity has exactly one
    definer, structurally), and collapsing to a single edge per
    relationship keeps this simple. This choice gets revisited when
    `calls` edges are added, since a function can call another
    multiple times - decision deferred to that step, not made here
    speculatively.
 
    Files that failed to parse (result.success == False) are skipped
    entirely - they contribute no nodes. This means the graph is
    silently incomplete for those files; parse_repository()'s log
    output is what tells you how many files that affects, so always
    check that log line, don't assume 100% coverage.
 
    Time complexity: O(total functions + classes across the repo) -
    two passes per file (nodes, then edges via dict lookup), each
    O(entities in that file).
    """
    graph = nx.DiGraph()
 
    for result in parse_results:
        if not result.success:
            continue
 
        module_id = result.filepath
        graph.add_node(module_id, type="module", filepath=result.filepath)
 
        # Maps this file's qualified_name -> node_id, so the edge pass
        # below can find a parent node without re-deriving IDs.
        qualified_to_id: dict[str, str] = {}
 
        for cls in result.classes:
            node_id = f"{result.filepath}::{cls.qualified_name}"
            qualified_to_id[cls.qualified_name] = node_id
            graph.add_node(
                node_id,
                type="class",
                name=cls.name,
                qualified_name=cls.qualified_name,
                filepath=result.filepath,
                bases=cls.bases,
                decorators=cls.decorators,
                docstring=cls.docstring or "",
                start_line=cls.start_line,
                end_line=cls.end_line,
            )
 
        for fn in result.functions:
            node_id = f"{result.filepath}::{fn.qualified_name}"
            qualified_to_id[fn.qualified_name] = node_id
            graph.add_node(
                node_id,
                type="method" if fn.is_method else "function",
                name=fn.name,
                qualified_name=fn.qualified_name,
                filepath=result.filepath,
                is_async=fn.is_async,
                args=fn.args,
                decorators=fn.decorators,
                docstring=fn.docstring or "",
                start_line=fn.start_line,
                end_line=fn.end_line,
            )
 
        # Second pass: now every entity in this file has a node_id, so
        # parent lookup by qualified_name is safe regardless of the
        # order classes/functions were visited in.
        for qualified_name, node_id in qualified_to_id.items():
            if "." in qualified_name:
                parent_qualified_name = qualified_name.rsplit(".", 1)[0]
                # Falls back to module_id if the parent qualified name
                # wasn't captured as its own node (shouldn't happen
                # given how the parser builds qualified names, but a
                # silent wrong edge is worse than a defensive fallback).
                parent_id = qualified_to_id.get(parent_qualified_name, module_id)
            else:
                parent_id = module_id
            graph.add_edge(parent_id, node_id, type="defines")
 
    return graph
 
 
def save_graph(graph: nx.DiGraph, output_path: str | Path) -> None:
    """
    Persist the graph via plain pickle.
 
    NOTE: networkx removed nx.write_gpickle/read_gpickle in 3.0. The
    roadmap's `graph.gpickle` filename is kept as a naming convention
    only - the actual serialization here is Python's stdlib `pickle`
    module directly, which is what current networkx docs recommend
    for full-fidelity graph persistence (preserves all node/edge
    attributes exactly, unlike e.g. GraphML which restricts attribute
    types).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(graph, f)
    logger.info("Saved graph to %s", output_path)
 
 
def load_graph(input_path: str | Path) -> nx.DiGraph:
    with Path(input_path).open("rb") as f:
        return pickle.load(f)
 
 
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="graph_builder.py",
        description="Build a dependency graph from a parsed Python repository.",
    )
    p.add_argument("repo_path", type=str, help="Path to the root of the Python repository to parse.")
    p.add_argument(
        "-o", "--output", type=str, default=None,
        help="Optional path to save the graph (pickle format), e.g. data/graph/graph.gpickle",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG-level logging.")
    return p
 
 
def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
 
    parse_results = parse_repository(args.repo_path)
    graph = build_graph(parse_results)
 
    node_counts: dict[str, int] = {}
    for _, data in graph.nodes(data=True):
        node_counts[data["type"]] = node_counts.get(data["type"], 0) + 1
    edge_counts: dict[str, int] = {}
    for _, _, data in graph.edges(data=True):
        edge_counts[data["type"]] = edge_counts.get(data["type"], 0) + 1
 
    print(f"\nTotal nodes: {graph.number_of_nodes()}")
    for node_type, count in sorted(node_counts.items()):
        print(f"  {node_type}: {count}")
    print(f"Total edges: {graph.number_of_edges()}")
    for edge_type, count in sorted(edge_counts.items()):
        print(f"  {edge_type}: {count}")
 
    if args.output:
        save_graph(graph, args.output)
 
    return 0
 
 
if __name__ == "__main__":
    sys.exit(main())
 