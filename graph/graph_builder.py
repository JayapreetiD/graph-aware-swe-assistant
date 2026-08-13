

"""
graph/graph_builder.py
 
Builds a NetworkX DiGraph from parser.py output.
 
Edge types implemented so far:
    - defines   (module -> class/function, class -> method) - trivial,
      needs no resolution, structural only.
    - inherits  (subclass -> base class) - needs RESOLUTION: the
      parser only gives us the base class as a plain string like
      "Greeter" or "click.Command" - we have to figure out which
      actual node that refers to, if any.
 
node_id SCHEME:
    Module nodes:                "<filepath>"
    Class/function/method nodes: "<filepath>::<qualified_name>"
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
 
 
# ---------------------------------------------------------------------------
# Node + defines edges (unchanged from the previous step)
# ---------------------------------------------------------------------------
 
def build_graph(parse_results: list[FileParseResult]) -> nx.DiGraph:
    """
    Build a DiGraph with one node per module/class/function/method and
    `defines` edges connecting each entity to its parent (class or
    module). See module docstring for the node_id scheme.
    """
    graph = nx.DiGraph()
 
    for result in parse_results:
        if not result.success:
            continue
 
        module_id = result.filepath
        graph.add_node(module_id, type="module", filepath=result.filepath)
 
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
 
        for qualified_name, node_id in qualified_to_id.items():
            if "." in qualified_name:
                parent_qualified_name = qualified_name.rsplit(".", 1)[0]
                parent_id = qualified_to_id.get(parent_qualified_name, module_id)
            else:
                parent_id = module_id
            graph.add_edge(parent_id, node_id, type="defines")
 
    return graph
 
 
# ---------------------------------------------------------------------------
# inherits edge resolution
# ---------------------------------------------------------------------------
 
def build_module_index(parse_results: list[FileParseResult]) -> dict[str, str]:
    """
    Maps a "guessed dotted module path" -> filepath, so an import
    statement's module string (e.g. "click.core") can be turned back
    into the actual file that defines it (e.g. "click/core.py").
 
    Two kinds of entries are added:
      1. Full dotted path from the file's relative path, e.g.
         "click/core.py" -> dotted key "click.core".
      2. The file's stem alone (e.g. "core"), but ONLY if that stem
         is unique across the whole repo. This specifically supports
         RELATIVE imports like `from .core import Command` - Python's
         ast module reports the module string for these as just
         "core" (the leading dots are a separate `level` field that
         this project's parser does not currently capture - see
         LIMITATION below). Guarding with uniqueness avoids a stem
         match confidently picking the WRONG file when multiple
         subpackages have a same-named module - in that ambiguous
         case we deliberately resolve nothing rather than guess wrong.
 
    LIMITATION: because parser.py's ImportInfo does not record the
    relative-import `level` (number of leading dots), a relative
    import is only resolved correctly when its target module's stem
    happens to be globally unique in the repo. For a mostly-flat
    package (like click's src/click/*.py) this works well in
    practice. For a repo with deeply nested subpackages containing
    same-named files, some relative imports will go unresolved rather
    than silently resolving to the wrong file - a safe failure mode,
    not a correctness bug, but worth knowing before trusting
    resolution stats on a very differently-shaped repo.
    """
    index: dict[str, str] = {}
    stem_counts: dict[str, int] = {}
    filepath_by_stem: dict[str, str] = {}
 
    for result in parse_results:
        if not result.success:
            continue
        dotted = Path(result.filepath).with_suffix("").as_posix().replace("/", ".")
        index[dotted] = result.filepath
 
        stem = Path(result.filepath).stem
        stem_counts[stem] = stem_counts.get(stem, 0) + 1
        filepath_by_stem[stem] = result.filepath
 
    for stem, count in stem_counts.items():
        if count == 1:
            index.setdefault(stem, filepath_by_stem[stem])
 
    return index
 
 
def build_class_index(graph: nx.DiGraph) -> dict[str, dict[str, str]]:
    """
    Maps filepath -> {simple_class_name: node_id}, built from the
    graph's own class nodes. Used to look up "the class named X in
    file Y" while resolving base classes.
 
    LIMITATION: keyed by simple name, not qualified_name. If a file
    has two classes with the same simple name at different nesting
    levels (rare, but possible with nested classes), the later one
    encountered wins. Not a concern for the flat class structures
    typical of most Python code, including click's.
    """
    index: dict[str, dict[str, str]] = {}
    for node_id, data in graph.nodes(data=True):
        if data["type"] != "class":
            continue
        index.setdefault(data["filepath"], {})[data["name"]] = node_id
    return index
 
 
def resolve_inheritance(
    graph: nx.DiGraph,
    parse_results: list[FileParseResult],
    module_index: dict[str, str],
    class_index: dict[str, dict[str, str]],
) -> dict[str, int]:
    """
    Adds `inherits` edges (subclass -> base class), matching the same
    directional convention as `defines`: the edge points FROM the
    thing doing the referencing TO the thing being referenced.
 
    RESOLUTION RULES (in order of attempt, per base class string):
      1. Same-file: base name matches a class in the SAME file.
      2. Imported, explicit prefix (e.g. "click.Command"): the prefix
         ("click") is looked up against this file's imports; if it
         resolves to a known module/file, look for the class there.
      3. Imported, bare name (e.g. "Command" after
         `from click.core import Command`): the bare name is looked
         up directly against this file's imports.
      4. Otherwise: UNRESOLVED. This is the expected, correct outcome
         for built-ins (Exception, object) and third-party/stdlib
         base classes not present in this repo - not a bug.
 
    Returns resolved/unresolved counts so the caller can report
    coverage rather than assuming 100%.
 
    LIMITATION, confirmed against real behavior on click's source
    (src/click/types.py): the parser has no concept of conditional
    code paths (if/else, `if TYPE_CHECKING:` blocks). A class or
    function defined twice under such a branch - a common pattern for
    providing a generics-annotated version for type-checkers and a
    plain version for runtime - is seen as two separate ClassDef
    nodes with the identical qualified_name. Both get the same
    node_id, so the second definition's node attributes silently
    overwrite the first's, and resolving inheritance for both
    collapses onto a single edge. `resolved` count from this function
    can therefore be slightly HIGHER than the actual number of
    distinct `inherits` edges added to the graph - this is expected,
    not a bug, and the gap is a proxy for "how much conditionally-
    defined code exists in this repo," which is itself a data point.
    """
    resolved = 0
    unresolved = 0
 
    for result in parse_results:
        if not result.success:
            continue
 
        import_module_by_name: dict[str, str] = {}
        for imp in result.imports:
            for name in imp.names:
                import_module_by_name[name] = imp.module if imp.module else name
 
        for cls in result.classes:
            child_id = f"{result.filepath}::{cls.qualified_name}"
 
            for base in cls.bases:
                base = base.strip()
                if not base or base == "object":
                    continue
 
                parts = base.split(".")
                simple_name = parts[-1]
                prefix = parts[0] if len(parts) > 1 else None
 
                parent_id: str | None = None
 
                if prefix and prefix in import_module_by_name:
                    target_file = module_index.get(import_module_by_name[prefix])
                    if target_file:
                        parent_id = class_index.get(target_file, {}).get(simple_name)
                elif simple_name in class_index.get(result.filepath, {}):
                    parent_id = class_index[result.filepath][simple_name]
                elif simple_name in import_module_by_name:
                    target_file = module_index.get(import_module_by_name[simple_name])
                    if target_file:
                        parent_id = class_index.get(target_file, {}).get(simple_name)
 
                if parent_id and parent_id in graph:
                    graph.add_edge(child_id, parent_id, type="inherits")
                    resolved += 1
                else:
                    unresolved += 1
 
    return {"resolved": resolved, "unresolved": unresolved}
 
 
# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
 
def save_graph(graph: nx.DiGraph, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(graph, f)
    logger.info("Saved graph to %s", output_path)
 
 
def load_graph(input_path: str | Path) -> nx.DiGraph:
    with Path(input_path).open("rb") as f:
        return pickle.load(f)
 
 
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
 
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
 
    module_index = build_module_index(parse_results)
    class_index = build_class_index(graph)
    inherit_stats = resolve_inheritance(graph, parse_results, module_index, class_index)
 
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
 
    print(
        f"\ninherits resolution: {inherit_stats['resolved']} resolved, "
        f"{inherit_stats['unresolved']} unresolved (built-ins / external bases)"
    )
 
    if args.output:
        save_graph(graph, args.output)
 
    return 0
 
 
if __name__ == "__main__":
    sys.exit(main())
