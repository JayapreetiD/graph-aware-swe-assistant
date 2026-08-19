

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


def _add_typed_edge(graph: nx.DiGraph, u: str, v: str, edge_type: str) -> None:
    """
    Adds an edge, but SAFELY when an edge between u and v already
    exists with a different relationship type.

    WHY THIS EXISTS (found via real testing, not anticipated
    speculatively): a plain `graph.add_edge(u, v, type=X)` on a
    DiGraph OVERWRITES any existing edge between the same (u, v) pair
    - DiGraph allows only one edge per node pair. This silently
    destroys real relationships in cases like a function that both
    DEFINES a nested helper function and later CALLS that same
    helper - confirmed happening 8 times in click's actual source
    (e.g. click/formatting.py::wrap_text defines AND calls its own
    nested `_flush_par` helper). Without this guard, the `calls` edge
    would silently erase the `defines` edge for that exact pair,
    losing real structural information with no error or warning.

    This function merges into a `types` list on the edge instead of
    switching to a MultiDiGraph, deliberately: a MultiDiGraph would
    also un-collapse the "multiple call-sites to the same target
    become one edge" behavior we specifically chose for `calls`
    edges (see resolve_calls) - that trade-off was intentional, not
    an oversight, and switching graph types would have silently
    undone it as a side effect while fixing this bug.
    """
    if graph.has_edge(u, v):
        existing_types = graph[u][v].get("types", [graph[u][v]["type"]])
        if edge_type not in existing_types:
            existing_types = existing_types + [edge_type]
        graph[u][v]["types"] = existing_types
        # `type` stays as the first-seen type for simple single-type
        # lookups elsewhere; `types` is the authoritative full list.
    else:
        graph.add_edge(u, v, type=edge_type, types=[edge_type])


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
            _add_typed_edge(graph, parent_id, node_id, "defines")

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
                    _add_typed_edge(graph, child_id, parent_id, "inherits")
                    resolved += 1
                else:
                    unresolved += 1

    return {"resolved": resolved, "unresolved": unresolved}


def resolve_imports(
    graph: nx.DiGraph,
    parse_results: list[FileParseResult],
    module_index: dict[str, str],
) -> dict[str, int]:
    """
    Adds `imports` edges: module -> module, one edge per distinct
    (source, target) pair regardless of how many names were imported
    from that target. This is coarser granularity than `inherits`
    (class -> class) or `defines` (module/class -> function/class) by
    design - `imports` per the roadmap's edge type list is a
    module-level relationship, not a "which specific name" record
    (the specific imported names are still available on FileParseResult
    if ever needed - just not modeled as separate graph edges).

    Handles both import forms:
      - `from X import a, b` (imp.module = "X", imp.names = ["a","b"])
      - `import X` (imp.module = None, imp.names = ["X"] - here each
        name IS a module to resolve directly, not an attribute of one)

    Resolution reuses module_index exactly as inherits resolution
    does, so it inherits (no pun intended) the same relative-import
    limitation documented on build_module_index: a relative import
    only resolves correctly when its target's file stem is globally
    unique in the repo.

    Self-imports (a file resolving to itself) are skipped - not
    something normal Python code does, but guarded defensively rather
    than assumed impossible.
    """
    resolved = 0
    unresolved = 0

    for result in parse_results:
        if not result.success:
            continue

        module_id = result.filepath
        edges_added_to: set[str] = set()

        for imp in result.imports:
            # Collect the module string(s) this import statement could
            # resolve to. `from X import a, b` -> just "X", checked
            # once. `import X, Y` -> each of X, Y is itself a module.
            target_keys = [imp.module] if imp.module else imp.names

            for target_key in target_keys:
                target_file = module_index.get(target_key)
                if target_file and target_file != module_id:
                    if target_file not in edges_added_to:
                        _add_typed_edge(graph, module_id, target_file, "imports")
                        edges_added_to.add(target_file)
                    resolved += 1
                else:
                    unresolved += 1

    return {"resolved": resolved, "unresolved": unresolved}


# ---------------------------------------------------------------------------
# calls edge resolution
# ---------------------------------------------------------------------------

def build_function_indices(
    graph: nx.DiGraph,
) -> tuple[dict[str, dict[str, str]], dict[tuple[str, str], dict[str, str]]]:
    """
    Builds two lookup structures needed to resolve call targets:

      top_level_index: filepath -> {function_name: node_id}
          For bare calls to top-level functions, e.g. `add(1, 2)`.

      method_index: (filepath, class_qualified_name) -> {method_name: node_id}
          For `self.method_name(...)` / `cls.method_name(...)` calls,
          keyed by the CALLER's own class.

    AMBIGUITY HANDLING (found via real collision check on click's
    source, not anticipated speculatively): a file can contain
    multiple NESTED functions sharing the same simple name under
    different enclosing functions (e.g. click/decorators.py defines
    five separate nested functions all named `decorator`, one inside
    each of make_pass_decorator, pass_meta_key, command, argument,
    option). Since top_level_index is keyed by simple name only (not
    full qualified_name, and not enclosing-scope-aware), a naive
    "last one wins" index would silently resolve a bare call to the
    WRONG node - a wrong edge, not just a missing one, which is worse
    than unresolved for the reasons documented in resolve_calls'
    docstring.

    RESOLUTION: any (filepath, name) key with more than one matching
    node is treated as AMBIGUOUS and excluded entirely from the
    index, so calls to it fall through to "unresolved" - consistent
    with this module's existing philosophy (resolve_calls,
    resolve_inheritance): anything uncertain is left unresolved
    rather than guessed. This only affects NESTED functions sharing
    a name with a sibling nested function in the same file - true
    top-level, module-scope functions are effectively never
    ambiguous in practice (Python itself would raise most naming
    conflicts at that scope).
    """
    top_level_index: dict[str, dict[str, str]] = {}
    method_index: dict[tuple[str, str], dict[str, str]] = {}

    # Collect ALL candidates per key first, so we can detect
    # collisions before committing anything to the index - a
    # single-pass "last one wins" approach can't distinguish
    # "no collision" from "collision, silently overwritten."
    top_level_candidates: dict[tuple[str, str], list[str]] = {}
    method_candidates: dict[tuple[str, str, str], list[str]] = {}

    for node_id, data in graph.nodes(data=True):
        if data["type"] == "function":
            key = (data["filepath"], data["name"])
            top_level_candidates.setdefault(key, []).append(node_id)
        elif data["type"] == "method":
            qualified_name = data["qualified_name"]
            if "." in qualified_name:
                class_qualified_name = qualified_name.rsplit(".", 1)[0]
                key = (data["filepath"], class_qualified_name, data["name"])
                method_candidates.setdefault(key, []).append(node_id)

    ambiguous_functions = 0
    for (filepath, name), node_ids in top_level_candidates.items():
        if len(node_ids) > 1:
            ambiguous_functions += 1
            continue  # excluded from index - falls through to unresolved
        top_level_index.setdefault(filepath, {})[name] = node_ids[0]

    ambiguous_methods = 0
    for (filepath, class_qn, name), node_ids in method_candidates.items():
        if len(node_ids) > 1:
            ambiguous_methods += 1
            continue
        method_index.setdefault((filepath, class_qn), {})[name] = node_ids[0]

    if ambiguous_functions or ambiguous_methods:
        logger.warning(
            "Excluded %d ambiguous function name(s) and %d ambiguous method name(s) "
            "from call resolution indices (multiple same-named nested functions/methods "
            "in the same scope-file) - calls to these will resolve as unresolved rather "
            "than risk a wrong edge.",
            ambiguous_functions, ambiguous_methods,
        )

    return top_level_index, method_index

def resolve_calls(
    graph: nx.DiGraph,
    parse_results: list[FileParseResult],
    module_index: dict[str, str],
    class_index: dict[str, dict[str, str]],
    top_level_index: dict[str, dict[str, str]],
    method_index: dict[tuple[str, str], dict[str, str]],
) -> dict[str, int]:
    """
    Adds `calls` edges: caller function/method -> callee function/
    method/class, per the roadmap's explicit scope: "same-module calls
    and directly imported calls only - no dynamic dispatch resolution."

    RESOLUTION RULES, IN ORDER (a call resolves via the first rule
    that matches; anything else is left unresolved rather than
    guessed):

      1. `self.x(...)` / `cls.x(...)` inside a method -> resolved
         against the CALLER'S OWN class's methods only. A call to a
         method that's only defined on a PARENT class (inherited, not
         overridden) will NOT resolve here - that would require
         walking the `inherits` edges at resolution time, which is a
         reasonable v2 addition, not attempted in v1.
      2. Bare name matching a top-level function in the SAME file.
      3. Bare name matching a CLASS in the same file (constructor
         call, e.g. `Greeter("world")` -> edge to the `Greeter` class
         node, not to `__init__` - treating instantiation as "uses
         this class" is the more useful graph relationship for
         retrieval purposes than pointing at a specific dunder method).
      4. Bare name imported from elsewhere (function OR class),
         resolved the same way `inherits` resolves imported base
         classes.
      5. Everything else - unresolved. This explicitly includes: any
         "<complex>" callee (calls on the result of another call),
         any dotted call not on self/cls (e.g. `os.path.join`,
         `some_object.method()` where we have no type information
         about `some_object`), and calls that happen at module scope
         outside any function (caller_qualified_name is None - there
         is no function/method node to draw the edge FROM, so these
         are skipped rather than attributed to the module as a whole).

    Returns resolved/unresolved counts. A LOW resolution rate here is
    expected and NOT itself a bug - real Python code calls methods on
    objects whose type isn't known without a type checker or runtime
    trace, and this project's stated scope explicitly excludes that
    (see project scope doc: "no dynamic Python call resolution").
    The resolution rate is a useful number to report in the final
    write-up precisely because it quantifies that limitation with
    real data instead of a vague caveat.
    """
    resolved = 0
    unresolved = 0

    for result in parse_results:
        if not result.success:
            continue

        import_target_by_name: dict[str, str] = {}
        for imp in result.imports:
            for name in imp.names:
                import_target_by_name[name] = imp.module if imp.module else name

        for call in result.calls:
            if call.callee == "<complex>" or call.caller_qualified_name is None:
                unresolved += 1
                continue

            caller_id = f"{result.filepath}::{call.caller_qualified_name}"
            parts = call.callee.split(".")
            target_id: str | None = None

            if parts[0] in ("self", "cls") and len(parts) == 2 and "." in call.caller_qualified_name:
                caller_class_qn = call.caller_qualified_name.rsplit(".", 1)[0]
                target_id = method_index.get((result.filepath, caller_class_qn), {}).get(parts[1])

            elif len(parts) == 1:
                name = parts[0]
                target_id = top_level_index.get(result.filepath, {}).get(name)
                if target_id is None:
                    target_id = class_index.get(result.filepath, {}).get(name)
                if target_id is None and name in import_target_by_name:
                    target_file = module_index.get(import_target_by_name[name])
                    if target_file:
                        target_id = top_level_index.get(target_file, {}).get(name)
                        if target_id is None:
                            target_id = class_index.get(target_file, {}).get(name)

            if target_id and target_id in graph and caller_id in graph:
                _add_typed_edge(graph, caller_id, target_id, "calls")
                resolved += 1
            else:
                unresolved += 1

    return {"resolved": resolved, "unresolved": unresolved}

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
    import_stats = resolve_imports(graph, parse_results, module_index)
    top_level_index, method_index = build_function_indices(graph)
    call_stats = resolve_calls(graph, parse_results, module_index, class_index, top_level_index, method_index)

    node_counts: dict[str, int] = {}
    for _, data in graph.nodes(data=True):
        node_counts[data["type"]] = node_counts.get(data["type"], 0) + 1
    # An edge can now carry MULTIPLE relationship types (see
    # _add_typed_edge), so edge_counts sums per-type occurrences,
    # which can exceed graph.number_of_edges() when overlaps exist -
    # both numbers are printed so that's visible, not hidden.
    edge_counts: dict[str, int] = {}
    multi_type_edges = 0
    for _, _, data in graph.edges(data=True):
        types = data.get("types", [data["type"]])
        if len(types) > 1:
            multi_type_edges += 1
        for t in types:
            edge_counts[t] = edge_counts.get(t, 0) + 1

    print(f"\nTotal nodes: {graph.number_of_nodes()}")
    for node_type, count in sorted(node_counts.items()):
        print(f"  {node_type}: {count}")
    print(f"Total edges: {graph.number_of_edges()} (edges carrying >1 relationship type: {multi_type_edges})")
    for edge_type, count in sorted(edge_counts.items()):
        print(f"  {edge_type}: {count}")

    print(
        f"\ninherits resolution: {inherit_stats['resolved']} resolved, "
        f"{inherit_stats['unresolved']} unresolved (built-ins / external bases)"
    )
    print(
        f"imports resolution:  {import_stats['resolved']} resolved, "
        f"{import_stats['unresolved']} unresolved (stdlib / third-party)"
    )
    print(
        f"calls resolution:    {call_stats['resolved']} resolved, "
        f"{call_stats['unresolved']} unresolved (dynamic/unknown-type calls, by design)"
    )

    if args.output:
        save_graph(graph, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())