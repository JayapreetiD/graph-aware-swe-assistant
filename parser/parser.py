

from __future__ import annotations

import argparse
import ast
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Directories we never walk into. Hardcoded, not configurable - no
# concrete requirement yet to make this a config option (YAGNI).
_SKIP_DIRS = {
    ".git", ".venv", "venv", "env", "__pycache__", "node_modules",
    ".mypy_cache", ".pytest_cache", ".tox", "build", "dist", ".eggs",
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ImportInfo:
    """
    One import statement.

    module: the dotted module being imported from (None for plain
        `import x` - see `names` instead).
    names: the names brought into scope. For `import os` -> ["os"].
        For `from a.b import c, d` -> ["c", "d"]. For `import os as o`
        -> ["o"] (we record the *bound* name, since that's what call
        resolution later will actually see used in the code, not the
        original name).
    lineno: 1-indexed line of the import statement.
    """
    module: str | None
    names: list[str]
    lineno: int


@dataclass
class CallInfo:
    """
    One function/method call expression.

    callee: best-effort dotted name of what's being called, e.g.
        "add", "self.greet", "os.path.join". For call expressions we
        can't reduce to a simple dotted name (e.g. calling the result
        of another call: `get_handler()()`), we record "<complex>"
        rather than crashing or silently dropping the call - a caller
        counting "how many calls does this function make" still gets
        an accurate count even when it can't get an accurate name.
    caller_qualified_name: the qualified name of the enclosing
        function/method this call happens inside, or None if the call
        is at module level (e.g. inside a top-level script line, not
        inside any def). This is what graph_builder.py will later use
        to draw a `calls` edge FROM.
    lineno: 1-indexed line of the call.

    LIMITATION, stated plainly: this is syntactic name capture, not
    call resolution. `self.greet` is captured as the string
    "self.greet" - we are not resolving which class `self` actually
    is, or whether `greet` is inherited. That resolution step belongs
    in graph_builder.py (per the roadmap: "same-module calls and
    directly imported calls only" - no dynamic dispatch resolution).
    """
    callee: str
    caller_qualified_name: str | None
    lineno: int


@dataclass
class FunctionInfo:
    """
    One function, async function, or method definition.

    qualified_name: dotted path from module scope, e.g. "add" for a
        top-level function, or "Greeter.greet" for a method. This is
        NOT the full node_id yet (node_id also needs the file path -
        that gets prefixed in graph_builder.py, not here, because
        this file has no concept of "relative to repo root").
    is_async: True for `async def`.
    is_method: True if this def is directly nested inside a
        ClassDef (one level - see LIMITATION below).
    class_name: the enclosing class's simple name if is_method,
        else None.
    args: parameter names as written (no type/default resolution -
        just names, sufficient for a chunk signature string).
    decorators: decorator expressions as source-like strings (e.g.
        "staticmethod", "click.command"), best-effort via ast.unparse.
    docstring: first statement's string literal if present, else None.
    start_line / end_line: 1-indexed, inclusive.

    LIMITATION: "is_method" is determined by direct nesting inside a
    class body. A function defined inside another function, inside a
    class (a closure inside a method) will NOT be marked is_method,
    which is correct - it isn't one. But a function assigned as a
    class attribute after definition (some metaclass/decorator
    patterns) won't be caught here at all since we only visit actual
    `def` nodes textually inside the class body. This matches the
    roadmap's explicit scope limit: no dynamic/runtime resolution.
    """
    name: str
    qualified_name: str
    is_async: bool
    is_method: bool
    class_name: str | None
    args: list[str]
    decorators: list[str]
    docstring: str | None
    start_line: int
    end_line: int


@dataclass
class ClassInfo:
    """
    One class definition.

    bases: base class expressions as best-effort strings (e.g.
        "click.Command"). Not resolved to actual classes/modules -
        that resolution (does "click.Command" refer to an imported
        symbol?) is graph_builder.py's job, using the ImportInfo list
        from the same file.
    """
    name: str
    qualified_name: str
    bases: list[str]
    decorators: list[str]
    docstring: str | None
    start_line: int
    end_line: int


@dataclass
class FileParseResult:
    """
    Everything extracted from one file, or the reason extraction
    failed.

    success=False means ast.parse() itself failed (syntax error,
    encoding error). In that case functions/classes/imports/calls are
    all empty lists, not partial data - see module docstring on why
    `ast` can't give partial results.
    """
    filepath: str  # relative to repo root
    success: bool
    error: str | None = None
    functions: list[FunctionInfo] = field(default_factory=list)
    classes: list[ClassInfo] = field(default_factory=list)
    imports: list[ImportInfo] = field(default_factory=list)
    calls: list[CallInfo] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def find_python_files(repo_root: Path) -> list[Path]:
    """
    Find every .py file under repo_root, pruning noise directories
    at the directory level (not filtering after the fact) so we
    never descend into e.g. a large .venv.

    Returns a list (not a generator): callers need the count up
    front for progress/summary reporting, and repo-scale file counts
    (hundreds to low thousands) don't justify streaming here.
    """
    found: list[Path] = []
    stack = [repo_root]
    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError as e:
            logger.warning("Cannot list directory %s: %s", current, e)
            continue
        for entry in entries:
            if entry.is_dir():
                if entry.name not in _SKIP_DIRS:
                    stack.append(entry)
            elif entry.is_file() and entry.suffix == ".py":
                found.append(entry)
    return found


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _unparse_safe(node: ast.AST | None) -> str:
    """
    ast.unparse can itself raise on exotic nodes in edge cases across
    Python versions. Wrapped so one weird decorator/base-class
    expression can't take down extraction for the whole file -
    consistent with the "one bad thing shouldn't kill the whole run"
    principle applied at every layer, not just file-level.
    """
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return "<unparseable>"


def _call_name(node: ast.Call) -> str:
    """
    Best-effort dotted name for a call's target.

    Handles the common cases directly (Name: `foo()`, Attribute:
    `obj.method()`, chained Attribute: `a.b.c()`) without falling
    back to full ast.unparse for the common path, since unparse
    reconstructs source text generically and is slower / can include
    call-arguments-shaped noise for edge cases. Falls back to
    "<complex>" for anything else (e.g. calling a call's result, a
    subscript, a lambda) rather than guessing.
    """
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parts = [func.attr]
        cur = func.value
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))
    return "<complex>"


class _StructureVisitor(ast.NodeVisitor):
    """
    Single-pass visitor collecting functions, classes, imports, and
    calls, while tracking a qualified-name context stack so nested
    defs get correct dotted names (e.g. "Greeter.greet") and calls
    get attributed to the correct enclosing function.

    Design choice: one visitor, one pass, mutating lists on self -
    not four separate ast.walk() passes each filtering by node type.
    A single visitor is O(n) total (n = AST node count); four
    separate ast.walk() filters would be O(4n) and, worse, each pass
    would have to independently reconstruct the same qualified-name
    context stack to know "which class/function is this node inside"
    - duplicated state-tracking logic is a maintenance risk, not just
    a performance one.
    """

    def __init__(self) -> None:
        self.functions: list[FunctionInfo] = []
        self.classes: list[ClassInfo] = []
        self.imports: list[ImportInfo] = []
        self.calls: list[CallInfo] = []
        # Stack of (name, is_class) tracking current nesting, used to
        # build dotted qualified names and to know the current
        # enclosing *function* (not class) for call attribution.
        self._scope_stack: list[tuple[str, bool]] = []

    def _qualified_name(self, name: str) -> str:
        prefix = ".".join(n for n, _is_class in self._scope_stack)
        return f"{prefix}.{name}" if prefix else name

    def _current_function_qualified_name(self) -> str | None:
        """Nearest enclosing function/method scope, walking outward.
        Skips class scopes because a call directly inside a class
        body (rare - e.g. a default argument evaluated at class
        definition time) has no enclosing *function*."""
        for name, is_class in reversed(self._scope_stack):
            if not is_class:
                return self._qualified_name_for_scope_prefix_up_to(name)
        return None

    def _qualified_name_for_scope_prefix_up_to(self, target_name: str) -> str:
        names = []
        for name, _is_class in self._scope_stack:
            names.append(name)
            if name == target_name:
                break
        return ".".join(names)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            bound_name = alias.asname or alias.name
            self.imports.append(ImportInfo(module=None, names=[bound_name], lineno=node.lineno))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        bound_names = [alias.asname or alias.name for alias in node.names]
        self.imports.append(ImportInfo(module=node.module, names=bound_names, lineno=node.lineno))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self.calls.append(CallInfo(
            callee=_call_name(node),
            caller_qualified_name=self._current_function_qualified_name(),
            lineno=node.lineno,
        ))
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qname = self._qualified_name(node.name)
        self.classes.append(ClassInfo(
            name=node.name,
            qualified_name=qname,
            bases=[_unparse_safe(b) for b in node.bases],
            decorators=[_unparse_safe(d) for d in node.decorator_list],
            docstring=ast.get_docstring(node),
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
        ))
        self._scope_stack.append((node.name, True))
        self.generic_visit(node)
        self._scope_stack.pop()

    def _visit_function_like(self, node: ast.FunctionDef | ast.AsyncFunctionDef, is_async: bool) -> None:
        qname = self._qualified_name(node.name)
        is_method = bool(self._scope_stack) and self._scope_stack[-1][1] is True
        class_name = self._scope_stack[-1][0] if is_method else None
        self.functions.append(FunctionInfo(
            name=node.name,
            qualified_name=qname,
            is_async=is_async,
            is_method=is_method,
            class_name=class_name,
            args=[a.arg for a in node.args.args],
            decorators=[_unparse_safe(d) for d in node.decorator_list],
            docstring=ast.get_docstring(node),
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
        ))
        self._scope_stack.append((node.name, False))
        self.generic_visit(node)
        self._scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function_like(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function_like(node, is_async=True)


def parse_file(filepath: Path, repo_root: Path) -> FileParseResult:
    """
    Parse and extract structure from one file. Never raises - any
    failure (read error, syntax error, decode error) is captured in
    the returned FileParseResult so the caller can keep going.
    """
    relative_path = filepath.relative_to(repo_root).as_posix()
    try:
        source = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        logger.warning("Skipping %s: could not read file (%s)", relative_path, e)
        return FileParseResult(filepath=relative_path, success=False, error=str(e))

    try:
        tree = ast.parse(source, filename=relative_path)
    except SyntaxError as e:
        logger.warning("Skipping %s: syntax error (%s)", relative_path, e)
        return FileParseResult(filepath=relative_path, success=False, error=str(e))

    visitor = _StructureVisitor()
    visitor.visit(tree)
    return FileParseResult(
        filepath=relative_path,
        success=True,
        functions=visitor.functions,
        classes=visitor.classes,
        imports=visitor.imports,
        calls=visitor.calls,
    )


def parse_repository(repo_root: str | Path) -> list[FileParseResult]:
    """
    Entry point: discover and parse every .py file in a repo.

    Returns a list of FileParseResult, one per file found - including
    failed ones, so callers get an accurate picture of coverage
    (e.g. "37/40 files parsed cleanly") rather than silently losing
    failures.
    """
    repo_root = Path(repo_root).resolve()
    if not repo_root.is_dir():
        raise NotADirectoryError(f"repo_root does not exist or is not a directory: {repo_root}")

    files = find_python_files(repo_root)
    results = [parse_file(f, repo_root) for f in files]

    ok = sum(1 for r in results if r.success)
    failed = len(results) - ok
    logger.info("Parsed %d/%d files successfully (%d failed) from %s", ok, len(results), failed, repo_root)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="parser.py",
        description="Parse a Python repository and report extracted structure (functions, classes, imports, calls).",
    )
    p.add_argument("repo_path", type=str, help="Path to the root of the Python repository to parse.")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG-level logging.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        results = parse_repository(args.repo_path)
    except NotADirectoryError as e:
        logger.error(str(e))
        return 1

    total_functions = sum(len(r.functions) for r in results)
    total_classes = sum(len(r.classes) for r in results)
    total_imports = sum(len(r.imports) for r in results)
    total_calls = sum(len(r.calls) for r in results)
    failed_files = [r for r in results if not r.success]

    print(f"\nFiles parsed: {len(results) - len(failed_files)}/{len(results)}")
    print(f"Functions found: {total_functions}")
    print(f"Classes found:   {total_classes}")
    print(f"Imports found:   {total_imports}")
    print(f"Calls found:     {total_calls}")

    if failed_files:
        print(f"\nFailed files ({len(failed_files)}):")
        for r in failed_files:
            print(f"  {r.filepath}: {r.error}")

    return 0


if __name__ == "__main__":
    sys.exit(main())