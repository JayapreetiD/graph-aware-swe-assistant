

"""
parser/parser.py

Phase 1 - first working parser, built on Python's built-in `ast` module.
"""

from __future__ import annotations

import argparse
import ast
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_SKIP_DIRS = {
    ".git", ".venv", "venv", "env", "__pycache__", "node_modules",
    ".mypy_cache", ".pytest_cache", ".tox", "build", "dist", ".eggs",
}


@dataclass
class ImportInfo:
    module: str | None
    names: list[str]
    lineno: int


@dataclass
class CallInfo:
    callee: str
    caller_qualified_name: str | None
    lineno: int


@dataclass
class FunctionInfo:
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
    name: str
    qualified_name: str
    bases: list[str]
    decorators: list[str]
    docstring: str | None
    start_line: int
    end_line: int


@dataclass
class FileParseResult:
    filepath: str
    success: bool
    error: str | None = None
    functions: list[FunctionInfo] = field(default_factory=list)
    classes: list[ClassInfo] = field(default_factory=list)
    imports: list[ImportInfo] = field(default_factory=list)
    calls: list[CallInfo] = field(default_factory=list)


def find_python_files(repo_root: Path) -> list[Path]:
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


def _unparse_safe(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return "<unparseable>"


def _call_name(node: ast.Call) -> str:
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
    def __init__(self) -> None:
        self.functions: list[FunctionInfo] = []
        self.classes: list[ClassInfo] = []
        self.imports: list[ImportInfo] = []
        self.calls: list[CallInfo] = []
        self._scope_stack: list[tuple[str, bool]] = []

    def _qualified_name(self, name: str) -> str:
        prefix = ".".join(n for n, _is_class in self._scope_stack)
        return f"{prefix}.{name}" if prefix else name

    def _current_function_qualified_name(self) -> str | None:
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

    def _visit_function_like(self, node, is_async: bool) -> None:
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
    relative_path = str(filepath.relative_to(repo_root))
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


def parse_repository(repo_root) -> list[FileParseResult]:
    repo_root = Path(repo_root).resolve()
    if not repo_root.is_dir():
        raise NotADirectoryError(f"repo_root does not exist or is not a directory: {repo_root}")

    files = find_python_files(repo_root)
    results = [parse_file(f, repo_root) for f in files]

    ok = sum(1 for r in results if r.success)
    failed = len(results) - ok
    logger.info("Parsed %d/%d files successfully (%d failed) from %s", ok, len(results), failed, repo_root)
    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="parser.py",
        description="Parse a Python repository and report extracted structure (functions, classes, imports, calls).",
    )
    p.add_argument("repo_path", type=str, help="Path to the root of the Python repository to parse.")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG-level logging.")
    return p


def main(argv=None) -> int:
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