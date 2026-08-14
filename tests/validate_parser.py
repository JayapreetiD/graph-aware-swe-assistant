

from __future__ import annotations
 
import sys
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from parser.parser import parse_repository  # noqa: E402
 
 
def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(f"Usage: python {argv[0]} <repo_root> <relative_file_path> [<relative_file_path> ...]")
        return 1
 
    repo_root = argv[1]
    target_files = set(argv[2:])
 
    all_results = parse_repository(repo_root)
    results_by_path = {r.filepath: r for r in all_results}
 
    for target in target_files:
        # Accept both forward and backward slash input, since the
        # user may type either depending on OS - always compare
        # against the tool's own (now posix-normalized) filepath.
        normalized = target.replace("\\", "/")
        result = results_by_path.get(normalized)
 
        print("=" * 70)
        print(f"FILE: {normalized}")
        print("=" * 70)
 
        if result is None:
            print("  NOT FOUND in parsed results (check the path is correct).")
            continue
 
        if not result.success:
            print(f"  PARSE FAILED: {result.error}")
            continue
 
        print(f"  Functions/methods: {len(result.functions)}")
        for fn in result.functions:
            kind = "async " if fn.is_async else ""
            print(f"    - {kind}{fn.qualified_name}({', '.join(fn.args)})  [lines {fn.start_line}-{fn.end_line}]")
 
        print(f"  Classes: {len(result.classes)}")
        for cls in result.classes:
            bases = f"({', '.join(cls.bases)})" if cls.bases else ""
            print(f"    - {cls.qualified_name}{bases}  [lines {cls.start_line}-{cls.end_line}]")
 
        print(f"  Imports: {len(result.imports)}")
        for imp in result.imports:
            source = imp.module if imp.module else "(direct import)"
            print(f"    - from {source}: {', '.join(imp.names)}  [line {imp.lineno}]")
 
        print(f"  Calls: {len(result.calls)}")
        for call in result.calls:
            caller = call.caller_qualified_name or "(module level)"
            print(f"    - {caller} -> {call.callee}()  [line {call.lineno}]")
 
        print()
 
    return 0
 
 
if __name__ == "__main__":
    sys.exit(main(sys.argv))
 