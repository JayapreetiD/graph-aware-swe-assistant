# Phase 1 Validation Report: Parser & Graph Construction

## Purpose

Per the project roadmap (Phase 1, Step 1.6), the parser and graph builder
must be validated against manually-checked ground truth before moving to
Phase 2. This document records that validation.

**Target:** ~90% agreement between the tool's extracted structure and a
human's manual count of the same file.

**Result: 100% agreement across all 8 validated files**, for functions,
classes, and imports. `calls` counts were spot-checked rather than
exhaustively hand-verified (see Methodology).

## Methodology

1. Selected 8 real files from `pallets/click` (not the synthetic
   `fake_repo` fixture), ranging from 36 to 627 lines, chosen to cover a
   spread of complexity: near-trivial utility files, class-hierarchy-heavy
   files, decorator-heavy files, and a near-pure-imports re-export file.
2. Ran `tests/validate_parser.py` to print the tool's extracted counts per
   file.
3. Independently derived the "ground truth" count for each file, primarily
   via `grep`-based line inspection of the real source, cross-checked
   against the tool's per-item listing (not just the totals).
4. Compared and reconciled every discrepancy found, rather than accepting
   round numbers at face value.

**Important methodological note:** several apparent discrepancies during
this process turned out to be errors in the *manual* count, not the tool.
This is disclosed in detail below rather than omitted, because it's
informative: naive text-based counting (e.g. `grep`) is not a reliable
ground truth on its own, and the errors found illustrate specific reasons
why an AST-based tool can outperform naive text scanning.

## Files Validated

| File | Lines | Functions/Methods | Classes | Imports | Agreement |
|---|---|---|---|---|---|
| `click/_utils.py` | 36 | 1 | 1 | 3 | Exact |
| `click/globals.py` | 67 | 6 | 0 | 4 | Exact* |
| `click/_textwrap.py` | 188 | 5 | 1 | 6 | Exact |
| `click/exceptions.py` | 378 | 23 | 12 | 14 | Exact |
| `click/__init__.py` | 144 | 1 | 0 | 72 | Exact* |
| `click/formatting.py` | 320 | 17 | 1 | 8 | Exact |
| `click/parser.py` | 533 | 21 | 4 | 20 | Exact* |
| `click/decorators.py` | 627 | 35 | 0 | 15 | Exact* |

\* See discrepancy notes below — first-pass manual counts differed from
the tool's output; all were resolved as manual-counting errors upon
re-inspection.

**Agreement rate: 8/8 files (100%) for functions, classes, and imports.**

## Discrepancies Found and Resolved

Every discrepancy encountered during validation is recorded here, with
resolution, rather than silently corrected.

### 1. `click/globals.py` — initial manual undercount of calls (6 vs 7)

First manual pass missed `_local = local()` at module level (line 9), a
call to `local()` imported from `threading`. **Resolution: manual error;
tool was correct (7 calls).**

### 2. `click/parser.py` — initial manual undercount of imports (14 vs 20)

First manual pass counted only top-level and function-local imports,
missing 6 additional imports inside an `if t.TYPE_CHECKING:` block
(lines 41-46). **Resolution: manual error; tool was correct (20
imports).** This block-scoped import pattern is common in modern typed
Python and the parser handles it correctly — `ast` does not distinguish
"conditionally executed" from "always executed" code paths when
extracting import statements, so both are captured, which is the correct
behavior for this project's purposes (the imported symbol exists in the
codebase and is a real relationship worth graphing, even if only
active during type-checking).

### 3. `click/decorators.py` — manual overcount via naive text search (37 vs 35 functions; 16 vs 15 imports)

`grep -n "    def "` matched two lines (63, 65) that are **not real code**
— they are example code shown inside a docstring (a triple-quoted string
in `make_pass_decorator`'s documentation, lines 59-73). The same docstring
also contains a `from functools import update_wrapper` line that a naive
import-line grep also falsely matched.

**Resolution: manual (grep-based) method was wrong; the tool was
correct.** This is the most instructive finding in this validation pass:
`parser.py` operates on the Python AST, which structurally distinguishes
a string literal (the docstring) from executable code. A text-based
search cannot make this distinction without also parsing the file. This
is direct evidence for one of the specific advantages of the AST-based
approach chosen for this project over simpler text-processing
alternatives.

### 4. Known limitation (previously documented, reconfirmed here): `TYPE_CHECKING`/`@overload` duplicate definitions

`click/globals.py`'s `get_current_context` is defined three times
(two `@overload` stub signatures plus the real implementation) — all
three are correctly extracted as separate `FunctionInfo` entries by
`parser.py` at the raw-extraction level validated in this report. Note
that this is distinct from the previously-documented graph-construction-time
behavior (see `graph_builder.py`), where multiple same-named definitions
in one file collapse onto a single graph node, since `node_id` is based
on qualified name, not source location. This report validates the parser's
raw extraction only, not that downstream collapsing behavior.

## Calls Validation (Spot-Checked, Not Exhaustive)

Call counts (46 to 104 per file, 76 total across all 8 files at the
highest) were not exhaustively hand-verified line-by-line — the volume
makes full manual verification impractical within this validation pass.
Instead, calls were spot-checked by:

- Confirming caller attribution (which function/method a call is inside)
  matches the actual source structure in all inspected cases.
- Confirming callee name extraction matches the actual call expression
  text for straightforward cases (`name()`, `self.attr()`,
  `module.sub.func()`).
- Relying on the already-documented, code-level-flagged limitations for
  known-incomplete cases (chained calls like `super().greet().upper()`
  losing chain context — see `parser.py` module docstring and the
  `_call_name` function).

**This is disclosed as a limitation of this validation report, not
hidden**: the `calls` agreement rate is not backed by the same
exhaustive process as functions/classes/imports. A full call-accuracy
audit would require either significantly more manual effort or a
separate automated cross-check (e.g. comparing against a reference
tool), which is out of scope for this validation pass but noted as a
possible extension if call-graph accuracy becomes a bottleneck in later
evaluation (Phase 5).

## Conclusion

The parser and its raw extraction logic meet and exceed the roadmap's
~90% agreement target, achieving 100% agreement on functions, classes,
and imports across all 8 validated files. All discrepancies encountered
during this process were traced to manual-counting errors, not tool
errors, and are disclosed above rather than omitted. This provides
sufficient confidence to proceed to Phase 2 (chunking and embeddings),
which will build directly on this validated extraction.

Known, documented limitations carried forward from Phase 1 (not
re-litigated here, see inline code comments for full detail):

- Call resolution is syntactic, not semantic (no type inference) —
  by explicit project scope, not a bug.
- Chained method calls (e.g. `super().x().y()`) lose intermediate
  context in callee name extraction.
- Relative imports resolve correctly only when the target module's
  filename stem is unique repo-wide.
- Conditionally-defined symbols (`TYPE_CHECKING`, `@overload`) with
  identical qualified names collapse onto a single graph node at
  graph-construction time (not at parser extraction time, which is
  what this report validates).

## Post-Validation Fix (2026-08-18)

A code review after this report was written found that
`build_function_indices` in `graph_builder.py` indexed nested
functions by simple name only, causing same-named nested functions
in the same file (e.g. `click/decorators.py`'s multiple `decorator`
and `new_func` helpers) to silently collide, resolving `calls` edges
to the wrong target node. Confirmed via a direct collision check on
the graph (4 collisions found). Fixed by excluding ambiguous
(filepath, name) keys from the resolution index entirely, so affected
calls now correctly fall through to "unresolved" rather than
resolving wrong. Edge count changed from 1103 to 1099 (4 wrong edges
removed). See `build_function_indices` docstring for full detail.
