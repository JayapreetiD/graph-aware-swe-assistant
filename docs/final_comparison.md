# Final Comparison: Graph-Aware Hybrid Retrieval Across Two Codebases

**Status:** Draft — synthesizes complete, verified results from both repositories.
**Sources:** `docs/phase5_click_analysis.md`, `docs/phase5_django_analysis.md`, `docs/analysis_notes.md`

---

## Research Question

> Does graph-aware hybrid retrieval improve software-engineering question answering compared with semantic retrieval alone?

## The Answer: It Depends Entirely on the Codebase's Structural Shape

Run identically across two real Python codebases, the same hybrid retrieval technique produced two different outcomes:

| | click (54/54 pairs) | Django subset (34/34 pairs) |
|---|---|---|
| Recall@K, semantic → hybrid | 0.546 → 0.583 (real improvement) | 0.608 → 0.608 (no change) |
| Precision@K, semantic → hybrid | 0.160 → 0.167 | 0.164 → 0.164 (no change) |
| MRR, semantic → hybrid | 0.598 → 0.598 (identical) | 0.569 → 0.569 (identical) |
| Win/loss (strict rule) | 2 hybrid wins / 0 semantic wins / 25 tied | 0 hybrid wins / 0 semantic wins / 17 tied |
| Grounding accuracy, semantic → hybrid | 0.525 → 0.485 (hybrid worse) | 0.343 → 0.373 (hybrid slightly better) |

Click showed a real, if modest, hybrid advantage concentrated in `cross_file_1hop` queries. Django showed **no measurable difference at all** — every single query tied, with precision/recall/MRR numerically identical between modes across the entire benchmark.

This is not a contradiction between the two results — it is the experiment answering the actual research question correctly. Neither repository's result should be treated as "the" answer; together they show the real answer is conditional.

## Why the Same Technique Produced Different Results

The explanation lives in what each codebase's graph actually looks like:

**Click** has zero cross-file inheritance edges, but a real (if shallow) set of multi-hop cross-file `calls` chains — 113 cross-file call edges, some reaching 2-3 hops deep. This is exactly the structure graph expansion is built to exploit: when the answer to a question lives in a *different function in a different file*, reachable via a call relationship, hybrid retrieval can surface it even when semantic search alone ranks it too low.

**Django's scoped subset** (`forms/`+`db/`) has the opposite profile: only 25 cross-file call edges (versus click's 113), and essentially no real multi-hop cross-file call chains — but a rich, deep same-file class inheritance structure (421 inherits edges, some 3+ levels deep). The benchmark built for Django, by necessity of what the codebase actually offered, ended up testing almost entirely same-file inheritance (14 of 17 queries). When the correct answer is a sibling or parent class **in the same file** as the top semantic match, semantic search already retrieves it on relevance alone — graph expansion adds nothing, because there was never a cross-file gap for it to bridge.

**The honest generalization: graph-aware retrieval helps when the relevant code is reachable via a real cross-file structural relationship that semantic similarity alone misses. It does not help — and should not be expected to help — when the relevant code is already semantically close and structurally local.** This is a testable, falsifiable claim, not a hedge — and both datasets are consistent with it.

## Secondary Finding: The Citation-Abstention Problem

Independent of the semantic-vs-hybrid question, both repositories surfaced a real problem with the LLM's citation behavior:

- **Click**: 1 of 54 calls (query q14) produced zero citations, trivially scoring a "perfect" 1.0 citation correctness for citing nothing.
- **Django**: 13 of 34 calls (38%) showed the same pattern — a far larger, systemic rate. Five queries were zero-citation in *both* modes simultaneously.

This means **`citation_correctness_rate` cannot be reported as a standalone positive metric anywhere in this project** — a perfect score is at least as likely to reflect abstention as accuracy. This finding only became visible at scale; the single click occurrence looked like noise until Django's 38% confirmed it as a systemic weakness in how the LLM handles cases where it isn't confident enough to cite specific code.

## Ablation Note

Only click has hop-depth ablation data (0/1/2/3 hops, 24 pairs). It showed precision/recall/MRR identical across all four depths, with all genuine 3-hop chains in click funneling through a single shared function (`utils.py::echo`) — meaning deeper hop traversal exists in the graph but is rarely exercised by real retrieval at the project's locked scoring defaults. No ablation was run for Django; given Django's benchmark showed zero retrieval difference between modes at all, an ablation would very plausibly show the same flat pattern across hop depths — but this is a prediction based on the pattern already observed, not tested data, and should not be presented as if it were.

## Limitations of This Comparison

- **Sample sizes are both small** (27 and 17 queries respectively), and several categories in each have n≤4. Neither dataset supports strong statistical claims — the comparison is directional and illustrative, not a rigorous statistical test.
- **The two benchmarks are not perfectly parallel.** Click's queries were built primarily around cross-file calls; Django's were built primarily around same-file inheritance, because that's what each codebase's actual scoped structure offered. This is itself the finding, not a flaw to apologize for — but it means "click's cross_file_1hop category" and "Django's same_file_inherit categories" are not directly comparable apples-to-apples; they're testing different relationship types by design.
- **Django was scoped to `forms/`+`db/` only.** A broader scope, or a different medium-sized repository with richer cross-file call structure, might show a result closer to click's pattern. This result characterizes hybrid retrieval's behavior *on this specific scoped subset*, not on "Django" or "codebases with inheritance" as a general category.
- **The scoring formula's known ceiling** (a graph-expanded node's score can never exceed its seed's semantic score, per `hybrid_retriever.py`'s documented design) applies to both repos and may itself be part of why deeper structural relationships rarely surface as the top-ranked results in either dataset.

## Overall Conclusion (edit before final submission)

Across two structurally different Python codebases, graph-aware hybrid retrieval showed a real but conditional benefit: it helped click, where relevant code was often reachable through genuine cross-file call relationships, and it made no measurable difference on this Django subset, where the relevant code was already semantically close and structurally local via same-file inheritance. The project's original research question — "does graph-aware retrieval improve SWE question answering over semantic-only?" — does not have a single yes/no answer; the honest answer is **"it depends on whether the codebase's actual dependency structure creates cross-file relationships that semantic similarity alone misses."** This is a more useful and more defensible finding than a blanket claim in either direction would have been, and it directly reflects the project's stated design principle of not assuming graph retrieval is better without letting the experiment determine the result. A separate, secondary finding — the citation-abstention pattern affecting up to 38% of calls in one dataset — is an important caveat for interpreting every citation-correctness number reported anywhere in this project, and is worth flagging as a concrete direction for future work on the LLM/prompt layer.