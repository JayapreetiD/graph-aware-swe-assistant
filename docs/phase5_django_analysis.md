# Phase 5 Analysis: Semantic vs. Graph-Aware Hybrid Retrieval (Django)

**Status:** Draft — based on complete, verified data (34/34 benchmark pairs).
**Source data:** `data/results/django/evaluation_report.json`, `data/results/django/metrics_summary.json`
**Scope note:** This evaluation covers `django/forms` and `django/db` only (per project scope), parsed from a local scoped copy pinned to commit `cccc004b46f71b4e54d87b376be691a17de6b903`, not the full Django codebase. No hop-depth ablation was run for Django (a separate, smaller effort scoped only for click).

---

## Research Question

> Does graph-aware hybrid retrieval improve software-engineering question answering compared with semantic retrieval alone? Does the answer differ on a codebase with a different structural profile than click?

## Headline Finding

**On this Django subset, hybrid retrieval made no measurable difference in retrieval quality.** By the same win/loss rule used for click, **all 17 queries tied** (0 hybrid wins, 0 semantic wins, 17 tied). Precision@K, Recall@K, and MRR are **numerically identical** between modes across the entire benchmark — not just similar, the same values to four decimal places. The only metric that differed at all was grounding accuracy, where hybrid held a small edge (0.3725 vs 0.3431).

This is a genuinely different result from click, where hybrid showed a real (if modest) recall advantage. The most likely explanation, grounded in how this benchmark was designed: Django's queries were built almost entirely around **same-file class inheritance** (14 of 17 queries), since the scoped `forms/`+`db/` subset had almost no multi-hop cross-file `calls` structure to draw on (only 25 cross-file call edges existed at all, versus click's 113). When the correct answer sits in the same file as the semantically-best match — which is exactly what same-file inheritance means — graph expansion has nothing to add, because semantic search already retrieves the right file's other classes on relevance alone.

## Headline Numbers

| Metric | Semantic | Hybrid | Direction |
|---|---|---|---|
| Precision@K | 0.1638 | 0.1638 | Identical |
| Recall@K | 0.6078 | 0.6078 | Identical |
| MRR | 0.5686 | 0.5686 | Identical |
| Grounding accuracy | 0.3431 | 0.3725 | Hybrid slightly higher (+0.029) |
| Citation correctness | 1.0000 | 1.0000 | Identical — **38% abstention, see caveat below, not a clean result** |
| Latency (s) | 27.6264 | 25.9451 | Hybrid slightly faster |

**Win/loss (same rule as click: recall must be strictly higher AND grounding not worse):**
- `hybrid_win`: 0 queries
- `semantic_win`: 0 queries
- `tied_or_mixed`: 17 queries (100%)
- `incomplete`: 0

## The Citation-Correctness Caveat Still Applies — And Is Far More Severe Here Than In Click

Both modes show a perfect 1.0 citation correctness rate here, but this is **not evidence of good grounding behavior in this benchmark** — it is overwhelmingly an artifact of abstention. A direct check of the raw results found **13 of 34 successful calls (38%) produced zero citations at all**, each trivially scoring a "perfect" 1.0 by the `valid/total` formula. This is a much larger and more systemic pattern than the single isolated case documented for click (`docs/analysis_notes.md`, q14) — there it was one query out of 54; here it's over a third of the entire benchmark.

Notably, five queries (dj04, dj06, dj09, dj10, dj15) were zero-citation in **both** modes simultaneously — meaning the LLM consistently declined to cite anything for these specific questions regardless of retrieval method. This points to something about the questions themselves or the shape of the retrieved context for this codebase, not a semantic-vs-hybrid difference. **Any reporting of citation correctness for Django must lead with this 38% abstention rate, not the misleadingly perfect 1.0 average** — the two headline numbers (`mean citation correctness: 1.0` for both modes) should not appear in a summary table without this context immediately beside them.

## By Category

| Category | n | Semantic recall | Hybrid recall | Semantic grounding | Hybrid grounding |
|---|---|---|---|---|---|
| cross_file_1hop | 7 | 0.500 | 0.500 | 0.4286 | 0.4286 |
| same_file_inherit_shallow | 6 | 0.6667 | 0.6667 | 0.4167 | 0.5000 |
| same_file_inherit_deep | 4 | 0.7084 | 0.7084 | 0.0833 | 0.0833 |

**Every category shows identical recall between modes**, including `cross_file_1hop` — the category most analogous to click's cross-file categories, where click showed hybrid's clearest advantage. This is the strongest evidence in the dataset that Django's structural shape (in this scoped subset) simply doesn't give graph expansion room to help, regardless of query type.

**`same_file_inherit_deep` has the lowest grounding accuracy in the whole benchmark (0.0833, both modes)** — these are the hardest queries (dj13, dj15: multi-level inheritance chains, some requiring 3-4 ground-truth nodes), and both modes struggled to produce citations covering the full chain. Worth a manual look at these specific answers before drawing conclusions, since n=4 is a small sample.

**`same_file_inherit_shallow` is the only category where grounding differs at all** (0.4167 → 0.5000, hybrid ahead) — a modest, real difference, but on n=6.

## Honest Limitations

- **Very small sample.** 17 queries total, with the largest category at n=7. No category here reaches click's more substantial n=10-12 categories. Treat every category-level number as suggestive, not conclusive.
- **No ablation was run.** Unlike click, there's no hop-depth (0/1/2/3) breakdown for Django. Given hybrid showed zero measurable retrieval difference from semantic here, an ablation would likely show flat results across all depths too — but that's a prediction, not tested data.
- **Scope is narrow.** This evaluates `forms/`+`db/` only, and even within that scope, the query set is weighted heavily toward inheritance questions (14/17) because that's where Django's structure in this subset actually had testable depth. A broader Django scope (including more of `db/migrations/`, `db/backends/`, or unscoped modules) might show different results.
- **Grounding accuracy is notably lower than click's across the board** (~0.34-0.37 vs click's ~0.48-0.52). Worth investigating whether this is a genuine Django-specific pattern (harder to cite correctly when many candidate classes share similar names/structure — recall the CharField/NullBooleanField name-collision issue documented in the query notes) or a smaller-sample artifact.

## Draft Conclusion (edit before final submission)

On this scoped Django subset, graph-aware hybrid retrieval showed **no measurable improvement over semantic-only retrieval** in precision, recall, or MRR — a result that stands in real contrast to click, where hybrid showed a modest but consistent recall advantage. The most likely explanation is structural: click's benchmark drew on genuine multi-hop cross-file call chains, which is exactly the kind of relationship graph expansion is built to surface; this Django subset's benchmark, by necessity of what the codebase actually offered, drew almost entirely on same-file inheritance, which semantic search can already find without graph traversal. This is not a null result to discard — it's evidence that **hybrid retrieval's value is contingent on a codebase's actual dependency shape**, not a universal property of the technique.

A second, independent finding is at least as important: **38% of Django's benchmark calls produced no citations at all**, inflating citation correctness to a meaningless 1.0. This citation-abstention pattern was already flagged as a risk from a single click example (`docs/analysis_notes.md`); Django's results confirm it generalizes and is severe enough to actively distort any summary statistic that reports citation correctness without citation count alongside it. Any future iteration of this project's LLM/prompt design should treat citation abstention as a first-class failure mode to measure and reduce, not a footnote.

The comparison between these two repos is more informative together than either is alone: click shows hybrid can help when cross-file call chains exist; Django shows it adds nothing when the relevant structure is same-file inheritance instead — and Django additionally surfaces a citation-behavior problem that was easy to miss when it appeared only once in click's data.