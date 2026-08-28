# Phase 5 Analysis: Semantic vs. Graph-Aware Hybrid Retrieval (click)

**Status:** Draft — based on complete, verified data (54/54 benchmark pairs, 24/24 ablation pairs).
**Source data:** `data/results/click/evaluation_report.json`, `data/results/click/metrics_summary.json`, `docs/analysis_notes.md`

---

## Research Question

> Does graph-aware hybrid retrieval improve software-engineering question answering compared with semantic retrieval alone?

## Headline Finding

**The answer for click is: marginally and unevenly, not decisively.** Hybrid retrieval improved recall in aggregate and in most categories, ran faster on average, and never lost outright to semantic-only under our win/loss rule — but it also showed a **worse grounding-accuracy** average, and by a strict recall-and-grounding win condition, it only clearly "won" on 2 of 27 queries (7%). The dominant result across the benchmark is **parity**: 25 of 27 queries (93%) landed in `tied_or_mixed`. This does not support a strong claim that graph expansion meaningfully improves answer quality on click — it supports a more limited claim that it helps retrieval recall at a real, but modest, cost to citation faithfulness.

## Headline Numbers

| Metric | Semantic | Hybrid | Direction |
|---|---|---|---|
| Precision@K | 0.1596 | 0.1674 | Hybrid slightly higher |
| Recall@K | 0.5463 | 0.5833 | Hybrid higher (+0.037, ~7% relative) |
| MRR | 0.5976 | 0.5976 | Identical |
| Grounding accuracy | 0.5247 | 0.4846 | **Semantic higher** (hybrid −0.040) |
| Citation correctness | 0.9815 | 1.0000 | Hybrid higher — **see caveat below** |
| Latency (s) | 15.9956 | 12.7812 | Hybrid faster |

**Win/loss (recall must be strictly higher AND grounding not worse, both queries succeeded):**
- `hybrid_win`: 2 queries (q04, q12)
- `semantic_win`: 0 queries
- `tied_or_mixed`: 25 queries
- `incomplete`: 0

## The Citation-Correctness Number Is Misleading in Isolation

Hybrid's perfect 1.0 citation correctness looks like a clean win, but it is not reliable evidence of better grounding on its own. `citation_correctness_rate` is `valid_citations / total_citations` — a call that produces **zero citations** scores a trivial, meaningless 1.0. This is not hypothetical: query q14 demonstrates it directly. Both modes retrieved the identical (wrong) set of chunks and both scored `grounding_accuracy = 0.0`, yet semantic — which at least attempted two citations, one of which was a large invalid line range — scored 0.5, while hybrid, which cited nothing at all, scored a "perfect" 1.0. **Any claim about citation correctness in the final report should be paired with citation count/coverage, never reported alone.** (Full case study in `docs/analysis_notes.md`.)

## By Category

| Category | n | Semantic recall | Hybrid recall | Semantic grounding | Hybrid grounding |
|---|---|---|---|---|---|
| control | 1 | 0.500 | 0.500 | 0.500 | 0.500 |
| local_0hop | 3 | 0.667 | 0.667 | 0.667 | 0.500 |
| cross_file_1hop | 12 | 0.625 | 0.708 | 0.639 | 0.597 |
| cross_file_2hop | 10 | 0.450 | 0.450 | 0.400 | 0.392 |
| cross_file_3hop | 1 | 0.250 | 0.250 | 0.000 | 0.000 |

**`cross_file_1hop` is where hybrid's recall advantage actually lives** — recall improves from 0.625 to 0.708 across 12 queries, the largest, most-populated category showing a real difference. But grounding accuracy in that same category still favors semantic (0.639 vs 0.597), the same tradeoff pattern as the aggregate numbers.

**`cross_file_2hop` shows *identical* recall between modes** (0.450 both) across all 10 queries — not just similar, literally the same value. This confirms a finding noted earlier in the project: for 2-hop queries in this dataset, hybrid's graph expansion was not surfacing chunks beyond what semantic search already found. This is consistent with the retriever's documented scoring limitation — a graph-expanded node's score can never exceed the semantic score of the seed it came from, so at the project's locked defaults (`top_n=15`, `decay=0.6`), deeper candidates are frequently outranked before they can matter.

**`cross_file_3hop` has only one query (q26)** and both modes scored `grounding_accuracy = 0.0` despite `MRR = 1.0` (the correct node was ranked first) — the LLM found the right code but didn't cite it in either mode. With n=1, this should be reported as a single case, not generalized.

## Hop-Depth Ablation

Across hop depths 0, 1, 2, and 3 (6 queries each, 24 total), **precision, recall, and MRR were identical at every depth** (0.1779 / 0.4583 / 0.5555). Grounding accuracy varied (0.5556 → 0.3889 → 0.5000 → 0.5556) but with no consistent trend as depth increases.

This is a genuine, reportable structural finding, already documented in the project's benchmark files: **all 16 genuine 3-hop cross-file chains in click funnel through a single node, `utils.py::echo`**, and click has **zero cross-file inheritance edges**. Click's cross-file structure is shallow enough that varying hop depth beyond 1 produced no measurable retrieval difference in this benchmark — the ablation shows hybrid's theoretical depth capability is not being exercised by click's actual codebase shape, not that deeper hops are inherently unhelpful.

## Honest Limitations

- **Sample size is small.** 27 queries, several categories with n≤3 (control, local_0hop, cross_file_3hop). Category-level numbers, especially the 3-hop single-query result, should not be treated as statistically robust.
- **Win/loss rule is one defensible choice, not the only one.** It weights recall as primary and treats grounding as a non-regression gate. A precision-primary or MRR-primary rule could show a different picture. The rule is fully disclosed in `evaluation_report.json` for scrutiny.
- **Citation correctness needs citation count context every time it's cited**, per the finding above.
- **This is one small repository.** Click has an unusual structural profile — zero cross-file inheritance, shallow multi-hop chains concentrated through one function. The Django evaluation (in progress, separate track) is expected to show a different shape, given Django already has substantial same-file inheritance where click has none.

## Draft Conclusion (edit before final submission)

On click, graph-aware hybrid retrieval produced a real, consistent recall improvement — most visible in the cross_file_1hop category, which had the largest sample — and ran measurably faster, but did not produce better grounded answers on average, and the citation-correctness metric alone overstates hybrid's advantage due to a scoring artifact of zero-citation calls. The evidence supports a qualified conclusion: **hybrid retrieval helped click's retrieval recall modestly, at a real cost to answer grounding, and the benefit was concentrated in one category rather than uniform across the benchmark.** Whether this pattern holds, strengthens, or reverses on a codebase with richer cross-file and inheritance structure is the open question the Django evaluation is designed to test.