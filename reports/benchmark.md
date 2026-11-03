# Apriori Benchmark — Boréal Marché

## Experimental setup

- Dataset: 1,000,000 shopping baskets, 8,379,837 transaction rows, 98 products.
- Basket matrix: 1,000,000 × 98, represented with sparse boolean data.
- Required implementation: `mlxtend.frequent_patterns.apriori`.
- For every reported run: `low_memory=True`, `use_colnames=True`.
- Association rules generated with `metric="confidence"` and `min_threshold=0.30`.
- Timing: elapsed wall-clock time; memory: measured peak process RSS **above the baseline** as reported by the executed notebook. These values are not total system RAM usage.

## Measured results

| min_support | Itemsets | Rules | Time (s) | Peak memory above baseline (MB) | Outcome |
|---:|---:|---:|---:|---:|---|
| 0.05 | 191 | 282 | 3.64 | 212.36 | Success |
| 0.02 | 658 | 2,707 | 7.52 | 262.26 | Success |
| 0.01 | 2,267 | 7,817 | 15.86 | 219.13 | Success |
| 0.005 | 7,106 | 23,918 | 44.94 | 231.38 | Success |

All four rows reflect **observed successful runs**, not estimates or fabricated failures. See `notebooks/01_mining.ipynb` and `reports/benchmark_results.csv` for the recorded analysis.

## Why memory is a concern with Apriori

Apriori searches for frequent item combinations and repeatedly generates and evaluates candidate itemsets. With a million baskets, candidate generation and intermediate arrays can consume substantially more memory than the original sparse basket matrix. As `min_support` decreases, more itemsets qualify, which generally increases the work required; in our measurements the number of itemsets grew from 191 to 7,106, and elapsed time increased from 3.64 s to 44.94 s.

The `low_memory=True` option in `mlxtend.frequent_patterns.apriori` uses a lower-memory approach to generating candidates, trading possible runtime costs for reduced memory demand. We chose it to make the full-scale runs more practical. All measured runs succeeded using this option. We **did not record a controlled default (`low_memory=False`) failure**, so this report does not claim an observed `MemoryError`. Such a failure is a risk identified in the assignment brief, not an experimental outcome evidenced by this table.

Peak memory does not need to increase monotonically with lower support: intermediate allocations, process memory management and measurement effects can vary between runs. The figures above must therefore be read as measured values, not as a strict scaling law.

## Final selection and rule quality

We selected `min_support=0.005` and `min_confidence=0.30`, then removed rules with `lift <= 1` because a lift near 1 indicates little added association beyond the consequent's ordinary popularity. For example, `Sac réutilisable` is present in approximately 90% of baskets, making confidence-only rankings misleading.

- Rules generated at the selected thresholds, before lift filtering: **23,918**.
- Filtered rules with `lift > 1`: **22,476**.
- Recommendable products: **58 out of 98**.
- Catalogue coverage: **59.18%**.

## Limitations and proposed improvements

1. **Coverage and cold start.** Association rules cannot suggest every catalogue product and cannot fire for an empty basket or some rare-product baskets. The application falls back to popular items. A hybrid recommender incorporating content-based similarity or product-category information could improve coverage and diversity.
2. **Seasonality and changing purchasing habits.** Rules mined from a full year do not necessarily capture current preferences. The seasonal analysis found meaningful differences between summer and winter. Re-mining periodically (for example, quarterly and before major seasonal changes) would improve relevance.

## Operational separation

Mining is performed offline in the notebook. The Streamlit application loads `models/rules.json` and `models/metadata.json` and does **not** invoke Apriori while serving recommendations.
