# Apriori benchmark

Run `notebooks/01_mining.ipynb` locally and copy the measured values from the generated benchmark table.

| min_support | Itemsets | Rules | Time (s) | Peak memory above baseline (MB) | Outcome |
|---:|---:|---:|---:|---:|---|
| 0.05 |  |  |  |  |  |
| 0.02 |  |  |  |  |  |
| 0.01 |  |  |  |  |  |
| 0.005 |  |  |  |  |  |

## Memory note

The required algorithm is `mlxtend.frequent_patterns.apriori`. On one million baskets, candidate generation can use substantial memory. The notebook runs Apriori with `low_memory=True` and measures wall-clock time plus peak process RSS. Any failed run should be documented honestly instead of replaced with estimated numbers.
