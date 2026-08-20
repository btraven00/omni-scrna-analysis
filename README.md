# omni-scrna-analysis

Turns one omnibenchmark run (an `out/` folder) into a single parquet for
archiving and posterior analysis.

```sh
./collect.py ../split-stages-plan/out
# metrics_c707a622.parquet: 54 nodes, 71 columns, run 488eeafd-9a6b-...

./collect.py ../split-stages-plan/out --csv          # also as a .csv
./collect.py ../split-stages-plan/out -o results/    # write into a folder
```

`--out` (`-o`) names a **directory**, created if missing, defaulting to the
current one. The file is always `metrics_{run_key}.parquet`, so several runs
can collect into one folder without colliding — which is what `compare.py`
wants:

```sh
for d in out out_2 out_3; do ./collect.py ../split-stages-plan/$d -o results/; done
./compare.py results/*.parquet
```

Dependencies declared as a uv script preamble, so there is no env to set up.

The filename reflects `run_key`, the first 8 hex of `md5(run_id)`, where
`run_id` comes from `out/.metadata/manifest.json`. Parquet is the archive
copy; the CSV dumps the same table, ~20x bigger size and untyped.

## Shape

Flat: **one row per node**, one column per fact. A node is a
`STAGE/method/.hash` triple in the output tree. No nested JSON anywhere — the
CSV opens in a spreadsheet as-is.

| column | |
|---|---|
| `stage`, `method`, `param_hash`, `commit`, `repo` | which module produced this row |
| `kind` | `metric` for stages ending in `-M`, else `method` |
| `evaluates`, `evaluates_method` | for a metric node, the method stage it scores |
| `metric_group` | `embedding`, `graph`, `cluster`, … |
| `<STAGE>_method`, `<STAGE>_<param>` | lineage — method and each parameter at every ancestor stage |
| `CLUST_n_clusters` | how many clusters the method *found*, counted from `clusters.tsv` |
| `ARI`, `silhouette`, `SI`, … | the metrics themselves, one column each |
| `perf_s`, `perf_max_rss`, … | snakemake resource counters |
| `run_id`, `run_key`, `timestamp`, `ob_version`, `hostname` | run provenance |

Metric stages are identified by the `-M` suffix in the plan's stage ids
(`EMBED-M`, `GRAPH-M`, `CLUST-M`), and a metric node scores the method
directly above it. A benchmark that doesn't follow the convention gets `kind`,
`evaluates` and `evaluates_method` left **empty**.

```python
df.filter(pl.col("stage") == "CLUST-M").select(
    "PCA_method", "PCA_solver", "CLUST_resolution", "ARI", "AMI")
# pc-rapids  randomized-halko  1.0  0.9297  0.90
```

Runtime and memory come from snakemake's `*_performance.txt`, prefixed
`perf_` so they never collide with a metric name. `h:m:s` is dropped as a
duplicate of `s`.

`n_clusters` is derived, not reported. The metric modules emit `n_labels`,
which is the *truth* label count and therefore constant across every row —
nothing records how many clusters a method actually found, even though ARI is
very sensitive to it. Counting the distinct labels in the CLUST node's
`clusters.tsv` fills the gap, and it rides the lineage down to the CLUST-M row
where ARI lives.


## Comparing determinism across runs

```sh
./compare.py data/*.parquet            # exit 1 if any outcome differs
```

See [docs/determinism.md](docs/determinism.md) for tolerances, how nodes are
matched, and the runtime control.

## Check

```sh
./test_collect.py && ./test_compare.py
```
