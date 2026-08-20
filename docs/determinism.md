# Comparing determinism across runs

```sh
./compare.py data/*.parquet            # exit 1 if any outcome differs
./compare.py data/*.parquet -t 0       # demand bit-exact equality
./compare.py data/*.parquet -t 1e-3    # looser
```

`--tolerance` (`-t`, default `1e-5`) is the numeric spread treated as
agreement, taken relative to the larger magnitude with a floor of 1 — so one
number works for an ARI near 0.8 and a `calinski_harabasz` near 300. Text
columns are always exact; a solver name has no near-enough. It matters: two
full runs of the be1 slice agree to 1e-5 but not bit-exactly, drifting ~3e-11
on `calinski_harabasz` and ~9e-14 on `silhouette`.

Runs are merged with a diagonal concat, so tables with different columns (a
newer collector, a run missing a stage) line up with nulls. Nodes match on the
`node` path, which encodes the whole lineage including every parameter hash —
the same path in two runs means the same recipe, so it is the join key. Seeds
ride along as `<STAGE>_random_seed` and are part of that hash.

Runtimes are checked as a **control**, not an outcome. If nothing in `perf_`
moved, the runs almost certainly share outputs — an incremental snakemake run
re-executes only stale jobs — and agreement on the metrics proves nothing.
`compare.py` says so loudly rather than reporting a clean state:


```
WARNING: no runtime differs between these runs -- they most likely share
outputs ..., so agreement below is not evidence of determinism.
```

Two full runs of the be1 slice agree on all 64 nodes and every outcome
(ARI, AMI, silhouette, graph metrics, `n_clusters`) with runtimes moving on
520 counter pairs — so the `random_seed: 42` plumbing holds end to end,
randomized solvers included.
