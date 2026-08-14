#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0"]
# ///
"""Compare metric tables from two or more runs -- a determinism check.

    ./compare.py metrics_*.parquet

Nodes are matched on their `node` path, which encodes the whole lineage
including every parameter hash, so the same path in two runs means the same
recipe. For each node present in more than one run, every outcome column is
checked for agreement. Exit status is 1 if any differ, so this works in CI.

Runtimes are checked too, but as a control rather than an outcome: if nothing
in `perf_` moved, the runs very likely reused each other's outputs (an
incremental snakemake run re-executes only stale jobs) and agreement on the
metrics proves nothing at all.
"""

import argparse
import sys
from pathlib import Path

import polars as pl

# Per-run provenance -- expected to differ, never an outcome.
PROVENANCE = {"run_key", "run_id", "timestamp", "ob_version", "hostname", "node"}


def load(paths: list[Path]) -> pl.DataFrame:
    """Diagonal concat: runs may carry different columns (a newer collector
    adds one, a run may lack a stage), and those gaps become nulls."""
    return pl.concat([pl.read_parquet(p) for p in paths], how="diagonal")


def disagreements(df: pl.DataFrame, columns: list[str],
                  tolerance: float = 0.0) -> pl.DataFrame:
    """(node, column) pairs whose value is not the same in every run holding it.

    Values are first compared as text -- that catches string columns too, and
    exact equality is the common case. Numeric columns then get a second look:
    a spread within `tolerance`, taken relative to the larger magnitude but
    with a floor of 1, is not a disagreement. So the same tolerance is sensible
    for an ARI near 0.8 and a calinski_harabasz near 300. Pass 0 to demand
    exact equality.
    """
    if not columns:
        return pl.DataFrame(schema={"node": pl.String, "column": pl.String})
    diff = (
        df.select("node", "run_key", pl.col(columns).cast(pl.String))
        .unpivot(index=["node", "run_key"], variable_name="column",
                 value_name="value")
        .drop_nulls("value")
        .group_by("node", "column")
        .agg(values=pl.col("value").unique(), runs=pl.len())
        .filter(pl.col("values").list.len() > 1)
    )
    # Non-numeric values cast to null, leaving `spread` null, so they stay
    # disagreements no matter the tolerance -- text has no near-enough.
    numbers = pl.col("values").list.eval(pl.element().cast(pl.Float64, strict=False))
    scale = pl.max_horizontal(pl.lit(1.0), numbers.list.max().abs(),
                              numbers.list.min().abs())
    return (
        diff.with_columns(spread=numbers.list.max() - numbers.list.min())
        .filter(pl.col("spread").is_null()
                | (pl.col("spread") > tolerance * scale))
        .sort("node", "column")
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parquets", nargs="+", type=Path, help="two or more metric tables")
    ap.add_argument("--show", type=int, default=20,
                    help="max differing rows to print (default 20)")
    ap.add_argument("-t", "--tolerance", type=float, default=1e-5,
                    help="numeric spread treated as agreement, relative to the "
                         "larger magnitude with a floor of 1 (default 1e-5). "
                         "Use 0 to demand bit-exact equality")
    args = ap.parse_args()

    df = load(args.parquets)
    print(df.group_by("run_key", maintain_order=True)
            .agg(pl.col("run_id").first(), pl.col("timestamp").first(),
                 nodes=pl.len()))

    n_runs = df["run_key"].n_unique()
    if n_runs < 2:
        sys.exit("need at least two distinct runs to compare")

    seen = df.group_by("node").agg(runs=pl.col("run_key").n_unique())
    shared = seen.filter(pl.col("runs") == n_runs).height
    print(f"nodes: {df['node'].n_unique()} total, {shared} in all {n_runs} runs, "
          f"{seen.filter(pl.col('runs') < n_runs).height} partial")

    outcomes = [c for c in df.columns
                if c not in PROVENANCE and not c.startswith("perf_")]
    diff = disagreements(df.filter(pl.col("node").is_in(
        seen.filter(pl.col("runs") > 1)["node"].implode())), outcomes,
        args.tolerance)

    # The control stays exact: any movement at all means the jobs really ran.
    moved = disagreements(df, [c for c in df.columns if c.startswith("perf_")])
    if moved.is_empty():
        print("\nWARNING: no runtime differs between these runs -- they most "
              "likely share outputs (an incremental re-run executes only stale "
              "jobs), so agreement below is not evidence of determinism.")
    else:
        print(f"\nruntime moved on {moved.height} (node, counter) pairs, as expected")

    within = "identical" if args.tolerance == 0 else f"within {args.tolerance:g}"
    if diff.is_empty():
        print(f"{within}: every outcome agrees across {n_runs} runs")
        return
    print(f"\n{diff.height} disagreement(s) beyond {args.tolerance:g}:")
    with pl.Config(tbl_rows=args.show, fmt_str_lengths=60, tbl_cols=4):
        print(diff)
    sys.exit(1)


if __name__ == "__main__":
    main()
