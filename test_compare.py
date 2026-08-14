#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0"]
# ///
"""Self-check for compare.py.

    ./test_compare.py
"""

import polars as pl

from compare import disagreements, load


def run(key, ari, secs, extra=None):
    row = {"run_key": key, "run_id": f"id-{key}", "timestamp": "t",
           "ob_version": "0", "hostname": "h", "node": "DATA/be1/.h/CLUST-M/m/.d",
           "PCA_solver": "arpack", "ARI": ari, "perf_s": secs}
    return pl.DataFrame([row | (extra or {})])


def test_identical_runs():
    df = pl.concat([run("a", 0.839, 1.0), run("b", 0.839, 2.0)], how="diagonal")
    outcomes = [c for c in df.columns if c in ("ARI", "PCA_solver")]
    assert disagreements(df, outcomes).is_empty()
    # the control: runtime must move, or the comparison was vacuous
    assert disagreements(df, ["perf_s"]).height == 1


def test_metric_drift_is_caught():
    df = pl.concat([run("a", 0.839, 1.0), run("b", 0.842, 2.0)], how="diagonal")
    diff = disagreements(df, ["ARI", "PCA_solver"])
    assert diff.height == 1
    assert diff["column"][0] == "ARI"
    assert sorted(diff["values"][0]) == ["0.839", "0.842"]


def test_column_missing_from_one_run():
    # an older run lacking a column must not read as a disagreement
    df = pl.concat([run("a", 0.839, 1.0),
                    run("b", 0.839, 2.0, {"CLUST_n_clusters": 11})], how="diagonal")
    assert disagreements(df, ["ARI", "CLUST_n_clusters"]).is_empty()


def test_load_tolerates_differing_columns(tmp_path):
    run("a", 0.839, 1.0).write_parquet(tmp_path / "a.parquet")
    run("b", 0.839, 2.0, {"CLUST_n_clusters": 11}).write_parquet(tmp_path / "b.parquet")
    df = load([tmp_path / "a.parquet", tmp_path / "b.parquet"])
    assert df.height == 2 and "CLUST_n_clusters" in df.columns


if __name__ == "__main__":
    import tempfile

    test_identical_runs()
    test_metric_drift_is_caught()
    test_column_missing_from_one_run()
    with tempfile.TemporaryDirectory() as d:
        from pathlib import Path
        test_load_tolerates_differing_columns(Path(d))
    print("ok")
