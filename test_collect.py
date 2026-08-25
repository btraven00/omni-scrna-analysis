#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0", "pyyaml"]
# ///
"""Self-check: build a miniature out/ tree and assert collect() reads it.

    ./test_collect.py
"""

import json
from pathlib import Path


from collect import collect, lineage_of, read_modules

# io_in is 'NA': snakemake emits that for a counter it could not measure
PERF = "s\th:m:s\tmax_rss\tio_in\tcpu_time\n1.5\t0:00:01\t20.41\tNA\t7.14\n"

CONFIG = """
stages:
  - id: DATA
    modules:
      - id: be1
        repository: {url: https://e.com/d, commit: aaa}
  - id: PCA
    modules:
      - id: pc-scanpy
        repository: {url: https://e.com/s, commit: bbb}
  - id: EMBED-M
    modules:
      - id: em-metrics-py
        repository: {url: https://e.com/m, commit: ccc}
  - id: CLUST
    modules:
      - id: cl-scrapper
        repository: {url: https://e.com/s, commit: ddd}
  - id: CLUST-M
    modules:
      - id: cl-metrics-r
        repository: {url: https://e.com/m, commit: eee}
"""


def build(root: Path) -> None:
    meta = root / ".metadata"
    meta.mkdir(parents=True)
    (meta / "manifest.json").write_text(json.dumps(
        {"run_id": "488eeafd-9a6b-4f05-8e61-bc7cc29ba096", "hostname": "x"}))
    (meta / "benchmark.yaml").write_text(CONFIG)
    # modules.txt is incomplete in real runs, and its SHAs win where present
    (meta / "modules.txt").write_text(
        "DATA/be1:\n  Repository: https://e.com/d\n  Commit: aaa111\n")
    # a dot-dir at the top must be skipped even though it looks like a node
    snake = root / ".snakemake" / "conda" / "x"
    snake.mkdir(parents=True)
    (snake / "be1_performance.txt").write_text(PERF)

    data = root / "DATA" / "be1" / ".f612184b"
    data.mkdir(parents=True)
    (data / "be1_performance.txt").write_text(PERF)

    pca = data / "PCA" / "pc-scanpy" / ".07ef8b80"
    pca.mkdir(parents=True)
    (pca / "be1_performance.txt").write_text(PERF)
    (pca / "parameters.json").write_text('{"solver": "arpack", "n_components": 50}')

    em = pca / "EMBED-M" / "em-metrics-py" / ".default"
    em.mkdir(parents=True)
    (em / "be1_embedding_metrics.json").write_text(
        json.dumps({"n_cells": 1715, "silhouette": 0.2648, "note": "hi"}))

    clust = pca / "CLUST" / "cl-scrapper" / ".6a558c16"
    clust.mkdir(parents=True)
    (clust / "performance.txt").write_text(PERF)  # older runs omit the stem
    # both collision cases at once: a module declaring `method` (cl-rapids does)
    # and one declaring `n_clusters` (sc3s does), against 3 observed clusters
    (clust / "parameters.json").write_text('{"method": "rapids-leiden", "n_clusters": 8}')
    (clust / "be1_clusters.tsv").write_text(
        "cell_id\tcluster\na\t1\nb\t2\nc\t2\nd\t3\n")  # 3 distinct clusters
    cm = clust / "CLUST-M" / "cl-metrics-r" / ".default"
    cm.mkdir(parents=True)
    (cm / "be1_cluster_metrics.json").write_text(
        json.dumps({"n_labels": 8, "ARI": 0.839}))


def test_lineage_of():
    assert lineage_of(Path("DATA/be1/.f612184b")) == [("DATA", "be1", "f612184b")]
    assert lineage_of(Path("DATA/be1")) == []  # not a whole triple


def test_read_modules(tmp_path):
    build(tmp_path)
    mods = read_modules(tmp_path / ".metadata")
    assert mods["DATA/be1"]["commit"] == "aaa111"  # modules.txt beats the config
    assert mods["PCA/pc-scanpy"]["commit"] == "bbb"  # only the config has it
    assert mods["EMBED-M/em-metrics-py"]["repo"] == "https://e.com/m"


def test_kind_blanked_without_the_convention(tmp_path):
    build(tmp_path)
    # rename the metric stages to the spelled-out style other benchmarks use
    for old, new in (("EMBED-M", "embedding-metrics"), ("CLUST-M", "cluster-metrics")):
        for d in tmp_path.rglob(old):
            d.rename(d.with_name(new))
    df = collect(tmp_path, processes=1)
    assert df["kind"].null_count() == df.height  # unknown, not "method"
    assert df["evaluates"].null_count() == df.height
    assert df["ARI"].drop_nulls().to_list() == [0.839]  # metrics still collected


def test_collect(tmp_path):
    build(tmp_path)
    (tmp_path / "stray.txt").write_text("")  # a top-level file must not break us
    df = collect(tmp_path, processes=2)

    assert df.height == 5  # one row per node; the .snakemake one is excluded
    assert df["node"].is_duplicated().sum() == 0
    assert df["run_key"].n_unique() == 1 and len(df["run_key"][0]) == 8

    rows = {r["stage"]: r for r in df.iter_rows(named=True)}

    # -M naming marks metric stages, and they name the method they score
    assert rows["EMBED-M"]["kind"] == "metric"
    assert rows["PCA"]["kind"] == "method"
    assert (rows["EMBED-M"]["evaluates"], rows["EMBED-M"]["evaluates_method"]) \
        == ("PCA", "pc-scanpy")
    assert rows["PCA"]["evaluates"] is None

    # metrics are columns, flat -- no nested json anywhere
    assert rows["EMBED-M"]["silhouette"] == 0.2648
    assert rows["EMBED-M"]["note"] == "hi"  # non-numeric metrics survive
    assert rows["EMBED-M"]["metric_group"] == "embedding"
    assert rows["PCA"]["silhouette"] is None

    # performance is prefixed, h:m:s dropped, values numeric
    assert rows["PCA"]["perf_max_rss"] == 20.41
    assert not [c for c in df.columns if c.endswith("h:m:s")]
    # an unmeasured counter is null, not an error and not a zero
    assert "perf_io_in" in df.columns
    assert rows["PCA"]["perf_io_in"] is None

    # lineage: the metric row carries the PCA solver that produced it
    assert rows["EMBED-M"]["PCA_module"] == "pc-scanpy"
    assert rows["EMBED-M"]["PCA_solver"] == "arpack"
    assert rows["EMBED-M"]["PCA_n_components"] == 50
    assert rows["DATA"]["PCA_solver"] is None

    # k found is derived from clusters.tsv, and rides the lineage down to the
    # CLUST-M row where ARI lives -- n_labels there is the truth count, not k
    assert rows["CLUST"]["CLUST_observed_n_clusters"] == 3
    assert rows["CLUST-M"]["CLUST_observed_n_clusters"] == 3
    assert rows["CLUST-M"]["n_labels"] == 8
    assert rows["PCA"]["CLUST_observed_n_clusters"] is None

    # identity, declared parameters and derived facts must not overwrite each
    # other: all three survive a module that declares `method` and `n_clusters`
    for r in ("CLUST", "CLUST-M"):
        assert rows[r]["CLUST_module"] == "cl-scrapper"        # identity
        assert rows[r]["CLUST_method"] == "rapids-leiden"      # declared
        assert rows[r]["CLUST_n_clusters"] == 8                # declared, not 3
        assert rows[r]["CLUST_observed_n_clusters"] == 3       # derived

    # commits resolve for every module, from either metadata file
    assert df["commit"].null_count() == 0
    assert rows["PCA"]["commit"] == "bbb"
    assert rows["DATA"]["commit"] == "aaa111"

    # identity leads, resources trail
    assert df.columns[:4] == ["run_key", "node", "stage", "method"]
    assert df.columns.index("PCA_solver") < df.columns.index("silhouette")
    assert df.columns.index("silhouette") < df.columns.index("perf_s")


if __name__ == "__main__":
    import tempfile

    test_lineage_of()
    for case in (test_read_modules, test_collect, test_kind_blanked_without_the_convention):
        with tempfile.TemporaryDirectory() as d:
            case(Path(d))
    print("ok")
