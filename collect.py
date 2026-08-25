#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0", "pyyaml"]
# ///
"""Collect one omnibenchmark run tree (an `out/` folder) into a flat table.

One row per node, where a node is a `STAGE/method/.param_hash` triple in the
output tree. Everything about that node is a column: its identity, its
parameters, its metrics, and its snakemake resource counters.

    ./collect.py ../split-stages-plan/out --csv

Stage ids ending in `-M` are metric stages (EMBED-M, GRAPH-M, CLUST-M); the
rest are methods. A metric node scores the method directly above it, named in
`evaluates` / `evaluates_method`. Every node also carries its whole lineage as
`{STAGE}_module` and `{STAGE}_{param}` columns, so a row is self-describing:
the EMBED-M row holding a silhouette also holds the PCA solver that earned it.
Identity, declared parameters and derived facts get separate prefixes, so a
module is free to declare a parameter called `method` or `n_clusters`.
"""

import argparse
import hashlib
import json
import os
import re
from multiprocessing import Pool
from pathlib import Path

import polars as pl
import yaml


def read_modules(meta: Path) -> dict[str, dict]:
    """{"STAGE/method": {"commit": ..., "repo": ...}} from the run metadata.

    benchmark.yaml is the complete list; modules.txt only records a subset
    (omnibenchmark 0.6 drops modules that share a repo with an earlier stage)
    but carries full SHAs, so it wins where it overlaps.
    """
    out: dict[str, dict[str, str]] = {}

    config = meta / "benchmark.yaml"
    if config.exists():
        spec = yaml.safe_load(config.read_text()) or {}
        for stage in spec.get("stages") or []:
            for module in stage.get("modules") or []:
                repo = module.get("repository") or {}
                out[f"{stage['id']}/{module['id']}"] = {
                    "commit": repo.get("commit"), "repo": repo.get("url")}

    listing = meta / "modules.txt"
    if listing.exists():
        pattern = r"^(\S+):\n\s+Repository:\s*(\S+)\n\s+Commit:\s*(\S+)"
        for key, repo, commit in re.findall(pattern, listing.read_text(), re.M):
            out.setdefault(key, {}).update(commit=commit, repo=repo)
    return out


def lineage_of(rel: Path) -> list[tuple[str, str, str]]:
    """Out-relative node dir -> [(stage, method, param_hash), ...].

    The tree alternates STAGE / method / .hash all the way down, so the parts
    come in threes; anything else is not a node.
    """
    parts = rel.parts
    if not parts or len(parts) % 3:
        return []
    return [(parts[i], parts[i + 1], parts[i + 2].lstrip("."))
            for i in range(0, len(parts), 3)]


def as_float(text: str) -> float | None:
    """Snakemake writes 'NA' for a counter it could not measure (io_in/io_out
    on filesystems it can't poll, rss on a job that ended too fast). That is a
    missing reading, not a zero, so it becomes null."""
    try:
        return float(text)
    except ValueError:
        return None


def read_perf(path: Path) -> dict[str, float | None]:
    """Snakemake's two-line benchmark TSV: header, then one row of numbers.

    `h:m:s` is dropped -- it is `s` in human form, and the only non-number.
    """
    head, row = path.read_text().splitlines()[:2]
    return {f"perf_{k}": as_float(v)
            for k, v in zip(head.split("\t"), row.split("\t")) if k != "h:m:s"}


def read_params(node_dir: Path) -> dict:
    f = node_dir / "parameters.json"
    return json.loads(f.read_text()) if f.exists() else {}


def count_clusters(node_dir: Path) -> dict:
    """How many clusters a CLUST node actually found.

    The metric modules report `n_labels`, which is the *truth* label count and
    so is constant everywhere; nothing records what the method produced. ARI
    is highly sensitive to it, so derive it from the clusters.tsv itself.
    """
    files = list(node_dir.glob("*_clusters.tsv"))
    if not files:
        return {}
    rows = files[0].read_text().splitlines()[1:]
    return {"n_clusters": len({r.rsplit("\t", 1)[-1] for r in rows if r})}


def node_row(job: tuple[str, str]) -> dict | None:
    """Flatten one node into a single row. Runs in a worker process."""
    root, node_dir = job
    d = Path(node_dir)
    rel = d.relative_to(root)
    chain = lineage_of(rel)
    if not chain:
        return None
    stage, method, param_hash = chain[-1]

    # Ancestor dirs are prefixes of ours, so their parameters.json is one read
    # away. Flattening them here is what makes a metric row self-describing.
    lineage = {}
    for depth, (s, m, _) in enumerate(chain, start=1):
        ancestor = Path(root).joinpath(*rel.parts[:depth * 3])
        # Three separate namespaces. Sharing one silently loses data whenever a
        # module declares a parameter named like the identity or a derived fact:
        # cl-rapids declares `method: rapids-leiden`, which used to overwrite its
        # own module id, and sc3s declares `n_clusters`, which used to be
        # replaced by the observed count that happened to equal it.
        lineage[f"{s}_module"] = m
        lineage |= {f"{s}_{k}": v for k, v in read_params(ancestor).items()}
        lineage |= {f"{s}_observed_{k}": v for k, v in count_clusters(ancestor).items()}

    # Metric stages carry a `-M` suffix (EMBED-M, GRAPH-M, CLUST-M) and score
    # the method stage directly above them. Runs that don't follow the
    # convention get these three columns blanked in collect() -- null means
    # "unknown", which beats calling a metric stage a method.
    is_metric = stage.endswith("-M")
    parent = chain[-2] if is_metric and len(chain) > 1 else None
    row = {
        "node": str(rel),
        "stage": stage,
        "method": method,
        "kind": "metric" if is_metric else "method",
        "param_hash": param_hash,
        "metric_group": None,
        "evaluates": parent[0] if parent else None,
        "evaluates_method": parent[1] if parent else None,
        **lineage,
    }

    # Not "*_metrics.json": older runs write these unprefixed, without the
    # dataset stem (out_ci has a bare performance.txt).
    for f in d.glob("*metrics.json"):
        row["metric_group"] = re.sub(r"_?metrics\.json$", "", f.name).split("_")[-1] \
            or "metrics"
        row |= json.loads(f.read_text())
    for f in d.glob("*performance.txt"):
        row |= read_perf(f)
    return row


def node_dirs(root: Path) -> list[str]:
    """Every directory holding results. Sorted, so parents come before their
    children. Skips out/'s dot-dirs (.snakemake, .modules, ...); param-hash
    dirs are dotted too, but only ever below a stage."""
    return sorted({str(f.parent)
                   for top in root.iterdir()
                   if top.is_dir() and not top.name.startswith(".")
                   for pattern in ("**/*performance.txt", "**/*metrics.json")
                   for f in top.glob(pattern)})


def order_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Identity first, resources and run provenance last. The middle keeps the
    order the rows were built in: lineage shallow to deep, then own metrics."""
    lead = ["run_key", "node", "stage", "method", "kind", "commit", "repo",
            "param_hash", "metric_group", "evaluates", "evaluates_method"]
    tail = ["run_id", "timestamp", "ob_version", "hostname"]
    perf = [c for c in df.columns if c.startswith("perf_")]
    rest = [c for c in df.columns if c not in set(lead + tail + perf)]
    return df.select(lead + rest + perf + tail)


def collect(root: Path, processes: int | None = None) -> pl.DataFrame:
    meta = root / ".metadata" / "manifest.json"
    if not meta.exists():
        raise SystemExit(f"{meta} not found -- is {root} an omnibenchmark run?")
    manifest = json.loads(meta.read_text())
    modules = read_modules(root / ".metadata")

    jobs = [(str(root), d) for d in node_dirs(root)]
    with Pool(processes or min(os.cpu_count() or 1, len(jobs) or 1)) as pool:
        # imap, not imap_unordered: jobs are sorted parent-first, so keeping
        # that order is what puts the lineage columns in DAG order for free.
        rows = [r for r in pool.imap(node_row, jobs, chunksize=8) if r]
    # A results dir whose depth is not a multiple of three is not a node we can
    # place in the DAG. Say so rather than dropping it quietly.
    if skipped := len(jobs) - len(rows):
        print(f"warning: {skipped} of {len(jobs)} results dirs are not "
              f"STAGE/method/.hash triples and were skipped")

    run_id = manifest["run_id"]
    key = pl.concat_str("stage", "method", separator="/")
    df = (
        # infer_schema_length=None is load-bearing: rows carry one key per
        # stage in their lineage, so deep stages (CLUST-M) only appear in late
        # rows. Inferring from the first 100 silently drops their columns.
        pl.DataFrame(rows, infer_schema_length=None)
        .with_columns(
            run_id=pl.lit(run_id),
            run_key=pl.lit(hashlib.md5(run_id.encode()).hexdigest()[:8]),
            timestamp=pl.lit(manifest.get("timestamp")),
            ob_version=pl.lit(manifest.get("ob_version")),
            hostname=pl.lit(manifest.get("hostname")),
            commit=key.replace_strict(
                {k: v.get("commit") for k, v in modules.items()}, default=None),
            repo=key.replace_strict(
                {k: v.get("repo") for k, v in modules.items()}, default=None),
        )
        .sort("node")
    )

    # `kind`/`evaluates` mean nothing if this benchmark doesn't use the `-M`
    # suffix (out_ci spells it `cluster-metrics`). Blank them rather than
    # labelling every metric stage a method.
    if not df["stage"].str.ends_with("-M").any():
        print("warning: no stage id ends in '-M'; leaving kind/evaluates empty")
        df = df.with_columns(pl.lit(None, dtype=pl.String)
                             .alias(c) for c in
                             ("kind", "evaluates", "evaluates_method"))
    return order_columns(df)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path, help="an omnibenchmark out/ folder")
    ap.add_argument("-o", "--out", type=Path, default=Path("."),
                    help="directory to write into, created if missing "
                         "(default: the current one). Files are always named "
                         "metrics_{run_key}.parquet, so several runs can share it")
    ap.add_argument("-j", "--processes", type=int, default=None)
    ap.add_argument("--csv", action="store_true",
                    help="also write a sibling .csv, for eyeballing in a spreadsheet")
    args = ap.parse_args()

    df = collect(args.run_dir, args.processes)
    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / f"metrics_{df['run_key'][0]}.parquet"
    df.write_parquet(dest, compression="zstd")
    written = [dest]
    if args.csv:
        written.append(dest.with_suffix(".csv"))
        df.write_csv(written[-1])

    missing = df.filter(pl.col("commit").is_null())["stage", "method"].unique()
    if missing.height:
        print(f"warning: no commit in run metadata for {missing.height} module(s): "
              + ", ".join(f"{s}/{m}" for s, m in missing.iter_rows()))
    print(f"{', '.join(map(str, written))}: {df.height} nodes, "
          f"{len(df.columns)} columns, run {df['run_id'][0]}")


if __name__ == "__main__":
    main()
