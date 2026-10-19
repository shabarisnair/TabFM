#!/usr/bin/env python
"""Freeze the tiny-V benchmark's tables into a folder git tracks (results/ is git-ignored), so they can be
reloaded and analysed later, also on another machine.

    python scripts/archive_bench_tinyv.py [--root results/bench_tinyv] [--out results_archive/bench_tinyv] [--tag DATE]

Run from the repository root. Regenerates every table from <root>'s runs into <out>/<tag>/ (report_bench_tinyv.py,
elo_bench_tinyv.py) and adds:
  - analysis-ready CSVs: by_v_r.csv (the per V x R tables, unrounded), per_dataset.csv, datasets.csv and
    instances.csv;
  - raw/runs.jsonl.gz, every run record as bench_tinyv.py wrote it (configurations, timings, all rows);
  - raw/draws_and_masks.tar.gz, the draws (V, the poisoned rows, the given labels) and the saved removal masks;
  - code/, the scripts that made the results and the tables;
  - findings.md (the written analysis, copied from <tables>), README.md (files, columns, how to load them) and
    manifest.json (git commit, counts and the sha256 of every file).
The baselines' configuration grids (<root>/grids, about 250 MB) are left out; they can be recomputed.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import gzip
import hashlib
import io
import json
import shutil
import socket
import subprocess
import sys
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_tinyv as B  # noqa: E402
import elo_bench_tinyv as EB  # noqa: E402
import report_bench_tinyv as RB  # noqa: E402

CODE = ("bench_tinyv.py", "tinyv_learned.py", "report_bench_tinyv.py", "elo_bench_tinyv.py", "archive_bench_tinyv.py")
VR_METRICS = {"prec": "prec", "auc": "auc", "acc": "acc", "ce": "logloss"}  # results.csv column: by_v_r.csv suffix


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout.rstrip()  # porcelain lines start with a space


def datasets_table() -> pd.DataFrame:
    """Type, class shares and balance of each dataset, read from its split files."""
    import gate_context as G
    rows = []
    for ds, (ctx, test) in B.DATASETS.items():
        y, yt = (np.asarray(G.load(Path(f), "cpu")[1].tolist()).astype(int) for f in (ctx, test))
        p, pt = np.bincount(y) / len(y), np.bincount(yt) / len(yt)
        assert np.allclose(p, RB.CLASS_SHARES[ds], atol=1e-3), (ds, p)  # the shares the report's tables print
        rows.append({"dataset": ds, "tag": RB.TAG[ds], "type": "binary" if len(p) == 2 else "multi-class",
                     "n_classes": len(p), "balance": RB.balance(p), "max_min_ratio": round(p.max() / p.min(), 2),
                     "class_shares_context": " / ".join(f"{s:.3f}" for s in p),
                     "class_shares_test": " / ".join(f"{s:.3f}" for s in pt),
                     "n_context": len(y), "n_test": len(yt), "context_file": ctx, "test_file": test})
    return pd.DataFrame(rows)


def instances_table(root: Path) -> pd.DataFrame:
    rows = []
    for f in sorted((root / "draws").glob("*.npz")):
        z = np.load(f)
        ds, rest = f.stem.rsplit("_R", 1)
        R, c, s = rest.split("_")
        rows.append({"instance": f.stem, "dataset": ds, "c": int(c[1:]), "R": float(R), "seed": int(s[1:]),
                     "v_size": len(z["v_rows"]), "pool_size": len(z["ctx_rows"]), "k_poisoned": int(z["k"])})
    return pd.DataFrame(rows)


def by_v_r_table(res: pd.DataFrame) -> pd.DataFrame:
    """One row per (c, R, entry); per dataset its precision, ROC-AUC, accuracy and log loss, unrounded."""
    long = res.melt(id_vars=["c", "R", "entry", "dataset"], value_vars=list(VR_METRICS), var_name="metric")
    long["col"] = long.dataset.map(RB.TAG) + "_" + long.metric.map(VR_METRICS)
    wide = long.pivot(index=["c", "R", "entry"], columns="col", values="value")
    wide = wide.reindex(columns=[f"{t}_{m}" for t in RB.TAG.values() for m in VR_METRICS.values()]).reset_index()
    order = {e: i for i, (e, _) in enumerate(RB.ENTRIES)}
    wide.insert(3, "method", wide.entry.map(RB.NAME))
    return wide.sort_values(["c", "R", "entry"], key=lambda s: s.map(order) if s.name == "entry" else s).reset_index(drop=True)


def per_dataset_table(res: pd.DataFrame) -> pd.DataFrame:
    out = res.groupby(["dataset", "entry"]).agg(
        instances=("instance", "size"), acc=("acc", "mean"), bacc=("bacc", "mean"), auc=("auc", "mean"),
        logloss=("ce", "mean"), prec=("prec", "mean"), n_removed=("n_removed", "mean")).reset_index()
    out.insert(2, "method", out.entry.map(RB.NAME))
    return out


def readme(tag: str, meta: dict, ds: pd.DataFrame, entries: list[str]) -> str:
    L = [f"# Tiny-V poisoning benchmark: frozen results ({tag})", "",
         "TabPFN v2 with V (c clean rows per class) + a method's cleaned version of the poisoned train context, scored "
         "on each dataset's held-out test set. 6 datasets x c in {1, 3, 5} x R in {10, 20, 30}% hub-flip-local "
         "poisoning, seed 0: 54 instances. The runbook is `docs/bench_tinyv.md`; the analysis is `findings.md`.", "",
         "## Provenance", "",
         f"- Made {meta['created']} on {meta['host']} by `scripts/archive_bench_tinyv.py` from `{meta['root']}`.",
         f"- Git commit {meta['git_commit']}"
         + (f", with uncommitted changes to {', '.join(meta['git_dirty'])}." if meta["git_dirty"] else "."),
         "  `code/` holds the scripts as they were when this snapshot was made.",
         f"- {meta['instances']} instances, {meta['entries']} entries, {meta['run_files']} run files. Runs per host: "
         + ", ".join(f"{h} {n}" for h, n in meta["runs_per_host"].items()) + ".", "",
         "## Files", "",
         "| file | what |", "|---|---|",
         "| `results.csv` | one row per instance x entry: every metric (the source of every other table) |",
         "| `by_v_r.md`, `by_v_r.csv` | one table per V size x poisoning rate: per dataset, precision, ROC-AUC, accuracy, log loss |",
         "| `report.md` | summary over the instances and per-dataset tables of each metric |",
         "| `per_dataset.csv` | means over each dataset's 9 instances, per entry |",
         "| `elo.md`, `elo.csv` | TabArena-style Elo (main fit and sensitivity fits) |",
         "| `datasets.csv` | per dataset: type, classes, class shares, balance, split files and sizes |",
         "| `instances.csv` | per instance: V size, poisoned-set size and number of poisoned rows k |",
         "| `findings.md` | the written analysis |",
         "| `raw/runs.jsonl.gz` | every run record (configurations, timings, all rows), one JSON object per line |",
         "| `raw/draws_and_masks.tar.gz` | the draws (`draws/<instance>.npz`: v_rows, ctx_rows, y_given, pois_rows, k) and the removal masks (`runs/<instance>/<method>.npz`) |",
         "| `code/` | the scripts that ran the benchmark and made these tables |",
         "| `manifest.json` | git commit, counts, sha256 of every file |", "",
         "## Columns", "",
         "`results.csv`:",
         "- `instance` (`<dataset>_R<R>_c<c>_s<seed>`), `dataset`, `R` (% of the poisoned set given a wrong label), "
         "`c` (clean rows per class in V), `seed`;",
         "- `entry`: the method (legend below);",
         "- `acc`, `bacc`: test accuracy and balanced accuracy (0-1); `auc`: test ROC-AUC (binary: of class 1; "
         "multi-class: one-vs-rest macro); `ce`: test log loss;",
         "- `prec`: share of the removed rows that were poisoned (relabel modes: of the relabelled rows); empty when "
         "nothing is removed. Each method removes k rows, so removing at random scores R/100 on average;",
         "- `n_removed`: rows removed; `cfg`, `mode`: a baseline's configuration and mode (V-selected rows: the "
         "chosen ones); `imported`: taken from the schedule study's identical runs.",
         "",
         "`by_v_r.csv`: `c`, `R`, `entry`, `method`, then `<tag>_<metric>` per dataset tag (below) and metric "
         "(`prec`, `auc`, `acc`, `logloss`).",
         "",
         "`elo.csv`: `variant` (`main` is TabArena's procedure; the others are sensitivity fits), `entry`, `elo`, "
         "`lo_tasks`/`hi_tasks` (95% CI, task bootstrap), `lo_datasets`/`hi_datasets` (dataset bootstrap), "
         "`gap_lo_*`/`gap_hi_*` (95% interval of the Elo gap to poisoned + V), `win_rate`, `improvability`, `tasks`, "
         "`W`/`T`/`L` (wins, ties, losses against poisoned + V). Error: 1 - ROC-AUC on binary datasets, log loss "
         "on multi-class ones; Bradley-Terry fit; poisoned + V calibrated to 1000; oracle + V not rated.", "",
         "## Datasets", "",
         "| tag | dataset | type | balance | class shares (context) | context / test rows |", "|---|---|---|---|---|---|"]
    L += [f"| {r.tag} | {r.dataset} | {r.type} ({r.n_classes} classes) | {r.balance} | {r.class_shares_context} | "
          f"{r.n_context} / {r.n_test} |" for r in ds.itertuples()]
    L += ["", "Balance: largest over smallest class share up to 2 is roughly balanced, up to 4 moderately imbalanced.",
          "", "## Entries", "", "| entry | method |", "|---|---|"]
    L += [f"| `{e}` | {RB.NAME.get(e, e)} |" for e in entries]
    L += ["", "## Loading", "", "```python", "import gzip, json, tarfile", "import numpy as np, pandas as pd", "",
          "r = pd.read_csv('results.csv')                         # one row per instance x entry",
          "cell = r[(r.c == 1) & (r.R == 10) & (r.dataset == 'heloc')]",
          "elo = pd.read_csv('elo.csv').query(\"variant == 'main'\")",
          "runs = [json.loads(l) for l in gzip.open('raw/runs.jsonl.gz', 'rt')]",
          "tarfile.open('raw/draws_and_masks.tar.gz').extractall('raw/')  # then np.load('raw/draws/<instance>.npz')",
          "```", "",
          "To make a new snapshot after more runs: `python scripts/archive_bench_tinyv.py --tag <date>`."]
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=B.ROOT)
    ap.add_argument("--tables", type=Path, default=Path("results/tables/bench_tinyv"), help="where findings.md is")
    ap.add_argument("--out", type=Path, default=Path("results_archive/bench_tinyv"))
    ap.add_argument("--tag", default=dt.date.today().isoformat())
    a = ap.parse_args(argv)
    out = a.out / a.tag
    (out / "raw").mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):  # the two reports print their tables
        RB.main(["--root", str(a.root), "--out", str(out)])
        EB.main(["--root", str(a.root), "--out", str(out)])
    res = pd.read_csv(out / "results.csv")
    by_v_r_table(res).to_csv(out / "by_v_r.csv", index=False)
    per_dataset_table(res).to_csv(out / "per_dataset.csv", index=False)
    ds = datasets_table()
    ds.to_csv(out / "datasets.csv", index=False)
    inst = instances_table(a.root)
    inst.to_csv(out / "instances.csv", index=False)
    if (a.tables / "findings.md").exists():
        shutil.copy2(a.tables / "findings.md", out / "findings.md")
    run_files = sorted((a.root / "runs").glob("*/*.json"))
    hosts = Counter()
    with gzip.open(out / "raw" / "runs.jsonl.gz", "wt") as fh:
        for f in run_files:
            rec = json.loads(f.read_text())
            hosts[rec.get("host", "imported" if "provenance" in rec else "unknown")] += 1
            fh.write(json.dumps(rec) + "\n")
    with tarfile.open(out / "raw" / "draws_and_masks.tar.gz", "w:gz") as tar:
        for f in sorted((a.root / "draws").glob("*.npz")) + sorted((a.root / "runs").glob("*/*.npz")):
            tar.add(f, arcname=str(f.relative_to(a.root)))
    (out / "code").mkdir(exist_ok=True)
    for name in CODE:
        shutil.copy2(Path(__file__).resolve().parent / name, out / "code" / name)
    dirty = [line[3:] for line in git("status", "--porcelain", "--", "scripts", "docs").splitlines()
             if "__pycache__" not in line]
    meta = {"created": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "host": socket.gethostname(),
            "root": str(a.root), "git_commit": git("rev-parse", "HEAD"), "git_dirty": dirty,
            "instances": int(res.instance.nunique()), "entries": int(res.entry.nunique()), "run_files": len(run_files),
            "runs_per_host": dict(hosts.most_common())}
    entries = [e for e, _ in RB.ENTRIES if e in set(res.entry)]
    (out / "README.md").write_text(readme(a.tag, meta, ds, entries))
    files = {}
    for f in sorted(p for p in out.rglob("*") if p.is_file() and p.name != "manifest.json"):
        entry = {"bytes": f.stat().st_size, "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
        if f.suffix == ".csv":
            entry["rows"] = len(pd.read_csv(f))
        files[str(f.relative_to(out))] = entry
    (out / "manifest.json").write_text(json.dumps(meta | {"files": files}, indent=1) + "\n")
    total = sum(v["bytes"] for v in files.values())
    print(f"-> {out}: {len(files)} files, {total / 1e6:.1f} MB")
    for name, v in files.items():
        print(f"   {name:34s} {v['bytes'] / 1e3:9.1f} kB" + (f"  {v['rows']} rows" if "rows" in v else ""))


if __name__ == "__main__":
    main()
