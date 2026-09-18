#!/usr/bin/env python
"""One result table per dataset for the main context-poisoning grid.

Reads ``results/main_experiments/<ds>/<attack>_random_r<NNN>/summary.json`` and reports,
for each attack x R cell, the poisoned metric as mean +- std over the per-run winners
together with the absolute and relative change from the clean baseline.

The clean baseline is a single deterministic number per dataset (one fit on the clean
context), so it is printed once in the header rather than repeated in every row.

Relative change is ``100 * delta / |clean|``. For MCC and F1 on a weak clean model the
denominator is tiny and the percentage is noise, so those cells are marked ``~`` and
should be read from the absolute delta instead.

    python scripts/make_main_tables.py                 # all datasets, all metrics
    python scripts/make_main_tables.py --metrics ce accuracy
    python scripts/make_main_tables.py --format markdown
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tabfm_experiments.transfer import agg, relative, tabpfn_run_metrics  # noqa: E402

DATASETS = ["coil2000_insurance_policies", "lcld_v2", "url_unique", "wids"]
ATTACKS = [("x-capgd", "X_train CAPGD"), ("label-flip", "Y_train label-flip (GA)")]
SHORT = {"ce": "CE", "accuracy": "accuracy", "balanced_accuracy": "bal. acc",
         "mcc": "MCC", "roc_auc": "ROC-AUC", "f1": "F1"}
WEAK = {"mcc": 0.15, "f1": 0.15}       # |clean| below this -> relative % is noise


def discover_rs(root: Path, subdirs=None) -> list[str]:
    """R values that actually have a finished run under ``root``, numerically sorted.

    Discovered rather than hard-coded so that adding an R to the grid needs no edit here.
    """
    found = set()
    for ds in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        if subdirs and ds.name not in subdirs:
            continue
        for p in ds.glob("*_random_r*"):
            if (p / "summary.json").exists():
                found.add(p.name.rsplit("_r", 1)[1])
    return sorted(found, key=int)


def load_cell(root: Path, ds: str, attack: str, r: str) -> dict | None:
    f = root / ds / f"{attack}_random_r{r}" / "summary.json"
    if not f.exists():
        return None
    s = json.loads(f.read_text())
    runs = tabpfn_run_metrics(s)
    # The clean baseline is identical across runs; take it from the first.
    return {"summary": s, "clean": runs["clean"][0],
            "poisoned": agg(runs["poisoned"]), "delta": agg(runs["delta"]),
            "n_runs": len(runs["clean"])}


def rel_cell(delta_mean: float | None, clean: float | None, metric: str) -> tuple[float | None, bool]:
    return relative(delta_mean, clean), clean is not None and abs(clean) < WEAK.get(metric, 0.0)


def fmt_rel(v: float | None, weak: bool) -> str:
    return "-" if v is None else f"{'~' if weak else ''}{v:+.1f}%"


def text_table(ds: str, cells: dict, metrics: list[str], rs: list[str]) -> str:
    any_cell = next(iter(cells.values()))
    s = any_cell["summary"]
    clean = any_cell["clean"]
    out = [
        f"{ds}   context {s['n_context']}x{s['n_features']}, test {s['n_test']}, "
        f"{any_cell['n_runs']} runs (best row-subsample per run by dCE)",
        "  clean TabPFNv2:  " + "   ".join(f"{SHORT[m]} {clean[m]:.4f}" for m in metrics),
        "",
    ]
    head = f"  {'attack':<11s} {'R%':>4s}"
    sub = f"  {'':<11s} {'':>4s}"
    for m in metrics:
        head += f"  {SHORT[m] + ' poisoned':>21s} {'delta':>18s} {'rel':>8s}"
        sub += f"  {'mean +- std':>21s} {'mean +- std':>18s} {'':>8s}"
    out += [head, sub, "  " + "-" * (len(head) - 2)]
    # The clean baseline is one deterministic fit, so it has no spread and no delta.
    clean_row = f"  {'(clean)':<11s} {'-':>4s}"
    for m in metrics:
        clean_row += f"  {f'{clean[m]:.4f}':>21s} {'-':>18s} {'-':>8s}"
    out += [clean_row, "  " + "-" * (len(head) - 2)]
    for attack, _ in ATTACKS:
        for r in rs:
            c = cells.get((attack, r))
            if c is None:
                continue
            row = f"  {attack:<11s} {int(r):>4d}"
            for m in metrics:
                p, d = c["poisoned"][m], c["delta"][m]
                v, weak = rel_cell(d["mean"], clean[m], m)
                pois = "{:.4f} +- {:.4f}".format(p["mean"], p["std"])
                dlt = "{:+.4f} +- {:.4f}".format(d["mean"], d["std"])
                row += f"  {pois:>21s} {dlt:>18s} {fmt_rel(v, weak):>8s}"
            out.append(row)
    return "\n".join(out)


def md_table(ds: str, cells: dict, metrics: list[str], rs: list[str]) -> str:
    any_cell = next(iter(cells.values()))
    s, clean = any_cell["summary"], any_cell["clean"]
    out = [
        f"### {ds}",
        "",
        f"Context {s['n_context']}x{s['n_features']}, test {s['n_test']}, "
        f"{any_cell['n_runs']} runs (best row-subsample per run by dCE). "
        "Cells are `poisoned mean ± std` / `Δ mean ± std` / `relative %`.",
        "",
        "| attack | R% | " + " | ".join(SHORT[m] for m in metrics) + " |",
        "|---|---|" + "---|" * len(metrics),
        "| **(clean baseline)** | - | " + " | ".join(f"**{clean[m]:.4f}**" for m in metrics) + " |",
    ]
    for attack, _ in ATTACKS:
        for r in rs:
            c = cells.get((attack, r))
            if c is None:
                continue
            row = [attack, str(int(r))]
            for m in metrics:
                p, d = c["poisoned"][m], c["delta"][m]
                v, weak = rel_cell(d["mean"], clean[m], m)
                row.append(f"{p['mean']:.4f} ± {p['std']:.4f}<br>{d['mean']:+.4f} ({fmt_rel(v, weak)})")
            out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("results/main_experiments"))
    ap.add_argument("--out", type=Path, default=Path("results/tables/main"))
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    # CE and ROC-AUC first: both are threshold-free. The three that follow read the
    # argmax, which is uninformative where the clean model never crosses 0.5 (coil2000).
    ap.add_argument("--metrics", nargs="+",
                    default=["ce", "roc_auc", "accuracy", "balanced_accuracy", "mcc"])
    ap.add_argument("--format", choices=["text", "markdown", "both"], default="both")
    ap.add_argument("--row-percents", nargs="+", default=None,
                    help="R values to include (default: every one found under --root)")
    a = ap.parse_args(argv)

    rs = a.row_percents or discover_rs(a.root, a.datasets)
    if not rs:
        raise SystemExit(f"no finished runs under {a.root}")
    print(f"R values: {', '.join(str(int(r)) for r in rs)}%")

    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in a.datasets:
        cells = {}
        for attack, _ in ATTACKS:
            for r in rs:
                c = load_cell(a.root, ds, attack, r)
                if c is not None:
                    cells[(attack, r)] = c
        if not cells:
            print(f"[skip] {ds}: no summary.json found under {a.root / ds}")
            continue
        clean = next(iter(cells.values()))["clean"]
        if a.format in ("text", "both"):
            print("\n" + text_table(ds, cells, a.metrics, rs) + "\n")
        if a.format in ("markdown", "both"):
            (a.out / f"{ds}.md").write_text(md_table(ds, cells, a.metrics, rs) + "\n")
        for (attack, r), c in cells.items():
            for m in a.metrics:
                v, weak = rel_cell(c["delta"][m]["mean"], clean[m], m)
                rows.append({"dataset": ds, "attack": attack, "row_percent": int(r), "metric": m,
                             "clean": clean[m], "poisoned_mean": c["poisoned"][m]["mean"],
                             "poisoned_std": c["poisoned"][m]["std"],
                             "delta_mean": c["delta"][m]["mean"], "delta_std": c["delta"][m]["std"],
                             "relative_pct": v, "relative_unstable": weak, "n_runs": c["n_runs"]})
    if rows:
        with open(a.out / "main_results.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {a.out / 'main_results.csv'} and {len(a.datasets)} markdown tables to {a.out}")
    return rows


if __name__ == "__main__":
    main()
