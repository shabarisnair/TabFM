#!/usr/bin/env python
"""One table per dataset for the XGBoost transfer of the TabPFNv2 context poison.

Reads ``results/xgb_transfer/<ds>/<attack>_random_r<NNN>/summary.json``. Each cell is
the poisoned metric as mean +- std over the five per-run-best poisoned contexts, with
the relative change from that protocol's own clean baseline.

The two protocols have DIFFERENT clean baselines and must not be compared with each
other's: A trains on 80% of the context and validates on the held-out 20% (poisoned
when the context is), B trains on 100% and validates on the separate clean file.

``--transfer`` prints the cross-dataset summary instead: the XGBoost delta next to the
TabPFNv2 delta it is trying to reproduce, and their ratio.

    python scripts/make_transfer_tables.py
    python scripts/make_transfer_tables.py --transfer --metric ce
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tabfm_experiments.transfer import relative  # noqa: E402

DATASETS = ["coil2000_insurance_policies", "lcld_v2", "url_unique", "wids"]
ATTACKS = ["x-capgd", "label-flip"]
PROTOCOLS = ["A", "B"]
METRICS = ["ce", "roc_auc", "accuracy", "balanced_accuracy", "mcc"]
SHORT = {"ce": "CE", "roc_auc": "ROC-AUC", "accuracy": "accuracy",
         "balanced_accuracy": "bal. acc", "mcc": "MCC", "f1": "F1"}
WEAK = {"mcc": 0.15, "f1": 0.15}       # |clean| below this -> relative % is noise


def discover_rs(root: Path, subdirs=None) -> list[str]:
    """R values that actually have a finished run under ``root``, numerically sorted."""
    found = set()
    for ds in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        if subdirs and ds.name not in subdirs:
            continue
        for p in ds.glob("*_random_r*"):
            if (p / "summary.json").exists():
                found.add(p.name.rsplit("_r", 1)[1])
    return sorted(found, key=int)


def load(root: Path, ds: str, attack: str, r: str) -> dict | None:
    f = root / ds / f"{attack}_random_r{r}" / "summary.json"
    return json.loads(f.read_text()) if f.exists() else None


def fmt_rel(v: float | None, weak: bool) -> str:
    return "-" if v is None else f"{'~' if weak else ''}{v:+.1f}%"


def cell(p: dict, d: dict, clean_mean: float | None, metric: str) -> tuple[str, str, str]:
    weak = clean_mean is not None and abs(clean_mean) < WEAK.get(metric, 0.0)
    return ("{:.4f} +- {:.4f}".format(p["mean"], p["std"]),
            "{:+.4f} +- {:.4f}".format(d["mean"], d["std"]),
            fmt_rel(relative(d["mean"], clean_mean), weak))


def text_table(ds: str, cells: dict, metrics: list[str], rs: list[str]) -> str:
    any_s = next(iter(cells.values()))
    out = [f"{ds}   XGBoost transfer   context {any_s['n_context']}x{any_s['n_features']}, "
           f"test {any_s['n_test']}, {len(any_s['contexts'])} poisoned contexts, "
           f"HPO {any_s['xgb']['hpo_trials']} optuna trials on {any_s['xgb']['hpo_objective']}", ""]
    head = f"  {'prot':<5s} {'attack':<11s} {'R%':>4s}"
    for m in metrics:
        head += f"  {SHORT[m] + ' poisoned':>21s} {'delta':>18s} {'rel':>8s}"
    out += [head, "  " + "-" * (len(head) - 2)]
    for prot in PROTOCOLS:
        base = any_s["protocols"][prot]["clean_agg"]
        row = f"  {prot:<5s} {'(clean)':<11s} {'-':>4s}"
        for m in metrics:
            row += f"  {'{:.4f} +- {:.4f}'.format(base[m]['mean'], base[m]['std']):>21s} {'-':>18s} {'-':>8s}"
        out += [row]
        for attack in ATTACKS:
            for r in rs:
                s = cells.get((attack, r))
                if s is None:
                    continue
                pr = s["protocols"][prot]
                row = f"  {prot:<5s} {attack:<11s} {int(r):>4d}"
                for m in metrics:
                    a, b, c = cell(pr["poisoned_agg"][m], pr["delta_agg"][m],
                                   pr["clean_agg"][m]["mean"], m)
                    row += f"  {a:>21s} {b:>18s} {c:>8s}"
                out.append(row)
        out.append("  " + "-" * (len(head) - 2))
    return "\n".join(out)


def md_table(ds: str, cells: dict, metrics: list[str], rs: list[str]) -> str:
    any_s = next(iter(cells.values()))
    out = [f"### {ds} -- XGBoost transfer", "",
           f"Context {any_s['n_context']}x{any_s['n_features']}, test {any_s['n_test']}, "
           f"{len(any_s['contexts'])} poisoned contexts, HPO {any_s['xgb']['hpo_trials']} optuna "
           f"trials on {any_s['xgb']['hpo_objective']}. "
           "Cells are `poisoned mean ± std` / `Δ (relative %)`. "
           "Each protocol has its own clean baseline.", "",
           "| prot | attack | R% | " + " | ".join(SHORT[m] for m in metrics) + " |",
           "|---|---|---|" + "---|" * len(metrics)]
    for prot in PROTOCOLS:
        base = any_s["protocols"][prot]["clean_agg"]
        out.append(f"| **{prot}** | **(clean)** | - | "
                   + " | ".join(f"**{base[m]['mean']:.4f} ± {base[m]['std']:.4f}**" for m in metrics) + " |")
        for attack in ATTACKS:
            for r in rs:
                s = cells.get((attack, r))
                if s is None:
                    continue
                pr = s["protocols"][prot]
                row = [prot, attack, str(int(r))]
                for m in metrics:
                    a, b, c = cell(pr["poisoned_agg"][m], pr["delta_agg"][m],
                                   pr["clean_agg"][m]["mean"], m)
                    row.append(f"{a.replace(' +- ', ' ± ')}<br>{b.split(' +- ')[0]} ({c})")
                out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def transfer_table(all_cells: dict, metric: str, rs: list[str]) -> str:
    """XGBoost delta vs the TabPFNv2 delta it is reproducing, per protocol."""
    out = [f"transfer of the {SHORT[metric]} degradation: XGBoost delta / TabPFNv2 delta", "",
           f"  {'dataset':<28s} {'attack':<11s} {'R%':>4s}  {'TabPFN d':>17s}"
           f"  {'XGB d (A)':>17s} {'A%':>7s}  {'XGB d (B)':>17s} {'B%':>7s}"]
    out.append("  " + "-" * (len(out[-1]) - 2))
    for ds in DATASETS:
        for attack in ATTACKS:
            for r in rs:
                s = all_cells.get((ds, attack, r))
                if s is None:
                    continue
                t = s["tabpfn"]["delta_agg"][metric]
                row = (f"  {ds:<28s} {attack:<11s} {int(r):>4d}  "
                       f"{'{:+.4f} +- {:.4f}'.format(t['mean'], t['std']):>17s}")
                for prot in PROTOCOLS:
                    d = s["protocols"][prot]["delta_agg"][metric]
                    ratio = (100.0 * d["mean"] / t["mean"]) if abs(t["mean"]) > 1e-12 else None
                    noisy = abs(t["mean"]) <= t["std"]
                    row += (f"  {'{:+.4f} +- {:.4f}'.format(d['mean'], d['std']):>17s}"
                            f" {fmt_rel(ratio, noisy):>7s}")
                out.append(row)
    out += ["", "  '~' marks a ratio whose TabPFNv2 denominator is inside its own across-run spread."]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("results/xgb_transfer"))
    ap.add_argument("--out", type=Path, default=Path("results/tables/transfer"))
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--metrics", nargs="+", default=METRICS)
    ap.add_argument("--transfer", action="store_true", help="print the transfer-ratio summary instead")
    ap.add_argument("--metric", default="ce", help="metric for --transfer")
    ap.add_argument("--format", choices=["text", "markdown", "both"], default="both")
    ap.add_argument("--row-percents", nargs="+", default=None,
                    help="R values to include (default: every one found under --root)")
    a = ap.parse_args(argv)

    rs = a.row_percents or discover_rs(a.root, a.datasets)
    if not rs:
        raise SystemExit(f"no finished runs under {a.root}")
    print(f"R values: {', '.join(str(int(r)) for r in rs)}%")

    a.out.mkdir(parents=True, exist_ok=True)
    all_cells, rows = {}, []
    for ds in a.datasets:
        cells = {}
        for attack in ATTACKS:
            for r in rs:
                s = load(a.root, ds, attack, r)
                if s is not None:
                    cells[(attack, r)] = s
                    all_cells[(ds, attack, r)] = s
        if not cells:
            print(f"[skip] {ds}: nothing under {a.root / ds}")
            continue
        if not a.transfer:
            if a.format in ("text", "both"):
                print("\n" + text_table(ds, cells, a.metrics, rs) + "\n")
            if a.format in ("markdown", "both"):
                (a.out / f"{ds}.md").write_text(md_table(ds, cells, a.metrics, rs) + "\n")
        for (attack, r), s in cells.items():
            for prot in PROTOCOLS:
                pr = s["protocols"][prot]
                for m in a.metrics:
                    cm = pr["clean_agg"][m]["mean"]
                    t = s["tabpfn"]["delta_agg"][m]
                    dm = pr["delta_agg"][m]["mean"]
                    rows.append({
                        "dataset": ds, "attack": attack, "row_percent": int(r), "protocol": prot,
                        "metric": m, "clean_mean": cm, "clean_std": pr["clean_agg"][m]["std"],
                        "poisoned_mean": pr["poisoned_agg"][m]["mean"],
                        "poisoned_std": pr["poisoned_agg"][m]["std"],
                        "delta_mean": dm, "delta_std": pr["delta_agg"][m]["std"],
                        "relative_pct": relative(dm, cm),
                        "tabpfn_delta_mean": t["mean"], "tabpfn_delta_std": t["std"],
                        "transfer_pct": (100.0 * dm / t["mean"]) if abs(t["mean"]) > 1e-12 else None,
                        "n_contexts": len(s["contexts"])})
    if a.transfer:
        print("\n" + transfer_table(all_cells, a.metric, rs) + "\n")
    if rows:
        with open(a.out / "transfer_results.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {a.out / 'transfer_results.csv'} ({len(rows)} rows)")
    return rows


if __name__ == "__main__":
    main()
