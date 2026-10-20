#!/usr/bin/env python
"""Report of the tiny-V poisoning benchmark (scripts/bench_tinyv.py; <root>/runs).

Writes <out>/report.md and <out>/results.csv (one line per instance x method row):
  - summary over the instances: mean accuracy, ROC-AUC (binary datasets: class 1; also all datasets with
    multi-class one-vs-rest), log loss and precision of the removed rows; the mean rank by each metric
    (over the instances every listed entry finished); wins and losses against the V-filter;
  - per dataset, one table per metric: rows = entries, columns = (c, R); the best non-oracle entry bold.
Accuracy is degenerate on coil2000 and is_this_a_good_customer (TabPFN predicts the majority class), so
read their ROC-AUC. Partial results are fine: a missing entry shows as '.', and the summary says how many
instances each mean covers.

    python scripts/report_bench_tinyv.py [--root results/bench_tinyv] [--out results/tables/bench_tinyv]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

BINARY = {"heloc", "ecommerce_shipping", "is_this_a_good_customer", "coil2000_insurance_policies"}
SHORT = {"gas-drift": "gas", "heloc": "heloc", "ecommerce_shipping": "ecommerce", "students_dropout": "students",
         "is_this_a_good_customer": "good_customer", "coil2000_insurance_policies": "coil2000"}
BASELINES = ("knn_sanitize", "label_spread", "knn_shapley", "data_oob", "cleanlab", "simifeat", "glc", "mwnet", "emlc", "ieg")
ENTRIES = [("v_only", "V only"), ("poisoned_plus_v", "poisoned + V"), ("oracle_plus_v", "oracle + V"),
           ("vfilter", "V-filter"), ("greedy_pct10", "greedy, 10% per step"), ("greedy_pct1", "greedy, 1% per step"),
           ("greedy_m1", "greedy, 1 row per step"), ("attn_topk", "attention weights, top-k (M1)"),
           ("attn_soft", "attention weights, soft (A3)"), ("attn_soft_softw", "A3, soft weights + V"),
           ("nan_topk", "NaN gate, top-k vote")]
for _b in BASELINES:
    ENTRIES += [(_b, f"{_b} (default)"), (f"{_b}_relabel", f"{_b} (default, relabel)"), (f"{_b}_vsel", f"{_b} (V-selected)")]
NAME = dict(ENTRIES)
METRICS = [("acc", "accuracy (%)", 100, True), ("auc", "ROC-AUC (%)", 100, True), ("ce", "log loss", 1, False),
           ("prec", "precision of the removed rows", 1, True)]
# by_v_r.md: one table per (c, R), the datasets in this order, each with these four columns
CLASS_SHARES = {  # class shares of the clean train contexts (the test sets are stratified the same way)
    "gas-drift": (0.184, 0.210, 0.118, 0.139, 0.216, 0.132), "students_dropout": (0.321, 0.180, 0.499),
    "heloc": (0.478, 0.522), "ecommerce_shipping": (0.407, 0.593), "is_this_a_good_customer": (0.886, 0.114),
    "coil2000_insurance_policies": (0.930, 0.070)}
TAG = {"gas-drift": "gas", "students_dropout": "stud", "heloc": "heloc", "ecommerce_shipping": "ecom",
       "is_this_a_good_customer": "good", "coil2000_insurance_policies": "coil"}
VR_METRICS = [("prec", "Prec", "{:.2f}", 1, True), ("auc", "AUC", "{:.3f}", 1, True), ("acc", "Acc %", "{:.1f}", 100, True),
              ("ce", "LL", "{:.3f}", 1, False)]


def load(root: Path) -> list[dict]:
    rows = []
    for f in sorted((root / "runs").glob("*/*.json")):
        r = json.loads(f.read_text())
        ds, rest = r["instance"].rsplit("_R", 1)
        R, c, s = rest.split("_")
        for e in r.get("rows", []):
            t = e.get("test", {})
            rows.append({"instance": r["instance"], "dataset": ds, "R": float(R), "c": int(c[1:]), "seed": int(s[1:]),
                         "entry": e["name"], "acc": t.get("acc"), "bacc": t.get("bacc"), "auc": t.get("auc"), "ce": t.get("ce"),
                         "prec": e.get("prec"), "n_removed": e.get("n_removed"), "cfg": e.get("cfg", ""), "mode": e.get("mode", ""),
                         "imported": "provenance" in r})
    return rows


def fmt(v, scale, nd=1):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "."
    return f"{scale * v:.{nd}f}" if scale != 1 else f"{v:.3f}"


def balance(shares) -> str:
    """Largest over smallest class share: up to 2 roughly balanced, up to 4 moderately imbalanced."""
    ratio = max(shares) / min(shares)
    return "roughly balanced" if ratio <= 2 else "moderately imbalanced" if ratio <= 4 else "imbalanced"


def kind(ds: str) -> str:
    p = CLASS_SHARES[ds]
    return f"{'binary' if len(p) == 2 else f'multi-class ({len(p)} classes)'}, {balance(p)}"


METRIC_KEY = ("Prec: precision of the removed rows (relabel modes: of the relabelled rows); random removal scores R/100 "
              "on average. AUC: ROC-AUC (binary: class 1; multi-class: one-vs-rest macro). Acc %: accuracy, degenerate "
              "on good and coil (TabPFN predicts the majority class). LL: log loss, lower is better.")


def dataset_legend(dss: list[str]) -> list[str]:
    return ["| tag | dataset | type | class shares of the clean train context |", "|---|---|---|---|"] + [
        f"| {TAG[d]} | {d} | {kind(d)} | {' / '.join(f'{100 * s:.0f}%' for s in CLASS_SHARES[d])} |" for d in dss]


def small_row(line: str) -> str:
    """A markdown table row with every cell in small type (<sub>; bold as <b>), which GitHub and VS Code render."""
    if set(line.replace(" ", "")) <= set("|-"):  # the separator row
        return line
    cells = [re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", x.strip()) for x in line.strip().strip("|").split("|")]
    return "| " + " | ".join(f"<sub>{x}</sub>" for x in cells) + " |"


def write_pooled(rows: list[dict], path: Path) -> None:
    """One table over all (c, R) settings: per entry and dataset, each metric's mean over the settings and the
    half-width of its 95% t-interval, the settings taken as the sample (so it spans how much the entry varies
    with c and R, not only draw noise); the numbers also go to <path>.csv."""
    from scipy import stats
    val = {(r["dataset"], r["c"], r["R"], r["entry"]): r for r in rows}
    entries = [e for e, _ in ENTRIES if any(k[3] == e for k in val)]
    dss = [d for d in TAG if any(k[0] == d for k in val)]
    settings = sorted({(k[1], k[2]) for k in val})
    stat, recs = {}, []
    for e in entries:
        for d in dss:
            for m, *_ in VR_METRICS:
                v = [val[d, c, R, e][m] for c, R in settings if (d, c, R, e) in val]
                v = np.array([x for x in v if x is not None and np.isfinite(x)], dtype=float)
                n = len(v)
                mean = float(v.mean()) if n else np.nan
                hw = float(stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
                stat[e, d, m] = (n, mean, hw)
                recs.append({"entry": e, "dataset": d, "metric": m, "n_settings": n, "mean": mean, "ci95_half_width": hw})
    best = {}
    for d in dss:
        for m, _, _, _, higher in VR_METRICS:
            ms = [stat[e, d, m][1] for e in entries if not e.startswith("oracle") and stat[e, d, m][0]]
            best[d, m] = (max(ms) if higher else min(ms)) if ms else None
    L = ["# Tiny-V benchmark: every V size and poisoning rate pooled", "",
         f"Each cell: the mean over the {len(settings)} settings (c = {', '.join(str(c) for c in sorted({c for c, _ in settings}))} "
         f"clean rows per class × R = {', '.join(f'{R:g}' for R in sorted({R for _, R in settings}))}%; one draw each) ± the "
         "half-width of its 95% t-interval, the settings taken as the sample: it spans how much the entry varies with "
         "V and R as well as draw noise.", ""]
    L += dataset_legend(dss) + ["", METRIC_KEY + " Bold: the best mean of the column, oracle + V excluded. '–': no rows "
                                "removed. A bracket gives the number of settings when it is short.", "",
                                "| entry | " + " | ".join(f"{TAG[d]} {name}" for d in dss for _, name, *_ in VR_METRICS) + " |",
                                "|---|" + "---|" * (len(dss) * len(VR_METRICS))]
    for e in entries:
        out = []
        for d in dss:
            for m, _, f, scale, _ in VR_METRICS:
                n, mean, hw = stat[e, d, m]
                if not n:
                    out.append("–")
                    continue
                s = f.format(scale * mean) + (f" ± {f.format(scale * hw)}" if np.isfinite(hw) else "")
                if not e.startswith("oracle") and mean == best[d, m]:
                    s = f"**{s}**"
                out.append(s + (f" ({n})" if n < len(settings) else ""))
        L.append(f"| {NAME.get(e, e)} | " + " | ".join(out) + " |")
    path.write_text("\n".join(L) + "\n")
    n_head = len(L) - len(entries) - 2  # the main table: its header, separator and one row per entry
    path.with_name(path.stem + "_small.md").write_text("\n".join(L[:n_head] + [small_row(x) for x in L[n_head:]]) + "\n")
    with open(path.with_suffix(".csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)


def write_by_v_r(rows: list[dict], path: Path) -> None:
    """One table per (c, R): a row per entry, and per dataset its precision, ROC-AUC, accuracy and log loss."""
    val = {(r["dataset"], r["c"], r["R"], r["entry"]): r for r in rows}
    entries = [e for e, _ in ENTRIES if any(k[3] == e for k in val)]
    dss = [d for d in TAG if any(k[0] == d for k in val)]
    L = ["# Tiny-V benchmark: one table per V size and poisoning rate", "",
         "V holds c clean rows per class (2c rows on the binary datasets, 3c on students, 6c on gas-drift); R% of the "
         "remaining train context is poisoned with hub-flip-local. One draw per table cell.", ""]
    L += dataset_legend(dss) + ["", METRIC_KEY + " Bold: the best entry of the column, oracle + V excluded. '–': not "
                                "run, or no rows removed.", ""]
    for c in sorted({k[1] for k in val}):
        for R in sorted({k[2] for k in val}):
            L += [f"## V = {c} row{'s' if c > 1 else ''} per class, R = {R:g}%", "",
                  "| entry | " + " | ".join(f"{TAG[d]} {name}" for d in dss for _, name, *_ in VR_METRICS) + " |",
                  "|---|" + "---|" * (len(dss) * len(VR_METRICS))]
            best = {}
            for d in dss:
                for m, _, _, _, higher in VR_METRICS:
                    vs = [val[d, c, R, e][m] for e in entries if not e.startswith("oracle") and (d, c, R, e) in val]
                    vs = [v for v in vs if v is not None and np.isfinite(v)]
                    best[d, m] = (max(vs) if higher else min(vs)) if vs else None
            for e in entries:
                out = []
                for d in dss:
                    for m, _, f, scale, _ in VR_METRICS:
                        v = val[d, c, R, e][m] if (d, c, R, e) in val else None
                        if v is None or not np.isfinite(v):
                            out.append("–")
                            continue
                        s = f.format(scale * v)
                        out.append(f"**{s}**" if not e.startswith("oracle") and v == best[d, m] else s)
                L.append(f"| {NAME.get(e, e)} | " + " | ".join(out) + " |")
            L.append("")
    path.write_text("\n".join(L) + "\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(os.environ.get("BENCH_TINYV_ROOT", "results/bench_tinyv")))
    ap.add_argument("--out", type=Path, default=Path("results/tables/bench_tinyv"))
    a = ap.parse_args(argv)
    rows = load(a.root)
    a.out.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise SystemExit(f"no results under {a.root / 'runs'}")
    with open(a.out / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    val = {(r["instance"], r["entry"]): r for r in rows}
    insts = sorted({r["instance"] for r in rows})
    meta = {r["instance"]: (r["dataset"], r["R"], r["c"]) for r in rows}
    entries = [e for e, _ in ENTRIES if any((i, e) in val for i in insts)]
    entries += sorted({r["entry"] for r in rows} - set(entries))

    def get(i, e, m):
        r = val.get((i, e))
        v = None if r is None else r[m]
        return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else float(v)

    L = [f"# Tiny-V poisoning benchmark ({len(insts)} instances with results)", "",
         "V + the cleaned poisoned rows as TabPFN's context, scored on each dataset's held-out test set. "
         "ROC-AUC (binary) is of class 1; '(all)' adds the multi-class datasets' one-vs-rest macro AUC. "
         "Precision: the share of the removed rows (relabelling: of the relabelled rows) that are poisoned. "
         "Accuracy is degenerate on coil2000 and good_customer (the majority class everywhere): read their AUC.", ""]
    # ---- summary
    full = [i for i in insts if all((i, e) in val for e in entries if not e.startswith("oracle"))]
    L += ["## Summary", "",
          f"Means over the instances each entry has (n); mean ranks over the {len(full)} instances where every entry "
          "finished (1 = best; oracle + V excluded from the ranks); wins-losses against the V-filter on accuracy "
          "(multi-class datasets) or ROC-AUC (binary), over the instances both have.", "",
          "| entry | n | accuracy | ROC-AUC (binary) | ROC-AUC (all) | log loss | precision | rank acc | rank AUC | rank log loss | vs V-filter |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    ranked = [e for e in entries if not e.startswith("oracle")]
    ranks = {m: defaultdict(list) for m in ("acc", "auc", "ce")}
    for i in full:
        for m, higher in (("acc", True), ("auc", True), ("ce", False)):
            vals = {e: get(i, e, m) for e in ranked}
            vals = {e: v for e, v in vals.items() if v is not None}
            order = sorted(vals, key=lambda e: -vals[e] if higher else vals[e])
            for pos, e in enumerate(order):
                ranks[m][e].append(pos + 1)
    for e in entries:
        have = [i for i in insts if (i, e) in val]
        bin_ = [i for i in have if meta[i][0] in BINARY]
        prim = lambda i, ent: get(i, ent, "auc" if meta[i][0] in BINARY else "acc")  # noqa: E731
        both = [i for i in have if (i, "vfilter") in val and prim(i, e) is not None and prim(i, "vfilter") is not None]
        wins = sum(prim(i, e) > prim(i, "vfilter") for i in both)
        losses = sum(prim(i, e) < prim(i, "vfilter") for i in both)
        mean = lambda m, ii, sc: fmt(np.mean([get(i, e, m) for i in ii if get(i, e, m) is not None]) if any(  # noqa: E731
            get(i, e, m) is not None for i in ii) else None, sc)
        rk = lambda m: f"{np.mean(ranks[m][e]):.1f}" if ranks[m].get(e) else "."  # noqa: E731
        L.append(f"| {NAME.get(e, e)} | {len(have)} | {mean('acc', have, 100)} | {mean('auc', bin_, 100)} | {mean('auc', have, 100)} | "
                 f"{mean('ce', have, 1)} | {mean('prec', have, 1)} | {rk('acc')} | {rk('auc')} | {rk('ce')} | "
                 f"{f'{wins}-{losses}' if e != 'vfilter' and both else '.'} |")
    # ---- per dataset and metric
    for ds in [d for d in SHORT if any(meta[i][0] == d for i in insts)]:
        cells = sorted({(meta[i][2], meta[i][1]) for i in insts if meta[i][0] == ds})
        inst_of = {(meta[i][2], meta[i][1]): i for i in insts if meta[i][0] == ds}
        L += ["", f"## {ds}", ""]
        for m, title, scale, higher in METRICS:
            if m == "acc" and ds in ("coil2000_insurance_policies", "is_this_a_good_customer"):
                title += " -- degenerate here (majority class)"
            L += [f"### {title}", "", "| entry | " + " | ".join(f"c{c} R{R:g}" for c, R in cells) + " |",
                  "|---|" + "---|" * len(cells)]
            best = {}
            for cell in cells:
                vs = [get(inst_of[cell], e, m) for e in entries if not e.startswith("oracle")]
                vs = [v for v in vs if v is not None]
                best[cell] = (max(vs) if higher else min(vs)) if vs else None
            for e in entries:
                if not any((inst_of[cell], e) in val for cell in cells):
                    continue
                out = []
                for cell in cells:
                    v = get(inst_of[cell], e, m)
                    s = fmt(v, scale)
                    if v is not None and best[cell] is not None and not e.startswith("oracle") and abs(v - best[cell]) < 1e-12:
                        s = f"**{s}**"
                    out.append(s)
                L.append(f"| {NAME.get(e, e)} | " + " | ".join(out) + " |")
            L.append("")
    imported = sorted({(r["instance"], r["entry"]) for r in rows if r["imported"]})
    L += ["", f"Imported from the schedule study (identical draws and code): {len(imported)} entries."]
    (a.out / "report.md").write_text("\n".join(L) + "\n")
    write_by_v_r(rows, a.out / "by_v_r.md")
    write_pooled(rows, a.out / "pooled.md")
    print("\n".join(L[:len(entries) + 12]))
    print(f"\n-> {a.out / 'report.md'}, {a.out / 'by_v_r.md'}, {a.out / 'pooled.md'}, {a.out / 'results.csv'}")


if __name__ == "__main__":
    main()
