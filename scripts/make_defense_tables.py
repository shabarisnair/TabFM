#!/usr/bin/env python
"""Tables for the k-fold defence: our method vs perfect removal vs no corruption.

For every (dataset, attack, R) cell it lines up four contexts on the same test set:

    clean       never corrupted                        -- the ceiling
    poisoned    corrupted, undefended                  -- the floor
    defended    corrupted, our R% removed by k-fold    -- the method
    oracle      corrupted, the REAL flipped rows removed -- perfect removal

``clean`` is one deterministic fit, so it has no spread. The other three are means over
the cells of the crossed design (25 for label-flip-random = 5 poison x 5 fold draws,
5 for the deterministic hub-flip = 1 x 5), and their std is the TOTAL spread. The two
components are carried into the CSV separately:

    between_poison_std       how much the ATTACK draw moves the number (0 for hub-flip)
    within_poison_fold_std   how much the DEFENCE's fold draw moves it, poison fixed

``recovered`` is the share of the oracle's repair that the method actually achieved,
(defended - poisoned) / (oracle - poisoned): 1.0 means it matched perfect removal, 0
means it did nothing, and NEGATIVE means it left the model worse than no defence at all.
It is only meaningful where the oracle actually repairs something, so it is blanked when
|oracle - poisoned| is negligible.

    python scripts/make_defense_tables.py
    python scripts/make_defense_tables.py --metrics ce accuracy --format text
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

DATASETS = ["credit-approval", "url_unique", "coil2000_insurance_policies",
            "heloc", "lcld_v2"]
ATTACKS = [("label-flip-random", "random label flips"), ("hub-flip", "hub label flips")]
VARIANTS = ["clean", "poisoned", "defended", "oracle"]
METRICS = ["ce", "roc_auc", "accuracy", "balanced_accuracy", "mcc"]
SHORT = {"ce": "CE", "roc_auc": "ROC-AUC", "accuracy": "accuracy",
         "balanced_accuracy": "bal. acc", "mcc": "MCC", "f1": "F1"}
# Lower is better for CE only; everything else is higher-is-better. Used to decide
# whether a move away from the floor counts as repair.
LOWER_BETTER = {"ce"}


def load(root: Path) -> dict:
    out: dict = {}
    for f in sorted(root.glob("*/*/summary.json")):
        s = json.loads(f.read_text())
        out[(s["dataset"], s["attack"], int(round(s["row_percent"])))] = s
    return out


def cell(s: dict, variant: str, m: str) -> dict:
    """One variant's value for metric ``m``, in a uniform shape."""
    if variant == "clean":
        return {"mean": float(s["clean"][m]), "std": 0.0,
                "between_poison_std": 0.0, "within_poison_fold_std": 0.0, "n": 1}
    d = s["aggregate"].get(variant, {}).get(m)
    return d if d else {"mean": None, "std": None, "between_poison_std": None,
                        "within_poison_fold_std": None, "n": 0}


# Below this, the oracle's own repair is too small for the ratio to mean anything: the
# denominator is noise and the quotient explodes (heloc random R=5 headroom is 0.23% of
# the poisoned CE, which produced a "recovered" of +51). Mirrors the WEAK convention in
# make_main_tables.py -- such cells are marked "~" and should be read off the raw means.
RECOVERED_REL_FLOOR = 0.02


def recovered(defended: float, poisoned: float, oracle: float) -> tuple[float | None, bool]:
    """(share of the oracle's repair the method achieved, is_unstable).

    Unstable means perfect removal barely moved the metric at this R, so there was no
    repair to capture and the ratio is dividing by noise.
    """
    gap = oracle - poisoned
    if abs(gap) < 1e-12:
        return None, True
    unstable = abs(gap) < RECOVERED_REL_FLOOR * abs(poisoned)
    return float((defended - poisoned) / gap), unstable


def fmt_recovered(v: float | None, unstable: bool) -> str:
    if v is None:
        return "-"
    return ("~" if unstable else "") + f"{v:+.2f}"


def text_table(ds: str, attack: str, label: str, rows: dict, metrics: list[str],
               rs: list[int]) -> str:
    s0 = rows[rs[0]]
    head = (f"{ds}   {label} (R = corruption rate, known to the defender)   "
            f"context {s0['n_context']}x{s0['n_features']}, test {s0['n_test']}   "
            f"{s0['n_poison_draws']} poison x {s0['n_fold_draws']} fold draws "
            f"= {s0['n_runs']} cells, {s0['n_folds']} folds")
    out = [head, ""]
    cols = f"  {'R%':>4s} {'k':>5s}  {'context':<10s}"
    for m in metrics:
        cols += f"  {SHORT[m]:>17s}"
    cols += f"  {'recovered':>10s}"
    out += [cols, "  " + "-" * (len(cols) - 2)]
    clean_line = f"  {'-':>4s} {'-':>5s}  {'clean':<10s}"
    for m in metrics:
        clean_line += f"  {format(cell(s0, 'clean', m)['mean'], '.4f'):>17s}"
    out += [clean_line + f"  {'-':>10s}", "  " + "-" * (len(cols) - 2)]

    for r in rs:
        s = rows[r]
        for i, v in enumerate(("poisoned", "defended", "oracle")):
            line = (f"  {r if i == 0 else '':>4} {s['k_flipped'] if i == 0 else '':>5}  "
                    f"{v:<10s}")
            for m in metrics:
                c = cell(s, v, m)
                line += f"  {format(c['mean'], '.4f') + ' +- ' + format(c['std'], '.4f'):>17s}"
            if v == "defended":
                rec, unst = recovered(cell(s, "defended", metrics[0])["mean"],
                                      cell(s, "poisoned", metrics[0])["mean"],
                                      cell(s, "oracle", metrics[0])["mean"])
                line += f"  {fmt_recovered(rec, unst):>10s}"
            else:
                line += f"  {'':>10s}"
            out.append(line)
        out.append("")
    out.append(f"  recovered = (defended - poisoned) / (oracle - poisoned) on {SHORT[metrics[0]]}; "
               "1.00 = matched perfect removal, <0 = worse than no defence.")
    out.append(f"  '~' = perfect removal itself moved {SHORT[metrics[0]]} by less than "
               f"{RECOVERED_REL_FLOOR:.0%}, so the ratio divides by noise -- read the means instead.")
    return "\n".join(out)


def md_table(ds: str, attack: str, label: str, rows: dict, metrics: list[str],
             rs: list[int]) -> str:
    s0 = rows[rs[0]]
    out = [f"### {ds} — {label}", "",
           f"Context {s0['n_context']}×{s0['n_features']}, test {s0['n_test']}. "
           f"{s0['n_poison_draws']} poison × {s0['n_fold_draws']} fold draws = "
           f"{s0['n_runs']} cells of {s0['n_folds']} folds. Cells are `mean ± std` "
           f"(total spread). `clean` is a single deterministic fit.", "",
           "| R% | k | context | " + " | ".join(SHORT[m] for m in metrics) + " | recovered |",
           "|---|---|---|" + "---|" * (len(metrics) + 1),
           "| – | – | **clean** | "
           + " | ".join(f"**{cell(s0, 'clean', m)['mean']:.4f}**" for m in metrics)
           + " | – |"]
    for r in rs:
        s = rows[r]
        for i, v in enumerate(("poisoned", "defended", "oracle")):
            cells = " | ".join(f"{cell(s, v, m)['mean']:.4f} ± {cell(s, v, m)['std']:.4f}"
                               for m in metrics)
            rec = ""
            if v == "defended":
                x, unst = recovered(cell(s, "defended", metrics[0])["mean"],
                                    cell(s, "poisoned", metrics[0])["mean"],
                                    cell(s, "oracle", metrics[0])["mean"])
                rec = fmt_recovered(x, unst)
            out.append(f"| {r if i == 0 else ''} | {s['k_flipped'] if i == 0 else ''} "
                       f"| {v} | {cells} | {rec} |")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("results/defense_kfold"))
    ap.add_argument("--out", type=Path, default=Path("results/tables/defense_kfold"))
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--metrics", nargs="+", default=METRICS)
    ap.add_argument("--format", choices=["text", "markdown", "both"], default="both")
    a = ap.parse_args(argv)

    data = load(a.root)
    if not data:
        raise SystemExit(f"no summary.json found under {a.root}")
    a.out.mkdir(parents=True, exist_ok=True)

    rows_csv = []
    for ds in a.datasets:
        for attack, label in ATTACKS:
            rs = sorted(r for (d, at, r) in data if d == ds and at == attack)
            if not rs:
                print(f"[skip] {ds} / {attack}: nothing under {a.root}")
                continue
            rows = {r: data[(ds, attack, r)] for r in rs}
            if a.format in ("text", "both"):
                print("\n" + text_table(ds, attack, label, rows, a.metrics, rs) + "\n")
            if a.format in ("markdown", "both"):
                (a.out / f"{ds}__{attack}.md").write_text(
                    md_table(ds, attack, label, rows, a.metrics, rs) + "\n")
            for r in rs:
                s = rows[r]
                for m in a.metrics:
                    c = {v: cell(s, v, m) for v in VARIANTS}
                    rec, rec_unstable = recovered(c["defended"]["mean"], c["poisoned"]["mean"],
                                                  c["oracle"]["mean"])
                    for v in VARIANTS:
                        rows_csv.append({
                            "dataset": ds, "attack": attack, "row_percent": r,
                            "k_flipped": s["k_flipped"], "metric": m, "context": v,
                            "mean": c[v]["mean"], "std": c[v]["std"],
                            "between_poison_std": c[v]["between_poison_std"],
                            "within_poison_fold_std": c[v]["within_poison_fold_std"],
                            "n_cells": c[v]["n"],
                            "delta_vs_clean": (None if c[v]["mean"] is None
                                               else c[v]["mean"] - c["clean"]["mean"]),
                            "delta_vs_poisoned": (None if c[v]["mean"] is None
                                                  else c[v]["mean"] - c["poisoned"]["mean"]),
                            "recovered_fraction": rec if v == "defended" else None,
                            "recovered_unstable": rec_unstable if v == "defended" else None,
                            "oracle_headroom": c["oracle"]["mean"] - c["poisoned"]["mean"],
                            "lower_is_better": m in LOWER_BETTER,
                            "n_poison_draws": s["n_poison_draws"],
                            "n_fold_draws": s["n_fold_draws"], "n_folds": s["n_folds"],
                        })
    if rows_csv:
        p = a.out / "defense_results.csv"
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_csv[0]))
            w.writeheader()
            w.writerows(rows_csv)
        print(f"wrote {p} ({len(rows_csv)} rows) and per-(dataset,attack) tables in {a.out}")
    return rows_csv


if __name__ == "__main__":
    main()
