#!/usr/bin/env python
"""Analyse the gate-valsplit sweep (results/gate_valsplit): is gating worth it, and does a
held-out validation-test set choose a good iteration?

For every run (one dataset, R, V and strategy):
  references   clean rows only (val_as_context, all V rows as the context), the poisoned
               context, the poisoned context + the V rows, the clean context, the oracle gate
  rules        the test scores at the iteration each stopping rule picks (gate_context.
               selection_rules), recomputed from log.csv for every run and checked against
               test_metrics.json where it is stored (round 2):
                 valtest_acc / valtest_ce  held-out accuracy (ties: CE) / held-out CE
                 val_acc                   accuracy on the fitted rows (round 1's rule)
                 val_first_max             first time the fitted-row accuracy peaks
                 last                      the last update
                 test_acc_oracle           the best test checkpoint (peeks; a ceiling)
  chosen       the run's own best gates: holdout runs -> valtest_acc; full runs -> the
               highest fitted-row accuracy over all iterations (round 1's best_iteration)
  bootstrap    a paired bootstrap over test rows (2000 resamples) of the accuracy
               difference (chosen gates) - (clean rows only), from test_probs.npz
  proxy        Spearman correlation, across the test checkpoints, between held-out and
               test accuracy (holdout runs)

Writes results/tables/gate_valsplit/runs.csv (every run, every validation seed) and
summary.txt (the runs of one validation seed, --seed, default 0).

    python scripts/analyze_gate_valsplit.py [--seed 1]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from gate_context import selection_rules  # noqa: E402

ROOT = Path("results/gate_valsplit")
OUT = Path("results/tables/gate_valsplit")
DATASETS = ("gas-drift", "url_unique", "heloc", "students_dropout", "credit-approval")
RULES = ("valtest_acc", "valtest_ce", "val_acc", "val_first_max", "last", "test_acc_oracle")


def read_log(path: Path) -> list[dict]:
    rows = []
    for r in csv.DictReader(open(path)):
        row = {k: float(v) for k, v in r.items() if v not in ("", None)}
        row["iter"] = int(row["iter"])
        rows.append(row)
    return rows


def spearman(a, b) -> float:
    ra, rb = (np.argsort(np.argsort(x)).astype(float) for x in (a, b))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def bootstrap_diff(p_a, p_b, y, n_boot=2000, seed=0) -> tuple[float, float, float]:
    """Accuracy of a minus accuracy of b, with a 95% paired bootstrap interval."""
    ca = (p_a.argmax(1) == y).astype(float)
    cb = (p_b.argmax(1) == y).astype(float)
    d = ca - cb
    idx = np.random.default_rng(seed).integers(0, len(y), size=(n_boot, len(y)))
    boots = d[idx].mean(1)
    return float(d.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def load_runs() -> list[dict]:
    runs = []
    for tm_path in sorted(ROOT.glob("*/*/*/test_metrics.json")):
        d = tm_path.parent
        cfg = json.load(open(d / "config.json"))
        a = cfg["args"]
        tm = json.load(open(tm_path))
        log = read_log(d / "log.csv")
        holdout = bool(a.get("val_holdout", 0))
        rules = selection_rules(log, holdout)
        if "selection_rules" in tm:  # round 2 stored them: the recomputation must agree
            for k, v in tm["selection_rules"].items():
                assert v["iteration"] == rules[k]["iteration"], (d.name, k)
                assert abs(v["test_soft"]["accuracy"] - rules[k]["test_soft"]["accuracy"]) < 1e-12
        M = tm["metrics"]
        tp = np.load(d / "test_probs.npz")
        y = tp["y_test"].astype(int)
        diff = bootstrap_diff(tp["poisoned_best_gate_soft"], tp["val_as_context"], y)
        tests = [r for r in log if "test_soft_accuracy" in r]
        proxy = (spearman([r["valtest_accuracy"] for r in tests], [r["test_soft_accuracy"] for r in tests])
                 if holdout else float("nan"))
        runs.append({
            "dataset": a["dataset"], "R": int(a["row_percent"]), "V": a["val_from_context"],
            "seed": a["val_seed"],
            "strategy": "holdout" if holdout else "full", "n_test": len(y), "K": cfg["n_classes"],
            "ref": {k: M[k] for k in ("clean_context", "poisoned_no_gate", "poisoned_oracle_gate",
                                     "val_as_context", "poisoned_plus_val")},
            "chosen": {"iteration": tm["best_iteration"], "soft": M["poisoned_best_gate_soft"],
                       "hard": M["poisoned_best_gate_hard"], "auroc": tm["best_val_row"]["auroc_poisoned"]},
            "rules": rules, "boot": diff, "proxy_spearman": proxy,
        })
    return runs


def fmt_rule(run, rule):
    r = run["rules"].get(rule)
    return f"{r['test_soft']['accuracy']:.3f}" if r else "  -  "


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0, help="validation seed of summary.txt")
    a = ap.parse_args(argv)
    runs = load_runs()
    OUT.mkdir(parents=True, exist_ok=True)
    by = {(r["dataset"], r["R"], r["V"], r["strategy"]): r for r in runs if r["seed"] == a.seed}

    # ---- runs.csv ----------------------------------------------------------------------
    with open(OUT / "runs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        head = ["dataset", "R", "V", "seed", "strategy", "n_test",
                "clean_rows_only_acc", "clean_rows_only_ce", "poisoned_acc", "poisoned_ce",
                "poisoned_plus_val_acc", "clean_context_acc", "oracle_gate_acc",
                "chosen_iter", "chosen_acc", "chosen_ce", "chosen_bal_acc", "chosen_hard_acc", "auroc",
                "gain_vs_clean_rows", "gain_ci_lo", "gain_ci_hi", "proxy_spearman"]
        head += [f"{rule}_{x}" for rule in RULES for x in ("iter", "acc", "ce")]
        w.writerow(head)
        for r in runs:
            ref = r["ref"]
            row = [r["dataset"], r["R"], r["V"], r["seed"], r["strategy"], r["n_test"],
                   ref["val_as_context"]["accuracy"], ref["val_as_context"]["ce"],
                   ref["poisoned_no_gate"]["accuracy"], ref["poisoned_no_gate"]["ce"],
                   ref["poisoned_plus_val"]["accuracy"], ref["clean_context"]["accuracy"],
                   ref["poisoned_oracle_gate"]["accuracy"],
                   r["chosen"]["iteration"], r["chosen"]["soft"]["accuracy"], r["chosen"]["soft"]["ce"],
                   r["chosen"]["soft"]["balanced_accuracy"], r["chosen"]["hard"]["accuracy"], r["chosen"]["auroc"],
                   *r["boot"], r["proxy_spearman"]]
            for rule in RULES:
                v = r["rules"].get(rule)
                row += ([v["iteration"], v["test_soft"]["accuracy"], v["test_soft"]["ce"]] if v else ["", "", ""])
            w.writerow(row)

    # ---- summary.txt ---------------------------------------------------------------------
    seeds = sorted({r["seed"] for r in runs})
    lines = [f"validation seed {a.seed} ({len(by)} runs); seeds present: {seeds}"]
    for ds in DATASETS:
        for R in (10, 20):
            sub = sorted({k[2] for k in by if k[0] == ds and k[1] == R})
            if not sub:
                continue
            any_run = by.get((ds, R, sub[0], "holdout")) or by.get((ds, R, sub[0], "full"))
            ref = any_run["ref"]
            lines.append(f"\n{ds}  R={R}%   n_test={any_run['n_test']}   oracle gate "
                         f"{ref['poisoned_oracle_gate']['accuracy']:.3f}   (test accuracy)")
            lines.append(f"  {'V':>4} {'clean':>6} {'pois':>6} {'p+val':>6} | {'FULL chosen':>11} {'ceiling':>7} | "
                         f"{'HOLDOUT chosen':>14} {'ce-rule':>7} {'fit-rule':>8} {'1st-max':>7} {'last':>6} {'ceiling':>7} | "
                         f"{'gain vs clean (holdout) [95% CI]':>32} {'rho':>5}")
            for V in sub:
                f, h = by.get((ds, R, V, "full")), by.get((ds, R, V, "holdout"))
                base = (h or f)["ref"]
                fc = f"{f['chosen']['soft']['accuracy']:.3f}" if f else "  -  "
                fo = fmt_rule(f, "test_acc_oracle") if f else "  -  "
                if h:
                    g, lo, hi = h["boot"]
                    hc = (f"{h['chosen']['soft']['accuracy']:.3f} {fmt_rule(h, 'valtest_ce'):>7} {fmt_rule(h, 'val_acc'):>8} "
                          f"{fmt_rule(h, 'val_first_max'):>7} {fmt_rule(h, 'last'):>6} {fmt_rule(h, 'test_acc_oracle'):>7}")
                    gtxt = f"{g:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                    rho = f"{h['proxy_spearman']:.2f}"
                else:
                    hc, gtxt, rho = "-", "-", "-"
                lines.append(f"  {V:>4} {base['val_as_context']['accuracy']:.3f} {base['poisoned_no_gate']['accuracy']:.3f} "
                             f"{base['poisoned_plus_val']['accuracy']:.3f} | {fc:>11} {fo:>7} | {hc:>14} | {gtxt:>32} {rho:>5}")
    text = "\n".join(lines)
    (OUT / "summary.txt").write_text(text + "\n")
    print(text)
    print(f"\nwrote {OUT}/runs.csv ({len(runs)} runs, seeds {seeds}) and summary.txt (seed {a.seed})")


if __name__ == "__main__":
    main()
