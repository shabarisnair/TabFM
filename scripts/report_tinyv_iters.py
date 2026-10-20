#!/usr/bin/env python
"""Report of the tiny-V update-count study (scripts/tinyv_iter_study.py): where each learner peaks on the test set,
and the one update count to use for every dataset and V size.

Per run (dataset x c) and metric (ROC-AUC, accuracy, log loss) over the evaluated updates: the peak (the update with
the best test value; ties: the earliest) and the regret curve, (best - value) / (best - worst), 0 at the run's best
update and 1 at its worst (0 throughout if the metric never moves). The count chosen is the evaluated update with the
lowest regret averaged over the runs and the three metrics (ties: the earlier); its paired gains over update 0 and
over update 20 are shown, overall and per dataset and c.

    python scripts/report_tinyv_iters.py [--out results/tables/tinyv_iters]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tinyv_iter_study as T  # noqa: E402

METRICS = (("auc", "ROC-AUC", True), ("acc", "accuracy", True), ("ce", "log loss", False))
NAMES = {"m1": "M1 (top-k, hard + V)", "soft": "best soft form, lr 0.8 (soft + V)",
         "soft_lr0.2": "best soft form at lr 0.2 (soft + V)", "soft_lr0.05": "best soft form at lr 0.05 (soft + V)"}
SHORT = {"gas-drift": "gas-drift", "coil2000_insurance_policies": "coil2000", "heloc": "heloc",
         "students_dropout": "students", "ecommerce_shipping": "ecommerce", "is_this_a_good_customer": "good_customer"}


def load(runs_dir: str = "runs") -> dict:
    runs = {}
    for f in sorted((T.ROOT / runs_dir).glob("*/*.json")):
        r = json.loads(f.read_text())
        if r.get("skipped"):
            continue
        ds, rest = f.parent.name.rsplit("_R", 1)
        c = int(rest.split("_")[1][1:])
        its = [e["it"] for e in r["curve"]]
        vals = {m: np.array([e["test"][m] for e in r["curve"]], dtype=float) for m, _, _ in METRICS}
        runs[(r["method"], ds, c)] = {"its": its, "vals": vals, "prec": [e.get("prec", e.get("prec_zero")) for e in r["curve"]],
                                      "n_zero": [e.get("n_zero") for e in r["curve"]]}
    return runs


def regret(v: np.ndarray, higher: bool) -> np.ndarray:
    best, worst = (v.max(), v.min()) if higher else (v.min(), v.max())
    span = abs(best - worst)
    return np.zeros_like(v) if span < 1e-12 else np.abs(best - v) / span


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("results/tables/tinyv_iters"))
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    md = ["# Tiny-V: how many updates? (scripts/tinyv_iter_study.py)", "",
          "R = 30%, fresh draws (seed 1), both learners trained on all of V as in bench_tinyv, test metrics after a "
          "schedule of updates. Regret: per run and metric, (best - value) / (best - worst) over the evaluated updates. "
          "The chosen count has the lowest regret averaged over the runs and the three metrics.", ""]
    chosen = {}
    for runs_dir, label in (("runs", "80 updates, every dataset (gas-drift at c = 1 only)"),
                            ("runs_long", "longer M1 runs, 300 updates, the three cheapest datasets")):
        runs = load(runs_dir)
        if runs:
            md += [f"# {label}", ""]
            md += section(runs, chosen, runs_dir)
    md += ["## Chosen update counts", "", "| method, study | updates |", "|---|---|"] + [
        f"| {NAMES[m.split(':')[0]]}, {m.split(':')[1]} | {it} |" for m, it in chosen.items()]
    (a.out / "report.md").write_text("\n".join(md) + "\n")
    (a.out / "chosen.json").write_text(json.dumps(chosen, indent=1))
    print(f"-> {a.out}/report.md; chosen: {chosen}")


def section(runs: dict, chosen: dict, runs_dir: str) -> list[str]:
    md = []
    for method in T.METHODS:
        keys = sorted(k for k in runs if k[0] == method)
        if not keys:
            continue
        its = runs[keys[0]]["its"]
        assert all(runs[k]["its"] == its for k in keys)
        reg = {m: np.array([regret(runs[k]["vals"][m], hi) for k in keys]) for m, _, hi in METRICS}
        mean_reg = {m: r.mean(0) for m, r in reg.items()}
        comb = np.mean([mean_reg[m] for m, _, _ in METRICS], axis=0)
        best_i = int(np.argmin(comb))
        chosen[f"{method}:{runs_dir}"] = its[best_i]
        peaks = {m: [its[int(np.argmax(runs[k]["vals"][m]) if hi else np.argmin(runs[k]["vals"][m]))] for k in keys]
                 for m, _, hi in METRICS}
        md += [f"## {NAMES[method]}: {len(keys)} runs", "",
               "Peak update per run (the best test value; the earliest if tied): mean / median / range", "",
               "| metric | mean | median | range |", "|---|---|---|---|"]
        for m, t, _ in METRICS:
            p = np.array(peaks[m])
            md.append(f"| {t} | {p.mean():.1f} | {np.median(p):.0f} | {p.min()}-{p.max()} |")
        md += ["", "Mean regret over the runs (0 = every run at its best) and the mean test value, per evaluated update", "",
               "| update | " + " | ".join(f"regret {t}" for _, t, _ in METRICS) + " | regret, mean of the three | "
               + " | ".join(f"mean {t}" for _, t, _ in METRICS) + " |", "|---" * 8 + "|"]
        for i, it in enumerate(its):
            vals = [np.mean([runs[k]["vals"][m][i] for k in keys]) for m, _, _ in METRICS]
            mark = "**" if i == best_i else ""
            md.append(f"| {mark}{it}{mark} | " + " | ".join(f"{mean_reg[m][i]:.3f}" for m, _, _ in METRICS)
                      + f" | {mark}{comb[i]:.3f}{mark} | " + " | ".join(f"{v:.4f}" for v in vals) + " |")
        i20 = its.index(20)
        md += ["", f"Chosen: **{its[best_i]} updates** (mean regret {comb[best_i]:.3f}; 20 updates: {comb[i20]:.3f}; "
               f"no update: {comb[0]:.3f}).", "",
               f"Per run at {its[best_i]} updates: the change against no update and against 20 updates "
               "(ROC-AUC and accuracy in points of %, log loss as is; + is better for all three)", "",
               "| dataset | c | ROC-AUC vs 0 | accuracy vs 0 | log loss vs 0 | ROC-AUC vs 20 | accuracy vs 20 | log loss vs 20 |",
               "|---|---|---|---|---|---|---|---|"]
        tot = {(m, ref): [] for m, _, _ in METRICS for ref in (0, i20)}
        for k in keys:
            cells = []
            for ref in (0, i20):
                for m, _, hi in METRICS:
                    v = runs[k]["vals"][m]
                    dlt = (v[best_i] - v[ref]) * (1 if hi else -1) * (100 if m != "ce" else 1)
                    tot[(m, ref)].append(dlt)
                    cells.append(f"{dlt:+.{1 if m != 'ce' else 3}f}")
            md.append(f"| {SHORT[k[1]]} | {k[2]} | " + " | ".join(cells) + " |")
        md.append("| **mean (wins-losses)** | | " + " | ".join(
            f"{np.mean(tot[(m, ref)]):+.{2 if m != 'ce' else 3}f} ({sum(x > 1e-9 for x in tot[(m, ref)])}-"
            f"{sum(x < -1e-9 for x in tot[(m, ref)])})" for ref in (0, i20) for m, _, _ in METRICS) + " |")
        md.append("")
    return md


if __name__ == "__main__":
    main()
