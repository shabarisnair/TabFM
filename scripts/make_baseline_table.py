#!/usr/bin/env python
"""Table: every way of using the clean rows vs the gates, per dataset x V (R = 20%, seed 0).

Test accuracy (%) of: the V clean rows alone; the poisoned set alone; V + the poisoned set;
V + the filtered set (the R% least plausible labels under TabPFN with V as its context
removed); the soft gates (held-out-accuracy iteration); the soft gates + V (the V rows at
gate 1); the top-k vote; the top-k vote's kept rows + V; the oracle set (the poisoned rows
removed); the oracle set + V. Precision@R of the removed rows for the filter, the soft gates
(their k lowest) and the top-k vote. The ordering column ranks all ten from the highest test
accuracy to the lowest: '>' = the 95% paired bootstrap interval (test rows, 2000 resamples)
of the neighbours' difference excludes 0, '≈' = it does not, '=' = equal accuracy.

Source: the per-row test probabilities of scripts/eval_baseline_probs.py
(results/tables/gate_soft/baseline_probs_R20_s0). Also writes the accuracies and precisions
as results/tables/gate_soft/baselines_R20_s0.csv (for scripts/plot_baselines.py).

    python scripts/make_baseline_table.py   # -> results/tables/gate_soft/baselines_R20_s0.md
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from analyze_gate_valsplit import bootstrap_diff  # noqa: E402

DATASETS = ("gas-drift", "url_unique", "heloc", "students_dropout", "credit-approval")
COLS = (("v_only", "V only", "V"), ("poisoned", "poisoned", "P"), ("poisoned_plus_v", "V + poisoned", "V+P"),
        ("vfilter_plus_v", "V + filtered", "V+F"), ("soft", "soft gate", "S"), ("soft_plus_v", "soft gate + V", "S+V"),
        ("topk", "top-k", "T"), ("topk_plus_v", "top-k + V", "T+V"), ("oracle_train", "oracle", "O"),
        ("oracle_plus_v", "oracle + V", "O+V"))
PREC = (("prec_vfilter", "prec filter"), ("prec_soft", "prec soft"), ("prec_topk", "prec top-k"))
SRC = Path("results/tables/gate_soft/baseline_probs_R20_s0")
OUT = Path("results/tables/gate_soft")


def ordering(P: dict, y: np.ndarray) -> str:
    acc = {k: round(float((P[k].argmax(1) == y).mean()), 9) for k, _, _ in COLS}
    ms = sorted((k for k, _, _ in COLS), key=lambda k: -acc[k])  # stable: ties keep the column order
    abbr = {k: s for k, _, s in COLS}
    out = abbr[ms[0]]
    for a, b in zip(ms, ms[1:]):
        sym = "=" if acc[a] == acc[b] else (">" if bootstrap_diff(P[a], P[b], y)[1] > 0 else "≈")
        out += f" {sym} {abbr[b]}"
    return out


def main() -> None:
    rows, probs = [], {}
    for ds in DATASETS:
        for f in sorted(SRC.glob(f"{ds}_V*.npz"), key=lambda p: int(p.stem.rsplit("_V", 1)[1])):
            z = np.load(f)
            y = z["y_test"]
            P = {k: z[k] for k, _, _ in COLS}
            r = {"dataset": ds, "V": int(f.stem.rsplit("_V", 1)[1]), "n_test": len(y)}
            r |= {k: float((P[k].argmax(1) == y).mean()) for k, _, _ in COLS}
            r |= {k: float(z[k]) for k, _ in PREC}
            r["ordering"] = ordering(P, y)
            rows.append(r)
            probs[(ds, r["V"])] = (P, y)
    with open(OUT / "baselines_R20_s0.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    L = ["Test accuracy (%), R = 20% hub-flip-local, validation seed 0; precision@R of the removed rows (oracle = 1; "
         "soft gate + V and top-k + V gate or remove the same rows as soft gate and top-k). Best of the non-oracle "
         "methods in bold.",
         "Ordering: all ten methods from the highest test accuracy to the lowest. '>' = significant (95% paired "
         "bootstrap over test rows), '≈' = not significant, '=' = equal. V = V only, P = poisoned, V+P = V + poisoned, "
         "V+F = V + filtered, S = soft gate, S+V = soft gate + V, T = top-k, T+V = top-k + V, O = oracle, "
         "O+V = oracle + V. Each symbol compares its two neighbours only.", "",
         "| dataset | V | " + " | ".join(lbl for _, lbl, _ in COLS) + " | " + " | ".join(lbl for _, lbl in PREC)
         + " | ordering |",
         "|---|---|" + "---|" * (len(COLS) + len(PREC) + 1)]
    for r in rows:
        vals = {k: 100 * r[k] for k, _, _ in COLS}
        best = max(vals[k] for k, _, _ in COLS if not k.startswith("oracle"))
        cells = [(f"**{vals[k]:.1f}**" if not k.startswith("oracle") and abs(vals[k] - best) < 1e-9 else f"{vals[k]:.1f}")
                 for k, _, _ in COLS]
        L.append(f"| {r['dataset']} | {r['V']} | " + " | ".join(cells) + " | "
                 + " | ".join(f"{r[k]:.2f}" for k, _ in PREC) + f" | {r['ordering']} |")
    L.append(f"| **mean ({len(rows)} draws)** | | " + " | ".join(f"{np.mean([100 * r[k] for r in rows]):.1f}" for k, _, _ in COLS)
             + " | " + " | ".join(f"{np.mean([r[k] for r in rows]):.2f}" for k, _ in PREC) + " | |")
    wins = lambda a, b: (sum(r[a] > r[b] for r in rows), sum(r[a] < r[b] for r in rows))  # noqa: E731

    def sig(a, b):  # draws where a is significantly above / below b (95% paired bootstrap over test rows)
        ci = [bootstrap_diff(P[a], P[b], y)[1:] for P, y in (probs[(r["dataset"], r["V"])] for r in rows)]
        return sum(lo > 0 for lo, _ in ci), sum(hi < 0 for _, hi in ci)

    L.append("")
    for a, against in (("vfilter_plus_v", [k for k, _, _ in COLS if k != "vfilter_plus_v"]),
                       ("soft_plus_v", ["soft", "v_only", "vfilter_plus_v"]),
                       ("topk_plus_v", ["topk", "v_only", "vfilter_plus_v"])):
        name = next(lbl for k, lbl, _ in COLS if k == a)
        L.append(f"{name} vs: " + "; ".join(f"{lbl} {'-'.join(map(str, wins(a, k)))} (sig. {'-'.join(map(str, sig(a, k)))})"
                                            for k, lbl, _ in COLS if k in against)
                 + " (draws higher-lower, ties not counted; sig. = of them, significantly so).")
    text = "\n".join(L)
    (OUT / "baselines_R20_s0.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
