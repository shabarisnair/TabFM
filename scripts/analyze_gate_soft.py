#!/usr/bin/env python
"""Analyse the soft-gate variants of the overnight exploration (results/gate_soft) against
the baseline soft-gate runs on the same draws (results/gate_valsplit, PGD-L-inf lr 0.05).

Per dataset and variant (one run each; R, V and the validation seed fixed by --R --V --seed):
  CE rule       the test iteration with the lowest held-out CE (ties: earlier)
  acc rule      the test iteration with the highest held-out accuracy (ties: lower held-out
                CE, earlier)
  reported      at each rule: held-out CE / accuracy, test accuracy of the learned gates and of
                removing the k lowest-gated rows, precision@R, AUROC, and the removal mass
                sum(1 - g) (soft) or sum(1 - w) (attn); iteration 0 too, which for runs that
                start from the V-filter is the filter itself
Variants are ranked within each dataset by the held-out CE at the CE rule, and overall by the
mean rank over the datasets. Test numbers are reported, never used to choose.

    python scripts/analyze_gate_soft.py [--R 20 --V 500 --seed 0]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

DATASETS = ("gas-drift", "url_unique", "students_dropout")


def variant(run: Path) -> str:
    name = run.name
    if name.startswith("gate-valsplit_"):
        return "soft " + name.split("_H20_")[1].replace("_it300", "")
    return name.split("_H20_")[1].replace("_it300", "")


def summarise(run: Path) -> dict:
    rows = [r for r in csv.DictReader(open(run / "log.csv")) if r.get("valtest_ce")]
    f = lambda r, k: float(r[k]) if r.get(k) not in (None, "") else float("nan")  # noqa: E731
    n = json.loads((run / "config.json").read_text())["n_context"]
    by_ce = min(rows, key=lambda r: (f(r, "valtest_ce"), int(r["iter"])))
    by_acc = min(rows, key=lambda r: (-f(r, "valtest_accuracy"), f(r, "valtest_ce"), int(r["iter"])))
    out = {"run": run.name, "variant": variant(run)}
    for tag, r in (("ce", by_ce), ("acc", by_acc), ("it0", rows[0])):
        out |= {f"{tag}_iter": int(r["iter"]), f"{tag}_hold_ce": f(r, "valtest_ce"),
                f"{tag}_hold_acc": f(r, "valtest_accuracy"), f"{tag}_test": f(r, "test_soft_accuracy"),
                f"{tag}_test_removed": f(r, "test_hard_accuracy"), f"{tag}_prec": f(r, "precision_at_R"),
                f"{tag}_auroc": f(r, "auroc_poisoned"), f"{tag}_mass": n * (1 - f(r, "g_mean"))}
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--R", type=int, default=20)
    ap.add_argument("--V", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attack", default="hub-flip-local")
    ap.add_argument("--out", type=Path, default=Path("results/tables/gate_soft"))
    a = ap.parse_args(argv)
    runs = {ds: [] for ds in DATASETS}
    for ds in DATASETS:
        pat = f"{ds}/{a.attack}_r{a.R:03d}/*_R{a.R}_V{a.V}s{a.seed}_H20_*"
        cands = [d for d in Path("results/gate_soft").glob(pat) if (d / "test_metrics.json").exists()]
        cands += [d for d in Path("results/gate_valsplit").glob(pat)
                  if (d / "test_metrics.json").exists() and d.name.endswith("pgd-linf_lr0.05_it300")]
        # runs with another form of the clean set (--val-draw) are other draws: analyze_gate_forms.py
        cands = [d for d in cands if not variant(d).startswith(("local-", "imbalanced-"))]
        runs[ds] = sorted((summarise(d) for d in cands), key=lambda r: r["ce_hold_ce"])
    L = [f"Soft-gate variants, {a.attack}, R = {a.R}%, V = {a.V} (20% held out), validation seed {a.seed}. Rank by held-out "
         "CE at the CE rule; test reported. it0 = iteration 0 (the starting gates). mass = sum(1 - gate)."]
    head = (f"  {'variant':34s} | {'CE rule: it  hold CE  hold acc  test  test(rm)  prec  AUROC  mass':>66s} | "
            f"{'acc rule: it  hold acc  test  prec':>36s} | {'it0: test  prec':>16s}")
    for ds in DATASETS:
        if not runs[ds]:
            continue
        L += ["", ds, head]
        for r in runs[ds]:
            L.append(f"  {r['variant']:34s} | {r['ce_iter']:>12d} {r['ce_hold_ce']:8.4f} {r['ce_hold_acc']:8.3f} "
                     f"{100 * r['ce_test']:6.1f} {100 * r['ce_test_removed']:8.1f} {r['ce_prec']:5.2f} {r['ce_auroc']:6.2f} "
                     f"{r['ce_mass']:6.0f} | {r['acc_iter']:>12d} {r['acc_hold_acc']:9.3f} {100 * r['acc_test']:5.1f} "
                     f"{r['acc_prec']:5.2f} | {100 * r['it0_test']:10.1f} {r['it0_prec']:5.2f}")
    common = set.intersection(*(set(r["variant"] for r in runs[ds]) for ds in DATASETS if runs[ds]))
    if common:
        L += ["", f"overall (variants run on all of {', '.join(ds for ds in DATASETS if runs[ds])}): mean rank by held-out CE"]
        rank = {v: [] for v in common}
        for ds in DATASETS:
            order = [r["variant"] for r in runs[ds] if r["variant"] in common]
            for i, v in enumerate(order, 1):
                rank[v].append(i)
        test = {v: [100 * next(r["ce_test"] for r in runs[ds] if r["variant"] == v) for ds in DATASETS if runs[ds]]
                for v in common}
        for v in sorted(common, key=lambda v: (sum(rank[v]) / len(rank[v]), v)):
            L.append(f"  {v:34s} mean rank {sum(rank[v]) / len(rank[v]):.2f}  ranks {rank[v]}  test (CE rule) "
                     + " ".join(f"{t:.1f}" for t in test[v]))
    text = "\n".join(L)
    a.out.mkdir(parents=True, exist_ok=True)
    tag = ("" if a.attack == "hub-flip-local" else a.attack + "_") + f"R{a.R}_V{a.V}s{a.seed}"
    (a.out / f"variants_{tag}.txt").write_text(text + "\n")
    with open(a.out / f"variants_{tag}.csv", "w", newline="") as fh:
        allrows = [r | {"dataset": ds} for ds in DATASETS for r in runs[ds]]
        if allrows:
            w = csv.DictWriter(fh, fieldnames=list(allrows[0]))
            w.writeheader()
            w.writerows(allrows)
    print(text)


if __name__ == "__main__":
    main()
