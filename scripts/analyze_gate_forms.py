#!/usr/bin/env python
"""Gating vs the V-filter when the clean set is unrepresentative (--val-draw local | imbalanced).

For each round-4/5 draw (results/gate_soft, tags <form>-soft-lr0.05 and <form>-topk-init<i>;
hub-flip-local, R = 20%, V = 500, validation seeds 0-2) and, for reference, the stratified draw:
  V only, V-filter + V, oracle + V   results/tables/gate_soft/vforms_R20.csv (the same draw)
  soft                               the soft gate at its held-out-accuracy iteration (the run's
                                     own choice) and at its held-out-CE iteration
  top-k vote                         the vote of the four top-k restarts (vote_topk.py, run here
                                     on first use into <dataset>/hub-flip-local_r020/<vote dir>)
Test accuracy overall and on the sub-group the form threatens: the minor class (imbalanced) or
the half of the test rows farthest from the anchor (local; standardised over the context file
and the test set together, as eval_vfilter_vforms.py does).

    python scripts/analyze_gate_forms.py [--gpu 0]
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

ROOT = Path("results/gate_soft")
CASES = (("url_unique", "local"), ("url_unique", "imbalanced"), ("students_dropout", "imbalanced"),
         ("students_dropout", "local"), ("gas-drift", "local"))


def subgroup(run: Path, form: str, y_test: np.ndarray) -> np.ndarray:
    import torch
    import gate_context as G
    from tabfm_experiments.multiclass_attacks import standardise
    cfg = json.loads((run / "config.json").read_text())
    sp = np.load(run / "split.npz")
    if form == "imbalanced":
        return y_test == int(sp["val_minor_class"])
    X_file, _ = G.load(Path(cfg["context"]), "cpu")
    Xt, _ = G.load(Path(cfg["test"]), "cpu")
    Z = standardise(torch.cat([X_file, Xt]))
    a = int(sp["val_anchor"])
    d = torch.cdist(Z[a:a + 1], Z[len(X_file):])[0].numpy()
    return d >= np.median(d)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="0")
    a = ap.parse_args(argv)
    ref = {(r["dataset"], r["form"], int(r["seed"])): r for r in csv.DictReader(open("results/tables/gate_soft/vforms_R20.csv"))
           if r["V"] == "500"}
    L = ["Gating vs the V-filter with an unrepresentative clean set (hub-flip-local, R = 20%, V = 500).",
         "Test accuracy (%) overall / on the threatened sub-group (imbalanced: the minor class; local: the far "
         "half of the test rows).", "",
         f"  {'dataset':17s} {'form':11s} {'seed':>4s} | {'V only':>11s} {'V-filter+V':>11s} {'soft (acc)':>11s} {'soft (CE)':>9s} "
         f"{'top-k vote':>11s} {'oracle+V':>11s} | precision: V-filter soft top-k"]
    for ds, form, seed in [(ds, form, s) for ds, form in CASES for s in (0, 1, 2)]:
        d = ROOT / ds / "hub-flip-local_r020"
        soft = d / f"gate-soft_{ds}_hub-flip-local_R20_V500s{seed}_H20_{form}-soft-lr0.05_it300"
        restarts = [d / f"gate-soft_{ds}_hub-flip-local_R20_V500s{seed}_H20_{form}-topk-init{i}_it300" for i in range(4)]
        have_soft = (soft / "test_metrics.json").exists()
        have_topk = all((x / "test_metrics.json").exists() for x in restarts)
        if not (have_soft or have_topk):
            continue
        base = soft if have_soft else restarts[0]
        y = np.load(base / "test_probs.npz")["y_test"].astype(int)
        sub = subgroup(base, form, y)
        acc = lambda p, m=None: 100 * float(((p.argmax(1) == y)[m] if m is not None else (p.argmax(1) == y)).mean())  # noqa: E731
        r = ref.get((ds, form, seed), {})
        f = lambda k: 100 * float(r[k]) if r else float("nan")  # noqa: E731
        soft_txt, soft_ce, prec_soft = "-", float("nan"), float("nan")
        if have_soft:
            tm = json.loads((soft / "test_metrics.json").read_text())
            tp = np.load(soft / "test_probs.npz")
            soft_txt = f"{acc(tp['poisoned_best_gate_soft']):.1f}/{acc(tp['poisoned_best_gate_soft'], sub):.1f}"
            soft_ce = 100 * tm["selection_rules"]["valtest_ce"]["test_soft"]["accuracy"]
            prec_soft = tm["best_gate_stats"]["precision_at_R"]
        vote_txt, prec_vote = "(pending)", float("nan")
        if have_topk:
            vd = d / f"gate-topk-vote4_{ds}_hub-flip-local_R20_V500s{seed}_H20_{form}-topk_it300"
            if not (vd / "vote.json").exists():
                subprocess.run(["python", "scripts/vote_topk.py", "--gpu", a.gpu, "--out", str(vd)] + [str(x) for x in restarts],
                               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            vz = np.load(vd / "vote.npz")
            vote_txt = f"{acc(vz['test_probs']):.1f}/{acc(vz['test_probs'], sub):.1f}"
            prec_vote = json.loads((vd / "vote.json").read_text())["vote"]["precision_at_R"]
        L.append(f"  {ds:17s} {form:11s} {seed:>4d} | {f('v_only'):5.1f}/{f('sub_v_only'):5.1f} {f('vfilter_plus_v'):5.1f}/{f('sub_vfilter_plus_v'):5.1f} "
                 f"{soft_txt:>11s} {soft_ce:9.1f} {vote_txt:>11s} {f('oracle_plus_v'):5.1f}/{f('sub_oracle_plus_v'):5.1f} | "
                 f"{float(r['prec_vfilter']) if r else float('nan'):.2f} {prec_soft:.2f} {prec_vote:.2f}")
    text = "\n".join(L)
    Path("results/tables/gate_soft").mkdir(parents=True, exist_ok=True)
    Path("results/tables/gate_soft/forms_gating_R20.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
