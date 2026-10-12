#!/usr/bin/env python
"""Analyse the top-k gating runs (results/gate_topk): which setting is best?

A setting is (surrogate, initial score spread, lr); each has several restarts (init
seeds) per dataset. For the identity and centred surrogates only lr / spread matters
(scaling the scores and the step together leaves every mask unchanged), so the setting
is also labelled by that ratio.

Per dataset and setting:
  best restart   the restart with the lowest validation-test (held-out) CE at its chosen
                 iteration (each run chooses its iteration by that CE, --select-on ce)
  reported       its held-out CE and accuracy; its test accuracy and CE; precision@R
                 (share of the masked rows that are poisoned) and AUROC of -scores;
                 and, over all restarts, the mean and range of test accuracy
Settings are ranked within each dataset by the best restart's held-out CE, and overall by
the mean rank over the datasets where every restart of the setting has finished. Test
numbers are reported, never used to choose.

References per dataset (same validation draw, holdout and poison): the poisoned context;
the oracle gate; the 500 clean rows alone as the context; and the soft gates of
results/gate_valsplit at their lowest-held-out-CE iteration.

    python scripts/analyze_gate_topk.py [--root results/gate_topk] [--R 20]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

DATASETS = ("gas-drift", "url_unique", "students_dropout")


def setting_label(ste: str, std: float, lr: float) -> str:
    if ste in ("identity", "sigmoid-centered"):
        return f"{ste} lr/std={lr / std:g} (std {std:g}, lr {lr:g})"
    return f"{ste} std={std:g} lr={lr:g}"


def load_runs(root: Path, R: int) -> list[dict]:
    runs = []
    for tm_path in sorted(root.glob("*/*/*/test_metrics.json")):
        d = tm_path.parent
        a = json.load(open(d / "config.json"))["args"]
        if int(a["row_percent"]) != R or a.get("gate") != "topk":
            continue
        tm = json.load(open(tm_path))
        b, M, gs = tm["best_val_row"], tm["metrics"], tm["best_gate_stats"]
        log = list(csv.DictReader(open(d / "log.csv")))
        flips = [float(r["mask_flips"]) for r in log if r.get("mask_flips")]
        runs.append({
            "dataset": a["dataset"], "ste": a["ste"], "std": a["score_init_std"], "lr": a["lr"],
            "seed": a["init_seed"], "steps": a["steps"],
            "setting": (a["ste"], a["score_init_std"], a["lr"]),
            "iteration": tm["best_iteration"], "hold_ce": b["valtest_ce"], "hold_acc": b["valtest_accuracy"],
            "test_acc": M["poisoned_best_gate_soft"]["accuracy"], "test_ce": M["poisoned_best_gate_soft"]["ce"],
            "test_hard_acc": M["poisoned_best_gate_hard"]["accuracy"],
            "precision": gs["precision_at_R"], "auroc": gs["auroc_poisoned"],
            "late_flips": sum(flips[-50:]) / max(1, len(flips[-50:])),
            "ref": {k: M[k]["accuracy"] for k in ("poisoned_no_gate", "poisoned_oracle_gate", "val_as_context")},
        })
    return runs


def soft_gate_reference(dataset: str, R: int) -> dict | None:
    """The soft-gate run with the same draw (V = 500, seed 0, 20% holdout), at its lowest
    held-out CE checkpoint (the rule the top-k runs use)."""
    hits = list(Path("results/gate_valsplit").glob(f"{dataset}/*_r{R:03d}/*_V500s0_H20_*/test_metrics.json"))
    if not hits:
        return None
    tm = json.load(open(hits[0]))
    rule = tm["selection_rules"]["valtest_ce"]
    return {"hold_ce": rule["valtest"]["ce"], "test_acc": rule["test_soft"]["accuracy"],
            "iteration": rule["iteration"]}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("results/gate_topk"))
    ap.add_argument("--R", type=int, default=20)
    ap.add_argument("--restarts", type=int, default=4, help="restarts a complete setting has")
    ap.add_argument("--out", type=Path, default=Path("results/tables/gate_topk"))
    a = ap.parse_args(argv)
    runs = load_runs(a.root, a.R)
    if not runs:
        raise SystemExit(f"no top-k runs with R={a.R} under {a.root}")

    groups = defaultdict(list)
    for r in runs:
        groups[(r["dataset"], r["setting"])].append(r)
    lines = [f"top-k gating, R={a.R}%: best of {a.restarts} restarts by held-out CE; test numbers "
             "are reported, not used to choose"]
    best = {}
    for ds in DATASETS:
        keys = sorted(s for d, s in groups if d == ds)
        if not keys:
            continue
        ref = groups[(ds, keys[0])][0]["ref"]
        soft = soft_gate_reference(ds, a.R)
        lines.append(f"\n{ds}   poisoned {ref['poisoned_no_gate']:.3f}   oracle {ref['poisoned_oracle_gate']:.3f}   "
                     f"500 clean rows alone {ref['val_as_context']:.3f}"
                     + (f"   soft gates (held-out CE {soft['hold_ce']:.4f} @it{soft['iteration']}) "
                        f"{soft['test_acc']:.3f}" if soft else ""))
        lines.append(f"  {'setting':<46s} {'n':>2s} | {'hold CE':>8s} {'hold acc':>8s} {'it':>4s} | "
                     f"{'test acc':>8s} {'test CE':>8s} {'prec@R':>6s} {'AUROC':>5s} | "
                     f"{'test acc over restarts':>24s} {'late flips':>10s}")
        rows = []
        for s in keys:
            g = groups[(ds, s)]
            b = min(g, key=lambda r: (r["hold_ce"], -r["hold_acc"], r["seed"]))
            accs = [r["test_acc"] for r in g]
            best[(ds, s)] = b if len(g) >= a.restarts else None
            rows.append((b["hold_ce"], s, g, b, accs))
        for hold_ce, s, g, b, accs in sorted(rows, key=lambda x: x[0]):
            lines.append(f"  {setting_label(*s):<46s} {len(g):>2d} | {b['hold_ce']:>8.4f} {b['hold_acc']:>8.3f} "
                         f"{b['iteration']:>4d} | {b['test_acc']:>8.3f} {b['test_ce']:>8.4f} {b['precision']:>6.3f} "
                         f"{b['auroc']:>5.2f} | {sum(accs)/len(accs):>8.3f} [{min(accs):.3f}, {max(accs):.3f}] "
                         f"{sum(r['late_flips'] for r in g)/len(g):>10.1f}")

    # overall ranking over the settings complete on every dataset that has them
    settings = sorted({s for (_, s), v in best.items() if v is not None})
    dsets = [ds for ds in DATASETS if any(d == ds for d, _ in best)]
    complete = [s for s in settings if all(best.get((ds, s)) is not None for ds in dsets)]
    if complete:
        lines.append(f"\noverall: settings complete on {', '.join(dsets)}; rank 1 = lowest held-out CE")
        rank = {s: [] for s in complete}
        for ds in dsets:
            for i, s in enumerate(sorted(complete, key=lambda s: best[(ds, s)]["hold_ce"]), 1):
                rank[s].append(i)
        for s in sorted(complete, key=lambda s: (sum(rank[s]) / len(rank[s]), s)):
            lines.append(f"  {setting_label(*s):<46s} mean rank {sum(rank[s]) / len(rank[s]):.2f}  "
                         f"ranks {rank[s]}  test acc " + " ".join(f"{best[(ds, s)]['test_acc']:.3f}" for ds in dsets))
    text = "\n".join(lines)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"settings_R{a.R}.txt").write_text(text + "\n")
    with open(a.out / f"runs_R{a.R}.csv", "w", newline="") as fh:
        keys = [k for k in runs[0] if k not in ("ref", "setting")]
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(runs)
    print(text)


if __name__ == "__main__":
    main()
