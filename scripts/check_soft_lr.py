#!/usr/bin/env python
"""Does the best soft form reach the same level at lr 0.2 as at lr 0.8, given more updates?

The attention-form study's draws (V = 500, R = 30%, seed 0) and training (attn_form_study.train: w / V-filter / PGD,
held-out early stopping, the update with the lowest held-out log loss deployed), run for each --runs pair lr:budget
with a budget-times larger budget than the study's (patience 30 x budget updates, at most 200 x budget), so that a
smaller step is not cut short. The test set is scored at every evaluation too (never used to choose).

The runs are not deterministic on the GPU (two runs of one configuration can choose updates 45 and 65 on
good_customer), so by default TabPFN runs in strict deterministic mode (tabfm_experiments.runtime: deterministic
kernels, no flash / memory-efficient attention): reruns are then bit-identical, and the two learning rates are compared
without that noise. Strict attention needs far more memory, hence the default datasets leave out gas-drift and
coil2000.

    python scripts/check_soft_lr.py run --device cuda:0 [--runs 0.8:1 0.2:4] [--datasets ...] [--no-strict]
    python scripts/check_soft_lr.py report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

ROOT = Path(os.environ.get("SOFT_LR_ROOT", "results/soft_lr_check"))
FORM = Path("results/attn_form/runs")
DATASETS = ("heloc", "students_dropout", "ecommerce_shipping", "is_this_a_good_customer")
SHORT = {"gas-drift": "gas-drift", "coil2000_insurance_policies": "coil2000", "heloc": "heloc",
         "students_dropout": "students", "ecommerce_shipping": "ecommerce", "is_this_a_good_customer": "good_customer"}
RUNS = ("0.8:1", "0.2:4")


def name_of(lr: float, budget: int, strict: bool) -> str:
    return f"w_vfilter_pgd_lr{lr:g}_x{budget}" + ("_strict" if strict else "")


def run(device: str, runs, datasets, strict: bool) -> None:
    import torch
    import attn_form_study as S
    import bench_largev as BL
    from tabfm_experiments.config import Seeds
    from tabfm_experiments.runtime import build_tabpfn_v2
    torch.set_num_threads(4)
    for ds in datasets:
        tag = BL.tag_of(ds, 30.0, 500, 0)
        data = BL.Data(ds, device)
        data._clf = build_tabpfn_v2(device, Seeds.model_seed, deterministic_mode="strict" if strict else "best-effort",
                                    recompute_layers=True, n_classes=data.K)
        d = BL.load_draw(data, tag)
        for spec in runs:
            lr, budget = float(spec.split(":")[0]), int(spec.split(":")[1])
            out = ROOT / "runs" / tag / f"{name_of(lr, budget, strict)}.json"
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            stop = dict(S.STOP, max_steps=S.STOP["max_steps"] * budget, patience=S.STOP["patience"] * budget)
            t0 = time.time()
            print(f"{tag}: lr {lr:g}, at most {stop['max_steps']} updates, patience {stop['patience']}, "
                  f"{'strict' if strict else 'best-effort'}", flush=True)
            res, arrays = S.train({"ds": ds, "param": "w", "init": "vfilter", "opt": "pgd", "lr": lr}, data, d, stop,
                                  log=lambda m: print(m, flush=True), track_test=True)
            res["strict"] = strict
            np.savez_compressed(str(out)[:-len(".json")] + ".npz", **arrays)
            BL.write_json(out, res)
            s = res["sel"]["ce"]
            print(f"{tag}: lr {lr:g}: chosen update {s['it']} of {res['last_it']} ({res['stopped']}); test soft+V auc "
                  f"{s['test']['soft_v']['auc']:.4f} ce {s['test']['soft_v']['ce']:.4f} ({time.time() - t0:.0f} s)", flush=True)
        del data, d
        torch.cuda.empty_cache()


def report(runs, strict: bool) -> None:
    labels = [(spec, name_of(float(spec.split(":")[0]), int(spec.split(":")[1]), strict)) for spec in runs]
    md = [f"# The best soft form: lr {' vs '.join(s.split(':')[0] + ' (budget x' + s.split(':')[1] + ')' for s in runs)}, "
          f"V = 500, R = 30% ({'strict deterministic' if strict else 'best-effort'} mode)", "",
          "Deployed: the update with the lowest held-out log loss (the protocol). Best seen: the best test ROC-AUC (+ V) over "
          "all evaluated updates (not a usable choice; what the run could reach). Rows at w = 0 are deleted exactly.", "",
          "| dataset | lr (budget) | chosen update | stopped at | held-out log loss | rows at w = 0 | test ROC-AUC soft / + V | "
          "test accuracy soft / + V | test log loss soft / + V | best seen ROC-AUC + V (update) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    per = {}
    for ds in DATASETS + ("gas-drift", "coil2000_insurance_policies"):
        tag = f"{ds}_R30_V500_s0"
        for spec, name in labels:
            f = ROOT / "runs" / tag / f"{name}.json"
            if not f.exists():
                continue
            r = json.loads(f.read_text())
            s = r["sel"]["ce"]
            t, tv = s["test"]["soft"], s["test"]["soft_v"]
            best = max(r["evals"], key=lambda e: e["test"]["soft_v"]["auc"])
            md.append(f"| {SHORT[ds]} | {spec.replace(':', ' (x') + ')'} | {s['it']} | {r['last_it']} | {s['hold']['ce']:.4f} | "
                      f"{s['n_le1000_pois'] + s['n_le1000_clean']} | {t['auc']:.3f} / {tv['auc']:.3f} | {t['acc']:.3f} / {tv['acc']:.3f} | "
                      f"{t['ce']:.3f} / {tv['ce']:.3f} | {best['test']['soft_v']['auc']:.3f} ({best['it']}) |")
            per[(ds, spec)] = {"hold": s["hold"]["ce"], "auc": t["auc"], "auc_v": tv["auc"], "acc": t["acc"],
                               "acc_v": tv["acc"], "ce": t["ce"], "ce_v": tv["ce"]}
    both = [ds for ds in DATASETS + ("gas-drift", "coil2000_insurance_policies") if all((ds, s) in per for s in runs)]
    md += ["", f"Means over the {len(both)} datasets every run finished:", "",
           "| lr (budget) | held-out log loss | ROC-AUC soft / + V | accuracy soft / + V | log loss soft / + V |",
           "|---|---|---|---|---|"]
    for spec in runs if both else ():
        g = {k: np.mean([per[(ds, spec)][k] for ds in both]) for k in per[(both[0], spec)]}
        md.append(f"| {spec.replace(':', ' (x') + ')'} | {g['hold']:.4f} | {g['auc']:.3f} / {g['auc_v']:.3f} | "
                  f"{g['acc']:.3f} / {g['acc_v']:.3f} | {g['ce']:.3f} / {g['ce_v']:.3f} |")
    out = Path("results/tables/soft_lr_check.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("run", "report"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--runs", nargs="+", default=list(RUNS), help="lr:budget pairs")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--no-strict", action="store_true")
    a = ap.parse_args(argv)
    if a.stage == "run":
        run(a.device, a.runs, a.datasets, not a.no_strict)
    else:
        report(a.runs, not a.no_strict)


if __name__ == "__main__":
    main()
