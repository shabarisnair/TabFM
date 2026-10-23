#!/usr/bin/env python
"""How much does random label noise hurt TabPFNv2 on the benchmark datasets?

Each context row's label is flipped independently with probability R (to one of the other K - 1 labels, uniformly:
multiclass_attacks' label-flip-bernoulli, the same rows as attack_context.py's at K = 2), TabPFNv2 is fitted on the
whole flipped context and scored on the held-out test set: R = 30, 40, 45%, RUNS runs each (a fresh draw per run),
against the clean context. The benchmarks' contexts, test sets and model (bench_tinyv.DATASETS, BenchData). For
reference, the benchmarks' own poison (hub-flip-local: the R% densest rows, each relabelled to its locally closest
other class; deterministic, so one run) at R = 30% on the same full context.

Pure noise is reached at R = (K - 1) / K: 50% for binary labels, 67% for students_dropout (3 classes), 83% for gas-drift
(6 classes), so the same R is a weaker corruption of the multi-class datasets.

    python scripts/random_flip_degradation.py run --device cuda:0 [--datasets gas-drift heloc ...]
    python scripts/random_flip_degradation.py report     # -> results/tables/random_flip/report.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

ROOT = Path("results/random_flip")
DATASETS = ("gas-drift", "students_dropout", "heloc", "ecommerce_shipping", "is_this_a_good_customer",
            "coil2000_insurance_policies")
RS, RUNS, HUB_R = (30.0, 40.0, 45.0), 5, 30.0
METRICS = (("auc", "ROC-AUC", True), ("acc", "accuracy", True), ("bacc", "balanced accuracy", True),
           ("mcc", "MCC", True), ("ce", "log loss", False))


def scores(p: np.ndarray, y: np.ndarray) -> dict:
    import attn_learn as A
    from sklearn.metrics import matthews_corrcoef
    out = A.metrics(p, y)
    out["mcc"] = float(matthews_corrcoef(y, p.argmax(1)))
    return out


def run(a) -> None:
    import bench_tinyv as B
    import torch
    from tabfm_experiments.config import Seeds
    from tabfm_experiments.multiclass_attacks import poison
    ROOT.mkdir(parents=True, exist_ok=True)
    for ds in a.datasets:
        out = ROOT / f"{ds}.json"
        if out.exists():
            continue
        t0 = time.time()
        bd = B.BenchData(ds, a.device)
        y = np.asarray(bd.y).astype(int)
        yt = np.asarray(bd.yt).astype(int)
        mix = lambda v: (np.bincount(v, minlength=bd.K) / len(v)).round(4).tolist()  # noqa: E731

        def score(y_ctx):
            return scores(bd.probs(bd.X, y_ctx, bd.Xt, yt), yt)

        res = {"dataset": ds, "K": bd.K, "n_context": len(y), "n_features": int(bd.X.shape[1]), "n_test": len(yt),
               "class_mix": mix(y), "clean": score(y), "random": {}, "hub": []}
        for R in RS:
            res["random"][f"{R:g}"] = []
            for r in range(RUNS):
                _, yf = poison("label-flip-bernoulli", bd.X, y, bd.K, R, run_id=r, row_seed=Seeds.row_seed)
                res["random"][f"{R:g}"].append(score(yf) | {"flipped": float((yf != y).mean()), "class_mix": mix(yf)})
        _, yp = poison("hub-flip-local", bd.X, y, bd.K, HUB_R, run_id=0, row_seed=Seeds.row_seed)
        res["hub"].append(score(yp) | {"flipped": float((yp != y).mean()), "class_mix": mix(yp)})
        res["seconds"] = round(time.time() - t0, 1)
        out.write_text(json.dumps(res, indent=1))
        c = res["clean"]
        print(f"{ds}: clean auc {c['auc']:.4f} acc {c['acc']:.4f} ce {c['ce']:.4f}; " + "; ".join(
            f"R{R}: auc {np.mean([e['auc'] for e in v]):.4f} acc {np.mean([e['acc'] for e in v]):.4f}"
            for R, v in res["random"].items()) + f"  ({res['seconds']:.0f} s)", flush=True)
        del bd
        torch.cuda.empty_cache()


def report(a) -> None:
    res = {ds: json.loads((ROOT / f"{ds}.json").read_text()) for ds in DATASETS if (ROOT / f"{ds}.json").exists()}
    md = ["# Random label noise: how much does it hurt TabPFNv2?", "",
          f"Each context row flipped independently with probability R to one of the other labels (uniformly); the whole "
          f"context (no V), the benchmarks' contexts and test sets; {RUNS} runs per R (mean +- std over runs; Δ = change "
          f"from the clean context). hub-flip-local: the benchmarks' structured poison at R = {HUB_R:g}% on the same "
          f"context (deterministic: one run). Pure noise is R = (K - 1) / K: 50% for binary labels, 67% for 3 classes, "
          f"83% for 6.", "",
          "## Change from the clean context (mean over runs)", ""]
    for m, title, hi in METRICS:
        md += [f"### {title}" + ("" if hi else " (higher is worse)"), "",
               "| dataset | K | clean | " + " | ".join(f"random R = {R:g}%" for R in RS) + f" | hub-flip-local R = {HUB_R:g}% |",
               "|---|---|---|" + "---|" * (len(RS) + 1)]
        for ds, r in res.items():
            cells = [f"{np.mean([e[m] for e in r['random'][f'{R:g}']]) - r['clean'][m]:+.4f}" for R in RS]
            cells.append(f"{np.mean([e[m] for e in r['hub']]) - r['clean'][m]:+.4f}")
            md.append(f"| {ds} | {r['K']} | {r['clean'][m]:.4f} | " + " | ".join(cells) + " |")
        md.append("")
    md += ["## Per dataset (mean +- std over runs)", ""]
    for ds, r in res.items():
        md += [f"### {ds}", "", f"Context {r['n_context']} x {r['n_features']}, test {r['n_test']}, {r['K']} classes; "
               f"class mix {r['class_mix']}.", "",
               "| context | rows flipped | class mix | " + " | ".join(t for _, t, _ in METRICS) + " |",
               "|---|---|---|" + "---|" * len(METRICS)]
        md.append("| clean | 0 | " + f"{r['class_mix']} | " + " | ".join(f"{r['clean'][m]:.4f}" for m, _, _ in METRICS) + " |")
        for label, es in [*((f"random R = {R:g}%", r["random"][f"{R:g}"]) for R in RS), (f"hub-flip-local R = {HUB_R:g}%", r["hub"])]:
            cm = np.mean([e["class_mix"] for e in es], 0).round(3).tolist()
            md.append(f"| {label} | {np.mean([e['flipped'] for e in es]):.3f} | {cm} | " + " | ".join(
                f"{np.mean([e[m] for e in es]):.4f}" + (f" +- {np.std([e[m] for e in es], ddof=1):.4f}" if len(es) > 1 else "")
                for m, _, _ in METRICS) + " |")
        md.append("")
    out = Path("results/tables/random_flip")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(md) + "\n")
    print(f"-> {out}/report.md ({len(res)} datasets)")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("run", "report"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    a = ap.parse_args(argv)
    {"run": run, "report": report}[a.stage](a)


if __name__ == "__main__":
    main()
