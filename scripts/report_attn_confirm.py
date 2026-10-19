#!/usr/bin/env python
"""Report of the attention-weighting confirmation runs (scripts/attn_learn.py, results/attn_learn/confirm).

Metric: test ROC-AUC on the binary datasets (heloc, coil2000; higher is better) and test log loss
on the multi-class one (students_dropout; lower is better). Draws: 3 datasets x R = 10/20/30% x
V = 50/500 clean rows (20% of them held out to choose the iteration) x seeds 0-2.

Per dataset and V, a table with the methods as rows and R as columns (mean +- sd over seeds):
the references of every draw -- V only, poisoned + V, V-filter, V-filter + V, oracle + V -- and
every configuration's four deployments: soft (the learned weights on the context), soft + V (the
clean rows added at weight 1), hard (the k = R% lowest-weight rows deleted) and hard + V. The
iteration is the one with the lowest held-out log loss (per deployment kind). Also the post-hoc
rank ensembles of scripts/attn_posthoc.py (M1-M3; M1-M3 and the V-filter; M1 and the V-filter). Then:
  - overall: per method, the mean difference to V only and to V-filter + V over all draws of a
    dataset, and the draws it is better / worse (in the metric's own direction);
  - selection rules: the test metric at the iteration chosen by held-out log loss, accuracy, AUC
    (binary), at the last iteration, and the best test value over all iterations (an upper
    bound no rule can reach);
  - overfitting: how often the held-out data chose iteration 0 (the V-filter start untouched)
    and the median chosen iteration.

    python scripts/report_attn_confirm.py [results/attn_learn/confirm]
        -> results/tables/attn_learn/confirm.md, results/plots/attn_learn/confirm.png
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

DATASETS = (("heloc", "auc", "heloc (ROC-AUC, %)"), ("coil2000_insurance_policies", "auc", "coil2000 (ROC-AUC, %)"),
            ("students_dropout", "ce", "students_dropout (log loss, lower is better)"))
CONFIGS = (("M1_topk_lr0.2", dict(param="topk", opt="pgd-linf", lr=0.2, init="vfilter", prior=0.0, pseudo=0.0)),
           ("M2_adam_prior0.1", dict(param="log", opt="adam", lr=0.1, init="vfilter", prior=0.1, pseudo=0.0)),
           ("M3_pgd_lr1", dict(param="log", opt="pgd-linf", lr=1.0, init="vfilter", prior=0.0, pseudo=0.0)),
           ("A1_adam_zero_init", dict(param="log", opt="adam", lr=0.1, init="zero", prior=0.0, pseudo=0.0)),
           ("A2_pgd_lr1_pseudo", dict(param="log", opt="pgd-linf", lr=1.0, init="vfilter", prior=0.0, pseudo=0.5)),
           ("A3_adam_no_prior", dict(param="log", opt="adam", lr=0.1, init="vfilter", prior=0.0, pseudo=0.0)))
REFS = (("v_only", "V only"), ("poisoned_plus_v", "poisoned + V"), ("vfilter", "V-filter"), ("vfilter_plus_v", "V-filter + V"),
        ("oracle_plus_v", "oracle + V"))
DEPS = (("soft", "test_soft", "soft"), ("soft + V", "test_soft_v", "soft"), ("hard", "test_hard", "hard"),
        ("hard + V", "test_hard_v", "hard"))
RS, VS, SEEDS = (10.0, 20.0, 30.0), (50, 500), (0, 1, 2)
SELECT = "ce"  # the held-out criterion that chooses the iteration in the main tables


def config_of(cfg: dict) -> str | None:
    for name, want in CONFIGS:
        if all(cfg.get(k) == v for k, v in want.items()):
            return name
    return None


def scale(metric: str, x: float) -> float:
    return 100 * x if metric in ("auc", "acc") else x


def select(evals: list, kind: str, rule: str, metric: str) -> dict:
    if rule == "ce":
        return min(evals, key=lambda e: (e[f"hold_{kind}_ce"], e["it"]))
    if rule in ("acc", "auc"):
        return max(evals, key=lambda e: (np.nan_to_num(e[f"hold_{kind}_{rule}"], nan=-1.0), -e[f"hold_{kind}_ce"], -e["it"]))
    if rule == "last":
        return evals[-1]
    raise ValueError(rule)


def main() -> None:
    global DATASETS
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    global SELECT
    logloss = "--logloss" in sys.argv  # every dataset in log loss (a secondary report)
    acc = "--acc" in sys.argv  # every dataset in accuracy
    if logloss:
        DATASETS = tuple((ds, "ce", t.split(" (")[0] + " (log loss, lower is better)") for ds, _, t in DATASETS)
    if acc:
        DATASETS = tuple((ds, "acc", t.split(" (")[0] + " (accuracy, %)") for ds, _, t in DATASETS)
    if "--select-acc" in sys.argv:  # choose the iteration by held-out accuracy (ties: log loss) instead
        SELECT = "acc"
    suffix = ("_logloss" if logloss else "_acc" if acc else "") + ("_selacc" if SELECT == "acc" else "")
    root = Path(args[0]) if args else Path("results/attn_learn/confirm")
    refs, runs, posthoc = {}, {}, {}
    for draw in sorted(p for p in root.iterdir() if p.is_dir()):
        rp = draw / "references.json"
        for f in draw.glob("*.json"):
            if f.name in ("references.json", "posthoc.json"):
                continue
            r = json.loads(f.read_text())
            name = config_of(r["cfg"])
            if name is None:
                continue
            d = r["draw"]
            key = (d["ds"], float(d["R"]), int(d["V"]), int(d["seed"]))
            runs[key + (name,)] = r
            if rp.exists():
                refs[key] = json.loads(rp.read_text())
            if (draw / "posthoc.json").exists():
                posthoc[key] = json.loads((draw / "posthoc.json").read_text())
    better = lambda metric, a, b: (a > b) if metric in ("auc", "acc") else (a < b)  # noqa: E731
    L = ["# Attention weighting: confirmation and ablations", "",
         "hub-flip-local; V clean rows drawn from the context file (20% held out to choose the iteration by log loss); "
         "seeds 0-2; test ROC-AUC (%) for heloc and coil2000, test log loss for students_dropout (lower is better). "
         "Mean +- sd over the seeds; [n] = number of seeds when fewer than 3. soft: the learned weights; hard: the k = R% "
         "lowest-weight rows deleted; + V: the clean rows added to the context.", ""]
    overall = defaultdict(list)  # (ds, method) -> [(value, v_only, vfilter_plus_v, V)]
    for ds, metric, title in DATASETS:
        for V in VS:
            keys = [(ds, R, V, s) for R in RS for s in SEEDS if (ds, R, V, s) in refs]
            if not keys:
                continue
            L += [f"## {title}, V = {V}", "", "| method | R = 10% | R = 20% | R = 30% | mean |", "|---|---|---|---|---|"]

            def row(label, getter):
                cells, allv = [], []
                for R in RS:
                    vals = [getter((ds, R, V, s)) for s in SEEDS if (ds, R, V, s) in refs]
                    vals = [v for v in vals if v is not None]
                    allv += vals
                    cells.append("" if not vals else f"{np.mean(vals):.{1 if metric in ('auc', 'acc') else 3}f} ± "
                                 f"{np.std(vals):.{1 if metric in ('auc', 'acc') else 3}f}" + (f" [{len(vals)}]" if len(vals) < 3 else ""))
                mean = f"{np.mean(allv):.{1 if metric in ('auc', 'acc') else 3}f}" if allv else ""
                L.append(f"| {label} | " + " | ".join(cells) + f" | {mean} |")

            for rk, rl in REFS:
                row(rl, lambda k, rk=rk: scale(metric, refs[k][rk][metric]))
                for k in keys:
                    overall[(ds, rl)].append((scale(metric, refs[k][rk][metric]), scale(metric, refs[k]["v_only"][metric]),
                                              scale(metric, refs[k]["vfilter_plus_v"][metric]), V))
            for name, _ in CONFIGS:
                if not any(k + (name,) in runs for k in keys):
                    continue
                for dl, prefix, kind in DEPS:
                    def get(k, name=name, prefix=prefix, kind=kind):
                        r = runs.get(k + (name,))
                        return None if r is None else scale(metric, select(r["evals"], kind, SELECT, metric)[f"{prefix}_{metric}"])
                    row(f"{name} {dl}", get)
                    for k in keys:
                        v = get(k)
                        if v is not None:
                            overall[(ds, f"{name} {dl}")].append((v, scale(metric, refs[k]["v_only"][metric]),
                                                                  scale(metric, refs[k]["vfilter_plus_v"][metric]), V))
            for vname in ("ens_M", "ens_M_vf", "ens_M1_vf"):
                if not any(k in posthoc for k in keys):
                    continue
                for dl, field in (("hard", "hard"), ("hard + V", "hard_v")):
                    def getp(k, vname=vname, field=field):
                        return scale(metric, posthoc[k][vname][field][metric]) if k in posthoc else None
                    row(f"{vname} {dl}", getp)
                    for k in keys:
                        v = getp(k)
                        if v is not None:
                            overall[(ds, f"{vname} {dl}")].append((v, scale(metric, refs[k]["v_only"][metric]),
                                                                  scale(metric, refs[k]["vfilter_plus_v"][metric]), V))
            L.append("")

    L += ["## Overall: difference to V only and to V-filter + V", "",
          "Per dataset, over every draw run (both V sizes): mean difference (method minus reference; for log loss a "
          "negative difference is better) and the draws better / worse than the reference.", "",
          "| dataset | method | V = 50: vs V only | vs V-filter + V | V = 500: vs V only | vs V-filter + V |", "|---|---|---|---|---|---|"]
    for ds, metric, _ in DATASETS:
        for (d_, method), vals in overall.items():
            if d_ != ds or method in ("V only",):
                continue
            cells = []
            for V in VS:
                vv = [t for t in vals if t[3] == V]
                if not vv:
                    cells += ["", ""]
                    continue
                for j in (1, 2):
                    diff = np.mean([t[0] - t[j] for t in vv])
                    w = sum(better(metric, t[0], t[j]) for t in vv)
                    l_ = sum(better(metric, t[j], t[0]) for t in vv)
                    cells.append(f"{diff:+.{1 if metric in ('auc', 'acc') else 3}f} ({w}-{l_})")
            L.append(f"| {ds.split('_')[0]} | {method} | " + " | ".join(cells) + " |")
    L.append("")

    L += ["## Selection rules (hard + V; mean over all draws of the dataset)", "",
          "The test metric at the iteration chosen on the held-out 20% of V by: log loss, accuracy, ROC-AUC (binary only), "
          "the last iteration, and the best test value over all iterations (an upper bound, not a rule).", "",
          "| dataset | config | V | log loss | accuracy | ROC-AUC | last | best on test | chose iteration 0 | median iteration |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for ds, metric, _ in DATASETS:
        for name, _ in CONFIGS:
            for V in VS:
                rs = [runs[k] for k in runs if k[0] == ds and k[2] == V and k[4] == name]
                if not rs:
                    continue
                cells = []
                for rule in ("ce", "acc", "auc", "last"):
                    if rule == "auc" and metric != "auc":
                        cells.append("")
                        continue
                    cells.append(f"{np.mean([scale(metric, select(r['evals'], 'hard', rule, metric)[f'test_hard_v_{metric}']) for r in rs]):.{1 if metric in ('auc', 'acc') else 3}f}")
                best = [(max if metric in ("auc", "acc") else min)(scale(metric, e[f"test_hard_v_{metric}"]) for e in r["evals"]) for r in rs]
                its = [select(r["evals"], "hard", "ce", metric)["it"] for r in rs]
                L.append(f"| {ds.split('_')[0]} | {name} | {V} | " + " | ".join(cells)
                         + f" | {np.mean(best):.{1 if metric in ('auc', 'acc') else 3}f} | {sum(i == 0 for i in its)}/{len(its)} | {int(np.median(its))} |")
    text = "\n".join(L) + "\n"
    out = Path("results/tables/attn_learn")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"confirm{suffix}.md").write_text(text)
    print(text)
    plot(refs, runs, suffix)


def plot(refs: dict, runs: dict, suffix: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    series = (("V only", "#eda100", lambda k: refs[k]["v_only"]), ("poisoned + V", "#8a8985", lambda k: refs[k]["poisoned_plus_v"]),
              ("V-filter + V", "#2a78d6", lambda k: refs[k]["vfilter_plus_v"]),
              ("attention, M1 hard + V", "#eb6834", lambda k: _dep(runs, k, "M1_topk_lr0.2", "test_hard_v", "hard")),
              ("attention, M2 hard + V", "#1baf7a", lambda k: _dep(runs, k, "M2_adam_prior0.1", "test_hard_v", "hard")),
              ("oracle + V", "#2b2a28", lambda k: refs[k]["oracle_plus_v"]))
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.4), constrained_layout=True)
    for j, (ds, metric, title) in enumerate(DATASETS):
        for i, V in enumerate(VS):
            ax = axes[i, j]
            for label, color, fn in series:
                xs, ys = [], []
                for R in RS:
                    vals = []
                    for s in SEEDS:
                        k = (ds, R, V, s)
                        if k in refs:
                            m = fn(k)
                            if m is not None:
                                vals.append(scale(metric, m[metric]))
                    if vals:
                        xs.append(R)
                        ys.append(np.mean(vals))
                if xs:
                    ax.plot(xs, ys, color=color, lw=1.8, marker="o", ms=5, ls="--" if label == "oracle + V" else "-",
                            label=label if (i, j) == (0, 0) else None)
            ax.set_title(f"{title.split(' (')[0]}, V = {V}", loc="left", fontsize=10, color="#2b2a28")
            ax.set_xticks(list(RS), [f"{r:g}%" for r in RS])
            ax.set_ylabel({"auc": "test ROC-AUC (%)", "acc": "test accuracy (%)"}.get(metric, "test log loss (lower is better)"),
                          color="#2b2a28")
            ax.grid(color="#e4e3df", lw=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(colors="#6b6a66", labelsize=8.5)
            if i == 1:
                ax.set_xlabel("poisoned share of the context (R)", color="#2b2a28")
    fig.legend(loc="outside lower center", ncol=6, frameon=False, fontsize=9.5)
    fig.suptitle("Learned attention weights vs the references (mean of 3 seeds; iteration chosen by held-out log loss)",
                 x=0.01, ha="left", fontsize=11, color="#2b2a28")
    out = Path("results/plots/attn_learn")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"confirm{suffix}.png", dpi=150, facecolor="#fcfcfb")
    print(out / f"confirm{suffix}.png")


def _dep(runs, k, name, prefix, kind):
    r = runs.get(k + (name,))
    if r is None:
        return None
    e = select(r["evals"], kind, SELECT, "")
    return {m: e[f"{prefix}_{m}"] for m in ("acc", "ce", "bacc", "auc")}


if __name__ == "__main__":
    main()
