#!/usr/bin/env python
"""Report of the M1 regularisation study (scripts/m1_reg_study.py): trained long on all of V, does M1 beat its start,
which is exactly the V-filter (hard + V)?

Deployed hard + V on the test set. Per arm: the mean metrics at updates 0 (= the V-filter) / 50 / 100 / 200 / 300 and
the snapshot ensemble over 100-300 (test probabilities averaged), overall, per V regime and per dataset; wins (and
clear wins / clear losses: ROC-AUC or accuracy by 0.01, log loss by 0.02) against the V-filter and against V only at
100 and 300; what the removed set did (precision, overlap with the V-filter's, swaps). A figure of the trajectories.

    python scripts/report_m1_reg.py [--out results/tables/m1_reg] [--plots results/plots/m1_reg]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import m1_reg_study as M  # noqa: E402

METRICS = (("auc", "ROC-AUC", True, 0.01), ("acc", "accuracy", True, 0.01), ("ce", "log loss", False, 0.02))
AT = (0, 50, 100, 200, 300)
SHORT = {"heloc": "heloc", "students_dropout": "students", "is_this_a_good_customer": "good_customer",
         "gas-drift": "gas-drift", "ecommerce_shipping": "ecommerce"}


def ds_of(tag: str) -> str:
    return tag.split("_R30")[0].split("_", 1)[1]


def regime_of(tag: str) -> str:
    return "5 per class" if tag.startswith("tiny") else "V = " + tag.rsplit("_", 2)[1][1:]


def load() -> tuple[dict, dict]:
    import attn_learn as A
    refs, runs = {}, {}
    for f in sorted((M.ROOT / "runs").glob("*/*.json")):
        if f.parent.name not in M.instances():
            continue
        r = json.loads(f.read_text())
        if f.stem == "refs":
            refs[f.parent.name] = r
            continue
        z = np.load(f.with_suffix(".npz"))
        its = z["its"].tolist()
        sel = [i for i, it in enumerate(its) if 100 <= it <= 300]
        ens = A.metrics(z["probs_v"][sel].mean(0), z["yt"]) if sel else None
        runs[(f.parent.name, f.stem)] = {"curve": {e["it"]: e for e in r["curve"]}, "ens": ens}
    return refs, runs


def table(md, refs, runs, arms, pick, title):
    tags_all = sorted({t for (t, _) in runs if pick(t) and t in refs})
    if not tags_all:
        return
    md += [f"## {title} ({len(tags_all)} instances)", ""]
    for m, mt, hi, marg in METRICS:
        md += [f"### {mt}", "", "| arm | " + " | ".join(f"update {it}" for it in AT) + " | ens 100-300 | vs V-filter at 100 | "
               "vs V-filter at 300 | vs V only at 300 |", "|---|" + "---|" * (len(AT) + 4)]
        md.append("| (V only) | " + " | ".join(f"{np.mean([refs[t]['rows']['v_only'][m] for t in tags_all]):.4f}"
                                              for _ in AT) + " | | | | |")
        for arm in arms:
            tags = [t for t in tags_all if (t, arm) in runs]
            if not tags:
                continue
            cells = []
            for it in AT:
                v = [runs[(t, arm)]["curve"].get(it, {}).get("test_v", {}).get(m) for t in tags]
                cells.append(f"{np.mean(v):.4f}" if all(x is not None for x in v) else ".")
            ens = [runs[(t, arm)]["ens"] for t in tags]
            cells.append(f"{np.mean([e[m] for e in ens]):.4f}" if all(e is not None for e in ens) else ".")

            def wins(it, ref):
                d = []
                for t in tags:
                    e = runs[(t, arm)]["curve"].get(it)
                    if e is None:
                        return "."
                    rv = refs[t]["rows"][ref][m]
                    d.append((e["test_v"][m] - rv) if hi else (rv - e["test_v"][m]))
                d = np.array(d)
                return f"{(d > 1e-9).sum()} / {(d > marg).sum()} / {(d < -marg).sum()}"

            cells += [wins(100, "vfilter_hard_v"), wins(300, "vfilter_hard_v"), wins(300, "v_only")]
            md.append(f"| {arm} | " + " | ".join(cells) + " |")
        md.append("")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("results/tables/m1_reg"))
    ap.add_argument("--plots", type=Path, default=Path("results/plots/m1_reg"))
    a = ap.parse_args(argv)
    refs, runs = load()
    arms = [arm for tier in M.TIERS for arm in tier if any(k[1] == arm for k in runs)]
    a.out.mkdir(parents=True, exist_ok=True)
    md = ["# M1 trained on all of V: regularisation (R = 30%, development draws, seed 1)", "",
          "Deployed hard + V on the test set (the k = R% lowest-scored rows removed, all of V added); update 0 is exactly "
          "the V-filter. Wins: instances where the arm beats the reference (clear: ROC-AUC or accuracy by 0.01, log loss "
          "by 0.02), as wins / clear wins / clear losses. ens 100-300: test probabilities averaged over updates 100-300.",
          "", "## References (mean over instances; ROC-AUC / accuracy / log loss)", "",
          "| dataset | instances | V only | poisoned + V | V-filter hard + V (= update 0) | oracle + V |", "|---|---|---|---|---|---|"]
    for ds in [None, *M.DATA]:
        tags = [t for t in refs if ds is None or ds_of(t) == ds]
        if tags:
            cell = lambda name: " / ".join(f"{np.mean([refs[t]['rows'][name][m] for t in tags]):.3f}" for m, _, _, _ in METRICS)  # noqa: E731
            md.append(f"| {'all' if ds is None else SHORT[ds]} | {len(tags)} | {cell('v_only')} | {cell('poisoned_plus_v')} | "
                      f"{cell('vfilter_hard_v')} | {cell('oracle_plus_v')} |")
    md.append("")
    table(md, refs, runs, arms, lambda t: True, "All instances")
    for reg in ("V = 50", "V = 100", "5 per class"):
        table(md, refs, runs, arms, lambda t, reg=reg: regime_of(t) == reg, reg)
    for ds in M.DATA:
        table(md, refs, runs, arms, lambda t, ds=ds: ds_of(t) == ds, SHORT[ds])
    md += ["## What the removed set did (means over all instances)", "",
           "| arm | precision at 0 | precision at 300 | shared with the V-filter's at 300 | rows that left the removed set "
           "(sum over the run, share of k) | V's log loss at 0 / 300 (objective) |", "|---|---|---|---|---|---|"]
    for arm in arms:
        es = [(runs[(t, aa)]["curve"][0], runs[(t, aa)]["curve"].get(300), refs[t]["k"]) for (t, aa) in runs if aa == arm]
        es = [x for x in es if x[1] is not None]
        if es:
            md.append(f"| {arm} | {np.mean([a['prec'] for a, _, _ in es]):.3f} | {np.mean([b['prec'] for _, b, _ in es]):.3f} | "
                      f"{np.mean([b['overlap_vfilter'] for _, b, _ in es]):.3f} | {np.mean([b['swaps'] / k for _, b, k in es]):.2f} | "
                      f"{np.mean([a['train_ce'] for a, _, _ in es]):.3f} / {np.mean([b['train_ce'] for _, b, _ in es]):.3f} |")
    (a.out / "report.md").write_text("\n".join(md) + "\n")
    plot(runs, refs, arms, a.plots)
    print(f"-> {a.out}/report.md, {a.plots}/curves.png ({len(runs)} runs, {len(refs)} instances)")


def plot(runs, refs, arms, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    palette = ["#0b0b0b", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#8a5cd1"]
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8), facecolor="#fcfcfb")
    for ax, (m, title, hi, _) in zip(axes, METRICS):
        for i, arm in enumerate(arms):
            tags = [t for (t, aa) in runs if aa == arm and t in refs]
            its = [it for it in M.CKPTS if all(it in runs[(t, arm)]["curve"] for t in tags)]
            ys = [np.mean([runs[(t, arm)]["curve"][it]["test_v"][m] for t in tags]) for it in its]
            ax.plot(its, ys, color=palette[i % len(palette)], marker="o", markersize=4, linewidth=2.6 if arm == "m1" else 1.8,
                    label=arm + (" (control)" if arm == "m1" else ""))
        tags = list(refs)
        for name, style, lab in (("vfilter_hard_v", "--", "V-filter hard + V (= update 0)"), ("v_only", ":", "V only")):
            ax.axhline(np.mean([refs[t]["rows"][name][m] for t in tags]), color="#52514e", linestyle=style, linewidth=1.2,
                       label=lab)
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xlim(0, max(M.CKPTS) * 1.1)
        ax.set_title(title + (" (lower is better)" if not hi else ""), loc="left", fontsize=11)
        ax.set_xlabel("update")
        ax.grid(color="#e4e3df", linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_facecolor("#fcfcfb")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("M1 trained on all of V at lr 0.2, regularised (R = 30%; mean over instances; test, hard + V)", fontsize=13, y=0.99)
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=140, facecolor=fig.get_facecolor())
    fig.savefig(out / "curves.pdf", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
