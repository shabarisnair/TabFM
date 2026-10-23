#!/usr/bin/env python
"""Report of the M1 regularisation study (scripts/m1_reg_study.py): trained long on all of V, does M1 beat its start,
which is exactly the V-filter (hard + V)?

Deployed hard + V on the test set. Per arm, overall, per V regime and per dataset: the mean metrics at updates 0 (= the
V-filter) / 50 / 100 / 200 / 300 and the snapshot ensemble over 100-300 (test probabilities averaged), the paired change
from update 0 at 300 (mean +- standard error over the instances), and the wins (and clear wins / clear losses: ROC-AUC
or accuracy by 0.01, log loss by 0.02) against the V-filter and against V only. Then each arm against the unregularised
control on the same instances, the best checkpoint in hindsight (picked on the test set: an upper bound for any stopping
rule), what the removed set did (precision, overlap with the V-filter's, rows that left it), and the calibration of the
test predictions (confident errors). A figure of the paired changes along the updates.

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


def calibration(p: np.ndarray, y: np.ndarray) -> dict:
    """Confident errors (probability of the true class below 0.05), their share of the log loss, mean confidence."""
    pt = np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)
    nll = -np.log(pt)
    bad = pt < 0.05
    return {"conf_err": float(bad.mean()), "ce_share": float(nll[bad].sum() / max(nll.sum(), 1e-12)),
            "conf": float(p.max(1).mean())}


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
        cal = {it: calibration(z["probs_v"][its.index(it)], z["yt"]) for it in (0, 300) if it in its}
        runs[(f.parent.name, f.stem)] = {"curve": {e["it"]: e for e in r["curve"]}, "ens": ens, "cal": cal}
    return refs, runs


def paired(runs, tags, arm, m, it, other=None) -> np.ndarray:
    """Per instance: the arm at `it` minus update 0 (the V-filter), or minus arm `other` at `it`."""
    out = []
    for t in tags:
        c = runs[(t, arm)]["curve"]
        if it not in c or (other is not None and ((t, other) not in runs or it not in runs[(t, other)]["curve"])):
            continue
        base = c[0] if other is None else runs[(t, other)]["curve"][it]
        out.append(c[it]["test_v"][m] - base["test_v"][m])
    return np.array(out)


def mean_se(x: np.ndarray) -> str:
    return "." if len(x) == 0 else f"{x.mean():+.4f} +- {x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0:.4f}"


def table(md, refs, runs, arms, pick, title):
    tags_all = sorted({t for (t, _) in runs if pick(t) and t in refs})
    if not tags_all:
        return
    md += [f"## {title} ({len(tags_all)} instances)", ""]
    for m, mt, hi, marg in METRICS:
        md += [f"### {mt}", "", "| arm | n | " + " | ".join(f"update {it}" for it in AT) + " | ens 100-300 | change at 300 "
               "(paired) | vs V-filter at 100 | vs V-filter at 300 | vs V only at 300 |", "|---|---|" + "---|" * (len(AT) + 5)]
        md.append(f"| (V only) | {len(tags_all)} | " + " | ".join(
            f"{np.mean([refs[t]['rows']['v_only'][m] for t in tags_all]):.4f}" for _ in AT) + " | | | | | |")
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
            cells.append(mean_se(paired(runs, tags, arm, m, 300)))

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
            md.append(f"| {arm} | {len(tags)} | " + " | ".join(cells) + " |")
        md.append("")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("results/tables/m1_reg"))
    ap.add_argument("--plots", type=Path, default=Path("results/plots/m1_reg"))
    a = ap.parse_args(argv)
    refs, runs = load()
    arms = [arm for tier in M.TIERS for arm in tier if any(k[1] == arm for k in runs)]
    missing = [f"{t} {arm}" for t in M.instances() for arm in M.ARMS if (t, arm) not in runs]
    a.out.mkdir(parents=True, exist_ok=True)
    md = ["# M1 trained on all of V: regularisation (R = 30%, development draws, seed 1)", "",
          "Deployed hard + V on the test set (the k = R% lowest-scored rows removed, all of V added); update 0 is exactly "
          "the V-filter. n: instances the arm finished. change at 300: the arm at update 300 minus update 0 on the same "
          "instance, mean +- standard error. Wins: instances where the arm beats the reference (clear: ROC-AUC or "
          "accuracy by 0.01, log loss by 0.02), as wins / clear wins / clear losses. ens 100-300: test probabilities "
          "averaged over updates 100-300.", "",
          f"Runs: {len(runs)} of {len(M.instances()) * len(M.ARMS)}" + (f"; missing: {', '.join(missing)}" if missing else ""), "",
          "## References (mean over instances; ROC-AUC / accuracy / log loss)", "",
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
    md += ["## Each arm against the control (m1), same instances, at update 300 (paired mean +- SE; wins-losses)", "",
           "| arm | n | ROC-AUC | accuracy | log loss (lower is better) |", "|---|---|---|---|---|"]
    tags = list(refs)
    for arm in arms[1:]:
        cells = []
        for m, _, hi, _ in METRICS:
            x = paired(runs, [t for t in tags if (t, arm) in runs], arm, m, 300, other="m1")
            s = 1 if hi else -1
            cells.append(f"{mean_se(x)} ({(s * x > 1e-9).sum()}-{(s * x < -1e-9).sum()})")
        md.append(f"| {arm} | {sum((t, arm) in runs for t in tags)} | " + " | ".join(cells) + " |")
    md += ["", "## Best checkpoint in hindsight (picked on the test set per instance; an upper bound for any stopping rule)",
           "", "Change from update 0, mean over the arm's instances.", "", "| arm | ROC-AUC | accuracy | log loss |",
           "|---|---|---|---|"]
    for arm in arms:
        cells = []
        for m, _, hi, _ in METRICS:
            s = 1 if hi else -1
            x = [max(s * (e["test_v"][m] - c[0]["test_v"][m]) for e in c.values())
                 for (t, aa), r in runs.items() if aa == arm for c in [r["curve"]]]
            cells.append(f"{s * np.mean(x):+.4f}")
        md.append(f"| {arm} | " + " | ".join(cells) + " |")
    md += ["", "## What the removed set did (means over all instances)", "",
           "| arm | precision at 0 | precision at 300 | shared with the V-filter's at 300 | rows that left the removed set "
           "(sum over the run, share of k) | V's log loss at 0 / 300 (objective: relaxed point) | V's log loss, hard context, "
           "0 / 300 |", "|---|---|---|---|---|---|---|"]
    for arm in arms:
        es = [(r["curve"][0], r["curve"].get(300), refs[t]["k"]) for (t, aa), r in runs.items() if aa == arm]
        es = [x for x in es if x[1] is not None]
        if es:
            md.append(f"| {arm} | {np.mean([a['prec'] for a, _, _ in es]):.3f} | {np.mean([b['prec'] for _, b, _ in es]):.3f} | "
                      f"{np.mean([b['overlap_vfilter'] for _, b, _ in es]):.3f} | {np.mean([b['swaps'] / k for _, b, k in es]):.2f} | "
                      f"{np.mean([a['train_ce'] for a, _, _ in es]):.3f} / {np.mean([b['train_ce'] for _, b, _ in es]):.3f} | "
                      f"{np.mean([a['train_ce_hard'] for a, _, _ in es]):.3f} / {np.mean([b['train_ce_hard'] for _, b, _ in es]):.3f} |")
    x = [(r["curve"][300]["prec"] - r["curve"][0]["prec"], r["curve"][300]["test_v"]["auc"] - r["curve"][0]["test_v"]["auc"],
          r["curve"][300]["test_v"]["acc"] - r["curve"][0]["test_v"]["acc"]) for r in runs.values() if 300 in r["curve"]]
    x = np.array(x)
    md += ["", f"Across all {len(x)} runs, the change in precision from update 0 to 300 correlates with the change in test "
           f"ROC-AUC at r = {np.corrcoef(x[:, 0], x[:, 1])[0, 1]:+.2f} and in accuracy at r = {np.corrcoef(x[:, 0], x[:, 2])[0, 1]:+.2f}.",
           "", "## Calibration of the test predictions (hard + V; means over the arm's instances)", "",
           "Confident errors: test rows whose true class gets probability below 0.05; their share of the log loss.", "",
           "| arm | confident errors at 0 / 300 | their share of the log loss at 0 / 300 | mean top probability at 0 / 300 |",
           "|---|---|---|---|"]
    for arm in arms:
        cs = [r["cal"] for (t, aa), r in runs.items() if aa == arm and 0 in r["cal"] and 300 in r["cal"]]
        if cs:
            md.append(f"| {arm} | " + " | ".join(f"{np.mean([c[0][k] for c in cs]):.3f} / {np.mean([c[300][k] for c in cs]):.3f}"
                                                   for k in ("conf_err", "ce_share", "conf")) + " |")
    (a.out / "report.md").write_text("\n".join(md) + "\n")
    plot(runs, refs, arms, a.plots)
    print(f"-> {a.out}/report.md, {a.plots}/curves.png ({len(runs)} runs, {len(refs)} instances)")


def plot(runs, refs, arms, out: Path) -> None:
    """The paired change from update 0 (the V-filter) along the updates, mean over each arm's instances; V only's
    difference from the V-filter as a reference line."""
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
            ys = [np.mean(paired(runs, tags, arm, m, it)) for it in its]
            ax.plot(its, ys, color=palette[i % len(palette)], marker="o", markersize=4, linewidth=2.6 if arm == "m1" else 1.8,
                    label=arm + (" (control)" if arm == "m1" else ""))
        tags = list(refs)
        ax.axhline(0, color="#52514e", linestyle="--", linewidth=1.2, label="V-filter hard + V (= update 0)")
        ax.axhline(np.mean([refs[t]["rows"]["v_only"][m] - refs[t]["rows"]["vfilter_hard_v"][m] for t in tags]),
                   color="#52514e", linestyle=":", linewidth=1.2, label="V only")
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xlim(0, max(M.CKPTS) * 1.1)
        ax.set_title(f"{title}: change from the V-filter" + (" (lower is better)" if not hi else ""), loc="left", fontsize=11)
        ax.set_xlabel("update")
        ax.grid(color="#e4e3df", linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_facecolor("#fcfcfb")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("M1 trained on all of V at lr 0.2, regularised (R = 30%; paired mean over instances; test, hard + V)",
                 fontsize=13, y=0.99)
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=140, facecolor=fig.get_facecolor())
    fig.savefig(out / "curves.pdf", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
