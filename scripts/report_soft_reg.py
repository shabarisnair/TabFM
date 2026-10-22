#!/usr/bin/env python
"""Report of the soft-form regularisation study (scripts/soft_reg_study.py): does long training on all of V, with a
lower learning rate or a regulariser, beat the V-filter start, and the benchmarks' hard V-filter?

Per arm, deployed soft + V on the test set (mean over the instances, overall and per V regime): the metrics at the
start (update 0) and at updates 50 / 100 / 200 / 300, and the wins (and clear wins: ROC-AUC or accuracy by 0.01, log
loss by 0.02) against the start and against the V-filter hard + V on the same instance. A snapshot ensemble (the test
probabilities averaged over updates 100-300) is scored from the saved probabilities. Then what the weights did at 300.
A figure: the mean test metrics against the update, the learning-rate arms and the regulariser arms side by side.

    python scripts/report_soft_reg.py [--out results/tables/soft_reg] [--plots results/plots/soft_reg]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import soft_reg_study as SR  # noqa: E402

METRICS = (("auc", "ROC-AUC", True, 0.01), ("acc", "accuracy", True, 0.01), ("ce", "log loss", False, 0.02))
REGIME = {"V50": "V = 50", "V100": "V = 100", "c5": "V = 5 per class"}
AT = (0, 50, 100, 200, 300)


def regime_of(tag: str) -> str:
    return "c5" if tag.startswith("tiny") else tag.rsplit("_", 2)[1]


def load() -> tuple[dict, dict]:
    refs, runs = {}, {}
    for f in sorted((SR.ROOT / "runs").glob("*/*.json")):
        r = json.loads(f.read_text())
        if f.stem == "refs":
            refs[f.parent.name] = r
        else:
            z = np.load(f.with_suffix(".npz"))
            runs[(f.parent.name, f.stem)] = {"curve": {e["it"]: e for e in r["curve"]}, "probs": z["probs_v"],
                                            "its": z["its"].tolist(), "yt": z["yt"]}
    return refs, runs


def ensemble_metrics(run: dict, lo: int = 100, hi: int = 300) -> dict | None:
    import attn_learn as A
    sel = [i for i, it in enumerate(run["its"]) if lo <= it <= hi]
    if not sel:
        return None
    return A.metrics(run["probs"][sel].mean(0), run["yt"])


def value(run, it, m):
    e = run["curve"].get(it)
    return None if e is None else e["test_v"][m]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("results/tables/soft_reg"))
    ap.add_argument("--plots", type=Path, default=Path("results/plots/soft_reg"))
    a = ap.parse_args(argv)
    refs, runs = load()
    arms = [arm for tier in SR.TIERS for arm in tier if any(k[1] == arm for k in runs)]
    a.out.mkdir(parents=True, exist_ok=True)
    md = ["# Soft form on all of V: learning rate and regularisation (R = 30%, development draws, seed 1)", "",
          "Deployed soft + V on the test set; mean over the instances an arm finished. start = update 0 (the V-filter "
          "soft weights); V-filter = the benchmarks' V-filter, hard + V (its k lowest rows removed). Wins: instances "
          "where the arm at that update beats the reference (clear: ROC-AUC or accuracy by 0.01, log loss by 0.02), as "
          "wins / clear wins / clear losses. ens 100-300: the test probabilities averaged over updates 100-300.", ""]
    # references
    md += ["## References (mean over instances)", "", "| regime | instances | V only | poisoned + V | V-filter hard + V | "
           "start (soft + V) | oracle + V |", "|---|---|---|---|---|---|---|"]
    for reg in (None, *REGIME):
        tags = [t for t in refs if reg is None or regime_of(t) == reg]
        if not tags:
            continue
        cell = lambda name: " / ".join(f"{np.mean([refs[t]['rows'][name][m] for t in tags]):.3f}" for m, _, _, _ in METRICS)  # noqa: E731
        md.append(f"| {'all' if reg is None else REGIME[reg]} | {len(tags)} | {cell('v_only')} | {cell('poisoned_plus_v')} | "
                  f"{cell('vfilter_hard_v')} | {cell('start_soft_v')} | {cell('oracle_plus_v')} |")
    md += ["", "(each cell: ROC-AUC / accuracy / log loss)", ""]
    for reg in (None, *REGIME):
        md += [f"## {'All instances' if reg is None else REGIME[reg]}", ""]
        for m, title, hi, marg in METRICS:
            md += [f"### {title}", "", "| arm | instances | " + " | ".join(f"update {it}" for it in AT) + " | ens 100-300 | "
                   "vs start at 300 | vs V-filter at 100 | vs V-filter at 300 |", "|---|---|" + "---|" * (len(AT) + 4)]
            for arm in arms:
                tags = [t for (t, aa) in runs if aa == arm and (reg is None or regime_of(t) == reg) and t in refs]
                if not tags:
                    continue
                cells = []
                for it in AT:
                    v = [value(runs[(t, arm)], it, m) for t in tags]
                    cells.append(f"{np.mean(v):.4f}" if all(x is not None for x in v) else ".")
                ens = [ensemble_metrics(runs[(t, arm)]) for t in tags]
                cells.append(f"{np.mean([e[m] for e in ens]):.4f}" if all(e is not None for e in ens) else ".")

                def wins(it, ref):
                    d = []
                    for t in tags:
                        v = value(runs[(t, arm)], it, m)
                        r = refs[t]["rows"][ref][m]
                        if v is None:
                            return "."
                        d.append((v - r) if hi else (r - v))
                    d = np.array(d)
                    return f"{(d > 1e-9).sum()} / {(d > marg).sum()} / {(d < -marg).sum()}"

                cells += [wins(300, "start_soft_v"), wins(100, "vfilter_hard_v"), wins(300, "vfilter_hard_v")]
                md.append(f"| {arm} | {len(tags)} | " + " | ".join(cells) + " |")
            md.append("")
    # what the weights did at the last update
    md += ["## Form against form: log w minus w, same regulariser, at update 300 (soft + V; + = log better)", "",
           "| regulariser | instances | ROC-AUC | accuracy | log loss |", "|---|---|---|---|---|"]
    for reg in SR.REGS:
        aw, al = SR.arm_name("w", reg), SR.arm_name("log", reg)
        tags = [t for (t, aa) in runs if aa == aw and (t, al) in runs]
        if not tags:
            continue
        cells = []
        for m, _, hi, _ in METRICS:
            d = np.array([(value(runs[(t, al)], 300, m) or np.nan) - (value(runs[(t, aw)], 300, m) or np.nan) for t in tags])
            d = d if hi else -d
            cells.append(f"{np.nanmean(d):+.4f} ({(d > 1e-9).sum()}-{(d < -1e-9).sum()})")
        md.append(f"| {reg.strip('_') or 'none (control)'} | {len(tags)} | " + " | ".join(cells) + " |")
    md += ["", "## What the weights did at update 300 (means over all instances)", "",
           "| arm | rows at w = 0 | their precision | rows at log w <= -10 | their precision | mean abs(w - w0) | "
           "rows moved > 0.1 | weights' AUROC vs poison | log loss of V |", "|---|---|---|---|---|---|---|---|---|"]
    for arm in arms:
        es = [runs[(t, aa)]["curve"].get(300) for (t, aa) in runs if aa == arm]
        es = [e for e in es if e is not None]
        if not es:
            continue
        pz = [e["prec_zero"] for e in es if e["prec_zero"] is not None]
        pl = [e["prec_le10"] for e in es if e.get("prec_le10") is not None]
        md.append(f"| {arm} | {np.mean([e['n_zero'] for e in es]):.0f} | {np.mean(pz) if pz else float('nan'):.3f} | "
                  f"{np.mean([e.get('n_le10', 0) for e in es]):.0f} | {np.mean(pl) if pl else float('nan'):.3f} | "
                  f"{np.mean([e['dev_mean'] for e in es]):.4f} | {np.mean([e['n_moved'] for e in es]):.0f} | "
                  f"{np.mean([e['auroc'] for e in es]):.3f} | {np.mean([e['train_ce'] for e in es]):.3f} |")
    (a.out / "report.md").write_text("\n".join(md) + "\n")
    plot(runs, refs, arms, a.plots)
    print(f"-> {a.out}/report.md, {a.plots}/curves.png ({len(runs)} runs, {len(refs)} instances)")


def plot(runs: dict, refs: dict, arms: list[str], out: Path) -> None:
    """Mean test metrics (soft + V) against the update: a row per form (w, log w), a line per regulariser (the form's
    unregularised control in black), with the V-filter hard + V and V only as reference lines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    palette = ["#0b0b0b", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#8a5cd1"]
    out.mkdir(parents=True, exist_ok=True)
    forms = [f for f in SR.FORMS if any(a.startswith(f + "_") for a in arms)]
    fig, axes = plt.subplots(len(forms), 3, figsize=(19, 5.6 * len(forms)), facecolor="#fcfcfb", squeeze=False)
    for row, form in enumerate(forms):
        for col, (m, title, hi, _) in enumerate(METRICS):
            ax = axes[row, col]
            for i, reg in enumerate(SR.REGS):
                arm = SR.arm_name(form, reg)
                tags = [t for (t, aa) in runs if aa == arm and t in refs]
                if not tags:
                    continue
                its = [it for it in SR.CKPTS if all(it in runs[(t, arm)]["curve"] for t in tags)]
                ys = [np.mean([runs[(t, arm)]["curve"][it]["test_v"][m] for t in tags]) for it in its]
                ax.plot(its, ys, color=palette[i % len(palette)], marker="o", markersize=4, linewidth=2.6 if not reg else 1.8,
                        label=(reg.strip("_") or "no regularisation (control)"))
            tags = list(refs)
            for name, style, lab in (("vfilter_hard_v", "--", "V-filter hard + V"), ("v_only", ":", "V only")):
                ax.axhline(np.mean([refs[t]["rows"][name][m] for t in tags]), color="#52514e", linestyle=style,
                           linewidth=1.2, label=lab)
            ax.set_xscale("symlog", linthresh=10)
            ax.set_xlim(0, max(SR.CKPTS) * 1.1)
            ax.set_title(f"{'w in [0, 1], lr 0.05' if form == 'w' else 'log w, lr 0.0625'}: {title}"
                         + (" (lower is better)" if not hi else ""), loc="left", fontsize=11)
            ax.set_xlabel("update")
            ax.grid(color="#e4e3df", linewidth=0.8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.set_facecolor("#fcfcfb")
        axes[row, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Soft weights trained on all of V, regularised (R = 30%; mean over instances; test, soft + V)",
                 fontsize=13, y=0.995)
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=140, facecolor=fig.get_facecolor())
    fig.savefig(out / "curves.pdf", facecolor=fig.get_facecolor())
    plt.close(fig)

if __name__ == "__main__":
    main()
