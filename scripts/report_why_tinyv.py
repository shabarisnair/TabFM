#!/usr/bin/env python
"""Tables for scripts/why_tinyv.py: results/why_tinyv/<attack>/<instance>/{judge,grad,landscape,trace}.json ->
results/tables/why_tinyv/tables.md (also printed). Means over the instances present; per attack.

    python scripts/report_why_tinyv.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path("results/why_tinyv")
TABLES = Path("results/tables/why_tinyv")
ATTACKS = ("hub-flip-local", "hub-flip-random", "rand-flip")


def load(name: str) -> dict:
    """attack -> {instance: result}."""
    res = defaultdict(dict)
    for f in sorted(OUT.glob(f"*/*/{name}.json")):
        r = json.loads(f.read_text())
        res[r["attack"]][r["instance"]] = r
    return res


def c_of(tag: str) -> int:
    return int(tag.rsplit("_c", 1)[1].split("_")[0])


def mean(xs) -> float:
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


def fmt(x, nd=2) -> str:
    return "." if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def judge_tables(lines: list[str]) -> None:
    J = load("judge")
    if not J:
        return
    lines += ["## E1. Which judge ranks the poison? (AUROC of -log p(given label); 0.5 = chance, < 0.5 = inverted)", ""]
    for attack in ATTACKS:
        if attack not in J:
            continue
        rs = J[attack]
        names = list(next(iter(rs.values()))["judges"])
        lines += [f"### {attack} ({len(rs)} instances)", "",
                  "| judge | AUROC, V = 1 / 3 / 5 per class | AUROC, all | precision of its k least plausible | "
                  "median p(given): poisoned / clean |", "|---|---|---|---|---|"]
        for nm in names:
            by_c = [mean([r["judges"][nm]["auroc"] for t, r in rs.items() if c_of(t) == c and nm in r["judges"]]) for c in (1, 3, 5)]
            allv = [r["judges"][nm] for r in rs.values() if nm in r["judges"]]
            lines.append(f"| {nm} | {' / '.join(fmt(x) for x in by_c)} | {fmt(mean([v['auroc'] for v in allv]))} | "
                         f"{fmt(mean([v['prec_at_k'] for v in allv]))} | {fmt(mean([v['p_poisoned_median'] for v in allv]))} / "
                         f"{fmt(mean([v['p_clean_median'] for v in allv]))} |")
        ep = len(next(iter(rs.values()))["mlp"])
        lines += ["", f"Meta-Sift's MLP, plain training on the poisoned rows + V: per-sample loss as the poison ranker",
                  "", "| epoch | AUROC | precision@k | mean loss poisoned / clean | fits its given label: poisoned / clean |",
                  "|---|---|---|---|---|"]
        for e in range(ep):
            v = [r["mlp"][e] for r in rs.values()]
            lines.append(f"| {e + 1} | {fmt(mean([x['auroc'] for x in v]))} | {fmt(mean([x['prec_at_k'] for x in v]))} | "
                         f"{fmt(mean([x['loss_poisoned'] for x in v]))} / {fmt(mean([x['loss_clean'] for x in v]))} | "
                         f"{fmt(mean([x['fits_given_poisoned'] for x in v]))} / {fmt(mean([x['fits_given_clean'] for x in v]))} |")
        lines += ["", "Local structure (10 nearest neighbours in z-scored features, among the poisoned set):", ""]
        for key in next(iter(rs.values()))["structure"]:
            lines.append(f"- {key}: {fmt(mean([r['structure'][key] for r in rs.values()]))}")
        meth = defaultdict(list)
        for r in rs.values():
            for m, v in r.get("methods", {}).items():
                meth[m].append(v)
        if meth:
            lines += ["", "Each method's final ranking / removal:", "", "| method | instances | AUROC | precision of its removal |",
                      "|---|---|---|---|"]
            for m, vs in meth.items():
                lines.append(f"| {m} | {len(vs)} | {fmt(mean([v.get('auroc') for v in vs]))} | {fmt(mean([v.get('prec_at_k') for v in vs]))} |")
        lines.append("")


def grad_tables(lines: list[str]) -> None:
    G = load("grad")
    if not G:
        return
    lines += ["## E2. Is there a first-order signal? d loss / d log w as a poison ranker (AUROC; context: the poisoned rows)", ""]
    keys = [("auroc_g_v", "V's loss (M1's objective)"), ("auroc_g_test", "test loss (oracle objective)"),
            ("auroc_g_v_near10", "V's loss, rows nearest V (10%)"), ("auroc_g_v_far50", "V's loss, rows farthest from V (50%)"),
            ("auroc_g_test_far50", "test loss, rows farthest from V (50%)"), ("auroc_g_v_kept", "V's loss, kept rows only"),
            ("auroc_g_test_kept", "test loss, kept rows only"), ("spearman_gv_gt", "Spearman(V gradient, test gradient)"),
            ("top1pct_mass_gv", "share of |V gradient| in the top 1% of rows"),
            ("top1pct_mass_gt", "share of |test gradient| in the top 1% of rows"),
            ("spearman_absgv_negdist", "Spearman(|V gradient|, closeness to V)")]
    for attack in ATTACKS:
        if attack not in G:
            continue
        rs = G[attack]
        pts = list(next(iter(rs.values()))["points"])
        lines += [f"### {attack} ({len(rs)} instances)", "", "| quantity | " + " | ".join(pts) + " |", "|---|" + "---|" * len(pts)]
        for key, lab in keys:
            vals = [mean([r["points"][p].get(key) for r in rs.values()]) for p in pts]
            if all(not np.isfinite(v) for v in vals):
                continue
            lines.append(f"| {lab} | " + " | ".join(fmt(v) for v in vals) + " |")
        by_c = {c: mean([r["points"][pts[0]]["auroc_g_v"] for t, r in rs.items() if c_of(t) == c]) for c in (1, 3, 5)}
        lines += ["", f"AUROC of V's gradient at w = 1 by V size (1 / 3 / 5 per class): {fmt(by_c[1])} / {fmt(by_c[3])} / {fmt(by_c[5])}", ""]


def landscape_tables(lines: list[str]) -> None:
    from scipy.stats import spearmanr
    L = load("landscape")
    if not L:
        return
    lines += ["## E3. What do the objectives reward? Splits of known precision and every method's split", "",
              "V-loss: V's log loss with the kept rows alone as the context (M1's hard objective, V-selection's criterion; "
              "lower = preferred). Sift objective: the removed rows' loss with the kept rows + V as the context, times k / n "
              "(TabPFN-sift / Meta-Sift's split objective; higher = preferred). Test: kept + V.", ""]
    for attack in ATTACKS:
        if attack not in L:
            continue
        rs = L[attack]
        lines += [f"### {attack} ({len(rs)} instances)", "",
                  "| split | precision | V-loss | sift objective | test ROC-AUC | test accuracy | test log loss |", "|---|---|---|---|---|---|---|"]
        groups = defaultdict(list)
        for r in rs.values():
            for s in r["splits"]:
                key = (f"constructed p = {s['precision_target']:.1f}, {s['clean_choice']} clean rows" if s["kind"] == "constructed"
                       else f"method: {s['method']}")
                groups[key].append(s)
        nc = [r["no_cleaning"] for r in rs.values()]
        lines.append(f"| no cleaning | - | - | - | {fmt(mean([x['auc'] for x in nc]), 3)} | {fmt(100 * mean([x['acc'] for x in nc]), 1)} | "
                     f"{fmt(mean([x['ce'] for x in nc]), 3)} |")
        for key in sorted(groups, key=lambda s: (s.startswith("method"), s)):
            v = groups[key]
            lines.append(f"| {key} ({len(v)}) | {fmt(mean([x['precision'] for x in v]))} | {fmt(mean([x['v_ce'] for x in v]), 3)} | "
                         f"{fmt(mean([x['sift_objective'] for x in v]), 3)} | {fmt(mean([x['test']['auc'] for x in v]), 3)} | "
                         f"{fmt(100 * mean([x['test']['acc'] for x in v]), 1)} | {fmt(mean([x['test']['ce'] for x in v]), 3)} |")
        # does each objective prefer the oracle? and does it track precision?
        beat_v, beat_s, rho_v, rho_s, rho_t = [], [], [], [], []
        for r in rs.values():
            ora = next(s for s in r["splits"] if s.get("precision_target") == 1.0)
            others = [s for s in r["splits"] if s is not ora]
            beat_v.append(np.mean([s["v_ce"] < ora["v_ce"] for s in others]))
            beat_s.append(np.mean([s["sift_objective"] > ora["sift_objective"] for s in others]))
            rnd = [s for s in r["splits"] if s["kind"] == "constructed" and s["clean_choice"] == "random"] + [ora]
            pr = [s["precision"] for s in rnd]
            rho_v.append(spearmanr(pr, [-s["v_ce"] for s in rnd])[0])
            rho_s.append(spearmanr(pr, [s["sift_objective"] for s in rnd])[0])
            rho_t.append(spearmanr(pr, [s["test"]["auc"] for s in rnd])[0])
        vf_vs_or = [next((s["v_ce"] for s in r["splits"] if s.get("method") == "vfilter"), np.nan)
                    - next(s["v_ce"] for s in r["splits"] if s.get("precision_target") == 1.0) for r in rs.values()]
        lines += ["", f"- Share of the other splits with a LOWER V-loss than the oracle split (mean over instances): {fmt(mean(beat_v))}",
                  f"- Share of the other splits with a HIGHER sift objective than the oracle split: {fmt(mean(beat_s))}",
                  f"- V-filter split's V-loss minus the oracle split's (mean): {fmt(mean(vf_vs_or), 3)}; lower on "
                  f"{sum(1 for x in vf_vs_or if np.isfinite(x) and x < 0)} of {sum(1 for x in vf_vs_or if np.isfinite(x))} instances",
                  f"- Spearman with precision over the random-clean splits (mean): objective-preferred V-loss {fmt(mean(rho_v))}, "
                  f"sift objective {fmt(mean(rho_s))}, test ROC-AUC {fmt(mean(rho_t))}", ""]


def removed_tables(lines: list[str]) -> None:
    Rm = load("removed")
    if not Rm:
        return
    lines += ["## E4. Which clean rows does each method remove?", "",
              "Hard clean row: p(true label) < 0.5 under the clean rows + V as the context, leave-one-out (the oracle judge). "
              "Minority: the clean rows of the rarest true class (binary datasets).", ""]
    for attack in ATTACKS:
        if attack not in Rm:
            continue
        rs = Rm[attack]
        methods = list(next(iter(rs.values()))["methods"])
        binary = [r for r in rs.values() if r["K"] == 2]
        lines += [f"### {attack} ({len(rs)} instances; {len(binary)} binary)", "",
                  "| method | precision | removed clean rows: median p(true) / share hard | share of all hard clean rows removed | "
                  "share of the minority clean rows removed (binary) |", "|---|---|---|---|---|"]
        for m in methods:
            v = [r["methods"][m] for r in rs.values() if m in r["methods"]]
            vb = [r["methods"][m] for r in binary if m in r["methods"]]
            lines.append(f"| {m} | {fmt(mean([x['precision'] for x in v]))} | {fmt(mean([x['p_true_removed_clean_median'] for x in v]))} / "
                         f"{fmt(mean([x['removed_clean_hard_share'] for x in v]))} | {fmt(mean([x['clean_hard_removed_share'] for x in v]))} | "
                         f"{fmt(mean([x['minority_clean_removed_share'] for x in vb]))} |")
        lines.append("")


def influence_tables(lines: list[str]) -> None:
    In = load("influence")
    if not In:
        return
    lines += ["## E7. Real deletions: the change in test log loss (negative = the deletion helps)", ""]
    for attack in ATTACKS:
        if attack not in In:
            continue
        rs = In[attack]
        lines += [f"### {attack}", "", "| instance | one poisoned row | one clean row | AUROC of the single-deletion effect | "
                  "per row: its cluster / as many scattered poisoned rows (cluster size) |", "|---|---|---|---|---|"]
        for t, r in sorted(rs.items()):
            lines.append(f"| {t} | {mean(r['single_poisoned']):+.1e} | {mean(r['single_clean']):+.1e} | {fmt(r['auroc_single_deletion'])} | "
                         f"{mean(r['cluster_per_row']):+.1e} / {mean(r['scattered_per_row']):+.1e} ({mean(r['cluster_sizes']):.1f}) |")
        lines += ["", f"Mean AUROC of the single-deletion effect: {fmt(mean([r['auroc_single_deletion'] for r in rs.values()]))}", ""]


def trace_tables(lines: list[str]) -> None:
    T = load("trace")
    if not T:
        return
    lines += ["## E6. Greedy's acceptance order: the poisoned share of each tenth of the accepted rows (base rate 0.30)", ""]
    for attack in ATTACKS:
        if attack not in T:
            continue
        rs = T[attack]
        lines += [f"### {attack}", "", "| method | instances | " + " | ".join(str(i + 1) for i in range(10)) + " | removed precision |",
                  "|---|---|" + "---|" * 11]
        for m in ("greedy_m1", "greedy_pct1", "greedy_pct10"):
            vs = [r["methods"][m] for r in rs.values() if m in r["methods"]]
            if vs:
                tenths = np.mean([v["poisoned share by tenth of acceptance"] for v in vs], axis=0)
                lines.append(f"| {m} | {len(vs)} | " + " | ".join(fmt(x) for x in tenths) + f" | {fmt(mean([v['removed precision'] for v in vs]))} |")
        lines.append("")


def main(argv=None) -> None:
    lines = ["# Why the learned filters fail at tiny V: diagnostic tables (scripts/why_tinyv.py)", ""]
    judge_tables(lines)
    grad_tables(lines)
    landscape_tables(lines)
    removed_tables(lines)
    influence_tables(lines)
    trace_tables(lines)
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1:])
