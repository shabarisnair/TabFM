#!/usr/bin/env python
"""Scores the candidate rules of scripts/vp_tune.py on the development draws: the V-filter's removal thresholds, greedy's
stopping rules and the baselines' two forms.

Each candidate's test ROC-AUC, balanced accuracy and log loss on every development draw; the headline is the mean rank
among the candidates of its family (per draw and metric, 1 = best; averaged over the three metrics and the 72 draws),
with means of the metrics, the share of P removed, and the split by poisoning and by V size.

    python scripts/vp_tune_report.py [--out results/tables/vp_tune/report.md]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path("results/vp_tune")
MET = (("auc", True), ("bacc", True), ("ce", False))


def load(stage: str) -> dict:
    return {f.stem: json.loads(f.read_text()) for f in sorted((ROOT / stage).glob("*.json"))}


def attack_of(tag: str) -> str:
    return "rand45" if "_rand45_" in tag else "hub30"


def c_of(tag: str) -> int:
    return int(tag.split("_c")[-1].split("_")[0])


def rank_table(cands: dict[str, dict[str, dict]], tags: list[str]) -> dict[str, dict]:
    """cands: name -> tag -> test metrics (or None). Mean rank over metrics and tags (ties: average), and means."""
    names = [n for n in cands if all(cands[n].get(t) is not None for t in tags)]
    out = {n: {"ranks": []} for n in names}
    for t in tags:
        for m, hi in MET:
            vals = np.array([cands[n][t][m] for n in names])
            r = rankdata(-vals if hi else vals)
            for n, x in zip(names, r):
                out[n]["ranks"].append(x)
    for n in names:
        out[n]["rank"] = float(np.mean(out[n]["ranks"]))
        for m, _ in MET:
            out[n][m] = float(np.mean([cands[n][t][m] for t in tags]))
    return out


def fmt_rows(tab: dict, extra: dict | None = None, top: int | None = None, keep: tuple = ()) -> list[str]:
    rows = sorted(tab.items(), key=lambda kv: kv[1]["rank"])
    if top:
        rows = [r for i, r in enumerate(rows) if i < top or r[0] in keep]
    lines = ["| candidate | mean rank | ROC-AUC | bal. acc. | log loss |" + ("".join(f" {c} |" for c in extra["cols"]) if extra else ""),
             "|---|---|---|---|---|" + ("---|" * len(extra["cols"]) if extra else "")]
    for n, v in rows:
        ex = "".join(f" {x} |" for x in extra["fn"](n)) if extra else ""
        lines.append(f"| {n} | {v['rank']:.2f} | {v['auc']:.3f} | {100 * v['bacc']:.1f} | {v['ce']:.3f} |{ex}")
    return lines


def vfilter_section(lines: list[str]) -> None:
    res = load("vfilter")
    tags = sorted(res)
    cands = {"V only": {t: res[t]["refs"]["v_only"] for t in tags}, "V + P": {t: res[t]["refs"]["v_plus_p"] for t in tags}}
    for name in res[tags[0]]["rules"]:
        cands[name] = {t: res[t]["rules"][name]["test"] for t in tags}
    share = {n: np.mean([res[t]["rules"][n]["share_removed"] for t in tags]) for n in res[tags[0]]["rules"]}
    prec = {n: np.nanmean([res[t]["rules"][n]["prec"] if res[t]["rules"][n]["prec"] is not None else np.nan for t in tags])
            for n in res[tags[0]]["rules"]}
    extra = {"cols": ["removed", "precision"], "fn": lambda n: (f"{share[n]:.2f}", f"{prec[n]:.2f}") if n in share else ("-", "-")}
    lines += ["## V-filter: removal rules (all 72 draws)", ""] + fmt_rows(rank_table(cands, tags), extra, top=16,
                                                                      keep=("R", "V only", "V + P")) + [""]
    for att in ("rand45", "hub30"):
        sub = [t for t in tags if attack_of(t) == att]
        lines += [f"### {att} only ({len(sub)} draws)", ""] + fmt_rows(rank_table(cands, sub), extra, top=10,
                                                                       keep=("R", "V only", "V + P")) + [""]
    for c in (1, 3, 5):
        sub = [t for t in tags if c_of(t) == c]
        tab = rank_table(cands, sub)
        best = sorted(tab.items(), key=lambda kv: kv[1]["rank"])[:5]
        lines.append(f"- c = {c}: best " + ", ".join(f"{n} ({v['rank']:.2f})" for n, v in best)
                     + f"; R {tab['R']['rank']:.2f}, V only {tab['V only']['rank']:.2f}, V + P {tab['V + P']['rank']:.2f}")
    lines.append("")


STATS = {  # stat -> (where, higher means keep accepting, thresholds)
    "p_mean": ("batch", True, (0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5)),
    "p_min": ("batch", True, (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)),
    "p_chance": ("batch", True, (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2)),
    "agree": ("batch", True, (0.99, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6)),
    "margin": ("batch", True, (0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)),
    "maxprob": ("batch", True, (0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6)),
    "entropy": ("batch", False, (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)),
    "gmm_clean": ("pool", True, (0.6, 0.5, 0.4, 0.3, 0.2, 0.1)),
    "pool_agree": ("pool", True, (0.8, 0.7, 0.6, 0.5, 0.4)),
}


def stop_state(r: dict, stat: str, thr: float, cap: bool) -> int:
    """The state greedy stops at: before the first batch whose statistic crosses thr (states[i] = after i batches);
    with cap, never past (1 - R)% (where greedy_pct1 stops)."""
    where, hi, _ = STATS[stat]
    key = "agree" if stat == "pool_agree" else stat
    n_cap = next(i for i, s in enumerate(r["states"]) if s["accepted"] >= r["n"] - r["k"])
    for i, st in enumerate(r["steps"]):
        v = st[where].get(key)
        if v is None:
            continue
        if (v < thr) if hi else (v > thr):
            return min(i, n_cap) if cap else i
        if cap and i + 1 >= n_cap:
            return n_cap
    return n_cap if cap else len(r["states"]) - 1


def greedy_section(lines: list[str]) -> None:
    res = load("greedy")
    tags = sorted(res)
    cands, stopped = {}, {}

    def add(name, idx_of):
        cands[name], stopped[name] = {}, {}
        for t in tags:
            i = idx_of(res[t])
            cands[name][t] = res[t]["states"][i]["test"]
            stopped[name][t] = res[t]["states"][i]["accepted"] / res[t]["n"]

    add("V only (state 0)", lambda r: 0)
    add("(1 - R)% (greedy_pct1)", lambda r: next(i for i, s in enumerate(r["states"]) if s["accepted"] >= r["n"] - r["k"]))
    add("V + P (all accepted)", lambda r: len(r["states"]) - 1)
    for stat, (_, _, thrs) in STATS.items():
        for thr in thrs:
            for cap in (False, True):
                add(f"{stat}{'<' if STATS[stat][1] else '>'}{thr:g}{' cap' if cap else ''}",
                    lambda r, s=stat, th=thr, cp=cap: stop_state(r, s, th, cp))
    best_state = {}  # the best state per draw on each metric, in hindsight
    for m, hi in MET:
        best_state[m] = float(np.mean([max(s["test"][m] for s in res[t]["states"]) if hi else min(s["test"][m] for s in res[t]["states"])
                                       for t in tags]))
    share = {n: np.mean(list(stopped[n].values())) for n in stopped}
    extra = {"cols": ["kept share of P"], "fn": lambda n: (f"{share[n]:.2f}",)}
    keep = ("V only (state 0)", "(1 - R)% (greedy_pct1)", "V + P (all accepted)")
    lines += ["## Greedy at 1% per step: stopping rules (all draws)", "",
              f"Hindsight bounds (each draw's best state, per metric): ROC-AUC {best_state['auc']:.3f}, balanced accuracy "
              f"{100 * best_state['bacc']:.1f}, log loss {best_state['ce']:.3f}.", ""]
    lines += fmt_rows(rank_table(cands, tags), extra, top=20, keep=keep) + [""]
    for att in ("rand45", "hub30"):
        sub = [t for t in tags if attack_of(t) == att]
        lines += [f"### {att} only ({len(sub)} draws)", ""] + fmt_rows(rank_table(cands, sub), extra, top=10, keep=keep) + [""]
    for c in (1, 3, 5):
        sub = [t for t in tags if c_of(t) == c]
        tab = rank_table(cands, sub)
        best = sorted(tab.items(), key=lambda kv: kv[1]["rank"])[:5]
        lines.append(f"- c = {c}: best " + ", ".join(f"{n} ({v['rank']:.2f})" for n, v in best)
                     + f"; (1 - R)% {tab['(1 - R)% (greedy_pct1)']['rank']:.2f}")
    lines.append("")


def baseline_section(lines: list[str]) -> None:
    res = load("bleval")
    if not res:
        lines += ["## Baselines: not evaluated yet", ""]
        return
    vf = load("vfilter")
    tags = sorted(t for t in res if t in vf)
    cands = {"V + P": {t: vf[t]["refs"]["v_plus_p"] for t in tags}, "V-filter R%": {t: vf[t]["rules"]["R"]["test"] for t in tags}}
    names = list(res[tags[0]]["rows"])
    for n in names:
        cands[n] = {t: res[t]["rows"][n]["test"] for t in tags}
    share = {n: np.mean([res[t]["rows"][n]["share_removed"] for t in tags]) for n in names}
    prec = {n: np.nanmean([res[t]["rows"][n]["prec"] if res[t]["rows"][n]["prec"] is not None else np.nan for t in tags]) for n in names}
    extra = {"cols": ["removed", "precision"], "fn": lambda n: (f"{share[n]:.2f}", f"{prec[n]:.2f}") if n in share else ("-", "-")}
    lines += [f"## Baselines: the top R% or their own rule ({len(tags)} draws)", ""] + fmt_rows(rank_table(cands, tags), extra) + [""]
    for att in ("rand45", "hub30"):
        sub = [t for t in tags if attack_of(t) == att]
        lines += [f"### {att} only ({len(sub)} draws)", ""] + fmt_rows(rank_table(cands, sub), extra) + [""]


def sigmoid_section(lines: list[str]) -> None:
    """Method 6 at a fixed iteration: every recorded iteration (0-10, then every 5 up to 200) of the development runs
    (bench_vp.py's sigmoid job on the development draws, results/vp_tune/sigmoid_dev) as a candidate deployment, with
    the held-out choice and the last iteration. Only runs of the study's protocol (a held-out part of V at c = 3, 5)
    count: since the study, method 6 fits all of V and its row `sigmoid` is the fixed iteration."""
    found = {f.parent.name: json.loads(f.read_text()) for f in sorted((ROOT / "sigmoid_dev" / "runs").glob("*/sigmoid.json"))}
    runs = {t: r for t, r in found.items() if not r["config"]["selection"].startswith("fixed")}
    if len(runs) < len(found):
        print(f"sigmoid: {len(found) - len(runs)} runs of the later fixed-iteration protocol left out")
    if not runs:
        lines += ["## Sigmoid (method 6): no development runs yet", ""]
        return
    tags = sorted(runs)
    its = [x["it"] for x in runs[tags[0]]["records"]]
    cands = {}
    for it in its:
        cands[f"it {it}"] = {t: next(x["test"] for x in runs[t]["records"] if x["it"] == it) for t in tags}
    for nm in ("sigmoid", "sigmoid_last"):
        cands[{"sigmoid": "held-out choice", "sigmoid_last": "last (200)"}[nm]] = {
            t: next(r["test"] for r in runs[t]["rows"] if r["name"] == nm) for t in tags}
    sel = [next(r["it"] for r in runs[t]["rows"] if r["name"] == "sigmoid") for t in tags if c_of(t) > 1]
    lines += [f"## Sigmoid (method 6): the deployment at a fixed iteration ({len(tags)} draws)", "",
              f"Held-out choice (c = 3, 5): median iteration {int(np.median(sel)) if sel else '-'} "
              f"(IQR {int(np.percentile(sel, 25)) if sel else '-'}-{int(np.percentile(sel, 75)) if sel else '-'}); c = 1: the last.", ""]
    lines += fmt_rows(rank_table(cands, tags), top=12, keep=("it 0", "held-out choice", "last (200)")) + [""]
    for att in ("rand45", "hub30"):
        sub = [t for t in tags if attack_of(t) == att]
        if not sub:
            continue
        lines += [f"### {att} only ({len(sub)} draws)", ""] + fmt_rows(rank_table(cands, sub), top=8,
                                                                       keep=("it 0", "held-out choice", "last (200)")) + [""]
    for c in (1, 3, 5):
        sub = [t for t in tags if c_of(t) == c]
        if not sub:
            continue
        tab = rank_table(cands, sub)
        best = sorted(tab.items(), key=lambda kv: kv[1]["rank"])[:5]
        lines.append(f"- c = {c}: best " + ", ".join(f"{n} ({v['rank']:.1f})" for n, v in best)
                     + f"; held-out choice {tab['held-out choice']['rank']:.1f}, last {tab['last (200)']['rank']:.1f}, it 0 {tab['it 0']['rank']:.1f}")
    lines.append("")
    lines += ["Mean curve over all draws (ROC-AUC / bal. acc. / log loss):", ""]
    pick = [i for i in its if i in (0, 1, 2, 5, 10, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200)]
    for m, hi in MET:
        vals = [np.mean([cands[f'it {i}'][t][m] for t in tags]) for i in pick]
        lines.append(f"- {m}: " + ", ".join(f"{i}: {100 * v:.1f}" if m == "bacc" else f"{i}: {v:.3f}" for i, v in zip(pick, vals)))
    lines.append("")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("results/tables/vp_tune/report.md"))
    ap.add_argument("--only", choices=["vfilter", "greedy", "baselines", "sigmoid"], default=None)
    a = ap.parse_args(argv)
    lines = ["# V + P benchmark: tuning on development draws (scripts/vp_tune.py)", "",
             "Mean rank: per draw and metric among the candidates of the table (1 = best), averaged over ROC-AUC, balanced "
             "accuracy and log loss and over the draws. Metrics: means over the draws.", ""]
    for name, fn in (("vfilter", vfilter_section), ("greedy", greedy_section), ("baselines", baseline_section),
                     ("sigmoid", sigmoid_section)):
        if a.only in (None, name) and (ROOT / {"baselines": "bleval", "sigmoid": "sigmoid_dev"}.get(name, name)).exists():
            fn(lines)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
