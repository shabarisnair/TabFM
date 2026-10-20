#!/usr/bin/env python
"""Report of the larger-V benchmark (scripts/bench_largev.py; <root>/runs). With one sampling of V (the default)
every cell is a single draw; with more, a cell is their mean with its 95% interval over the samplings.

Writes to <out> (default results/tables/bench_largev):
  results.csv  one row per instance x entry (every row of every result file; the top-k restarts are left out,
               their vote is nan_topk)
  summary.csv  per dataset x V x R x entry: the number of samplings, and per metric the mean and the 95%
               t-interval over them
  report.md    per dataset, the headline entries (all with V in the context) by V, pooled over R: ROC-AUC on
               a binary dataset or log loss on a multi-class one, accuracy and the precision of the removed
               rows. A cell is the mean over the samplings of the mean over R, with the 95% t-interval over
               the samplings. Then each learner's paired difference to the V-filter + V, the same way. Then a
               TabArena-style Elo over all instances (elo_bench_tinyv's fit: 1 - ROC-AUC on binary datasets,
               log loss on multi-class ones; poisoned + V = 1000; 95% intervals from resampling the instances
               and the datasets), and its point estimate per V.
Only the samplings of --seeds count (default: bench_largev.SEEDS). Partial results are fine: a cell with fewer
samplings than the rest shows their number in brackets.

    python scripts/report_bench_largev.py [--root results/bench_largev] [--out results/tables/bench_largev]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_largev as BL  # noqa: E402
import elo_bench_tinyv as EB  # noqa: E402
import report_bench_tinyv as RB  # noqa: E402

HEAD = [("v_only", "V only"), ("poisoned_plus_v", "poisoned + V"), ("oracle_plus_v", "oracle + V"),
        ("vfilter", "V-filter + V"), ("attn_topk", "attention weights, top-k (M1) + V"),
        ("attn_soft", "attention weights, soft (A3) + V"), ("nan_soft", "NaN gate, soft + V"),
        ("nan_topk", "NaN gate, top-k vote + V"),
        ("attn_wsoft", "attention weights, best soft form (w, V-filter, PGD; soft) + V")]
NAME = dict(HEAD)
LEARNERS = ["attn_topk", "attn_soft", "nan_soft", "nan_topk", "attn_wsoft"]
METRICS = ["acc", "bacc", "auc", "ce", "prec"]


def parse_tag(tag: str) -> tuple[str, float, int, int]:
    ds, rest = tag.rsplit("_R", 1)
    R, V, s = rest.split("_")
    return ds, float(R), int(V[1:]), int(s[1:])


def load(root: Path) -> pd.DataFrame:
    rows = []
    for f in sorted((root / "runs").glob("*/*.json")):
        if f.stem.startswith("nan_topk_r"):
            continue
        r = json.loads(f.read_text())
        ds, R, V, s = parse_tag(r["instance"])
        for e in r.get("rows", []):
            t = e["test"]
            rows.append({"instance": r["instance"], "dataset": ds, "V": V, "R": R, "seed": s, "entry": e["name"],
                         "acc": t["acc"], "bacc": t["bacc"], "auc": t["auc"], "ce": t["ce"], "prec": e.get("prec"),
                         "it": e.get("it"), "imported": "provenance" in r})
    return pd.DataFrame(rows)


def interval(x: pd.Series) -> tuple[int, float, float]:
    """(n, mean, half-width of the 95% t-interval) of the finite values."""
    x = x.dropna()
    n = len(x)
    if n == 0:
        return 0, np.nan, np.nan
    hw = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return n, float(x.mean()), float(hw)


def cell(n: int, mean: float, hw: float, n_full: int, fmt: str) -> str:
    if n == 0:
        return "."
    s = fmt.format(mean) + (f" ± {fmt.format(hw)}" if np.isfinite(hw) else "")
    return s + (f" ({n})" if n < n_full else "")


def per_seed(df: pd.DataFrame, value: str) -> pd.DataFrame:
    """The mean over R of `value`, per dataset x V x entry x seed."""
    return df.groupby(["dataset", "V", "entry", "seed"])[value].mean().reset_index()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(os.environ.get("BENCH_LARGEV_ROOT", "results/bench_largev")))
    ap.add_argument("--out", type=Path, default=Path("results/tables/bench_largev"))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(BL.SEEDS))
    a = ap.parse_args(argv)
    df = load(a.root)
    df = df[df.seed.isin(a.seeds)] if not df.empty else df
    if df.empty:
        raise SystemExit(f"no results under {a.root / 'runs'}")
    a.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out / "results.csv", index=False)
    summ = []
    for key, g in df.groupby(["dataset", "V", "R", "entry"]):
        rec = dict(zip(["dataset", "V", "R", "entry"], key))
        for m in METRICS:
            n, mean, hw = interval(pd.to_numeric(g[m], errors="coerce"))
            rec |= {f"{m}_n": n, f"{m}_mean": mean, f"{m}_lo": mean - hw, f"{m}_hi": mean + hw}
        summ.append(rec)
    pd.DataFrame(summ).to_csv(a.out / "summary.csv", index=False)

    n_seeds = df.seed.nunique()
    L = [f"# Larger-V poisoning benchmark ({df.instance.nunique()} instances with results)", "",
         "Every entry has all of V in its context. A cell is the mean over R = 10/20/30% of one sampling of V; with "
         "several samplings, their mean ± the half-width of its 95% t-interval over them, and a bracket gives the "
         "number of samplings when a cell is short. The learners fit on 80% of V and keep the iteration with the lowest log loss "
         "on the other 20%.", ""]
    for ds in [d for d in RB.SHORT if d in set(df.dataset)]:
        sub = df[(df.dataset == ds) & df.entry.isin(NAME)]
        Vs = sorted(sub.V.unique())
        main_m = "auc" if ds in RB.BINARY else "ce"
        L += ["", f"## {ds} ({'binary' if ds in RB.BINARY else 'multi-class'})", ""]
        for m, title, fmt in ((main_m, "ROC-AUC" if main_m == "auc" else "log loss (lower is better)", "{:.3f}"),
                              ("acc", "accuracy", "{:.3f}"), ("prec", "precision of the removed rows", "{:.2f}")):
            ps = per_seed(sub.assign(**{m: pd.to_numeric(sub[m], errors="coerce")}), m)
            L += [f"### {title}", "", "| entry | " + " | ".join(f"V = {V}" for V in Vs) + " |", "|---|" + "---|" * len(Vs)]
            for e, name in HEAD:
                if e not in set(ps.entry) or (m == "prec" and ps[ps.entry == e][m].isna().all()):
                    continue
                L.append(f"| {name} | " + " | ".join(
                    cell(*interval(ps[(ps.entry == e) & (ps.V == V)][m]), n_seeds, fmt) for V in Vs) + " |")
            L.append("")
        # paired differences to the V-filter + V, per instance, then as above
        piv = sub.pivot_table(index=["instance", "V", "seed", "R"], columns="entry", values=main_m).reset_index()
        if "vfilter" in piv:
            L += [f"### paired difference to the V-filter + V ({'ROC-AUC' if main_m == 'auc' else 'log loss; negative is better'})", "",
                  "| learner | " + " | ".join(f"V = {V}" for V in Vs) + " |", "|---|" + "---|" * len(Vs)]
            for e in LEARNERS:
                if e not in piv:
                    continue
                dd = piv.assign(diff=piv[e] - piv["vfilter"]).groupby(["V", "seed"])["diff"].mean().reset_index()
                L.append(f"| {NAME[e]} | " + " | ".join(cell(*interval(dd[dd.V == V]["diff"]), n_seeds, "{:+.3f}") for V in Vs) + " |")
            L.append("")

    # TabArena-style Elo over all instances (oracle + V not rated)
    rated = [e for e, _ in HEAD if e != "oracle_plus_v"]
    head = df[df.entry.isin(rated)].copy()
    head["err"] = np.where(head.dataset.isin(RB.BINARY), 1 - head.auc, head.ce)
    E = head.pivot(index="instance", columns="entry", values="err").reindex(columns=[e for e in rated if e in set(head.entry)])
    if "poisoned_plus_v" in E and E.shape[1] > 1 and len(E) > 1:
        fit = EB.fit(E, "poisoned_plus_v", impute=False)
        L += ["", "## TabArena-style Elo over all instances", "",
              "1 - ROC-AUC on binary datasets, log loss on multi-class ones; Bradley-Terry over every pair's battles on "
              "the instances both finished; poisoned + V = 1000; 95% intervals from 200 resamples of the instances "
              "(CI) and of the gap to poisoned + V over the instances and over the datasets.", "",
              "| entry | Elo | 95% CI | gap, instances | gap, datasets | instances | W-L vs poisoned + V |",
              "|---|---|---|---|---|---|---|"]
        for e, r in fit.iterrows():
            gaps = ". | ." if e == "poisoned_plus_v" else (f"{r.gap_lo_tasks:+.0f} to {r.gap_hi_tasks:+.0f} | "
                                                       f"{r.gap_lo_datasets:+.0f} to {r.gap_hi_datasets:+.0f}")
            wl = "." if e == "poisoned_plus_v" else f"{int(r.W)}-{int(r.L)}"
            L.append(f"| {NAME[e]} | {r.elo:.0f} | {r.lo_tasks:.0f} to {r.hi_tasks:.0f} | {gaps} | {int(r.tasks)} | {wl} |")
        by_v = {}
        for V in sorted(head.V.unique()):
            Ev = E[[parse_tag(t)[2] == V for t in E.index]]
            if len(Ev) > 1 and Ev.shape[1] > 1:
                by_v[V] = EB.fit(Ev, "poisoned_plus_v", impute=False)["elo"]
        if by_v:
            L += ["", "Elo per V (point estimates, each V fitted on its own instances):", "",
                  "| entry | " + " | ".join(f"V = {V}" for V in by_v) + " |", "|---|" + "---|" * len(by_v)]
            for e in fit.index:
                L.append(f"| {NAME[e]} | " + " | ".join(f"{s[e]:.0f}" if e in s and np.isfinite(s[e]) else "." for s in by_v.values()) + " |")
    (a.out / "report.md").write_text("\n".join(L) + "\n")
    print(f"-> {a.out / 'report.md'}, {a.out / 'summary.csv'}, {a.out / 'results.csv'}")


if __name__ == "__main__":
    main()
