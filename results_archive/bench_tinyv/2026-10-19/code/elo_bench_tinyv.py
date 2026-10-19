#!/usr/bin/env python
"""TabArena-style Elo of the tiny-V benchmark's entries (the rows report_bench_tinyv.py reads).

TabArena's procedure (github.com/autogluon/tabarena: packages/bencheval/src/bencheval/elo_utils.py and
evaluator.py's compute_elo), re-implemented here:
  - the error on a task: 1 - ROC-AUC on a binary dataset, log loss on a multi-class one;
  - a task: one instance (dataset x c x R). Its seed would be TabArena's split; there is one;
  - a battle: every two entries that both finished a task meet once on it; the lower error wins and an
    exact tie gives each half a win;
  - Bradley-Terry by maximum likelihood: the battles are summed per pair (A's wins + half the ties, B's
    wins + half the ties) and fitted by a logistic regression without intercept, x = +ln 10 for A and
    -ln 10 for B, C = 1e6, tol 1e-10; Elo = 400 * coefficient + 1000;
  - the 95% interval: 200 resamples of the tasks with replacement, each refitted, and their 2.5% and
    97.5% quantiles as widths around the full-data fit (clipped at 0), taken before the calibration;
  - the calibration: all ratings shifted so the anchor scores 1000 (TabArena: RF (default); here
    poisoned + V, i.e. no cleaning).
Also reported: a second interval from resampling the 6 datasets (all 9 instances of a drawn dataset; the
instances of one dataset are not independent, so it is wider), the win rate, TabArena's improvability
(the mean over tasks of 1 - best error / error, best among the compared entries) and the wins, ties and
losses against the anchor.

Missing results (the unfinished attention instances): by default an entry battles only on the tasks it
finished (TabArena's inner merge); the 'imputed' variant fills them with the anchor's result first, as
TabArena's leaderboard fills a failed task with RF (default)'s. oracle + V is left out (it knows the
poisoned rows) unless --with-oracle.

    python scripts/elo_bench_tinyv.py [--root results/bench_tinyv] [--out results/tables/bench_tinyv]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))

import report_bench_tinyv as RB  # noqa: E402

SCALE, INIT, ROUNDS = 400.0, 1000.0, 200
# (name, metric, tasks, impute): the main fit first, then the sensitivity checks
VARIANTS = [("main", "tabarena", "all", False), ("imputed", "tabarena", "all", True),
            ("binary only", "tabarena", "binary", False), ("multi-class only", "tabarena", "multiclass", False),
            ("multi-class by accuracy", "acc_multiclass", "all", False), ("without ecommerce", "tabarena", "no_ecommerce", False)]


def error_table(rows: list[dict], metric: str, tasks: str, entries: list[str]) -> pd.DataFrame:
    """tasks x entries; NaN where an entry has no result."""
    df = pd.DataFrame(rows)
    binary = df["dataset"].isin(RB.BINARY)
    df = {"binary": df[binary], "multiclass": df[~binary], "no_ecommerce": df[df["dataset"] != "ecommerce_shipping"]}.get(tasks, df)
    binary = df["dataset"].isin(RB.BINARY)
    auc, ce, acc = (pd.to_numeric(df[m], errors="coerce") for m in ("auc", "ce", "acc"))
    df = df.assign(err=np.where(binary, 1 - auc, ce if metric == "tabarena" else 1 - acc))
    E = df.pivot(index="instance", columns="entry", values="err")
    return E.reindex(columns=[e for e in entries if e in E.columns])


def outcomes(E: np.ndarray) -> np.ndarray:
    """O[t, a, b]: a's share of its battle with b on task t (1 win, 0.5 tie, 0 loss or no battle)."""
    a, b = E[:, :, None], E[:, None, :]
    O = np.where(a < b, 1.0, np.where(a == b, 0.5, 0.0)) * (~np.isnan(a) & ~np.isnan(b))
    i = np.arange(E.shape[1])
    O[:, i, i] = 0.0
    return O


def bt_elo(W: np.ndarray) -> np.ndarray:
    """Bradley-Terry Elo from the pair totals W[a, b] (a's wins over b + half their ties), uncalibrated;
    NaN for an entry without battles."""
    m = len(W)
    X, y, w = [], [], []
    for i in range(m):
        for j in range(i + 1, m):
            if W[i, j] + W[j, i] == 0:
                continue
            x = np.zeros(m)
            x[i], x[j] = np.log(10), -np.log(10)
            X += [x, x]
            y += [1.0, 0.0]
            w += [W[i, j], W[j, i]]
    lr = LogisticRegression(fit_intercept=False, C=1e6, tol=1e-10, max_iter=1000)
    lr.fit(np.array(X), np.array(y), sample_weight=np.array(w))
    elo = SCALE * lr.coef_[0] + INIT
    elo[(W + W.T).sum(1) == 0] = np.nan
    return elo


def bootstrap(O: np.ndarray, unit: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """[ROUNDS, entries] uncalibrated Elo of resamples that draw the units (unit[t] = task t's unit) with
    replacement; a task counts as often as its unit was drawn."""
    units, inv = np.unique(unit, return_inverse=True)
    out = []
    for _ in range(ROUNDS):
        cnt = np.bincount(rng.integers(0, len(units), len(units)), minlength=len(units))
        out.append(bt_elo(np.tensordot(cnt[inv].astype(float), O, axes=1)))
    return np.array(out)


def fit(E: pd.DataFrame, anchor: str, impute: bool) -> pd.DataFrame:
    if impute:
        E = E.apply(lambda col: col.fillna(E[anchor]))
    ent, A = list(E.columns), E.to_numpy(float)
    O = outcomes(A)
    k = ent.index(anchor)
    elo = bt_elo(O.sum(0))
    ds = np.array([t.rsplit("_R", 1)[0] for t in E.index])
    res = pd.DataFrame({"elo": elo}, index=pd.Index(ent, name="entry"))
    for name, unit, seed in (("tasks", np.arange(len(A)), 0), ("datasets", ds, 1)):
        boot = bootstrap(O, unit, np.random.default_rng(seed))
        res[f"lo_{name}"] = elo - np.clip(elo - np.nanquantile(boot, 0.025, axis=0), 0, None)
        res[f"hi_{name}"] = elo + np.clip(np.nanquantile(boot, 0.975, axis=0) - elo, 0, None)
        # the gap to the anchor within each resample: whether an entry beats the anchor beyond the noise
        res[f"gap_lo_{name}"] = np.nanquantile(boot - boot[:, [k]], 0.025, axis=0)
        res[f"gap_hi_{name}"] = np.nanquantile(boot - boot[:, [k]], 0.975, axis=0)
    rating = ["elo", "lo_tasks", "hi_tasks", "lo_datasets", "hi_datasets"]
    res[rating] += INIT - res.loc[anchor, "elo"]  # the calibration, after the intervals (TabArena's post_calibrate)
    have = ~np.isnan(A)
    battles = (have[:, :, None] & have[:, None, :]).sum(0)
    np.fill_diagonal(battles, 0)
    res["win_rate"] = O.sum((0, 2)) / battles.sum(1)
    res["improvability"] = np.nanmean(1 - np.nanmin(A, axis=1, keepdims=True) / A, axis=0)
    res["tasks"] = have.sum(0)
    for col, sel in (("W", O[:, :, k] == 1), ("T", O[:, :, k] == 0.5), ("L", O[:, k, :] == 1)):
        res[col] = sel.sum(0)
    res.loc[anchor, ["W", "T", "L"]] = 0
    return res.sort_values("elo", ascending=False)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(os.environ.get("BENCH_TINYV_ROOT", "results/bench_tinyv")))
    ap.add_argument("--out", type=Path, default=Path("results/tables/bench_tinyv"))
    ap.add_argument("--anchor", default="poisoned_plus_v", help="the entry calibrated to 1000")
    ap.add_argument("--with-oracle", action="store_true")
    a = ap.parse_args(argv)
    rows = RB.load(a.root)
    entries = [e for e, _ in RB.ENTRIES if a.with_oracle or not e.startswith("oracle")]
    fits = {}
    for name, metric, tasks, impute in VARIANTS:
        fits[name] = fit(error_table(rows, metric, tasks, entries), a.anchor, impute)
    main_ = fits["main"]
    n_tasks = len(error_table(rows, "tabarena", "all", entries))
    L = [f"# Tiny-V benchmark: TabArena-style Elo ({n_tasks} tasks)", "",
         "Error: 1 - ROC-AUC on the binary datasets (heloc, ecommerce, good_customer, coil2000), log loss on the "
         "multi-class ones (gas-drift, students). Task = one instance (dataset x c x R). Bradley-Terry by "
         "maximum likelihood over all pairwise battles (ties = half a win), Elo = 400 log10-odds; "
         f"calibrated so {RB.NAME[a.anchor]} = 1000. 95% CI: TabArena's, from {ROUNDS} bootstrap resamples of the "
         f"tasks. Gap to {RB.NAME[a.anchor]}: the 95% interval of Elo - its Elo within each resample, over the "
         "tasks and over the 6 datasets (all 9 instances of a drawn dataset; wider, since the instances of one "
         "dataset are not independent); an interval above 0 means it beats the anchor beyond the noise. "
         "Improvability (TabArena's): mean over tasks of 1 - best error / error (%, lower is better). "
         f"W-T-L: wins, ties and losses against {RB.NAME[a.anchor]}.", "",
         f"| # | entry | Elo | 95% CI | gap to {RB.NAME[a.anchor]}, over tasks | over datasets | win rate | "
         "improvability (%) | tasks | W-T-L |", "|---|---|---|---|---|---|---|---|---|---|"]
    for pos, (e, r) in enumerate(main_.iterrows(), 1):
        gaps = "| . | . |" if e == a.anchor else (f"| {r.gap_lo_tasks:+.0f} to {r.gap_hi_tasks:+.0f} | "
                                                  f"{r.gap_lo_datasets:+.0f} to {r.gap_hi_datasets:+.0f} |")
        L.append(f"| {pos} | {RB.NAME.get(e, e)} | {r.elo:.0f} | {r.lo_tasks:.0f} to {r.hi_tasks:.0f} {gaps} "
                 f"{r.win_rate:.2f} | {100 * r.improvability:.1f} | {int(r.tasks)} | "
                 f"{'.' if e == a.anchor else '-'.join(str(int(r[c])) for c in 'WTL')} |")
    L += ["", "## Sensitivity", "",
          "Elo under other choices, each calibrated the same way. imputed: the unfinished attention instances "
          "filled with the anchor's result (TabArena's handling of failed tasks). binary only: the 36 binary tasks "
          "(1 - AUC). multi-class only: the 18 multi-class tasks (log loss). multi-class by accuracy: as the main "
          "fit, with 1 - accuracy instead of log loss on gas-drift and students. without ecommerce: the 45 tasks "
          "left without ecommerce_shipping, where the poisoning costs nothing (oracle + V is no better than "
          "poisoned + V there).", "",
          "| entry | " + " | ".join(n for n, *_ in VARIANTS) + " |", "|---|" + "---|" * len(VARIANTS)]
    for e in main_.index:
        L.append(f"| {RB.NAME.get(e, e)} | " + " | ".join(
            f"{fits[n].loc[e, 'elo']:.0f}" if e in fits[n].index and np.isfinite(fits[n].loc[e, "elo"]) else "."
            for n, *_ in VARIANTS) + " |")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "elo.md").write_text("\n".join(L) + "\n")
    pd.concat({n: f for n, f in fits.items()}, names=["variant"]).to_csv(a.out / "elo.csv")
    print("\n".join(L))
    print(f"\n-> {a.out / 'elo.md'}, {a.out / 'elo.csv'}")


if __name__ == "__main__":
    main()
