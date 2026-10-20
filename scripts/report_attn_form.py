#!/usr/bin/env python
"""Report of the soft attention-weight form study (scripts/attn_form_study.py): parameterisation x start x update.

Every configuration at its chosen learning rate (chosen on the held-out log loss of the 4 cheap datasets), deployed
soft + V, on the 6 datasets; then the three factors compared pairwise, the learning-rate sensitivity, what the
weights did, and the secondary views (soft without V; the update chosen by held-out accuracy).

    python scripts/report_attn_form.py [--out results/tables/attn_form]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import attn_form_study as S  # noqa: E402

DS_SHORT = {"gas-drift": "gas-drift", "heloc": "heloc", "ecommerce_shipping": "ecommerce",
            "students_dropout": "students", "is_this_a_good_customer": "good_customer",
            "coil2000_insurance_policies": "coil2000"}
DATASETS = ("gas-drift", "coil2000_insurance_policies", *S.SMALL)
PRIMARY, SECONDARY = "soft", "soft_v"  # the deployment the weights were trained and chosen for; with V added
METRICS = (("auc", "ROC-AUC", True, "{:.3f}"), ("acc", "accuracy", True, "{:.3f}"), ("ce", "log loss", False, "{:.3f}"),
           ("bacc", "balanced accuracy", True, "{:.3f}"))
PARAM_NAME = {"w": "w in [0, 1]", "sigmoid": "sigmoid(t)", "log": "log w, floor -1000", "log2000": "log w, floor -2000"}


def cfg_label(param: str, init: str, opt: str) -> str:
    return f"{param} / {init} / {opt}"


def collect() -> dict:
    """Per configuration: its learning-rate state and, per dataset, the run at the chosen rate."""
    S._cache.clear()
    out = {}
    for param, init, opt in S.configs():
        state, lrs, best = S.lr_state(param, init, opt)
        runs = {}
        if best is not None:
            for ds in DATASETS:
                r = S.load(ds, param, init, opt, best)
                if r is not None and not r.get("failed"):
                    runs[ds] = r
        out[(param, init, opt)] = dict(state=state, lrs=lrs, lr=best, runs=runs)
    return out


def value(r: dict, metric: str, dep: str = PRIMARY, sel: str = "ce") -> float:
    return float(r["sel"][sel]["test"][dep][metric])


def complete(res: dict) -> list:
    return [c for c, v in res.items() if len(v["runs"]) == len(DATASETS)]


def ranks_of(res: dict, cfgs: list, metric: str, higher: bool, dep=PRIMARY, sel="ce") -> dict:
    """Mean over the datasets of each configuration's rank (1 = best; ties share the mean rank)."""
    out = {c: 0.0 for c in cfgs}
    for ds in DATASETS:
        v = np.array([value(res[c]["runs"][ds], metric, dep, sel) for c in cfgs])
        key = -v if higher else v
        order = key.argsort(kind="stable")
        rk = np.empty(len(cfgs))
        rk[order] = np.arange(1, len(cfgs) + 1)
        for u in np.unique(key):
            rk[key == u] = rk[key == u].mean()
        for c, x in zip(cfgs, rk):
            out[c] += x / len(DATASETS)
    return out


def bold_best(cells: list[str], vals: list[float], higher: bool) -> list[str]:
    ok = [v for v in vals if np.isfinite(v)]
    if not ok:
        return cells
    best = max(ok) if higher else min(ok)
    return [f"**{c}**" if np.isfinite(v) and abs(v - best) < 1e-12 else c for c, v in zip(cells, vals)]


def main_table(res: dict, cfgs: list, dep=PRIMARY, sel="ce") -> list[str]:
    md = ["| configuration | lr | " + " | ".join(f"{t} (mean)" for _, t, _, _ in METRICS[:3]) + " | "
          + " | ".join(f"rank {t}" for _, t, _, _ in METRICS[:3]) + " | update chosen (median) |",
          "|---|---|" + "---|" * 7]
    means = {m: [np.mean([value(res[c]["runs"][ds], m, dep, sel) for ds in DATASETS]) for c in cfgs]
             for m, _, _, _ in METRICS[:3]}
    rks = {m: ranks_of(res, cfgs, m, hi, dep, sel) for m, _, hi, _ in METRICS[:3]}
    cols = []
    for m, _, hi, f in METRICS[:3]:
        cols.append(bold_best([f.format(v) for v in means[m]], means[m], hi))
    for m, _, _, _ in METRICS[:3]:
        rv = [rks[m][c] for c in cfgs]
        cols.append(bold_best([f"{v:.1f}" for v in rv], rv, False))
    for i, c in enumerate(cfgs):
        its = [res[c]["runs"][ds]["sel"][sel]["it"] for ds in DATASETS]
        md.append(f"| {cfg_label(*c)} | {res[c]['lr']:g} | " + " | ".join(col[i] for col in cols)
                  + f" | {int(np.median(its))} |")
    return md


def per_dataset(res: dict, cfgs: list, metric: str, higher: bool, fmt: str, dep=PRIMARY) -> list[str]:
    md = ["| configuration | " + " | ".join(DS_SHORT[d] for d in DATASETS) + " | mean | mean, 5 (no good_customer) |",
          "|---|" + "---|" * (len(DATASETS) + 2)]
    table = np.array([[value(res[c]["runs"][ds], metric, dep) for ds in DATASETS] for c in cfgs])
    full = np.column_stack([table, table.mean(1), table[:, [i for i, d in enumerate(DATASETS)
                                                            if d != "is_this_a_good_customer"]].mean(1)])
    cols = [bold_best([fmt.format(v) for v in full[:, j]], list(full[:, j]), higher) for j in range(full.shape[1])]
    for i, c in enumerate(cfgs):
        md.append(f"| {cfg_label(*c)} | " + " | ".join(col[i] for col in cols) + " |")
    return md


def paired(res: dict, cfgs: list, factor: int, a: str, b: str, dep=PRIMARY) -> list[str]:
    """a against b on one factor, the other two held fixed: the mean difference over (other factors x datasets)
    and the wins / ties / losses of a."""
    pairs = [(c, tuple(b if i == factor else x for i, x in enumerate(c))) for c in cfgs if c[factor] == a]
    pairs = [(x, y) for x, y in pairs if y in cfgs]
    out = []
    for m, t, hi, _ in METRICS[:3]:
        d = np.array([value(res[x]["runs"][ds], m, dep) - value(res[y]["runs"][ds], m, dep) for x, y in pairs for ds in DATASETS])
        s = d if hi else -d
        w, ti, lo = int((s > 1e-9).sum()), int((np.abs(s) <= 1e-9).sum()), int((s < -1e-9).sum())
        out.append(f"{t} {np.mean(d):+.3f} ({w}-{ti}-{lo})")
    return [f"| {a} vs {b} | {len(pairs)} x {len(DATASETS)} | " + " | ".join(out) + " |"] if pairs else []


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("results/tables/attn_form"))
    a = ap.parse_args(argv)
    import pandas as pd
    res = collect()
    cfgs = complete(res)
    a.out.mkdir(parents=True, exist_ok=True)

    # every finished run, flat
    rows = []
    for f in sorted((S.ROOT / "runs").glob("*/*.json")):
        import json
        r = json.loads(f.read_text())
        if r.get("failed"):
            rows.append(r["job"] | {"failed": r["failed"]})
            continue
        row = dict(r["job"]) | {"n": r["n"], "k": r["k"], "stopped": r["stopped"], "last_it": r["last_it"],
                                "floor_hit": r["floor_hit"], "seconds": r["seconds"]}
        for sel in ("ce", "acc"):
            s = r["sel"][sel]
            row |= {f"{sel}_it": s["it"], f"{sel}_hold_ce": s["hold"]["ce"], f"{sel}_hold_acc": s["hold"]["acc"],
                    f"{sel}_auroc": s["auroc"], f"{sel}_logw_med_pois": s["logw_med_pois"],
                    f"{sel}_logw_med_clean": s["logw_med_clean"], f"{sel}_w_share_pois": s["w_share_pois"]}
            for th in (10, 100, 1000):
                row |= {f"{sel}_n_le{th}_pois": s[f"n_le{th}_pois"], f"{sel}_n_le{th}_clean": s[f"n_le{th}_clean"]}
            for dep in ("soft_v", "soft"):
                row |= {f"{sel}_{dep}_{m}": s["test"][dep][m] for m in ("auc", "acc", "ce", "bacc")}
        for dep in ("soft_v", "soft"):
            row |= {f"start_{dep}_{m}": r["start"][dep][m] for m in ("auc", "acc", "ce", "bacc")}
        row["start_hold_ce"] = r["evals"][0]["hold_ce"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(a.out / "runs.csv", index=False)

    md = ["# Soft attention weights: which parameterisation, start and update rule (V = 500, R = 30%)", "",
          "Learned soft, deployed soft: every poisoned row at its learned weight, nothing removed. The primary "
          "deployment is the one the weights were trained and chosen for: the weighted poisoned rows alone as the "
          "context (soft). Section 5 adds all 500 V rows at weight 1 (soft + V). One draw per dataset (the larger-V benchmark's, seed 0); 400 V rows fitted, "
          "100 held out. Every run stops when the held-out log loss has not improved by 1e-3 for 30 updates (at most "
          "200) and deploys the update with the lowest held-out log loss. The learning rate is chosen per "
          "configuration on the held-out log loss of heloc, students, ecommerce and good_customer (mean rank over a "
          "x4 grid) and reused on gas-drift and coil2000. Nothing is chosen on the test set. Bold: the best of a "
          "column. Ranks: per dataset among the configurations shown (1 = best), then averaged.", "",
          f"Configurations complete on all 6 datasets: {len(cfgs)} of {len(res)}.", ""]
    if not cfgs:
        (a.out / "report.md").write_text("\n".join(md) + "\n")
        print("nothing complete yet")
        return
    md += ["## 1. Every configuration (soft, test set)", ""] + main_table(res, cfgs) + [""]
    for m, t, hi, f in METRICS:
        md += [f"### {t}, per dataset (soft)", ""] + per_dataset(res, cfgs, m, hi, f) + [""]

    md += ["## 2. The factors, paired (the other two factors fixed; mean difference, then wins-ties-losses of the "
           "first over datasets x the fixed levels)", "",
           "| comparison | pairs | " + " | ".join(t for _, t, _, _ in METRICS[:3]) + " |", "|---|---|---|---|---|"]
    for x, y in (("log", "w"), ("log", "sigmoid"), ("sigmoid", "w"), ("log2000", "log")):
        md += paired(res, cfgs, 0, x, y)
    md += paired(res, cfgs, 1, "vfilter", "ones")
    for x, y in (("adam", "pgd"), ("adam", "sign"), ("pgd", "sign")):
        md += paired(res, cfgs, 2, x, y)
    md += [""]

    md += ["## 3. Learning-rate sensitivity (the 4 datasets the rate is chosen on: mean held-out log loss rank, "
           "then mean test, soft)", "",
           "| configuration | lr | rank of held-out log loss | held-out log loss | test ROC-AUC | test accuracy | "
           "test log loss | chosen |", "|---|---|---|---|---|---|---|---|"]
    for c, v in res.items():
        runs = {lr: [S.load(ds, *c, lr) for ds in S.SMALL] for lr in v["lrs"]}
        okl = [lr for lr in v["lrs"] if all(r is not None and not r.get("failed") for r in runs[lr])]
        if not okl:
            continue
        rk = {lr: 0.0 for lr in okl}
        for i in range(len(S.SMALL)):
            ces = np.array([S.hold_ce(runs[lr][i]) for lr in okl])
            order = ces.argsort(kind="stable")
            r_ = np.empty(len(okl))
            r_[order] = np.arange(1, len(okl) + 1)
            for j, lr in enumerate(okl):
                rk[lr] += r_[j] / len(S.SMALL)
        for lr in okl:
            rr = runs[lr]
            md.append(f"| {cfg_label(*c)} | {lr:g} | {rk[lr]:.2f} | {np.mean([S.hold_ce(r) for r in rr]):.3f} | "
                      f"{np.mean([value(r, 'auc') for r in rr]):.3f} | {np.mean([value(r, 'acc') for r in rr]):.3f} | "
                      f"{np.mean([value(r, 'ce') for r in rr]):.3f} | {'yes' if lr == v['lr'] else ''} |")
    md += [""]

    md += ["## 4. What the weights did (at the deployed update; medians over the 6 datasets)", "",
           "AUROC: the learned weights ranking the poisoned rows below the clean ones. Rows at or below log w = -10 / "
           "-1000, and in the zone -20 < log w < -1 where a soft weight distorts TabPFN (section 6): share of the "
           "poisoned rows | share of the clean rows.", "",
           "| configuration | update chosen | stopped at | AUROC | median log w, poisoned | median log w, clean | "
           "log w <= -10 | log w <= -1000 | -20 < log w < -1 | floor reached |", "|---|---|---|---|---|---|---|---|---|---|"]
    for c in cfgs:
        rs = [res[c]["runs"][ds] for ds in DATASETS]
        s = [r["sel"]["ce"] for r in rs]
        kp = [r["k"] for r in rs]
        kc = [r["n"] - r["k"] for r in rs]
        md.append(f"| {cfg_label(*c)} | {int(np.median([x['it'] for x in s]))} | "
                  f"{int(np.median([r['last_it'] for r in rs]))} | {np.median([x['auroc'] for x in s]):.3f} | "
                  f"{np.median([x['logw_med_pois'] for x in s]):.1f} | {np.median([x['logw_med_clean'] for x in s]):.1f} | "
                  f"{np.median([x['n_le10_pois'] / k for x, k in zip(s, kp)]):.2f} \\| "
                  f"{np.median([x['n_le10_clean'] / k for x, k in zip(s, kc)]):.2f} | "
                  f"{np.median([x['n_le1000_pois'] / k for x, k in zip(s, kp)]):.2f} \\| "
                  f"{np.median([x['n_le1000_clean'] / k for x, k in zip(s, kc)]):.2f} | "
                  f"{zone(res[c])} | {sum(r['floor_hit'] for r in rs)} of 6 |")
    md += [""]

    md += ["## 5. Secondary views", "", "### Soft + V (all 500 V rows added at weight 1; the weights were learned "
           "and chosen without them)", ""]
    md += main_table(res, cfgs, dep=SECONDARY) + [""]
    md += ["| comparison, soft + V | pairs | " + " | ".join(t for _, t, _, _ in METRICS[:3]) + " |", "|---|---|---|---|---|"]
    for x, y in (("log", "w"), ("log", "sigmoid"), ("sigmoid", "w")):
        md += paired(res, cfgs, 0, x, y, SECONDARY)
    md += paired(res, cfgs, 1, "vfilter", "ones", SECONDARY)
    for x, y in (("adam", "pgd"), ("adam", "sign"), ("pgd", "sign")):
        md += paired(res, cfgs, 2, x, y, SECONDARY)
    md += [""]
    md += ["### The update chosen by held-out accuracy instead (ties: log loss; within the same runs, which stopped "
           "on the log loss)", ""] + main_table(res, cfgs, sel="acc") + [""]
    md += ["### The starts, before any update (test set, soft | soft + V; the same for every update rule and rate)", "",
           "| start | parameterisation | " + " | ".join(DS_SHORT[d] for d in DATASETS) + " | mean |",
           "|---|---|" + "---|" * (len(DATASETS) + 1)]
    for m, t, hi, f in METRICS[:3]:
        for init in S.INITS:
            for param in ("w", "sigmoid", "log"):
                c = next((x for x in cfgs if x[0] == param and x[1] == init), None)
                if c is None:
                    continue
                v = [res[c]["runs"][ds]["start"][PRIMARY][m] for ds in DATASETS]
                u = [res[c]["runs"][ds]["start"][SECONDARY][m] for ds in DATASETS]
                md.append(f"| {init} ({t}) | {param} | " + " | ".join(f"{f.format(x)} \\| {f.format(y)}" for x, y in zip(v, u))
                          + f" | {f.format(np.mean(v))} \\| {f.format(np.mean(u))} |")
    md += [""]
    md += response_section(a.out)
    (a.out / "report.md").write_text("\n".join(md) + "\n")
    print(f"-> {a.out}/report.md, {a.out}/runs.csv ({len(cfgs)} of {len(res)} configurations complete)")


def zone(entry: dict) -> str:
    """Median over the datasets of the share of rows with -20 < log w < -1 at the deployed update: poisoned | clean."""
    shares = []
    for ds, r in entry["runs"].items():
        j = r["job"]
        z = np.load(S.result_path(ds, S.job_name(j["param"], j["init"], j["opt"], j["lr"])).with_suffix(".npz"))
        lw, pois = z["logw_sel"], z["pois"]
        inz = (lw > -20) & (lw < -1)
        shares.append((inz[pois].mean(), inz[~pois].mean()))
    a = np.median(np.array(shares), axis=0)
    return f"{a[0]:.2f} \\| {a[1]:.2f}"


def response_section(out: Path) -> list[str]:
    """TabPFN's response to soft weights (scripts/attn_soft_response.py), if its CSVs exist."""
    import pandas as pd
    md = []
    sw, ch = out / "weight_sweep.csv", out / "vfilter_channels.csv"
    if sw.exists():
        d = pd.read_csv(sw)
        levels = [0.0, -0.5, -1.0, -2.3, -4.6, -6.9, -9.2, -20.0, -100.0, -1000.0, -np.inf]
        head = "| dataset | " + " | ".join("w = 1" if lv == 0 else ("deleted" if lv == -np.inf else f"{lv:g}")
                                           for lv in levels) + " |"
        md += ["## 6. How TabPFN responds to a soft weight (scripts/attn_soft_response.py)", "",
               "### A random 30% of the poisoned rows at one log w (the rest at 1): test ROC-AUC, soft + V", "",
               head, "|---|" + "---|" * len(levels)]
        for ds, g in d[d.what == "random30"].groupby("dataset", sort=False):
            v = {round(r.logw, 2): r.test_v_auc for r in g.itertuples()}
            md.append(f"| {DS_SHORT.get(ds, ds)} | " + " | ".join(f"{v[round(lv, 2)]:.3f}" for lv in levels) + " |")
        md += ["", "### Every row given the smallest class at one log w: mean predicted probability of that class on "
               "the held-out rows (the poisoned rows only as the context; its true share in brackets)", "",
               head, "|---|" + "---|" * len(levels)]
        for ds, g in d[d.what == "class"].groupby("dataset", sort=False):
            v = {round(r.logw, 2): r.hold_p_cls for r in g.itertuples()}
            md.append(f"| {DS_SHORT.get(ds, ds)} ({g.true_share_cls.iloc[0]:.2f}) | "
                      + " | ".join(f"{v[round(lv, 2)]:.3f}" for lv in levels) + " |")
        md += [""]
    if ch.exists():
        d = pd.read_csv(ch)
        md += ["### The V-filter start (log w = s - max s) installed on one channel at a time: held-out accuracy | test "
               "ROC-AUC (soft + V)", "", "| dataset | none (w = 1) | all channels | attention only | label mean only | "
               "statistics only |", "|---|---|---|---|---|---|"]
        for ds, g in d.groupby("dataset", sort=False):
            v = {r.channels: f"{r.hold_acc:.2f} / {r.test_v_auc:.3f}" for r in g.itertuples()}
            md.append(f"| {DS_SHORT.get(ds, ds)} | " + " | ".join(v[c] for c in ("none", "all", "attention", "label_mean",
                                                                                 "statistics")) + " |")
        md += [""]
    return md


if __name__ == "__main__":
    main()
