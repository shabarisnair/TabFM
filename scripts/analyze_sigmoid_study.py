"""Paired analyses of the sigmoid study (scripts/sigmoid_study.py), soft + V unless noted: start vs chosen update,
the hard start vs deleting the same rows, every configuration minus each reference per regime, and the weights at the
chosen update. Output: results/tables/sigmoid_study/paired.txt

    python scripts/analyze_sigmoid_study.py > results/tables/sigmoid_study/paired.txt
"""
import sys, json
import numpy as np
sys.path.insert(0, "scripts")
import sigmoid_study as SS
res = SS.load_results()
DS = SS.DATASETS
M = ("auc", "acc", "ce")
INITS = ("ones", "vf_soft", "vf_hard", "random_s0", "random_ens")
def run(tag, init, lr):
    return res.get((tag, f"{init}_lr{lr:g}"))
def sv(r):
    return r["sel"]["ce"]["test"]["soft_v"]

print("== 1. start (update 0) vs chosen update, soft + V, mean over datasets and learning rates ==")
for rg in SS.REGIMES:
    for init in ("ones", "vf_soft", "vf_hard", "random_s0"):
        st, se = [], []
        for ds in DS:
            for lr in SS.LRS:
                r = run(SS.tag_of(ds, rg), init, lr)
                st.append([r["start"]["soft_v"][m] for m in M]); se.append([sv(r)[m] for m in M])
        st, se = np.array(st).mean(0), np.array(se).mean(0)
        print(f"  {rg:4s} {init:9s} start {st[0]:.3f}/{st[1]:.3f}/{st[2]:.3f} -> chosen {se[0]:.3f}/{se[1]:.3f}/{se[2]:.3f}  "
              f"(change {se[0]-st[0]:+.3f} / {se[1]-st[1]:+.3f} / {se[2]-st[2]:+.3f})")

print("\n== 2. the hard start (t = -30 / 4) at update 0 vs the V-filter's hard deletion + V, per regime ==")
for rg in SS.REGIMES:
    rows = []
    for ds in DS:
        ref = SS.references(ds, rg)["vfilter"]
        st = run(SS.tag_of(ds, rg), "vf_hard", 1.0)["start"]["soft_v"]
        rows.append([st[m] - ref[m] for m in M])
    x = np.array(rows)
    print(f"  {rg:4s} hard start minus V-filter (deleted): auc {x[:,0].mean():+.3f} acc {x[:,1].mean():+.3f} ce {x[:,2].mean():+.3f}  "
          f"(auc per dataset: {' '.join(f'{v:+.3f}' for v in x[:,0])})")

print("\n== 3. paired: each configuration minus a reference, soft + V, mean over 6 datasets [datasets better] ==")
refs = {rg: {ds: SS.references(ds, rg) for ds in DS} for rg in SS.REGIMES}
def refrow(rg, ds, name):
    if name == "best_soft" and rg == "c5":
        return sv(res[(SS.tag_of(ds, rg), "best_soft")])
    return refs[rg][ds].get({"best_soft": "attn_wsoft"}.get(name, name))
for rg in SS.REGIMES:
    print(f"  -- {rg}")
    for init in INITS:
        for lr in SS.LRS:
            cells = []
            for refname in ("vfilter", "v_only", "poisoned_plus_v", "best_soft"):
                d = np.array([[sv(run(SS.tag_of(ds, rg), init, lr))[m] - refrow(rg, ds, refname)[m] for m in M] for ds in DS])
                better = int((d[:, 0] > 0).sum())
                cells.append(f"{d[:,0].mean():+.3f}/{d[:,1].mean():+.3f}/{d[:,2].mean():+.2f} [{better}]")
            print(f"    {init:10s} lr {lr:<4g} vs V-filter {cells[0]:24s} vs V only {cells[1]:24s} vs poisoned+V {cells[2]:24s} vs best soft {cells[3]}")

print("\n== 4. weights at the chosen update (single runs; medians over datasets and learning rates) ==")
for rg in SS.REGIMES:
    for init in ("ones", "vf_soft", "vf_hard", "random_s0"):
        a, mp, mc, l10p, l10c, l100p = [], [], [], [], [], []
        for ds in DS:
            for lr in SS.LRS:
                r = run(SS.tag_of(ds, rg), init, lr); e = r["sel"]["ce"]; k, n = r["k"], r["n"]
                a.append(e["auroc"]); mp.append(e["logw_med_pois"]); mc.append(e["logw_med_clean"])
                l10p.append(e["n_le10_pois"] / k); l10c.append(e["n_le10_clean"] / (n - k)); l100p.append(e["n_le100_pois"] / k)
        print(f"  {rg:4s} {init:9s} AUROC(weights vs poison) {np.median(a):.3f} | median log w: poisoned {np.median(mp):7.2f} clean {np.median(mc):7.2f} | "
              f"log w <= -10: poisoned {np.median(l10p):.2f} clean {np.median(l10c):.2f} | <= -100: poisoned {np.median(l100p):.2f}")
