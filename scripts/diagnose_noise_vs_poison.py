#!/usr/bin/env python
"""Can any signal tell a genuine row with a surprising label from a poisoned row that looks plausible?

On noisy-label data the label-plausibility filters fail where genuine rows look wrong and flipped
rows look right. This scores every context row of one draw (dataset, V, validation seed; the
saved iterative-filter run gives V, the poison and the given labels) with candidate signals a
defender can compute, and asks how well each separates poisoned from genuine rows, overall and
inside the ambiguous band where a reference model trained on the CLEAN context (true labels,
5-fold cross-fit; unavailable to a defender) finds the given label equally plausible for both
(p in [--band-lo, --band-hi]).

V is split into V_fit (80%) and a clean holdout V_hold (20%). Signals (higher = more suspicious;
CE = -log p(given label)):
  v_filter      CE with V_fit as the context (the V-filter)
  cv_random     CE with V_fit + the context minus the row's random fold (5 folds): k-fold CV
                over the trusted rows, here all of them
  cv_block<B>   the same with the row's spatial block left out instead (k-means on standardised
                features, B blocks): the row's neighbourhood cannot vouch for it
  dependence    cv_block (largest B) minus cv_random: how much less plausible the row becomes
                once its own neighbourhood is gone
  holdout<B>    the row's block's harm to the clean holdout: CE of V_hold with V_fit + the whole
                context, minus with V_fit + the context without the block

Reports AUROC (poisoned vs genuine) on all rows and inside the band, and precision of the k most
suspicious rows (k = the number of poisoned rows), as removing them would.

    python scripts/diagnose_noise_vs_poison.py --gpu 0 --draws heloc:200 students_dropout:500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--draws", nargs="+", required=True, help="dataset:V (seed 0, R = 20%%, hub-flip-local, stratified)")
    ap.add_argument("--blocks", type=int, nargs="+", default=[50, 200])
    ap.add_argument("--band-lo", type=float, default=0.15)
    ap.add_argument("--band-hi", type=float, default=0.65)
    ap.add_argument("--out", type=Path, default=Path("results/tables/iterative_filter/noise_vs_poison.md"))
    a = ap.parse_args(argv)
    import torch
    from sklearn.cluster import KMeans
    from sklearn.metrics import roc_auc_score

    import gate_context as G
    from eval_vfilter_attacks import CONTEXT
    from tabfm_experiments.config import Seeds
    from tabfm_experiments.multiclass_attacks import standardise
    from tabfm_experiments.runtime import build_tabpfn_v2

    dev = G.device_from_gpu(a.gpu)
    lines = []
    for spec in a.draws:
        ds, V = spec.split(":")
        z = np.load(f"results/iterative_filter/e1/runs/{ds}_hub-flip-local_stratified_R20_V{V}s0/oneshot.npz")
        X, y = G.load(Path(CONTEXT[ds]), dev)
        info = G.DATASETS_DIR / ds / "splits" / "split_info.json"
        K = int((json.loads(info.read_text()).get("n_classes") if info.exists() else None) or len(np.unique(y)))
        clf = build_tabpfn_v2(dev, Seeds.model_seed, recompute_layers=True, n_classes=K)
        val = z["val_rows"]
        ctx = np.setdiff1d(np.arange(len(y)), val)
        n = len(ctx)
        y_true, y_given, pois = y[ctx].astype(int), z["y_given"].astype(int), z["pois"]
        fit_pos, hold_pos = G.split_holdout(y[val].astype(int), 0.2, 0)
        v_fit, v_hold = val[fit_pos], val[hold_pos]
        y_hold = y[v_hold].astype(int)
        t = lambda r: torch.as_tensor(np.asarray(r), device=dev)  # noqa: E731
        yf = lambda a_: torch.tensor(np.asarray(a_), dtype=torch.float32, device=dev)  # noqa: E731

        def probs(ctx_rows, ctx_labels, q_rows, q_labels):
            return G.evaluate(clf, X[t(ctx_rows)], yf(ctx_labels), X[t(q_rows)], t(q_labels), None, None,
                              need_grad=False, batch_size=None)[2]

        ce = lambda p, lab: -np.log(np.clip(p[np.arange(len(lab)), lab], 1e-12, None))  # noqa: E731
        S = {}
        # the reference: TabPFN on the clean context (true labels), 5-fold cross-fit
        fold = np.random.default_rng(0).permutation(n) % 5
        p_clean = np.zeros(n)
        for f in range(5):
            tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
            p_clean[te] = probs(ctx[tr], y_true[tr], ctx[te], y_given[te])[np.arange(len(te)), y_given[te]]
        # v_filter and random-fold CV over V_fit + the poisoned context
        S["v_filter"] = ce(probs(v_fit, y[v_fit], ctx, y_given), y_given)
        S["cv_random"] = np.zeros(n)
        for f in range(5):
            tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
            p = probs(np.concatenate([v_fit, ctx[tr]]), np.concatenate([y[v_fit], y_given[tr]]), ctx[te], y_given[te])
            S["cv_random"][te] = ce(p, y_given[te])
        # blocked CV and the holdout harm of each block
        Z = standardise(X[t(ctx)]).cpu().numpy()
        p_all_hold = probs(np.concatenate([v_fit, ctx]), np.concatenate([y[v_fit], y_given]), v_hold, y_hold)
        ce_hold_all = float(ce(p_all_hold, y_hold).mean())
        for B in a.blocks:
            block = KMeans(n_clusters=B, n_init=3, random_state=0).fit_predict(Z)
            S[f"cv_block{B}"], S[f"holdout{B}"] = np.zeros(n), np.zeros(n)
            for b in range(B):
                te, tr = np.flatnonzero(block == b), np.flatnonzero(block != b)
                if len(te) == 0:
                    continue
                p = probs(np.concatenate([v_fit, ctx[tr]]), np.concatenate([y[v_fit], y_given[tr]]),
                          np.concatenate([ctx[te], v_hold]), np.concatenate([y_given[te], y_hold]))
                S[f"cv_block{B}"][te] = ce(p[:len(te)], y_given[te])
                S[f"holdout{B}"][te] = ce_hold_all - float(ce(p[len(te):], y_hold).mean())
        S["dependence"] = S[f"cv_block{max(a.blocks)}"] - S["cv_random"]
        band = (p_clean >= a.band_lo) & (p_clean <= a.band_hi)
        k = int(pois.sum())
        lines += ["", f"## {ds}, V = {V} (V_fit {len(v_fit)}, holdout {len(v_hold)}; n = {n}, poisoned {k})", "",
                  f"Reference model on the clean context: genuine rows with p(own label) < 0.5: {(p_clean[~pois] < 0.5).mean():.2f}; "
                  f"poisoned rows with p(flipped label) > 0.5: {(p_clean[pois] > 0.5).mean():.2f}. Band p in "
                  f"[{a.band_lo}, {a.band_hi}]: {int((band & ~pois).sum())} genuine, {int((band & pois).sum())} poisoned rows.", "",
                  "| signal | AUROC, all rows | precision of the k most suspicious | AUROC inside the band |", "|---|---|---|---|"]
        for name, s in S.items():
            top = np.argsort(-s, kind="stable")[:k]
            auc_band = roc_auc_score(pois[band], s[band]) if band.sum() and 0 < pois[band].mean() < 1 else float("nan")
            lines.append(f"| {name} | {roc_auc_score(pois, s):.3f} | {pois[top].mean():.3f} | {auc_band:.3f} |")
        print("\n".join(lines[-(len(S) + 6):]), flush=True)
        np.savez_compressed(f"/tmp/claude-1023288547/-home-ssn899-Desktop-TabFM/f8d79958-c844-4f05-b093-22fd605d5321/scratchpad/nvp_{ds}_V{V}.npz",
                            pois=pois, p_clean=p_clean, **S)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
