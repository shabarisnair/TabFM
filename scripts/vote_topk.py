#!/usr/bin/env python
"""Vote the restarts of one top-k gating draw and score the voted mask.

The best top-k configuration (results/tables/gate_topk/night_log.md) runs several
restarts (init seeds) on one draw (dataset, R, V, validation seed, holdout) and combines
them by a vote. Each restart contributes its hard mask at its chosen iteration
(gates/top1.npz, the lowest held-out CE); the k rows masked by the most restarts are
switched off, ties going to the lower mean within-restart score rank. Like every
restart's mask, the vote mask switches off exactly k rows.

Scored with the same model and data as the runs (paths from their config.json):
  fitted     CE on the fitted validation rows (the training objective)
  valtest    the metrics of the held-out validation rows
  test       the metrics of the test set, the rows masked (as the runs score their best
             mask) and removed (plain TabPFN on the kept rows)
  detection  precision@R and recall@R of the switched-off rows; the vote histogram
Copied from the runs, not recomputed: each restart's logged numbers at its chosen
iteration, the best restart by held-out CE (the rule the vote replaces), and the
references of test_metrics.json (poisoned, oracle, the V clean rows alone, ...).
As a check, the first restart's mask is re-scored on the held-out rows and must
reproduce its logged held-out CE.

Writes <out>/vote.json and <out>/vote.npz (mask, votes, mean rank, test probabilities),
and with --wandb-mode online|offline one W&B run (job type "vote") named after <out>.

    python scripts/vote_topk.py --gpu 0 --out <vote dir> <run dir> <run dir> ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import gate_context as G  # noqa: E402

# config args that must agree across the restarts of one draw (init_seed must differ)
SAME = ("dataset", "context", "test", "val_from_context", "val_seed", "val_holdout", "attack",
        "row_percent", "remove_percent", "run_id", "row_seed", "hub_metric", "hub_k", "target_k",
        "gate", "ste", "ste_temp", "score_init", "score_init_std", "optimizer", "pgd_scale", "lr",
        "steps", "ste_probe", "lr_schedule", "lr_min_frac", "budget_warmup", "line_search",
        "swap_search", "fit_subsample", "select_on", "model_seed", "test_every")
REFERENCES = ("poisoned_no_gate", "poisoned_random_removal", "poisoned_oracle_gate",
              "poisoned_oracle_removal", "val_as_context", "poisoned_plus_val", "clean_context")


def score_ranks(scores: np.ndarray) -> np.ndarray:
    """Rank of each score within its restart, scaled to [0, 1]; 0 = the lowest score."""
    return np.argsort(np.argsort(scores, kind="stable"), kind="stable") / (len(scores) - 1)


def vote_mask(masks: list[np.ndarray], ranks: list[np.ndarray], k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mask, votes, mean rank): the k rows masked by the most restarts switched off (gate
    0), ties to the lower mean rank, then the earlier row."""
    votes = np.sum([1 - m for m in masks], axis=0).astype(int)
    mean_rank = np.mean(ranks, axis=0)
    mask = np.ones(len(votes))
    mask[np.lexsort((np.arange(len(votes)), mean_rank, -votes))[:k]] = 0.0
    return mask, votes, mean_rank


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", type=Path, nargs="+", help="the restarts' run directories")
    ap.add_argument("--out", type=Path, required=True, help="folder for vote.json and vote.npz")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--wandb-project", default="TFM_experiments")
    ap.add_argument("--wandb-group", default=None)
    ap.add_argument("--wandb-mode", default="disabled", choices=["online", "offline", "disabled"])
    a = ap.parse_args(argv)
    from tabfm_experiments.io import write_json
    from tabfm_experiments.metrics import multiclass_metrics
    from tabfm_experiments.runtime import build_tabpfn_v2

    if len(a.runs) < 2:
        raise SystemExit("a vote needs at least 2 restarts")
    cfgs, tms = [], []
    for d in a.runs:
        missing = [f for f in ("config.json", "test_metrics.json", "split.npz", "poison.npz", "gates/top1.npz")
                   if not (d / f).exists()]
        if missing:
            raise SystemExit(f"{d}: missing {missing}")
        cfgs.append(json.loads((d / "config.json").read_text()))
        tms.append(json.loads((d / "test_metrics.json").read_text()))
    args0 = cfgs[0]["args"]
    for d, c in zip(a.runs[1:], cfgs[1:]):
        diff = [key for key in SAME if c["args"].get(key) != args0.get(key)]
        if diff:
            raise SystemExit(f"{d} differs from {a.runs[0]} in {diff}: not restarts of one draw")
    seeds = [c["args"]["init_seed"] for c in cfgs]
    if args0.get("gate") != "topk" or len(set(seeds)) != len(seeds):
        raise SystemExit(f"need top-k runs with distinct init seeds, got gate {args0.get('gate')}, seeds {seeds}")
    sp, pz = np.load(a.runs[0] / "split.npz"), np.load(a.runs[0] / "poison.npz")
    if "context_rows" not in sp.files or "val_holdout_rows" not in sp.files:
        raise SystemExit("the runs must draw the validation rows from the context, with a holdout")
    for d in a.runs[1:]:
        s2, p2 = np.load(d / "split.npz"), np.load(d / "poison.npz")
        for key in ("context_rows", "val_fit_rows", "val_holdout_rows"):
            if not np.array_equal(s2[key], sp[key]):
                raise SystemExit(f"{d}: {key} differs from {a.runs[0]}")
        for key in ("poisoned_rows", "y_poisoned"):
            if not np.array_equal(p2[key], pz[key]):
                raise SystemExit(f"{d}: {key} differs from {a.runs[0]}")

    device = G.device_from_gpu(a.gpu)
    K, k = int(cfgs[0]["n_classes"]), int(cfgs[0]["k_remove"])
    X_all, y_all = G.load(Path(cfgs[0]["context"]), device)
    Xt, yt = G.load(Path(cfgs[0]["test"]), device)
    sel = lambda r: X_all[torch.as_tensor(r, device=device)]  # noqa: E731
    Xc, Xf, Xh = sel(sp["context_rows"]), sel(sp["val_fit_rows"]), sel(sp["val_holdout_rows"])
    yf, yh = y_all[sp["val_fit_rows"]], y_all[sp["val_holdout_rows"]]
    n = len(sp["context_rows"])
    yp = torch.tensor(pz["y_poisoned"], dtype=torch.float32, device=device)
    pois = np.zeros(n, dtype=bool)
    pois[pz["poisoned_rows"]] = True
    clf = build_tabpfn_v2(device, args0["model_seed"], recompute_layers=True, n_classes=K)
    null = G.capture_null_input(clf, Xc, yp)

    def probs(m, Xq, yq, ctx=None):
        X_ctx, y_ctx = (Xc, yp) if ctx is None else ctx
        gt = None if m is None else torch.as_tensor(m, dtype=torch.float32, device=device)
        return G.evaluate(clf, X_ctx, y_ctx, Xq, torch.as_tensor(yq, device=device), gt, null,
                          need_grad=False, batch_size=None)[:3:2]  # (mean CE, probabilities)

    def metrics(p, y):
        m = multiclass_metrics(p, y, K)
        return {key: float(m[key]) for key in G.METRICS}

    masks, ranks, restarts = [], [], []
    for d, c, tm in zip(a.runs, cfgs, tms):
        z = np.load(d / "gates" / "top1.npz")
        g = z["g"]
        if not (((g == 0) | (g == 1)).all() and int((g == 0).sum()) == k):
            raise SystemExit(f"{d}: top1.npz is not a hard mask with {k} rows switched off")
        masks.append(g)
        ranks.append(score_ranks(z["scores"]))
        b, gs = tm["best_val_row"], tm["best_gate_stats"]
        restarts.append({"run": d.name, "init_seed": c["args"]["init_seed"], "iteration": tm["best_iteration"],
                         "fitted_ce": b["val_ce"], "valtest_ce": b["valtest_ce"],
                         "valtest_accuracy": b["valtest_accuracy"],
                         "test": {m: tm["metrics"]["poisoned_best_gate_soft"][m] for m in G.METRICS},
                         "test_removed": {m: tm["metrics"]["poisoned_best_gate_hard"][m] for m in G.METRICS},
                         "precision_at_R": gs["precision_at_R"], "auroc_poisoned": gs["auroc_poisoned"]})

    ce_check = probs(masks[0], Xh, yh)[0]
    logged = restarts[0]["valtest_ce"]
    if abs(ce_check - logged) > 1e-3:
        raise SystemExit(f"re-scoring {a.runs[0].name}'s mask gives held-out CE {ce_check:.6f}, "
                         f"its log says {logged:.6f}: the data or model differ from the run's")

    mask, votes, mean_rank = vote_mask(masks, ranks, k)
    off = mask == 0
    ce_f = probs(mask, Xf, yf)[0]
    _, ph = probs(mask, Xh, yh)
    _, pt = probs(mask, Xt, yt)
    keep = np.flatnonzero(~off)
    if len(np.unique(pz["y_poisoned"][keep])) == K:
        _, pr = probs(None, Xt, yt, ctx=(Xc[torch.as_tensor(keep, device=device)], yp[torch.as_tensor(keep, device=device)]))
        removed = metrics(pr, yt)
    else:  # a class vanished from the kept rows: TabPFN would misalign the classes
        pr, removed = None, {m: float("nan") for m in G.METRICS}
    hist = np.bincount(votes, minlength=len(masks) + 1)
    best = min(restarts, key=lambda r: (r["valtest_ce"], -r["valtest_accuracy"], r["init_seed"]))
    out = {
        "vote_dir": a.out.name, "restart_runs": [str(d) for d in a.runs], "n_restarts": len(masks),
        "dataset": args0["dataset"], "row_percent": args0["row_percent"], "val_size": args0["val_from_context"],
        "val_seed": args0["val_seed"], "val_holdout": args0["val_holdout"], "k_remove": k, "n_context": n,
        "n_poisoned": int(pois.sum()), "n_fit": len(yf), "n_holdout": len(yh), "n_test": len(yt),
        "setting": {key: args0.get(key) for key in ("ste", "score_init_std", "lr", "steps", "ste_probe",
                                                     "lr_schedule", "pgd_scale", "select_on")},
        "rule": "vote of the restarts' masks at their lowest-held-out-CE iterations; ties -> lower mean score rank",
        "vote": {"fitted_ce": ce_f, "valtest": metrics(ph, yh), "test": metrics(pt, yt), "test_removed": removed,
                 "precision_at_R": float(pois[off].mean()), "recall_at_R": float(pois[off].sum() / pois.sum()),
                 "rows_by_votes": hist.tolist(),
                 "poisoned_share_by_votes": [float(pois[votes == v].mean()) if hist[v] else None
                                             for v in range(len(masks) + 1)]},
        "best_restart_by_valtest_ce": best,
        "restarts": restarts,
        "restart_means": {"valtest_ce": float(np.mean([r["valtest_ce"] for r in restarts])),
                          "test_accuracy": float(np.mean([r["test"]["accuracy"] for r in restarts])),
                          "precision_at_R": float(np.mean([r["precision_at_R"] for r in restarts]))},
        "references": {name: {m: tms[0]["metrics"][name][m] for m in G.METRICS}
                       for name in REFERENCES if name in tms[0]["metrics"]},
        "check": {"rescored_valtest_ce_restart0": ce_check, "logged": logged},
    }
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out / "vote.npz", mask=mask.astype(np.float32), votes=votes.astype(np.int16),
                        mean_rank=mean_rank.astype(np.float32), y_test=yt.astype(np.int16),
                        test_probs=pt.astype(np.float32),
                        **({"test_probs_removed": pr.astype(np.float32)} if pr is not None else {}))
    write_json(a.out / "vote.json", out)
    v = out["vote"]
    print(f"{a.out.name}: vote of {len(masks)} -> held-out CE {v['valtest']['ce']:.4f} acc {v['valtest']['accuracy']:.3f}"
          f" | test acc {v['test']['accuracy']:.3f} (removed {removed['accuracy']:.3f}) | precision@R "
          f"{v['precision_at_R']:.3f} | best restart: held-out CE {best['valtest_ce']:.4f}, test acc "
          f"{best['test']['accuracy']:.3f}, precision {best['precision_at_R']:.3f}")
    if a.wandb_mode != "disabled":
        import wandb
        from tabfm_experiments.io import jsonable
        run = wandb.init(project=a.wandb_project, group=a.wandb_group, name=a.out.name, job_type="vote",
                         mode=a.wandb_mode, dir=str(a.out), config=jsonable({k_: out[k_] for k_ in (
                             "dataset", "row_percent", "val_size", "val_seed", "val_holdout", "k_remove",
                             "n_restarts", "setting", "rule", "restart_runs")}))
        run.summary.update(jsonable({"vote": v, "best_restart_by_valtest_ce": best,
                                     "restart_means": out["restart_means"], "references": out["references"]}))
        run.finish()
    return out


if __name__ == "__main__":
    main()
