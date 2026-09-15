#!/usr/bin/env python
"""Temporary probe: compare the standard end repair vs a within-eps-budget end repair.

Both repairs are applied to the SAME CAPGD result inside one run, so the CE comparison
is exact even under best-effort nondeterminism. Constraints = full.
"""
import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tabfm_experiments.attacks import XCapgdConfig, run_x_capgd
from tabfm_experiments.constraints import FeatureSpec, build_constraints
from tabfm_experiments.data import infer_dataset_name, load_split, metadata_path_for, read_metadata
from tabfm_experiments.metrics import binary_metrics
from tabfm_experiments.runtime import build_tabpfn_v2, device_from_gpu, evaluate_context
from tabfm_experiments.sampling import k_from_percent, row_rng, sample_row_indices

def run(dataset, gpu, n_iter, seeds, ctx="context_5000.csv", row_percent=30, tbs=None):
    dev = device_from_gpu(gpu)
    base = Path("datasets")/dataset/"splits"
    Xtr, ytr, cols = load_split(base/"selected"/"natural"/ctx)
    Xte, yte, _ = load_split(base/"test_attack_1000.csv")
    Xc = torch.tensor(Xtr.to_numpy(np.float32), device=dev); yc = torch.tensor(ytr.astype(np.float32), device=dev)
    Xt = torch.tensor(Xte.to_numpy(np.float32), device=dev); yt = torch.tensor(yte.astype(np.int64), device=dev)
    meta = read_metadata(metadata_path_for(base/"selected"/"natural"/ctx), cols)
    spec = FeatureSpec.from_metadata(meta, device=dev, scaler="metadata", X_train=Xc)
    cons = build_constraints(infer_dataset_name(base/"selected"/"natural"/ctx), meta, cols)
    clf = build_tabpfn_v2(dev, 0, "best-effort", recompute_layers=(dataset=="wids"))
    _, _, p0 = evaluate_context(clf, Xc, yc, Xt, yt, need_grad=False, test_batch_size=tbs)
    clean = binary_metrics(p0.cpu().numpy(), yte)
    k = k_from_percent(len(ytr), row_percent)
    # budget_repair=True (now the default) reports the budgeted repair as primary and the
    # plain TabularBench repair under extra["standard_repair"].
    cfg = XCapgdConfig(n_iter=n_iter, constraints_mode="full", budget_repair=True)
    for rs in seeds:
        rows = sample_row_indices(ytr, k, attack_class=None, rng=row_rng(rs, 0))
        r = run_x_capgd(clf, Xc, yc, Xt, yt, rows, spec=spec, constraints=cons, cfg=cfg,
                        run_id=0, subsample_id=0, attack_seed=2, clean_metrics=clean, k_requested=k,
                        test_batch_size=tbs)
        s = r.extra["standard_repair"]
        print(f"  {dataset:6s} it{n_iter} rs{rs}: standard dCE={s['delta']['ce']:+.4f} (within_eps={s['within_eps']}, maxL2={s['max_l2_delta_scaled']:.3f}) | "
              f"budgeted dCE={r.delta['ce']:+.4f} (within_eps={r.extra['within_eps_after_repair']}, maxL2={r.extra['max_l2_delta_scaled']:.3f}) | "
              f"CE cost of budget = {s['delta']['ce']-r.delta['ce']:+.4f}", flush=True)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--gpu", default="1")
    ap.add_argument("--n-iter", type=int, nargs="+", default=[100])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1,2,3])
    ap.add_argument("--test-batch-size", type=int, default=None)
    a = ap.parse_args()
    for it in a.n_iter:
        run(a.dataset, a.gpu, it, a.seeds, tbs=a.test_batch_size)
