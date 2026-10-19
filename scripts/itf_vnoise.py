#!/usr/bin/env python
"""How noisy are the labels, judged from V alone? 5-fold cross-validated TabPFN on the V rows.

For every draw saved under results/iterative_filter/<exp>/runs (the V rows are in each run's npz),
TabPFN scores each fold of the V clean rows with the other four folds as its context: accuracy
and mean CE. Uses only V, never the context, the poison or the test set, so it can decide
before filtering whether to grow the trusted set (the greedy gains where labels are nearly
deterministic and loses where they are noisy).

Writes results/iterative_filter/vnoise.csv (draw, v_cv_acc, v_cv_ce), appending.

    python scripts/itf_vnoise.py --gpu 1 --exp e1 e3 e6
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--exp", nargs="+", required=True)
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args(argv)
    import torch
    import gate_context as G
    from eval_vfilter_attacks import CONTEXT
    from tabfm_experiments.config import Seeds
    from tabfm_experiments.runtime import build_tabpfn_v2

    device = G.device_from_gpu(a.gpu)
    out = Path("results/iterative_filter/vnoise.csv")
    done = {r["draw"] for r in csv.DictReader(open(out))} if out.exists() else set()
    new = not out.exists()
    fh = open(out, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=["draw", "v_cv_acc", "v_cv_ce"])
    if new:
        w.writeheader()
    data = {}
    for exp in a.exp:
        for d in sorted((Path("results/iterative_filter") / exp / "runs").iterdir()):
            if d.name in done:
                continue
            f = next((f for f in sorted(d.glob("*.npz")) if not f.name.startswith("base_")), None)
            if f is None:
                continue
            ds = next(k for k in CONTEXT if d.name.startswith(k + "_"))
            if ds not in data:
                X_file, y_file = G.load(Path(CONTEXT[ds]), device)
                info = G.DATASETS_DIR / ds / "splits" / "split_info.json"
                K = int((json.loads(info.read_text()).get("n_classes") if info.exists() else None) or len(np.unique(y_file)))
                data[ds] = (X_file, y_file, build_tabpfn_v2(device, Seeds.model_seed, recompute_layers=True, n_classes=K))
            X_file, y_file, clf = data[ds]
            val = np.load(f)["val_rows"]
            yv = y_file[val].astype(int)
            fold = np.random.default_rng(0).permutation(len(val)) % a.folds
            p = np.zeros((len(val), clf.n_classes_))
            for k in range(a.folds):
                tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
                X_tr = X_file[torch.as_tensor(val[tr], device=device)]
                y_tr = torch.tensor(yv[tr], dtype=torch.float32, device=device)
                p[te] = G.evaluate(clf, X_tr, y_tr, X_file[torch.as_tensor(val[te], device=device)],
                                   torch.as_tensor(yv[te], device=device), None, None, need_grad=False, batch_size=None)[2]
            ce = float(-np.log(np.clip(p[np.arange(len(yv)), yv], 1e-12, None)).mean())
            w.writerow({"draw": d.name, "v_cv_acc": float((p.argmax(1) == yv).mean()), "v_cv_ce": ce})
            fh.flush()
            done.add(d.name)
    fh.close()


if __name__ == "__main__":
    main()
