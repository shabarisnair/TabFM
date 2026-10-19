#!/usr/bin/env python
"""A label-free check of a filtered set: how well do the kept rows alone predict the V clean rows?

For every saved run of results/iterative_filter/<exp>/runs (scripts/iterative_filter.py), TabPFN
with only the kept context rows as its context (V left out) scores the V clean rows: mean CE and
accuracy. A kept set that is clean and covers the data should predict V well; one that drifted
(poison accepted, a class over-removed) should not. Nothing here uses the test set or the poison
labels, so the check can choose between filters per draw (scripts/analyze_itf_vcheck.py reports
how often it picks the better one).

Writes <exp dir>/vcheck.csv (draw, variant, v_ce, v_acc), appending; finished runs are skipped.

    python scripts/itf_vcheck.py --gpu 1 --exp e1
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
    ap.add_argument("--root", type=Path, default=Path("results/iterative_filter"))
    a = ap.parse_args(argv)
    import torch
    import gate_context as G
    from eval_vfilter_attacks import CONTEXT
    from tabfm_experiments.config import Seeds
    from tabfm_experiments.runtime import build_tabpfn_v2

    device = G.device_from_gpu(a.gpu)
    data = {}
    for exp in a.exp:
        out = a.root / exp / "vcheck.csv"
        done = {(r["draw"], r["variant"]) for r in csv.DictReader(open(out))} if out.exists() else set()
        new = not out.exists()
        fh = open(out, "a", newline="")
        w = csv.DictWriter(fh, fieldnames=["draw", "variant", "v_ce", "v_acc"])
        if new:
            w.writeheader()
        for d in sorted((a.root / exp / "runs").iterdir()):
            ds = next(k for k in CONTEXT if d.name.startswith(k + "_"))
            for f in sorted(d.glob("*.npz")):
                if f.name.startswith("base_") or (d.name, f.stem) in done:
                    continue
                z = np.load(f)
                if ds not in data:
                    X_file, y_file = G.load(Path(CONTEXT[ds]), device)
                    info = G.DATASETS_DIR / ds / "splits" / "split_info.json"
                    K = int((json.loads(info.read_text()).get("n_classes") if info.exists() else None) or len(np.unique(y_file)))
                    data[ds] = (X_file, y_file, build_tabpfn_v2(device, Seeds.model_seed, recompute_layers=True, n_classes=K))
                X_file, y_file, clf = data[ds]
                val = z["val_rows"]
                ctx = np.setdiff1d(np.arange(len(y_file)), val)
                kk = torch.as_tensor(ctx[z["keep"]], device=device)
                yk = torch.tensor(z["y_given"][z["keep"]], dtype=torch.float32, device=device)
                yv = y_file[val].astype(int)
                p = G.evaluate(clf, X_file[kk], yk, X_file[torch.as_tensor(val, device=device)], torch.as_tensor(yv, device=device),
                               None, None, need_grad=False, batch_size=None)[2]
                ce = float(-np.log(np.clip(p[np.arange(len(yv)), yv], 1e-12, None)).mean())
                w.writerow({"draw": d.name, "variant": f.stem, "v_ce": ce, "v_acc": float((p.argmax(1) == yv).mean())})
                fh.flush()
        fh.close()


if __name__ == "__main__":
    main()
