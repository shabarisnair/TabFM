#!/usr/bin/env python
"""Clean TabPFNv2 inference on a (context, test) CSV pair via the shared runtime.

Uses the same differentiable path as the attacks (``fit_with_differentiable_input`` +
``forward``), not sklearn ``fit`` / ``predict_proba``, so clean numbers here equal the
clean baselines reported by scripts/attack_context.py.

Example
-------
python scripts/infer.py --model tabpfnv2 --gpu 1 \
    --train datasets/url_unique/splits/selected/natural/context_1000.csv \
    --test  datasets/url_unique/splits/test_attack_1000.csv \
    --out   results/infer/url_unique_nat1000 --deterministic-mode best-effort
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from tabfm_experiments.io import ensemble_config_record, setup_logger, versions_info, write_json  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, type=Path)
    ap.add_argument("--test", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--target", default=None)
    ap.add_argument("--model", default="tabpfnv2", choices=["tabpfnv2", "tabiclv2"])
    ap.add_argument("--gpu", default="1", help="GPU index or 'cpu'")
    ap.add_argument("--model-seed", "--seed", dest="model_seed", type=int, default=0)
    ap.add_argument("--deterministic-mode", default="best-effort", choices=["best-effort", "strict"])
    ap.add_argument("--test-batch-size", type=int, default=None)
    ap.add_argument("--n-repeats", type=int, default=3, help="repeated clean forwards for max |dp|")
    a = ap.parse_args(argv)
    if a.model != "tabpfnv2":
        raise SystemExit("--model tabiclv2 is not supported on the shared TabPFNv2 runtime")

    from tabfm_experiments.data import load_split
    from tabfm_experiments.metrics import binary_metrics
    from tabfm_experiments.runtime import (
        build_tabpfn_v2, chunking_is_exact, device_from_gpu, evaluate_context, max_repeat_prob_delta,
    )

    log = setup_logger(a.out, "infer")
    write_json(a.out / "args.json", vars(a))
    device = device_from_gpu(a.gpu)
    Xtr, ytr, cols = load_split(a.train, a.target)
    Xte, yte, cols_te = load_split(a.test, a.target)
    if cols != cols_te:
        raise SystemExit("train and test feature columns differ")
    log.info(f"[tabpfnv2] train {Xtr.shape} test {Xte.shape} {device} mode={a.deterministic_mode}")

    clf = build_tabpfn_v2(device, a.model_seed, a.deterministic_mode)
    Xc = torch.tensor(Xtr.to_numpy(dtype=np.float32), device=device)
    yc = torch.tensor(ytr.astype(np.float32), device=device)
    Xt = torch.tensor(Xte.to_numpy(dtype=np.float32), device=device)
    yt = torch.tensor(yte.astype(np.int64), device=device)

    t0 = time.time()
    ce, _, probs = evaluate_context(clf, Xc, yc, Xt, yt, need_grad=False, test_batch_size=a.test_batch_size)
    secs = time.time() - t0
    m = binary_metrics(probs.cpu().numpy(), yte)
    max_dp = (max_repeat_prob_delta(clf, Xc, yc, Xt, yt, n_repeats=a.n_repeats, test_batch_size=a.test_batch_size)
              if a.n_repeats > 1 else None)

    write_json(a.out / "ensemble_config.json", ensemble_config_record(clf))
    write_json(a.out / "versions.json", versions_info(clf))
    write_json(a.out / "metrics.json", {
        "model": "tabpfnv2", "train": a.train, "test": a.test, "n_train": len(ytr), "n_test": len(yte),
        "n_features": len(cols), "model_seed": a.model_seed, "deterministic_mode": a.deterministic_mode,
        "device": device, "seconds": round(secs, 2), "max_repeat_abs_dprob": max_dp,
        "chunking_exact": chunking_is_exact(Xc, Xt), "metrics": m.as_dict(),
    })
    p = probs.cpu().numpy()
    pd.DataFrame({"y_true": yte, "y_pred": p.argmax(1), "proba_0": p[:, 0], "proba_1": p[:, 1]}).to_csv(
        a.out / "predictions.csv", index=False)
    log.info(f"  ce={m.ce:.5f} auc={m.roc_auc:.4f} acc={m.accuracy:.4f} f1={m.f1:.4f} "
             f"tn/fp/fn/tp={m.tn}/{m.fp}/{m.fn}/{m.tp}  max|dp|={max_dp}  [{secs:.1f}s] -> {a.out}")


if __name__ == "__main__":
    main()
