"""Usage: python scripts/soft_start_shapes.py <device> results/tables/soft_reg/start_shapes.json

Alternative starts for the soft weights, deployed soft + V with no training, on the 9 soft_reg instances:
the study's start (log w = s - max s), sharpened (T x that), class-balanced (s - max over the row's given label), both,
and the hard V-filter (k lowest s removed)."""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np, torch
import soft_reg_study as SR, attn_learn as A
from tabfm_experiments.row_weights import evaluate_weighted
dev = sys.argv[1]
rows = []
for tag in SR.instances():
    inst = SR.load_instance(tag, dev)
    s = SR.start_scores(inst); yp = inst["yp"]; n, k = inst["n"], inst["k"]
    smax_c = np.array([s[yp == c].max() for c in range(inst["K"])])[yp]
    starts = {"soft (study's start)": s - s.max(), "soft, sharpened x3": 3 * (s - s.max()), "soft, sharpened x10": 10 * (s - s.max()),
              "class-balanced": s - smax_c, "class-balanced, sharpened x10": 10 * (s - smax_c)}
    zv = torch.zeros(len(inst["yv"]), device=dev)
    Xcv, ycv = torch.cat([inst["Xc"], inst["Xv"]]), np.concatenate([yp, inst["yv"]])
    out = {}
    for name, lw in starts.items():
        lw_t = torch.as_tensor(np.clip(lw, -1000, 0), dtype=torch.float32, device=dev)
        p = evaluate_weighted(inst["clf"], Xcv, ycv, inst["Xt"], inst["yt"], torch.cat([lw_t, zv]), need_grad=False)[2]
        out[name] = A.metrics(p, inst["yt"])
    for name, score in (("hard V-filter", s), ("hard V-filter, class-balanced", s - smax_c)):
        rem = np.zeros(n, bool); rem[np.argsort(score, kind="stable")[:k]] = True
        lw_t = torch.where(torch.as_tensor(rem, device=dev), torch.full((n,), -float("inf"), device=dev), torch.zeros(n, device=dev))
        p = evaluate_weighted(inst["clf"], Xcv, ycv, inst["Xt"], inst["yt"], torch.cat([lw_t, zv]), need_grad=False)[2]
        out[name] = A.metrics(p, inst["yt"]) | {"prec": float(inst["pois"][rem].mean())}
    rows.append((tag, out))
    print(tag, {k2: round(v["auc"], 3) for k2, v in out.items()}, flush=True)
    del inst; torch.cuda.empty_cache()
json.dump(rows, open(sys.argv[2], "w"), indent=1)
