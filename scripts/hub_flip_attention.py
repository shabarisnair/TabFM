#!/usr/bin/env python
"""Test-agnostic hub label flipping, recording every attention map and prediction.

The attack (Bilos et al., "A Mechanistic Study of Tabular Foundation Models", S6/H.1):
flip the labels of the R% of context rows with the SMALLEST average k-NN distance to the
other context rows -- the "hubs" that sit in dense regions and therefore win attention
mass from many queries. The score uses the context features only: no test data, no
labels, no model. It is fully deterministic, so one run per R is enough.

For each condition this records, for every query row:
  * TabPFNv2's predicted class probabilities
  * along-column attention  (BETWEEN ROWS: query -> each context row), the map the
    paper's L9 vote reads, for every layer, head and (selected) column
  * along-row attention     (BETWEEN FEATURES: within the query row), every layer/head

TabPFNv2 never materialises attention weights -- both attention modules call a fused
scaled_dot_product_attention. They are recomputed here from the same projections and
verified against each module's real output (see --no-self-check).

    python scripts/hub_flip_attention.py --gpu 3 --dataset url_unique --row-percent 16
    python scripts/hub_flip_attention.py --gpu 3 --dataset coil2000_insurance_policies \
        --row-percent 0 --columns all --topk 128
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tabfm_experiments.config import DATASETS_DIR  # noqa: E402
from tabfm_experiments.io import write_json  # noqa: E402
from tabfm_experiments.metrics import binary_metrics, with_derived  # noqa: E402
from tabfm_experiments.sampling import k_from_percent  # noqa: E402


# --------------------------------------------------------------------- the hub attack

def hub_scores(X: torch.Tensor, metric: str = "knn", k: int = 10,
               chunk: int = 512) -> np.ndarray:
    """Hubness score per context row: lower = more central = more of a hub.

    ``knn``  mean distance to the k nearest OTHER context rows (the paper's definition).
    ``all``  mean distance to every other context row.

    Features are z-scored with the context's own statistics so no single wide column
    dominates the metric. Test rows are never touched.
    """
    Z = X.double()
    Z = (Z - Z.mean(0)) / Z.std(0).clamp_min(1e-12)
    n = Z.shape[0]
    out = torch.empty(n, dtype=torch.float64, device=Z.device)
    kk = min(k, n - 1)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = torch.cdist(Z[s:e], Z)                                   # [c, n]
        rows = torch.arange(e - s, device=d.device)
        d[rows, torch.arange(s, e, device=d.device)] = float("inf")  # exclude self
        if metric == "knn":
            out[s:e] = d.topk(kk, largest=False).values.mean(1)
        elif metric == "all":
            d[torch.isinf(d)] = 0.0
            out[s:e] = d.sum(1) / (n - 1)
        else:
            raise ValueError(f"unknown hub metric {metric!r}")
    return out.cpu().numpy()


def pick_hubs(scores: np.ndarray, k: int) -> np.ndarray:
    """The k smallest scores. Ties break on index, so the choice is reproducible."""
    return np.sort(np.argsort(scores, kind="stable")[:k])


# ------------------------------------------------------------- attention recomputation

class AttentionRecorder:
    """Recompute and capture both attention maps of every block, for the query rows.

    Hooks each block's two attention modules. A module may be invoked in chunks along its
    flattened leading dimension (TabPFN's memory saver), so each call is mapped back to
    its global row / column range and the pieces are stitched together in order.
    Assumes batch size 1 (the fixed single-estimator runtime).
    """

    def __init__(self, arch, n_ctx: int, n_cols: int, columns: str = "target",
                 topk: int = 0, col_chunk: int = 4, full_target: bool = False):
        self.n_ctx, self.n_cols, self.columns, self.topk = n_ctx, n_cols, columns, topk
        self.col_chunk = col_chunk
        # With --topk the other columns are truncated, but the target column is the one
        # the readout vote reads, so it is worth keeping exact.
        self.full_target = full_target and columns == "all"
        self.enabled = False
        self.handles: list = []
        self.row: dict[int, list] = {}
        self.col: dict[int, list] = {}
        self.tgt: dict[int, list] = {}
        self.check: dict[str, float] = {}
        for i, blk in enumerate(arch.blocks):
            self.handles.append(blk.per_sample_attention_between_features
                                .register_forward_hook(self._mk_row_hook(i)))
            self.handles.append(blk.per_column_attention_between_cells
                                .register_forward_hook(self._mk_col_hook(i)))

    def __enter__(self):
        self.enabled = True
        self.row, self.col, self.tgt = {}, {}, {}
        # Per LAYER, not global: every layer re-processes the same rows/columns from 0.
        self._row_seen: dict[int, int] = {}
        self._col_seen: dict[int, int] = {}
        return self

    def __exit__(self, *exc):
        self.enabled = False

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    # -- along-row: between the feature tokens of one row -----------------------------
    def _mk_row_hook(self, layer: int):
        def hook(module, args, output):
            if not self.enabled:
                return
            x = args[0]                                   # [Br, C, E], Br = B*R
            Br, C, _ = x.shape
            off = self._row_seen.get(layer, 0)
            self._row_seen[layer] = off + Br
            lo, hi = max(self.n_ctx, off), off + Br       # query rows only
            if hi <= lo:
                return
            sl = slice(lo - off, hi - off)
            xs = x[sl]
            H, D = module.num_heads, module.head_dim
            q = module.q_projection(xs).view(-1, C, H, D).transpose(1, 2)
            k = module.k_projection(xs).view(-1, C, H, D).transpose(1, 2)
            a = torch.softmax(q.float() @ k.float().transpose(-1, -2) / D ** 0.5, dim=-1)
            if layer == 0 and "between_features" not in self.check:
                v = module.v_projection(xs).view(-1, C, H, D).transpose(1, 2)
                ref = module.out_projection(
                    (a.to(v.dtype) @ v).transpose(1, 2).reshape(-1, C, H * D))
                self.check["between_features"] = float((ref - output[sl]).abs().max())
            self.row.setdefault(layer, []).append((lo, a.half().cpu()))
        return hook

    # -- along-column: between rows, within one column --------------------------------
    def _mk_col_hook(self, layer: int):
        def hook(module, args, output):
            if not self.enabled:
                return
            x = args[0]                                   # [Bc, R, E], Bc = B*C
            Bc = x.shape[0]
            off = self._col_seen.get(layer, 0)
            self._col_seen[layer] = off + Bc
            N, H, D = self.n_ctx, module.num_heads, module.head_dim
            if self.columns == "all":
                sel = list(range(Bc))
            else:                                          # the y slot = last column
                tgt = self.n_cols - 1
                if not (off <= tgt < off + Bc):
                    return
                sel = [tgt - off]
            pieces = []
            for s in range(0, len(sel), self.col_chunk):
                idx = sel[s:s + self.col_chunk]
                xs = x[idx]
                q = module.q_projection(xs[:, N:]).view(len(idx), -1, H, D).transpose(1, 2)
                kf = module.k_projection(xs[:, :N]).view(len(idx), N, H, D).transpose(1, 2)
                # Query rows use multi-query attention: every head reads key head 0.
                a = torch.softmax(
                    q.float() @ kf[:, :1].float().transpose(-1, -2) / D ** 0.5, dim=-1)
                if layer == 0 and "between_rows" not in self.check:
                    vf = module.v_projection(xs[:, :N]).view(len(idx), N, H, D).transpose(1, 2)
                    ref = module.out_projection(
                        (a.to(vf.dtype) @ vf[:, :1]).transpose(1, 2)
                        .reshape(len(idx), -1, H * D))
                    self.check["between_rows"] = float(
                        (ref - output[0][idx][:, N:]).abs().max())
                if self.full_target:
                    for j, c in enumerate(idx):
                        if off + c == self.n_cols - 1:
                            self.tgt.setdefault(layer, []).append(a[j].half().cpu())
                pieces.append(self._compress(a))
            self.col.setdefault(layer, []).append((off + sel[0], pieces))
        return hook

    def _compress(self, a: torch.Tensor):
        if self.topk <= 0:
            return (a.half().cpu(), None)
        v, i = a.topk(min(self.topk, a.shape[-1]), dim=-1)
        # int16 indexes a context of up to 32767 rows and halves the index storage,
        # which otherwise dominates the file.
        idx_dtype = torch.int16 if self.n_ctx < 32767 else torch.int32
        return (v.half().cpu(), i.to(idx_dtype).cpu())

    # -- assemble -------------------------------------------------------------------
    def stack_row(self) -> np.ndarray:
        """[L, M, H, C, C] float16 -- between-feature attention of every query row."""
        return torch.stack([
            torch.cat([t for _, t in sorted(self.row[layer], key=lambda p: p[0])], 0)
            for layer in sorted(self.row)]).numpy()

    def stack_col(self):
        """[L, Csel, H, M, N] float16, plus int32 indices when --topk is set."""
        vs, idxs = [], []
        for layer in sorted(self.col):
            calls = sorted(self.col[layer], key=lambda p: p[0])
            vs.append(torch.cat([torch.cat([p[0] for p in pieces], 0)
                                 for _, pieces in calls], 0))
            if self.topk > 0:
                idxs.append(torch.cat([torch.cat([p[1] for p in pieces], 0)
                                       for _, pieces in calls], 0))
        v = torch.stack(vs).numpy()
        return (v, torch.stack(idxs).numpy() if self.topk > 0 else None)

    def stack_target(self):
        """[L, H, M, N] float16 -- the target column, untruncated."""
        if not self.tgt:
            return None
        return torch.stack([self.tgt[layer][0] for layer in sorted(self.tgt)]).numpy()


# --------------------------------------------------------------------------- driving

def size_gb(L, C, H, M, N, topk) -> float:
    """GiB of between-row attention that one run will write."""
    per = topk * 4 if topk > 0 else N * 2          # fp16 values (+ int16 indices)
    return L * C * H * M * per / 2**30


def run_condition(clf, Xc, yc, Xt, yte, rec, evaluate_context, tag, out_dir, log):
    t0 = time.time()
    import contextlib
    with (rec if rec is not None else contextlib.nullcontext()):
        _, _, probs = evaluate_context(
            clf, Xc, yc, Xt,
            torch.as_tensor(np.array(yte), dtype=torch.long, device=Xc.device),
            need_grad=False)
    p = probs.float().cpu().numpy()
    m = with_derived(binary_metrics(p.astype(np.float64), yte))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "predictions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["test_index", "y_true", "p_class0", "p_class1", "pred", "correct"])
        for i in range(len(yte)):
            w.writerow([i, int(yte[i]), f"{p[i,0]:.6f}", f"{p[i,1]:.6f}",
                        int(p[i].argmax()), int(p[i].argmax() == yte[i])])
    arrays = {"probs": p.astype(np.float32), "y_test": np.asarray(yte, dtype=np.int64)}
    shapes = "attention not recorded"
    if rec is not None:
        col_v, col_i = rec.stack_col()
        feats = rec.stack_row()
        arrays["attn_between_rows"] = col_v
        arrays["attn_between_features"] = feats
        if col_i is not None:
            arrays["attn_between_rows_indices"] = col_i
        tgt = rec.stack_target()
        if tgt is not None:
            arrays["attn_between_rows_target_full"] = tgt
        shapes = f"between_rows{tuple(col_v.shape)} between_features{tuple(feats.shape)}"
    np.savez(out_dir / "attention.npz", **arrays)
    log(f"  {tag}: ce={m['ce']:.5f} acc={m['accuracy']:.4f} auc={m['roc_auc']:.4f} "
        f"mcc={m['mcc']:.4f} | {shapes} | {time.time()-t0:.0f}s")
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--train", type=Path, default=None,
                    help="context CSV (default: <dataset>/splits/selected/natural/context_5000.csv)")
    ap.add_argument("--test", type=Path, default=None,
                    help="query CSV (default: <dataset>/splits/test_attack_1000.csv)")
    ap.add_argument("--no-attention", action="store_true",
                    help="metrics and predictions only; skip recording any attention map")
    ap.add_argument("--row-percent", type=float, required=True, help="0 = clean baseline")
    ap.add_argument("--gpu", default="3")
    ap.add_argument("--out", type=Path, default=Path("results/hub_flip_attention"))
    ap.add_argument("--hub-metric", default="knn", choices=["knn", "all"])
    ap.add_argument("--hub-k", type=int, default=10, help="neighbours for --hub-metric knn")
    ap.add_argument("--n-runs", type=int, default=1,
                    help="the hub score is deterministic, so >1 only repeats the same flips")
    ap.add_argument("--columns", default="target", choices=["target", "all"],
                    help="which columns' between-row attention to store (all is huge)")
    ap.add_argument("--topk", type=int, default=0,
                    help="store only the top-K context weights per query (0 = full)")
    ap.add_argument("--n-test", type=int, default=0, help="0 = every test row")
    ap.add_argument("--max-gb", type=float, default=20.0)
    ap.add_argument("--force", action="store_true", help="write past --max-gb")
    ap.add_argument("--model-seed", type=int, default=0)
    ap.add_argument("--full-target-col", action="store_true",
                    help="with --columns all --topk K, also store the target column exactly")
    ap.add_argument("--no-self-check", action="store_true")
    a = ap.parse_args(argv)

    from tabfm_experiments.data import load_split
    from tabfm_experiments.runtime import build_tabpfn_v2, device_from_gpu, evaluate_context

    out_root = a.out / a.dataset / f"r{int(round(a.row_percent)):03d}"
    out_root.mkdir(parents=True, exist_ok=True)
    logf = open(out_root / "run.log", "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    device = device_from_gpu(a.gpu)
    ctx = a.train or DATASETS_DIR / a.dataset / "splits/selected/natural/context_5000.csv"
    tst = a.test or DATASETS_DIR / a.dataset / "splits/test_attack_1000.csv"
    Xtr_df, ytr, cols = load_split(ctx)
    Xte_df, yte, _ = load_split(tst)
    if a.n_test:
        yte, Xte_df = yte[:a.n_test], Xte_df.iloc[:a.n_test]
    Xc = torch.tensor(Xtr_df.to_numpy(np.float32), device=device)
    yc_clean = torch.tensor(ytr.astype(np.float32), device=device)
    Xt = torch.tensor(Xte_df.to_numpy(np.float32), device=device)
    N, M, F = len(ytr), len(yte), len(cols)
    L, H = 12, 6
    log(f"[{a.dataset}] context {N}x{F}, test {M}, R={a.row_percent}%")

    k = 0 if a.row_percent <= 0 else k_from_percent(N, a.row_percent)
    scores = hub_scores(Xc, a.hub_metric, a.hub_k)
    hubs = pick_hubs(scores, k) if k else np.array([], dtype=int)
    yc = yc_clean.clone()
    if k:
        t = torch.as_tensor(hubs, dtype=torch.long, device=device)
        yc[t] = 1.0 - yc[t]
        log(f"  hub metric={a.hub_metric} k={a.hub_k}: flipped {len(hubs)}/{N} rows; "
            f"their scores {scores[hubs].min():.4f}-{scores[hubs].max():.4f} "
            f"vs context {scores.min():.4f}-{scores.max():.4f}; "
            f"classes {np.bincount(ytr[hubs].astype(int), minlength=2).tolist()}")
    else:
        log("  clean baseline, no flips")

    clf = build_tabpfn_v2(device, a.model_seed)
    with torch.no_grad():
        clf.fit_with_differentiable_input(Xc, yc)
    # The fitted ensemble pipeline drops degenerate/constant columns (url_unique 63->62,
    # wids 108->106), so the token grid is narrower than ceil(F/2)+1 would suggest.
    f_eff = int(clf.executor_.ensemble_members[0].X_train.shape[1])
    C = -(-f_eff // 2) + 1
    n_cols = C if a.columns == "all" else 1
    gb = size_gb(L, n_cols, H, M, N, a.topk)
    feat_gb = L * H * M * C * C * 2 / 2**30
    log(f"  fitted pipeline keeps {f_eff}/{F} features -> {C} column tokens "
        f"({C - 1} feature groups + 1 target slot)")
    log(f"  per run: between_rows {L}x{n_cols}x{H}x{M}x{a.topk or N} = {gb:.2f} GiB"
        f"   between_features {L}x{M}x{H}x{C}x{C} = {feat_gb:.2f} GiB")
    if not a.no_attention and gb + feat_gb > a.max_gb and not a.force:
        raise SystemExit(f"would write {gb + feat_gb:.1f} GiB > --max-gb {a.max_gb}; "
                         f"use --topk / --n-test / --columns target, or --force")
    rec = None if a.no_attention else AttentionRecorder(
        clf.models_[0], N, C, a.columns, a.topk, full_target=a.full_target_col)

    results = []
    for run in range(a.n_runs):
        results.append(run_condition(clf, Xc, yc, Xt, yte, rec, evaluate_context,
                                     f"run{run}", out_root / f"run{run}", log))
        if rec is not None and not a.no_self_check:
            log("    self-check  max|recomputed attention @ V - module output|: "
                + "  ".join(f"{k_}={v:.2e}" for k_, v in rec.check.items()))
    if rec is not None:
        rec.close()

    write_json(out_root / "summary.json", {
        "dataset": a.dataset, "row_percent": a.row_percent, "k_flipped": int(k),
        "attack": "hub-flip (test-agnostic)", "hub_metric": a.hub_metric, "hub_k": a.hub_k,
        "deterministic": True, "n_runs": a.n_runs, "n_context": N, "n_test": M,
        "n_features": F, "n_features_after_pipeline": f_eff, "n_column_tokens": C, "n_layers": L, "n_heads": H,
        "columns_stored": a.columns, "topk": a.topk,
        "full_target_column": bool(a.full_target_col and a.columns == "all"),
        "attn_between_rows_axes": "[layer, column, head, query_row, context_row]",
        "attn_between_features_axes": "[layer, query_row, head, column, column]",
        "flipped_indices": hubs.tolist(),
        "attention_recorded": rec is not None,
        "self_check_max_abs_err": (rec.check if rec is not None else {}),
        "metrics_per_run": results,
    })
    log(f"wrote {out_root}")
    logf.close()
    return results


if __name__ == "__main__":
    main()
