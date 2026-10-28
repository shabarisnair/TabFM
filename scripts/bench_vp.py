#!/usr/bin/env python
"""The V + P benchmark: given a tiny clean set V and a label-poisoned train set P (and R), which method makes TabPFN v2
score best on a held-out test set? Any size of training context is allowed.

Instances: the 6 datasets of bench_tinyv (a fixed train set, the context file, and a fixed test set) x V = c clean rows
per class drawn from the train set (c = 1, 3, 5) x seeds 0-4 x two poisonings of the rest, P:
  rand45  label-flip-bernoulli at R = 45%: every row, with probability R, moved to a uniformly random other class;
          the flips follow the seed (run_id = seed)
  hub30   hub-flip-local at R = 30%: the R% densest rows moved to their locally closest other class (deterministic given
          P, so its five seeds differ mainly in V)
k = round(R% of |P|) is what a method that uses R removes. Every method is scored by TabPFN on the test set with the
context it builds: ROC-AUC (binary: class 1; multi-class: one-vs-rest macro), balanced accuracy, log loss and accuracy
(schedule_study.metrics); a method that keeps a subset of P also by the precision and recall of what it removes and
the poisoned share of what it keeps.

Jobs (one per instance; the rows each reports, numbered as the benchmark's methods):
  refs       1 v_only, 9 v_plus_p, 10 oracle (P's clean rows + V)
  vfilter    2 vfilter_R: the k rows whose given label TabPFN with V alone finds least probable, removed;
             3 vfilter_thr: the rows with p(given label) < 0.5 / K removed (tuned on development draws:
               results/tables/vp_tune/findings.md)
  greedy     4 greedy_R: greedy at 1% per step (bench_tinyv's greedy_pct1: the trusted set starts as V; every step
               TabPFN with V + the accepted rows scores the rest and the ceil(1% of |P|) most plausible join), stopped at
               (1 - R)% of P (that batch cut to reach it exactly);
             5 greedy_R10: the same run, 10 steps further
  baselines  8 knn_shapley (its own rule: rows of negative KNN-Shapley value removed) and ieg (its top R% removed), at
             bench_tinyv's default configurations, trained on CPU; their other forms are recorded as well
             (knn_shapley_R, ieg_own: the rows IEG relabels removed)
  sigmoid    6 attention weights w = sigmoid(t) trained on V's log loss (all of V the queries, P weighted as the
             context): hard V-filter start (t = -30 for vfilter_R's k rows, 4 for the rest), Adam lr 0.01, 70
             iterations; deployed soft + V (P at its weights, V at weight 1) at a fixed iteration, 10 (sigmoid: tuned on
             development draws, where training only overfits V and slowly raises the test log loss); the last
             iteration (sigmoid_last) and the start (sigmoid_start) are recorded too
  m1         7 M1: top-k scores started at the fitted V's V-filter log p, gradient at the relaxed point (probe 0.05),
             PGD-L-inf lr 0.2, 400 iterations; deployed hard + V (the k lowest-scored rows removed) at the last
             iteration (m1_last) and at the held-out choice (m1_sel)
  M1's held-out part of V: c = 3 and 5, one row per class (gate_context.split_holdout); the rest is fitted. c = 1:
  none (a class would lose its only row), so m1_sel is the last iteration there.
  greedy_thr 11 greedy_thr: greedy at 1% per step with a threshold: every step only the rows whose given label TabPFN
               with V + the accepted rows finds more probable than 0.5 may join, the ceil(1% of |P|) most probable of
               them; it stops at the first step where none qualifies. R is not used.
  The repair methods never look at P's given labels: P is unlabelled, and TabPFN labels it. With V and P's rows the
  same for both poisonings (build_draw), they give the same context under rand45 and hub30, so the second of the two
  instances copies the first's result (repair_twin) and recomputes only what depends on the poisoning.
  vrepair    12 vrepair: TabPFN with V alone as the context predicts every row of P; the rows it is most sure of
               (REPAIR["one_shot"], tuned on development draws: scripts/vp_tune_repair.py) join V with the predicted
               labels
  grepair    13 grepair: the iterative form: the trusted set starts as V; every step TabPFN with V + the rows labelled
               so far predicts the rest, and the ceil(1% of |P|) it is most sure of join with their predicted labels
               (REPAIR["greedy"]: how they are chosen and where it stops, tuned on development draws)
Intermediate records (in the result JSON and, with --wandb-mode, in a W&B run per job):
  greedy     every step: the next batch's and the pool's statistics (TabPFN's probability of the given label, the
             agreement of the given label with its prediction, margin, entropy, the true poisoned shares); the test
             metrics every 2 steps and at both stops
  greedy_thr as greedy, with the number of pool rows above the threshold; the test metrics every 2 steps and at the stop
  grepair    every step: the batch's and the pool's confidence, the accuracy of the labels given (against the true
             labels, never used by the method); the test metrics every REPAIR["greedy"]["test_every"] steps and at
             the stop
  sigmoid/m1 every iteration up to 10, then every 5: the fitted V's log loss (the training objective), the deployment's
             test metrics, the weights' AUROC against the poison; M1 also its held-out V's log loss / accuracy /
             ROC-AUC, its removal precision and its swaps since its start

    python scripts/bench_vp.py draws                                  # once, on the main machine (CPU)
    python scripts/bench_vp.py run --device cuda:0 [--job-list FILE] [--jobs sigmoid m1] [--wandb-mode online]
    python scripts/bench_vp.py run --device cpu --threads 8 ...      # on CPU: torch threads from --threads
    python scripts/bench_vp.py status
Results: results/bench_vp/runs/<instance>/<job>.json (+ .npz); BENCH_VP_ROOT moves the root (for tests).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_tinyv as B  # noqa: E402

ROOT = Path(os.environ.get("BENCH_VP_ROOT", "results/bench_vp"))
ATTACKS = {"rand45": ("label-flip-bernoulli", 45.0), "hub30": ("hub-flip-local", 30.0)}
DATASETS = tuple(B.DATASETS)
CS = (1, 3, 5)
SEEDS = (0, 1, 2, 3, 4)
JOBS = ("refs", "vfilter", "greedy", "baselines", "sigmoid", "m1", "greedy_thr", "vrepair", "grepair")
REPAIR_JOBS = ("vrepair", "grepair")
THR = 0.5          # vfilter_thr: p(given label | V) < THR / K removed
EXTRA_STEPS = 10   # greedy_R10: steps past (1 - R)%
TEST_EVERY = 2     # greedy, greedy_thr: test metrics every this many steps
GREEDY_THR = 0.5   # greedy_thr: a row may join only while p(given label | V + the accepted rows) > GREEDY_THR
# the repair methods: how the rows TabPFN is most sure of are chosen (by its max probability); to be set from
# scripts/vp_tune_repair.py's development runs (results/tables/vp_tune/report_repair.md). Until then REPAIR holds
# placeholders and the two repair jobs refuse to run (REPAIR_TUNED).
REPAIR = {"one_shot": {"rule": "top", "x": 1.0},
          "greedy": {"select": "global", "step": 0.01, "until": 1.0, "test_every": 5}}
REPAIR_TUNED = False
DENSE, EVERY = 10, 5  # sigmoid / m1: records at every iteration below DENSE, then every EVERY
SIGMOID = dict(steps=70, lr=0.01, t_kept=4.0, t_removed=-30.0, fixed_it=10)  # fixed_it: the deployed iteration (tuned)
M1 = dict(steps=400, lr=0.2, probe=0.05)
BASELINE_PRIMARY = {"knn_shapley": "own", "ieg": "R"}


def tag_of(ds: str, attack: str, c: int, seed: int) -> str:
    return f"{ds}_{attack}_c{c}_s{seed}"


def parse_tag(tag: str) -> tuple[str, str, int, int]:
    head, seed = tag.rsplit("_s", 1)
    head, c = head.rsplit("_c", 1)
    ds, attack = head.rsplit("_", 1)
    return ds, attack, int(c), int(seed)


def instances() -> list[str]:
    return [tag_of(ds, a, c, s) for ds in DATASETS for a in ATTACKS for c in CS for s in SEEDS]


# ---------------------------------------------------------------- draws

def build_draw(bd: B.BenchData, attack: str, c: int, seed: int) -> dict:
    """V: bench_small_v.draw_per_class (c rows per class, seeded); P: the rest, poisoned (scripts/vp_tune.py's draws
    with another seed)."""
    import vp_tune
    return vp_tune.build_draw(bd, attack, c, seed)


def load_draw(bd: B.BenchData, tag: str) -> dict:
    ds, attack, c, seed = parse_tag(tag)
    z = np.load(ROOT / "draws" / f"{tag}.npz")
    d = {key: z[key] for key in ("v_rows", "ctx_rows", "y_given", "pois_rows")}
    d |= {"k": int(z["k"]), "R": float(z["R"]), "n": len(d["ctx_rows"]), "seed": seed, "c": c, "attack": attack, "ds": ds,
          "K": bd.K, "tag": tag}
    d["pois"] = np.zeros(d["n"], dtype=bool)
    d["pois"][d["pois_rows"]] = True
    d["Xc"], d["Xv"] = bd.X[bd.idx(d["ctx_rows"])], bd.X[bd.idx(d["v_rows"])]
    d["yv"], d["y_true"] = np.asarray(bd.y[d["v_rows"]]).astype(np.int64), np.asarray(bd.y[d["ctx_rows"]]).astype(np.int64)
    return d


def holdout_split(d: dict) -> tuple[np.ndarray, np.ndarray]:
    """(fitted, held out) positions in V: one row per class held out when c >= 3; none when c = 1."""
    import gate_context as G
    if d["c"] < 3:
        return np.arange(len(d["yv"])), np.zeros(0, dtype=np.int64)
    fit, hold = G.split_holdout(d["yv"], 1.0 / d["c"], d["seed"])
    if not np.array_equal(np.bincount(d["yv"][hold], minlength=d["K"]), np.ones(d["K"], dtype=np.int64)):
        raise AssertionError(f"{d['tag']}: the held-out part is not one row per class")
    return fit, hold


# ---------------------------------------------------------------- shared pieces

def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def removal_stats(d: dict, removed: np.ndarray) -> dict:
    """The rows of P a method removes: how many, their poisoned share (precision), the share of the poison they catch
    (recall), and the poisoned share of what is kept."""
    pois = d["pois"]
    return {"n_removed": int(removed.sum()), "share_removed": float(removed.mean()),
            "prec": float(pois[removed].mean()) if removed.any() else None,
            "recall": float(pois[removed].sum() / max(int(pois.sum()), 1)),
            "kept_pois_share": float(pois[~removed].mean()) if (~removed).any() else None}


def v_fit(bd: B.BenchData, d: dict, keep: np.ndarray) -> dict:
    """The train-side measure of a kept subset: V's log loss and accuracy with the kept rows of P alone as the context
    (bench_tinyv.v_score; log loss None when the kept rows lack a class)."""
    ce, acc = B.v_score(bd, d, keep, None)
    return {"v_ce": float(ce) if math.isfinite(ce) else None, "v_acc": float(acc)}


def subset_row(bd: B.BenchData, d: dict, name: str, removed: np.ndarray, extra: dict | None = None) -> dict:
    return ({"name": name, "test": B.test_scores(bd, d, ~removed)} | removal_stats(d, removed) | v_fit(bd, d, ~removed)
            | (extra or {}))


def batch_stats(probs: np.ndarray, yg: np.ndarray, K: int) -> dict:
    """Statistics of a set of rows' predicted distributions (probs, m x K) against their given labels."""
    m = len(yg)
    p = probs[np.arange(m), yg]
    other = probs.copy()
    other[np.arange(m), yg] = -1.0
    ent = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(1) / math.log(K)
    return {"p_mean": float(p.mean()), "p_min": float(p.min()), "agree": float((probs.argmax(1) == yg).mean()),
            "margin": float((p - other.max(1)).mean()), "maxprob": float(probs.max(1).mean()), "entropy": float(ent.mean())}


class Recorder:
    """A W&B run for one job (or nothing, with --wandb-mode disabled): per-step metrics, the result rows as the summary,
    the result files as an artifact."""

    def __init__(self, a, job: str, d: dict, config: dict):
        self.run = None
        if a.wandb_mode == "disabled":
            return
        import wandb
        ROOT.mkdir(parents=True, exist_ok=True)
        self.run = wandb.init(project=a.wandb_project, entity=a.wandb_entity, group=a.wandb_group or f"vp_{job}",
                              job_type=job, name=f"{job}_{d['tag']}", tags=[job, d["ds"], d["attack"], f"c{d['c']}"],
                              mode=a.wandb_mode, dir=str(ROOT),
                              config={"job": job, "instance": d["tag"], "dataset": d["ds"], "attack": d["attack"],
                                      "R": d["R"], "c": d["c"], "seed": d["seed"], "n": d["n"], "k": d["k"],
                                      "n_flipped": int(d["pois"].sum()), "K": d["K"], "host": socket.gethostname()} | config)

    def log(self, data: dict, step: int) -> None:
        if self.run is not None:
            self.run.log(data, step=step)

    def finish(self, res: dict, files: list[Path], ok: bool = True) -> None:
        if self.run is None:
            return
        import wandb
        if ok:
            for r in res.get("rows", []):
                self.run.summary.update({f"final/{r['name']}/{m}": v for m, v in r["test"].items()})
                self.run.summary.update({f"final/{r['name']}/{m}": r[m] for m in
                                         ("prec", "recall", "share_removed", "kept_pois_share", "it", "v_ce", "v_acc", "train_ce",
                                          "share_kept", "label_acc", "steps") if r.get(m) is not None})
            art = wandb.Artifact(f"vp-{self.run.name}", type="vp-run")
            for f in files:
                if f.exists():
                    art.add_file(str(f))
            self.run.log_artifact(art)
        self.run.finish(exit_code=0 if ok else 1)


def flat(prefix: str, m: dict) -> dict:
    return {f"{prefix}/{k}": v for k, v in m.items() if v is not None}


# ---------------------------------------------------------------- jobs

def job_refs(bd: B.BenchData, d: dict, rec: Recorder) -> tuple[dict, dict]:
    rows = [{"name": "v_only", "test": B.metrics(bd.probs(d["Xv"], d["yv"], bd.Xt, bd.yt), bd.yt)},
            {"name": "v_plus_p", "test": B.test_scores(bd, d, np.ones(d["n"], dtype=bool))} | removal_stats(d, np.zeros(d["n"], dtype=bool))
            | v_fit(bd, d, np.ones(d["n"], dtype=bool)),
            subset_row(bd, d, "oracle", d["pois"])]
    return {"rows": rows}, {}


def job_vfilter(bd: B.BenchData, d: dict, rec: Recorder) -> tuple[dict, dict]:
    n, k, K, yg = d["n"], d["k"], d["K"], d["y_given"]
    P = bd.probs(d["Xv"], d["yv"], d["Xc"], yg)
    p = P[np.arange(n), yg]
    removed_R = np.zeros(n, dtype=bool)
    removed_R[np.argsort(p, kind="stable")[:k]] = True
    removed_thr = p < THR / K
    train = {"agree_with_given": float((P.argmax(1) == yg).mean()), "p_mean": float(p.mean())}
    rows = [subset_row(bd, d, "vfilter_R", removed_R), subset_row(bd, d, "vfilter_thr", removed_thr, {"threshold": THR / K})]
    return {"rows": rows, "train": train}, {"P": P.astype(np.float32)}


def job_greedy(bd: B.BenchData, d: dict, rec: Recorder) -> tuple[dict, dict]:
    n, k, K, yg = d["n"], d["k"], d["K"], d["y_given"]
    m = math.ceil(0.01 * n)
    stop4 = n - k
    stop5 = min(n, stop4 + EXTRA_STEPS * m)
    in_T = np.zeros(n, dtype=bool)
    pos = np.full(n, n, dtype=np.int64)
    test0 = B.metrics(bd.probs(d["Xv"], d["yv"], bd.Xt, bd.yt), bd.yt)
    steps, rows, step = [], [], 0
    rec.log(flat("test", test0) | {"greedy/accepted_share": 0.0}, step=0)
    while int(in_T.sum()) < stop5:
        accepted = int(in_T.sum())
        pool = np.flatnonzero(~in_T)
        acc_rows = np.flatnonzero(in_T)
        X_ctx = torch.cat([d["Xc"][bd.idx(acc_rows)], d["Xv"]])
        y_ctx = np.concatenate([yg[acc_rows], d["yv"]])
        probs = bd.probs(X_ctx, y_ctx, d["Xc"][bd.idx(pool)], yg[pool])
        p = probs[np.arange(len(pool)), yg[pool]]
        s = -np.log(np.clip(p, 1e-12, None))
        size = min(m, stop4 - accepted) if accepted < stop4 else min(m, stop5 - accepted)
        order = np.argsort(s, kind="stable")[:size]
        batch = pool[order]
        step += 1
        st = {"step": step, "accepted_before": accepted, "size": size, "batch": batch_stats(probs[order], yg[batch], K),
              "pool": batch_stats(probs, yg[pool], K), "batch_pois": float(d["pois"][batch].mean()),
              "pool_pois": float(d["pois"][pool].mean())}
        in_T[batch] = True
        pos[batch] = accepted + np.arange(size)
        acc_now = int(in_T.sum())
        st["kept_pois_share"] = float(d["pois"][in_T].mean())
        at_stop = acc_now in (stop4, stop5)
        if step % TEST_EVERY == 0 or at_stop:
            st["test"] = B.test_scores(bd, d, in_T)
            st["train"] = v_fit(bd, d, in_T)
        steps.append(st)
        rec.log({"greedy/accepted_share": acc_now / n, "greedy/kept_pois_share": st["kept_pois_share"],
                 "greedy/batch_pois": st["batch_pois"]} | flat("batch", st["batch"]) | flat("pool", st["pool"])
                | (flat("test", st["test"]) | flat("train", st["train"]) if "test" in st else {}), step=step)
        if acc_now == stop4:
            rows.append({"name": "greedy_R", "test": st["test"], "steps": step} | removal_stats(d, ~in_T) | st["train"])
        if acc_now == stop5:
            rows.append({"name": "greedy_R10", "test": st["test"], "steps": step} | removal_stats(d, ~in_T) | st["train"])
    return {"rows": rows, "steps": steps, "test_start": test0, "m": m, "stops": [stop4, stop5]}, {"pos": pos}


def job_greedy_thr(bd: B.BenchData, d: dict, rec: Recorder) -> tuple[dict, dict]:
    """Method 11: greedy at 1% per step as job_greedy, but each step takes only rows with p(given label) > GREEDY_THR,
    at most ceil(1% of |P|) of them, and it stops at the first step where none qualifies (or P is used up)."""
    n, K, yg = d["n"], d["K"], d["y_given"]
    m = math.ceil(0.01 * n)
    in_T = np.zeros(n, dtype=bool)
    pos = np.full(n, n, dtype=np.int64)
    test0 = B.metrics(bd.probs(d["Xv"], d["yv"], bd.Xt, bd.yt), bd.yt)
    steps, step, last = [], 0, {"test": test0, "train": v_fit(bd, d, in_T)}
    rec.log(flat("test", test0) | {"greedy/accepted_share": 0.0}, step=0)
    while not in_T.all():
        accepted = int(in_T.sum())
        pool = np.flatnonzero(~in_T)
        acc_rows = np.flatnonzero(in_T)
        X_ctx = torch.cat([d["Xc"][bd.idx(acc_rows)], d["Xv"]])
        y_ctx = np.concatenate([yg[acc_rows], d["yv"]])
        probs = bd.probs(X_ctx, y_ctx, d["Xc"][bd.idx(pool)], yg[pool])
        p = probs[np.arange(len(pool)), yg[pool]]
        n_above = int((p > GREEDY_THR).sum())
        size = min(m, n_above)
        st = {"step": step + 1, "accepted_before": accepted, "n_above": n_above, "size": size,
              "pool": batch_stats(probs, yg[pool], K), "pool_pois": float(d["pois"][pool].mean())}
        if size == 0:  # no row left whose given label is more probable than not: stop
            steps.append(st)
            break
        # most probable first, as job_greedy: the rows above the threshold are a prefix of this order
        order = np.argsort(-np.log(np.clip(p, 1e-12, None)), kind="stable")[:size]
        batch = pool[order]
        step += 1
        st |= {"batch": batch_stats(probs[order], yg[batch], K), "batch_pois": float(d["pois"][batch].mean())}
        in_T[batch] = True
        pos[batch] = accepted + np.arange(size)
        st["kept_pois_share"] = float(d["pois"][in_T].mean())
        if step % TEST_EVERY == 0:
            st["test"] = B.test_scores(bd, d, in_T)
            st["train"] = v_fit(bd, d, in_T)
            last = {"test": st["test"], "train": st["train"], "step": step}
        steps.append(st)
        rec.log({"greedy/accepted_share": float(in_T.mean()), "greedy/kept_pois_share": st["kept_pois_share"],
                 "greedy/batch_pois": st["batch_pois"], "greedy/n_above": n_above} | flat("batch", st["batch"])
                | flat("pool", st["pool"]) | (flat("test", st["test"]) | flat("train", st["train"]) if "test" in st else {}),
                step=step)
    if last.get("step") != step:  # the test metrics at the stop, unless that state was just scored
        last = {"test": B.test_scores(bd, d, in_T) if in_T.any() else test0, "train": v_fit(bd, d, in_T), "step": step}
        rec.log(flat("test", last["test"]) | flat("train", last["train"]), step=step)
    row = {"name": "greedy_thr", "test": last["test"], "steps": step} | removal_stats(d, ~in_T) | last["train"]
    return {"rows": [row], "steps": steps, "test_start": test0, "m": m, "threshold": GREEDY_THR}, {"pos": pos}


# ---------------------------------------------------------------- the repair methods (P's given labels unused)

def predict_pool(bd: B.BenchData, d: dict, rows: np.ndarray, labels: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """TabPFN's class probabilities for P's rows `pool`, with V + P's rows `rows` (labelled `labels`) as the context."""
    X_ctx = torch.cat([d["Xc"][bd.idx(rows)], d["Xv"]])
    return bd.probs(X_ctx, np.concatenate([np.asarray(labels, dtype=np.int64), d["yv"]]), d["Xc"][bd.idx(pool)])


def class_quota(m: int, pred: np.ndarray, K: int) -> np.ndarray:
    """m split over the predicted classes in proportion to their counts in pred (largest remainder)."""
    cnt = np.bincount(pred, minlength=K).astype(np.float64)
    share = m * cnt / max(cnt.sum(), 1.0)
    q = np.floor(share).astype(np.int64)
    for j in np.argsort(-(share - q), kind="stable")[: m - int(q.sum())]:
        q[j] += 1
    return np.minimum(q, cnt.astype(np.int64))


def most_sure(Q: np.ndarray, size: int, select: str) -> np.ndarray:
    """Positions (into Q's rows) of the `size` rows TabPFN is most sure of (max probability, ties in file order):
    over all rows (global) or within each predicted class, in proportion to the classes' predicted counts (cls)."""
    conf, pred = Q.max(1), Q.argmax(1)
    order = np.argsort(-conf, kind="stable")
    if select == "global":
        return order[:size]
    out = [order[pred[order] == j][:q] for j, q in enumerate(class_quota(size, pred, Q.shape[1]))]
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def one_shot_select(Q: np.ndarray, rule: str, x: float) -> np.ndarray:
    """The rows of P (a mask) that the one-shot repair labels: the top x of P by confidence (top), the top x of each
    predicted class (cls), or the rows with confidence > x (thr)."""
    n = len(Q)
    keep = np.zeros(n, dtype=bool)
    if rule == "thr":
        return Q.max(1) > x
    keep[most_sure(Q, int(round(x * n)), "global" if rule == "top" else "cls")] = True
    return keep


def label_acc_stats(d: dict, keep: np.ndarray, labels: np.ndarray) -> dict:
    """What a repair method puts in the context from P: how many rows, and the accuracy of the labels it gives them
    (against the true labels, never seen by the method). The same under both poisonings."""
    ok = labels[keep] == d["y_true"][keep]
    return {"n_kept": int(keep.sum()), "share_kept": float(keep.mean()),
            "label_acc": float(ok.mean()) if keep.any() else None,
            "kept_wrong_share": float(1 - ok.mean()) if keep.any() else None}


def label_stats(d: dict, keep: np.ndarray, labels: np.ndarray) -> dict:
    """label_acc_stats, and against this poisoning: the poisoned share of the kept rows' given labels, and how many
    poisoned rows got their true label back."""
    return label_acc_stats(d, keep, labels) | {
        "kept_pois_share": float(d["pois"][keep].mean()) if keep.any() else None,
        "repaired": int((keep & d["pois"] & (labels == d["y_true"])).sum())}


def repair_path(bd: B.BenchData, d: dict, select: str, step: float, until: float, test_every: int,
                taus: tuple = (0.5, 0.6, 0.7, 0.8, 0.9), rec: Recorder | None = None) -> dict:
    """The iterative repair: the trusted set starts as V; every step TabPFN with V + the rows labelled so far predicts
    the pool (the rest of P), and the m = ceil(step x |P|) rows it is most sure of (most_sure: global or cls) join with
    their predicted labels, until round(until x |P|) rows are labelled. Test metrics of V + the labelled rows every
    test_every steps and at the end."""
    n, K = d["n"], d["K"]
    m = math.ceil(step * n)
    target = min(n, int(round(until * n)))
    in_T = np.zeros(n, dtype=bool)
    lab = np.full(n, -1, dtype=np.int64)
    pos = np.full(n, n, dtype=np.int64)
    states = [{"step": 0, "accepted": 0, "test": B.metrics(bd.probs(d["Xv"], d["yv"], bd.Xt, bd.yt), bd.yt)}]
    steps, step_no = [], 0
    while int(in_T.sum()) < target:
        accepted = int(in_T.sum())
        pool = np.flatnonzero(~in_T)
        rows = np.flatnonzero(in_T)
        Q = predict_pool(bd, d, rows, lab[rows], pool)
        conf, pred = Q.max(1), Q.argmax(1)
        size = min(m, target - accepted)
        pick = most_sure(Q, size, select)
        batch = pool[pick]
        step_no += 1
        y_pool = d["y_true"][pool]
        st = {"step": step_no, "accepted_before": accepted, "size": int(len(batch)),
              "batch": {"conf_mean": float(conf[pick].mean()), "conf_min": float(conf[pick].min()),
                        "label_acc": float((pred[pick] == y_pool[pick]).mean()),
                        "classes": np.bincount(pred[pick], minlength=K).tolist()},
              "pool": {"conf_mean": float(conf.mean()), "acc": float((pred == y_pool).mean()),
                       "classes": np.bincount(pred, minlength=K).tolist()}
              | {f"above{t:g}": int((conf > t).sum()) for t in taus}}
        in_T[batch] = True
        lab[batch] = pred[pick]
        pos[batch] = accepted + np.arange(len(batch))
        st["label_acc_T"] = float((lab[in_T] == d["y_true"][in_T]).mean())
        if step_no % test_every == 0 or int(in_T.sum()) >= target:
            states.append({"step": step_no, "accepted": int(in_T.sum()), "test": B.test_scores(bd, d, in_T, lab)}
                          | label_acc_stats(d, in_T, lab))
        steps.append(st)
        if rec is not None:
            rec.log({"repair/accepted_share": float(in_T.mean()), "repair/label_acc_T": st["label_acc_T"],
                     "repair/batch_label_acc": st["batch"]["label_acc"], "repair/batch_conf": st["batch"]["conf_mean"],
                     "repair/pool_conf": st["pool"]["conf_mean"], "repair/pool_acc": st["pool"]["acc"]}
                    | (flat("test", states[-1]["test"]) if states[-1]["step"] == step_no else {}), step=step_no)
    return {"m": m, "target": target, "steps": steps, "states": states, "in_T": in_T, "lab": lab, "pos": pos}


def twin_tag(tag: str) -> str:
    ds, attack, c, seed = parse_tag(tag)
    return tag_of(ds, next(a for a in ATTACKS if a != attack), c, seed)


def repair_twin(d: dict, job: str) -> tuple[dict, dict] | None:
    """The other poisoning's finished result of a repair job, when its V and P rows are this instance's: (result,
    arrays), else None."""
    other = twin_tag(d["tag"])
    src = ROOT / "runs" / other / f"{job}.json"
    if not src.exists() or not src.with_suffix(".npz").exists():
        return None
    z = np.load(ROOT / "draws" / f"{other}.npz")
    if not (np.array_equal(z["v_rows"], d["v_rows"]) and np.array_equal(z["ctx_rows"], d["ctx_rows"])):
        return None
    arrays = dict(np.load(src.with_suffix(".npz")))
    return json.loads(src.read_text()), arrays


def require_tuned() -> None:
    if not REPAIR_TUNED:
        raise RuntimeError("REPAIR holds placeholders: set it from the development study (scripts/vp_tune_repair.py) "
                           "and REPAIR_TUNED = True first")


def job_vrepair(bd: B.BenchData, d: dict, rec: Recorder) -> tuple[dict, dict]:
    """Method 12: TabPFN with V alone predicts P; the rows chosen by REPAIR["one_shot"] join V, predicted labels."""
    require_tuned()
    cfg = REPAIR["one_shot"]
    twin = repair_twin(d, "vrepair")
    if twin is not None:  # the same context as the other poisoning's instance: its test metrics hold here
        res, arrays = twin
        keep, lab = arrays["keep"].astype(bool), arrays["labels"].astype(np.int64)
        row = {"name": "vrepair", "test": res["rows"][0]["test"]} | label_stats(d, keep, lab) | v_fit_labels(bd, d, keep, lab)
        return {"rows": [row], "config": cfg, "copied_from": twin_tag(d["tag"])}, arrays
    Q = predict_pool(bd, d, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.arange(d["n"]))
    keep = one_shot_select(Q, cfg["rule"], cfg["x"])
    lab = Q.argmax(1).astype(np.int64)
    test = B.test_scores(bd, d, keep, lab) if keep.any() else B.metrics(bd.probs(d["Xv"], d["yv"], bd.Xt, bd.yt), bd.yt)
    row = {"name": "vrepair", "test": test} | label_stats(d, keep, lab) | v_fit_labels(bd, d, keep, lab)
    rec.log(flat("test", test) | {"repair/share_kept": row["share_kept"], "repair/label_acc": row["label_acc"]}, step=0)
    return {"rows": [row], "config": cfg}, {"Q": Q.astype(np.float32), "keep": keep, "labels": lab.astype(np.int16)}


def job_grepair(bd: B.BenchData, d: dict, rec: Recorder) -> tuple[dict, dict]:
    """Method 13: the iterative repair (repair_path) with REPAIR["greedy"]."""
    require_tuned()
    cfg = REPAIR["greedy"]
    twin = repair_twin(d, "grepair")
    if twin is not None:
        res, arrays = twin
        keep, lab = arrays["pos"] < d["n"], arrays["labels"].astype(np.int64)
        row = {"name": "grepair", "test": res["rows"][0]["test"], "steps": res["rows"][0]["steps"]} \
            | label_stats(d, keep, lab) | v_fit_labels(bd, d, keep, lab)
        return {"rows": [row], "steps": res["steps"], "states": res["states"], "m": res["m"], "config": cfg,
                "copied_from": twin_tag(d["tag"])}, arrays
    path = repair_path(bd, d, cfg["select"], cfg["step"], cfg["until"], cfg["test_every"], rec=rec)
    keep, lab = path["in_T"], path["lab"]
    row = {"name": "grepair", "test": path["states"][-1]["test"], "steps": len(path["steps"])} \
        | label_stats(d, keep, lab) | v_fit_labels(bd, d, keep, lab)
    return ({"rows": [row], "steps": path["steps"], "states": path["states"], "m": path["m"], "config": cfg},
            {"pos": path["pos"], "labels": lab.astype(np.int16)})


def v_fit_labels(bd: B.BenchData, d: dict, keep: np.ndarray, labels: np.ndarray) -> dict:
    """v_fit for a repaired subset: V's log loss and accuracy with the labelled rows of P alone as the context."""
    if not keep.any():
        return {"v_ce": None, "v_acc": None}
    ce, acc = B.v_score(bd, d, keep, labels)
    return {"v_ce": float(ce) if math.isfinite(ce) else None, "v_acc": float(acc)}


def job_baselines(bd: B.BenchData, d: dict, rec: Recorder, threads: int) -> tuple[dict, dict]:
    """KNN-Shapley and IEG at their defaults, trained on CPU (as in the tuning), scored on bd's device."""
    import small_v_methods as M
    from bench_small_v import top_k
    torch.set_num_threads(threads)
    bd_cpu = B.BenchData(d["ds"], "cpu")
    d_cpu = load_draw(bd_cpu, d["tag"])
    feats = B.Feats(bd_cpu, d_cpu)
    rows, arrays, train = [], {}, {}
    for method in ("knn_shapley", "ieg"):
        kw = dict(B.DEFAULTS[method])
        feat = kw.pop("feat", "z")
        t0 = time.time()
        res = M.METHODS[method](feats.draw(feat), **kw)
        train[method] = {"seconds": round(time.time() - t0, 1), "config": B.DEFAULTS[method]}
        score = np.asarray(res.score, dtype=np.float64)
        arrays[f"{method}_score"] = score
        top = np.zeros(d["n"], dtype=bool)
        top[top_k(score, d["k"], d["seed"])] = True
        if method == "knn_shapley":
            forms = {"own": score > 0, "R": top}  # score = -value: value < 0
        else:
            relabel = np.asarray(res.relabel, dtype=np.int64)
            arrays["ieg_relabel"] = relabel.astype(np.int16)
            forms = {"R": top, "own": relabel != d["y_given"]}
        for form, removed in forms.items():
            name = method if form == BASELINE_PRIMARY[method] else f"{method}_{form}"
            rows.append(subset_row(bd, d, name, removed, {"form": form}))
    return {"rows": rows, "train": train}, arrays


def weight_stats(raw: torch.Tensor, lw: torch.Tensor, pois: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score
    key = raw.detach().cpu().numpy().astype(np.float64)
    w = lw.detach().exp().cpu().numpy().astype(np.float64)
    return {"auroc": float(roc_auc_score(pois, -key)) if np.ptp(key) > 0 else 0.5,
            "w_share_pois": float(w[pois].sum() / max(w.sum(), 1e-30)), "n_below_half": int((w < 0.5).sum()),
            "prec_below_half": float(pois[w < 0.5].mean()) if (w < 0.5).any() else None}


def learn(bd: B.BenchData, d: dict, rec: Recorder, kind: str, steps_override: int | None = None) -> tuple[dict, dict]:
    """sigmoid (method 6) or m1 (method 7): weights on P learned on the fitted V's log loss, with P weighted as the
    context (as in the sigmoid study and in bench_tinyv's attn_topk)."""
    import attn_form_study as S
    import attn_learn as A
    import tinyv_learned as L
    from tabfm_experiments.row_weights import evaluate_weighted
    clf, dev = bd.clf, bd.device
    n, k, yg, pois = d["n"], d["k"], d["y_given"], d["pois"]
    # sigmoid: all of V fitted, deployed at a fixed iteration; m1: a held-out part of V chooses m1_sel
    fit, hold = (np.arange(len(d["yv"])), np.zeros(0, dtype=np.int64)) if kind == "sigmoid" else holdout_split(d)
    Xf, yf = d["Xv"][bd.idx(fit)], d["yv"][fit]
    Xh, yh = (d["Xv"][bd.idx(hold)], d["yv"][hold]) if len(hold) else (None, None)
    dd = {"n": n, "k": k, "Xc": d["Xc"], "yp": yg, "Xf": Xf, "yf": yf}
    fixed_it = None
    if kind == "sigmoid":
        steps = steps_override or SIGMOID["steps"]
        fixed_it = min(SIGMOID["fixed_it"], steps)  # a shorter test run: its last iteration
        c = A.DEFAULTS | dict(param="sigmoid", opt="adam", lr=SIGMOID["lr"], floor=-math.inf, t_max=math.inf,
                              t0=SIGMOID["t_kept"], seed=d["seed"])
        raw0 = S.init_raw("sigmoid_free", "vf_hard", dd, clf, dev)  # 4 for all, -30 for vfilter_R's k rows
    else:
        steps = steps_override or M1["steps"]
        c = A.DEFAULTS | L.ATTN_CONFIGS["topk"] | dict(lr=M1["lr"], probe=M1["probe"], seed=d["seed"], steps=steps)
        torch.manual_seed(d["seed"])
        raw0 = A.init_raw(c, dd, clf, dev)  # the fitted V's V-filter log p
    p = A.Param(c["param"], raw0, c, k)
    p.project()
    st = A.Stepper(p, c)
    removed0 = p.deleted().cpu().numpy() if kind == "m1" else None
    zeros_v = torch.zeros(len(d["yv"]), device=dev)
    Xcv, ycv = torch.cat([d["Xc"], d["Xv"]]), np.concatenate([yg, d["yv"]])

    def deploy(lw: torch.Tensor) -> tuple[dict, np.ndarray | None]:
        if kind == "sigmoid":  # soft + V: P at its weights, all of V at weight 1
            probs = evaluate_weighted(clf, Xcv, ycv, bd.Xt, bd.yt, torch.cat([lw, zeros_v]), need_grad=False)[2]
            return B.metrics(probs, bd.yt), None
        removed = p.deleted().cpu().numpy()  # hard + V: the k lowest-scored rows removed exactly
        return B.test_scores(bd, d, ~removed), removed

    def held(lw: torch.Tensor) -> dict | None:
        if Xh is None:
            return None
        return B.metrics(evaluate_weighted(clf, d["Xc"], yg, Xh, yh, lw, need_grad=False)[2], yh)

    records, snaps, best, t0 = [], {}, (math.inf, -1), time.time()
    train_ce = None
    for it in range(steps + 1):
        recording = it < DENSE or it % EVERY == 0 or it in (steps, fixed_it)
        lw_dep = p.log_w().detach()  # sigmoid: the soft log-weights; m1: the hard split (deleted rows at -inf)
        if kind == "m1":
            lw_dep = torch.where(p.deleted(), torch.full_like(lw_dep, -1000.0), torch.zeros_like(lw_dep))
        if recording:
            test, removed = deploy(lw_dep)
            h = held(lw_dep)
            r = {"it": it, "test": test, "held": h, "sec": round(time.time() - t0, 1)} | weight_stats(p.raw, lw_dep, pois)
            if kind == "m1":
                r |= removal_stats(d, removed) | {"swapped": int((removed0 & ~removed).sum())}
            records.append(r)
            if h is not None and h["ce"] < best[0]:
                best = (h["ce"], it)
                snaps["sel"] = (p.raw.detach().clone(), it)
        if it == fixed_it:
            snaps["fixed"] = (p.raw.detach().clone(), it)
        if it == steps:
            break
        lw = p.probe_log_w() if kind == "m1" else p.log_w()
        lw_d = lw.detach().requires_grad_(True)
        train_ce, g, _ = evaluate_weighted(clf, d["Xc"], yg, Xf, yf, lw_d, need_grad=True, grad_channels="all")
        if recording:
            records[-1]["train_ce"] = float(train_ce)  # the fitted V's log loss at the weights just recorded (m1: relaxed point)
        if kind == "m1":
            st.step(torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0))
        else:
            p.raw.grad = None
            lw.backward(g)
            st.step(torch.nan_to_num(p.raw.grad.detach().clone(), nan=0.0, posinf=0.0, neginf=0.0))
        if recording:
            r = records[-1]
            rec.log(flat("test", r["test"]) | (flat("held", r["held"]) if r["held"] else {})
                    | {"train/fitted_ce": r["train_ce"], "weights/auroc_poison": r["auroc"], "weights/w_share_pois": r["w_share_pois"],
                       "weights/n_below_half": r["n_below_half"]}
                    | ({"removal/prec": r["prec"], "removal/swapped": r["swapped"]} if kind == "m1" else {}), step=it)
        if it % 50 == 0:
            print(f"      {kind} it {it}: fitted-V CE {float(train_ce):.4f}  test auc {records[-1]['test']['auc'] if recording else float('nan'):.3f}"
                  f"  ({time.time() - t0:.0f} s)", flush=True)
    # the last iteration: its fitted-V log loss (no step follows it)
    lw_last = p.probe_log_w() if kind == "m1" else p.log_w().detach()
    records[-1]["train_ce"] = float(evaluate_weighted(clf, d["Xc"], yg, Xf, yf, lw_last.detach(), need_grad=False)[0])
    r = records[-1]
    rec.log(flat("test", r["test"]) | (flat("held", r["held"]) if r["held"] else {})
            | {"train/fitted_ce": r["train_ce"], "weights/auroc_poison": r["auroc"]}
            | ({"removal/prec": r["prec"], "removal/swapped": r["swapped"]} if kind == "m1" else {}), step=steps)
    by_it = {x["it"]: x for x in records}
    if kind == "sigmoid":  # the fixed iteration: results/tables/vp_tune/findings.md
        picks = (("sigmoid", fixed_it), ("sigmoid_last", steps), ("sigmoid_start", 0))
    else:  # c = 1: no held-out part, m1_sel is the last iteration
        picks = (("m1_sel", snaps["sel"][1] if "sel" in snaps else steps), ("m1_last", steps), ("m1_start", 0))
    rows = []
    for nm, it in picks:
        x = by_it[it]
        row = {"name": nm, "it": it, "test": x["test"], "held": x["held"], "train_ce": x.get("train_ce")}
        row |= {key: x[key] for key in ("prec", "recall", "share_removed", "kept_pois_share", "n_removed") if key in x}
        row |= {key: x[key] for key in ("auroc", "w_share_pois", "n_below_half", "prec_below_half")}
        rows.append(row)
    arrays = {"raw_last": p.raw.detach().cpu().numpy().astype(np.float32), "raw_start": raw0.detach().cpu().numpy().astype(np.float32)}
    for key in ("sel", "fixed"):
        if key in snaps:
            arrays[f"raw_{key}"] = snaps[key][0].cpu().numpy().astype(np.float32)
    selection = (f"fixed iteration {fixed_it}" if kind == "sigmoid" else "lowest held-out log loss (ties: earliest)"
                 if len(hold) else "last iteration (c = 1: no held-out part)")
    config = {"kind": kind, "steps": steps, "lr": c["lr"], "opt": c["opt"], "param": c["param"],
              "fitted": [int(i) for i in fit], "held_out": [int(i) for i in hold], "dense": DENSE, "every": EVERY,
              "selection": selection}
    return {"rows": rows, "records": records, "config": config, "seconds_train": round(time.time() - t0, 1)}, arrays


JOB_CONFIG = {"refs": {}, "vfilter": {"threshold": f"{THR} / K"}, "greedy": {"step": "ceil(1% of |P|)", "extra_steps": EXTRA_STEPS,
              "test_every": TEST_EVERY}, "baselines": {"defaults": {m: B.DEFAULTS[m] for m in BASELINE_PRIMARY},
              "primary_forms": BASELINE_PRIMARY}, "sigmoid": SIGMOID | {"dense": DENSE, "every": EVERY},
              "m1": M1 | {"dense": DENSE, "every": EVERY},
              "greedy_thr": {"step": "ceil(1% of |P|)", "threshold": GREEDY_THR, "test_every": TEST_EVERY},
              "vrepair": REPAIR["one_shot"], "grepair": REPAIR["greedy"]}


def run_job(bd: B.BenchData, d: dict, job: str, a) -> tuple[dict, dict]:
    rec = Recorder(a, job, d, JOB_CONFIG[job])
    out = ROOT / "runs" / d["tag"] / f"{job}.json"
    try:
        if job == "refs":
            res, arrays = job_refs(bd, d, rec)
        elif job == "vfilter":
            res, arrays = job_vfilter(bd, d, rec)
        elif job == "greedy":
            res, arrays = job_greedy(bd, d, rec)
        elif job == "baselines":
            res, arrays = job_baselines(bd, d, rec, a.threads)
        elif job == "greedy_thr":
            res, arrays = job_greedy_thr(bd, d, rec)
        elif job == "vrepair":
            res, arrays = job_vrepair(bd, d, rec)
        elif job == "grepair":
            res, arrays = job_grepair(bd, d, rec)
        else:
            res, arrays = learn(bd, d, rec, job, a.steps_override)
    except BaseException:
        rec.finish({}, [], ok=False)
        raise
    res = {"job": job, "instance": d["tag"], "n": d["n"], "k": d["k"], "K": d["K"], "R": d["R"], "c": d["c"],
           "n_flipped": int(d["pois"].sum()), "host": socket.gethostname(), "device": bd.device} | res
    if arrays:
        np.savez_compressed(out.with_suffix(".npz"), **arrays)
    write_json(out, res)
    rec.finish(res, [out, out.with_suffix(".npz")])
    return res, arrays


# ---------------------------------------------------------------- driver

def read_job_list(path: Path) -> list[tuple[str, str]]:
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            tag, job = line.split()
            out.append((tag, job))
    return out


def worker(a) -> None:
    todo = read_job_list(a.job_list) if a.job_list else [(t, j) for t in instances() for j in (a.jobs or JOBS)]
    if a.jobs:
        todo = [(t, j) for t, j in todo if j in a.jobs]
    if a.datasets:
        todo = [(t, j) for t, j in todo if parse_tag(t)[0] in a.datasets]
    if a.device == "cpu":
        torch.set_num_threads(a.threads)
    bd = None
    for tag, job in todo:
        out = ROOT / "runs" / tag / f"{job}.json"
        if out.exists():
            continue
        if job in REPAIR_JOBS and (ROOT / "runs" / twin_tag(tag) / f"{job}.lock").exists():
            continue  # the other poisoning's instance is being run: copy it on a later pass instead
        out.parent.mkdir(parents=True, exist_ok=True)
        lock = out.with_suffix(".lock")
        if not B.claim(lock):
            continue
        t0 = time.time()
        try:
            ds = parse_tag(tag)[0]
            if bd is None or bd.ds != ds:
                bd = None
                if a.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                bd = B.BenchData(ds, a.device)
            d = load_draw(bd, tag)
            print(f"run {tag} {job}: start (n {d['n']}, k {d['k']}, V {len(d['yv'])}, flipped {int(d['pois'].sum())})", flush=True)
            res, _ = run_job(bd, d, job, a)
            summary = "  ".join(f"{r['name']}: auc {r['test']['auc']:.3f} bacc {r['test']['bacc']:.3f} ce {r['test']['ce']:.3f}"
                                for r in res["rows"])
            print(f"run {tag} {job}: done in {time.time() - t0:.0f} s  {summary}", flush=True)
        except Exception as e:  # noqa: BLE001 - one failing job must not stop the list; a second pass retries it
            print(f"run {tag} {job}: FAILED after {time.time() - t0:.0f} s: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            if a.device.startswith("cuda"):
                torch.cuda.empty_cache()
        finally:
            lock.unlink(missing_ok=True)


def draws(a) -> None:
    bd = None
    for tag in instances():
        path = ROOT / "draws" / f"{tag}.npz"
        if path.exists():
            continue
        ds, attack, c, seed = parse_tag(tag)
        if bd is None or bd.ds != ds:
            bd = B.BenchData(ds, "cpu")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **build_draw(bd, attack, c, seed))
        print(f"draw {tag}", flush=True)


def status(a) -> None:
    todo = read_job_list(a.job_list) if a.job_list else [(t, j) for t in instances() for j in JOBS]
    done = {j: 0 for j in JOBS}
    total = {j: 0 for j in JOBS}
    for tag, job in todo:
        total[job] += 1
        done[job] += (ROOT / "runs" / tag / f"{job}.json").exists()
    print("finished / planned: " + ", ".join(f"{j} {done[j]}/{total[j]}" for j in JOBS if total[j]))
    locks = sorted((ROOT / "runs").glob("*/*.lock"))
    print(f"locks held: {len(locks)}" + "".join(f"\n  {p.parent.name}/{p.name}: {p.read_text()}" for p in locks[:20]))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["draws", "run", "status"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--job-list", type=Path, default=None, help="lines '<instance> <job>', run in this order")
    ap.add_argument("--jobs", nargs="+", choices=JOBS, default=None)
    ap.add_argument("--datasets", nargs="+", choices=DATASETS, default=None)
    ap.add_argument("--threads", type=int, default=8, help="CPU threads for the baselines, and for every job on --device cpu")
    ap.add_argument("--steps-override", type=int, default=None, help="testing only: iterations of sigmoid / m1")
    ap.add_argument("--wandb-mode", default="disabled", choices=["disabled", "offline", "online"])
    ap.add_argument("--wandb-project", default="TFM_experiments")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--wandb-group", default=None, help="default: vp_<job>")
    a = ap.parse_args(argv)
    {"draws": draws, "run": worker, "status": status}[a.stage](a)


if __name__ == "__main__":
    main()
