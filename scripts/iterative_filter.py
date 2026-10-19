#!/usr/bin/env python
"""Grow a trusted set from the clean rows: an iterative V-filter.

The V-filter scores every context row once, with the V clean rows as TabPFN's context, and
removes the R% whose given (possibly flipped) label is least plausible. Here the trusted set T
starts as the V clean rows and grows instead:

  repeat until only k = R% of the context rows remain outside T:
      TabPFN with T as its context scores every context row not yet in T
      the m most plausible of them (lowest score) join T
  the k rows left outside T are removed; the final training set is T (V + the accepted rows)

With m = 1 this is the greedy one-row-at-a-time algorithm; with m = n - k it is the V-filter.
A variant spec (--variants) is

  oneshot                      the V-filter (m = n - k)
  grow:<schedule>[:<option>]   schedule m<int> (m rows per step), pct<p> (m = ceil(p% of n)),
                               frac<f> (m = ceil(f * |T|), T growing geometrically), shrink<f>
                               (m = ceil(f * the rows still to accept): big steps first, single
                               rows at the end) or both<f> (m = ceil(f * min(|T|, the rows still
                               to accept)): small at both ends); options
                               ce (default score: -log p(given label)), margin (max other-class
                               probability minus p(given label)), prior (probabilities re-weighted
                               from T's class frequencies to V's before scoring), cprior (the
                               same, toward the class frequencies of the context's given labels
                               instead of V's: for a clean set whose class mix is unrepresentative),
                               bal (each step
                               accepts per given-label class in proportion to the rows left),
                               anchor (score = the larger of the score with T and the score with
                               V alone: a row joins only if both find its label plausible),
                               conf<t> (with the CE score: only rows with p(given label) >= t
                               join step by step; once none is left, one last step accepts the
                               rest of the n - k at once, judged by the grown T)
  hold<w> (option of oneshot   a clean-holdout guard: 20% of V is held out and the rest starts T;
  and of grow)                 the context is cut into 50 regions (k-means on standardised
                               features) and every 10 steps each region's harm is measured, the
                               holdout's CE with T + the region's remaining candidates minus with
                               T alone; a candidate's score becomes (1 - w) x the rank of its own
                               score + w x the rank of its region's harm (hold0: the split, no
                               guard). holdr<w>: the same, with a region's harm measured as the
                               holdout's CE with V_fit + the whole context minus with the
                               region's remaining candidates removed from it (the removal view;
                               hold<w> adds them to T instead); holdp<w>: the removal view with
                               only harm above zero counted, so the guard can only delay rows of
                               harmful regions, never move other rows forward. The held-out rows
                               rejoin V in the final training set
  <base>+refine<r>             then up to r rounds of cross-fitted re-filtering: the context in
                               5 random folds, each fold's rows scored with V + the kept rows of
                               the other folds as the context, the k highest-scoring rows of the
                               whole context removed; stops early when the removed set repeats.
                               This is the hard, budgeted form of gating the context rows on the
                               query side (a row's loss counts only while it is kept)

Per draw (dataset, attack, form of the clean set, R, V, validation seed) and variant: the test
accuracy of TabPFN with V + the kept rows as its context, the precision of the removed rows,
the number of steps and the time. Also per draw: V only, oracle + V and poisoned. The draw is
built as gate_context.py builds it (eval_vfilter_attacks.py; hub-flip-local stratified draws
are checked against the saved soft-gate run). Each (draw, variant) also saves the step at which
every row joined T and its score then (<save-dir>/<draw>/<variant>.npz).

    python scripts/iterative_filter.py --gpu 1 --datasets credit-approval --sizes 50 100 200 \\
        --variants oneshot grow:pct1 grow:frac0.1 --out results/iterative_filter/e1/rows_a.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

FIELDS = ["dataset", "attack", "form", "R", "V", "seed", "n", "k", "variant", "acc", "prec", "n_pois_kept",
          "steps", "refine_rounds", "seconds"]
HOLD_FRAC, HOLD_BLOCKS, HOLD_EVERY = 0.2, 50, 10  # the clean-holdout guard (hold<w>)


@dataclass(frozen=True)
class Variant:
    name: str
    schedule: str          # "oneshot" | "m" | "pct" | "frac" | "shrink" | "both"
    value: float
    score: str = "ce"      # "ce" | "margin"
    prior: bool = False
    cprior: bool = False   # prior correction toward the context's given-label frequencies
    balance: bool = False
    anchor: bool = False
    conf: float = 0.0      # > 0: growth by confident rows only, then one final one-shot step
    hold: float = -1.0     # >= 0: 20% of V held out; the weight of the region-harm rank in the score
    hold_mode: str = "add"  # "add": candidates added to T; "remove": removed from the context; "penalty": remove, harm > 0 only
    refine: int = 0


def parse_variant(spec: str) -> Variant:
    base, _, ref = spec.partition("+refine")
    refine = int(ref) if ref else 0
    head, *opts = base.split(":")
    hold = [(float(o[5:]), "remove") if o.startswith("holdr") else (float(o[5:]), "penalty") if o.startswith("holdp")
            else (float(o[4:]), "add") for o in opts if o.startswith("hold")]
    opts = [o for o in opts if not o.startswith("hold")]
    if hold and not 0 <= hold[0][0] <= 1:
        raise ValueError(f"hold needs a weight in [0, 1]: {spec!r}")
    hold, hold_mode = hold[0] if hold else (-1.0, "add")
    if head == "oneshot":
        if opts:
            raise ValueError(f"oneshot takes no option but hold<w>: {spec!r}")
        return Variant(spec, "oneshot", 0.0, hold=hold, hold_mode=hold_mode, refine=refine)
    if head != "grow" or not opts:
        raise ValueError(f"bad variant {spec!r}")
    sched, opts = opts[0], opts[1:]
    conf = [float(o[4:]) for o in opts if o.startswith("conf")]
    opts = [o for o in opts if not o.startswith("conf")]
    if conf and not (0 < conf[0] < 1 and "margin" not in opts):
        raise ValueError(f"conf needs a threshold in (0, 1) and the CE score: {spec!r}")
    for kind in ("pct", "frac", "shrink", "both", "m"):
        if sched.startswith(kind):
            value = float(sched[len(kind):])
            break
    else:
        raise ValueError(f"bad schedule in {spec!r}")
    unknown = set(opts) - {"ce", "margin", "prior", "cprior", "bal", "anchor"}
    if unknown or value <= 0:
        raise ValueError(f"bad options in {spec!r}: {sorted(unknown)}")
    if "prior" in opts and "cprior" in opts:
        raise ValueError(f"prior and cprior exclude each other: {spec!r}")
    return Variant(spec, kind, value, score="margin" if "margin" in opts else "ce", prior="prior" in opts,
                   cprior="cprior" in opts, balance="bal" in opts, anchor="anchor" in opts,
                   conf=conf[0] if conf else 0.0, hold=hold, hold_mode=hold_mode, refine=refine)


def step_size(v: Variant, n: int, k: int, size_T: int, to_accept: int) -> int:
    """Rows to accept at this step (before capping at the rows still to accept)."""
    if v.schedule == "oneshot":
        return n - k
    if v.schedule == "m":
        return int(v.value)
    if v.schedule == "pct":
        return math.ceil(v.value / 100.0 * n)
    if v.schedule == "shrink":
        return math.ceil(v.value * to_accept)
    if v.schedule == "both":
        return math.ceil(v.value * min(size_T, to_accept))
    return math.ceil(v.value * size_T)  # frac


def blend(score: np.ndarray, harm: np.ndarray, w: float) -> np.ndarray:
    """(1 - w) x the rank of score + w x the rank of harm, ranks scaled to (0, 1] (ties: mean
    rank); lower = accepted first."""
    from scipy.stats import rankdata
    return (1 - w) * rankdata(score) / len(score) + w * rankdata(harm) / len(harm)


def quotas(m: int, counts: np.ndarray) -> np.ndarray:
    """m split over classes in proportion to counts (largest remainder), capped by counts."""
    share = m * counts / counts.sum()
    q = np.floor(share).astype(int)
    for c in np.argsort(-(share - q), kind="stable")[:m - q.sum()]:
        q[c] += 1
    q = np.minimum(q, counts)
    while q.sum() < m:  # a capped class's surplus goes to the classes with room, largest first
        room = counts - q
        q[np.argmax(room)] += 1
    return q


def grow(score_fn: Callable[[np.ndarray, np.ndarray], np.ndarray], n: int, k: int, n_v: int, v: Variant,
         y_given: np.ndarray | None = None,
         log_fn: Callable[[int, np.ndarray], None] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """The greedy growth of T. score_fn(accepted rows, query rows) -> scores of the query rows
    (lower = more plausible) with V + the accepted rows as the context. Returns (kept mask,
    step at which each row joined T or -1, its score then, steps). log_fn(step, in_T) is
    called after every step."""
    in_T = np.zeros(n, dtype=bool)
    step_of = np.full(n, -1, dtype=np.int32)
    score_at = np.full(n, np.nan, dtype=np.float32)
    step = 0
    while (n_left := n - in_T.sum()) > k:
        P = np.flatnonzero(~in_T)
        s = score_fn(np.flatnonzero(in_T), P)
        m = max(1, min(step_size(v, n, k, n_v + int(in_T.sum()), n_left - k), n_left - k))
        if v.conf:  # CE <= -log(t): confidently plausible; none left -> accept the rest at once
            n_ok = int((s <= -math.log(v.conf)).sum())
            m = min(m, n_ok) if n_ok else n_left - k
        if v.balance:
            classes = y_given[P]
            K = int(y_given.max()) + 1
            q = quotas(m, np.bincount(classes, minlength=K))
            pick = np.concatenate([P[classes == c][np.argsort(s[classes == c], kind="stable")[:q[c]]] for c in range(K)])
            pos = np.searchsorted(P, pick)
        else:
            pos = np.argsort(s, kind="stable")[:m]
            pick = P[pos]
        in_T[pick] = True
        step_of[pick] = step
        score_at[pick] = s[pos]
        step += 1
        if log_fn is not None:
            log_fn(step, in_T)
    return in_T, step_of, score_at, step


def refine(score_fn: Callable[[np.ndarray, np.ndarray], np.ndarray], keep: np.ndarray, k: int, rounds: int,
           folds: int = 5, seed: int = 0) -> tuple[np.ndarray, int]:
    """Up to `rounds` rounds of cross-fitted re-filtering (see the module docstring)."""
    n = len(keep)
    fold = np.random.default_rng(seed).permutation(n) % folds
    done = 0
    for _ in range(rounds):
        s = np.empty(n)
        for f in range(folds):
            q = np.flatnonzero(fold == f)
            s[q] = score_fn(np.flatnonzero(keep & (fold != f)), q)
        new = np.ones(n, dtype=bool)
        new[np.argsort(-s, kind="stable")[:k]] = False
        done += 1
        if np.array_equal(new, keep):
            break
        keep = new
    return keep, done


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[50, 100, 200, 300, 400, 500])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--attacks", nargs="+", default=["hub-flip-local"])
    ap.add_argument("--forms", nargs="+", default=["stratified"])
    ap.add_argument("--R", type=float, default=20.0)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--save-dir", type=Path, default=None, help="default: <out dir>/runs")
    ap.add_argument("--reuse-dir", type=Path, nargs="*", default=[],
                    help="for <base>+refine<r>: take the base's kept rows from a saved run of the same draw "
                         "(<dir>/<draw>/<base>.npz) instead of running the base again")
    a = ap.parse_args(argv)
    variants = [parse_variant(s) for s in a.variants]

    import torch
    import gate_context as G
    from eval_vfilter_attacks import CONTEXT, TEST
    from tabfm_experiments.config import Seeds
    from tabfm_experiments.multiclass_attacks import poison
    from tabfm_experiments.runtime import build_tabpfn_v2
    from tabfm_experiments.sampling import k_from_percent

    device = G.device_from_gpu(a.gpu)
    save_dir = a.save_dir or a.out.parent / "runs"
    done = set()
    if a.out.exists():
        done = {(r["dataset"], r["attack"], r["form"], float(r["R"]), int(r["V"]), int(r["seed"]), r["variant"])
                for r in csv.DictReader(open(a.out))}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    new = not a.out.exists()
    fh = open(a.out, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new:
        w.writeheader()
    for ds in a.datasets:
        X_file, y_file = G.load(Path(CONTEXT[ds]), device)
        Xt, yt = G.load(Path(TEST[ds]), device)
        yt_t = torch.as_tensor(yt, device=device)
        info = G.DATASETS_DIR / ds / "splits" / "split_info.json"
        K = int((json.loads(info.read_text()).get("n_classes") if info.exists() else None) or len(np.unique(y_file)))
        clf = build_tabpfn_v2(device, Seeds.model_seed, recompute_layers=True, n_classes=K)
        idx = lambda r: torch.as_tensor(np.asarray(r), device=device)  # noqa: E731

        def probs(X_ctx, y_ctx, Xq, yq):
            return G.evaluate(clf, X_ctx, y_ctx, Xq, torch.as_tensor(np.asarray(yq), device=device), None, None,
                              need_grad=False, batch_size=None)[2]

        for form in a.forms:
            for V in a.sizes:
                for s in a.seeds:
                    for attack in a.attacks:
                        todo = [v for v in variants if (ds, attack, form, a.R, V, s, v.name) not in done]
                        base_todo = [b for b in ("v_only", "oracle_plus_v", "poisoned")
                                     if (ds, attack, form, a.R, V, s, b) not in done]
                        if not todo and not base_todo:
                            continue
                        val_rows, _ = G.draw_validation_form(X_file, y_file, V, s, form)
                        ctx_rows = np.setdiff1d(np.arange(len(y_file)), val_rows)
                        Xv, Xc = X_file[idx(val_rows)], X_file[idx(ctx_rows)]
                        yv_np = y_file[val_rows].astype(int)
                        yv = torch.tensor(yv_np, dtype=torch.float32, device=device)
                        n = len(ctx_rows)
                        k = k_from_percent(n, a.R)
                        rows, y_pois = poison(attack, Xc, y_file[ctx_rows], K, a.R, run_id=0, row_seed=Seeds.row_seed)
                        y_pois = np.asarray(y_pois).astype(int)
                        if attack == "hub-flip-local" and form == "stratified":  # the saved soft-gate draw?
                            hit = sorted(Path("results/gate_valsplit").glob(
                                f"{ds}/*_r{int(a.R):03d}/gate-valsplit_{ds}_*_R{int(a.R)}_V{V}s{s}_H20_pgd-linf_lr0.05_it300"))
                            if hit:
                                assert np.array_equal(np.load(hit[0] / "split.npz")["context_rows"], ctx_rows), hit[0]
                                assert np.array_equal(np.load(hit[0] / "poison.npz")["y_poisoned"], y_pois), hit[0]
                        pois = np.zeros(n, dtype=bool)
                        pois[rows] = True
                        yp = torch.tensor(y_pois, dtype=torch.float32, device=device)
                        prior_v = np.bincount(yv_np, minlength=K) / len(yv_np)
                        prior_ctx = np.bincount(y_pois, minlength=K) / n
                        tag = f"{ds}_{attack}_{form}_R{a.R:g}_V{V}s{s}"
                        key = dict(dataset=ds, attack=attack, form=form, R=a.R, V=V, seed=s, n=n, k=k)

                        def test_correct(keep):  # per test row: right with V + the kept rows as the context
                            kk = idx(np.flatnonzero(keep))
                            p = probs(torch.cat([Xc[kk], Xv]), torch.cat([yp[kk], yv]), Xt, yt)
                            return p.argmax(1) == yt

                        blocks = None  # the regions of the hold<w> guard, made on first use
                        (save_dir / tag).mkdir(parents=True, exist_ok=True)
                        for b in base_todo:
                            t0 = time.time()
                            if b == "v_only":
                                correct = probs(Xv, yv, Xt, yt).argmax(1) == yt
                            elif b == "oracle_plus_v":
                                correct = test_correct(~pois)
                            else:
                                correct = probs(Xc, yp, Xt, yt).argmax(1) == yt
                            w.writerow(key | dict(variant=b, acc=float(correct.mean()), prec="", n_pois_kept="", steps=0,
                                                  refine_rounds=0, seconds=round(time.time() - t0, 1)))
                            fh.flush()
                            np.savez_compressed(save_dir / tag / f"base_{b}.npz", test_correct=correct)
                        for v in todo:
                            t0 = time.time()
                            if v.hold >= 0:  # V_fit starts T, V_hold is the clean check
                                fit_pos, hold_pos = G.split_holdout(yv_np, HOLD_FRAC, s)
                            else:
                                fit_pos, hold_pos = np.arange(len(val_rows)), np.zeros(0, dtype=int)
                            Xv_fit, yv_fit_np = Xv[idx(fit_pos)], yv_np[fit_pos]
                            yv_fit = torch.tensor(yv_fit_np, dtype=torch.float32, device=device)
                            Xv_hold, yv_hold_np = Xv[idx(hold_pos)], yv_np[hold_pos]

                            def base_score(acc_rows, q_rows, v=v):
                                X_ctx = torch.cat([Xc[idx(acc_rows)], Xv_fit])
                                y_ctx = torch.cat([yp[idx(acc_rows)], yv_fit])
                                p = probs(X_ctx, y_ctx, Xc[idx(q_rows)], y_pois[q_rows])
                                if v.prior or v.cprior:  # from T's class frequencies to V's (or the context's)
                                    f_T = np.bincount(np.concatenate([y_pois[acc_rows], yv_fit_np]), minlength=K) / (len(acc_rows) + len(yv_fit_np))
                                    p = p * ((prior_v if v.prior else prior_ctx) / np.maximum(f_T, 1e-12))[None, :]
                                    p = p / p.sum(1, keepdims=True)
                                pg = p[np.arange(len(q_rows)), y_pois[q_rows]]
                                if v.score == "margin":
                                    other = p.copy()
                                    other[np.arange(len(q_rows)), y_pois[q_rows]] = -np.inf
                                    return other.max(1) - pg
                                return -np.log(np.clip(pg, 1e-12, None))

                            if v.anchor:  # the score with V alone, once per draw
                                s_v = base_score(np.zeros(0, dtype=int), np.arange(n))

                            if v.hold > 0 and blocks is None:
                                from sklearn.cluster import KMeans
                                from tabfm_experiments.multiclass_attacks import standardise
                                blocks = KMeans(n_clusters=min(HOLD_BLOCKS, n), n_init=3, random_state=0).fit_predict(
                                    standardise(Xc).cpu().numpy())

                            def hold_ce(rows):  # mean CE of V_hold with V_fit + these context rows as the context
                                rows = np.asarray(rows, dtype=int)
                                p = probs(torch.cat([Xc[idx(rows)], Xv_fit]), torch.cat([yp[idx(rows)], yv_fit]),
                                          Xv_hold, yv_hold_np)
                                return float(-np.log(np.clip(p[np.arange(len(yv_hold_np)), yv_hold_np], 1e-12, None)).mean())

                            harm = {"calls": 0, "by_block": {}, "ce_all": None}

                            def score_fn(acc_rows, q_rows, v=v, harm=harm):
                                s_T = base_score(acc_rows, q_rows)
                                if v.anchor:
                                    s_T = np.maximum(s_T, s_v[q_rows])
                                if v.hold <= 0:
                                    return s_T
                                if harm["calls"] % HOLD_EVERY == 0 and v.hold_mode == "add":  # candidates added to T
                                    ce0 = hold_ce(acc_rows)
                                    harm["by_block"] = {int(b): hold_ce(np.concatenate([acc_rows, q_rows[blocks[q_rows] == b]])) - ce0
                                                        for b in np.unique(blocks[q_rows])}
                                elif harm["calls"] % HOLD_EVERY == 0:  # candidates removed from the whole context
                                    if harm["ce_all"] is None:
                                        harm["ce_all"] = hold_ce(np.arange(n))
                                    harm["by_block"] = {int(b): harm["ce_all"] - hold_ce(np.setdiff1d(np.arange(n), q_rows[blocks[q_rows] == b]))
                                                        for b in np.unique(blocks[q_rows])}
                                    if v.hold_mode == "penalty":  # only harmful regions are delayed
                                        harm["by_block"] = {b: max(h, 0.0) for b, h in harm["by_block"].items()}
                                harm["calls"] += 1
                                return blend(s_T, np.array([harm["by_block"][int(b)] for b in blocks[q_rows]]), v.hold)

                            last = [time.time()]

                            def log_fn(step, in_T, v=v):
                                if time.time() - last[0] >= 120:
                                    last[0] = time.time()
                                    print(f"  {tag} {v.name}: step {step}, |T| {len(val_rows) + int(in_T.sum())}, poisoned "
                                          f"accepted {int(pois[in_T].sum())}/{int(pois.sum())} ({time.time() - t0:.0f} s)",
                                          flush=True)

                            saved = None
                            if v.refine:
                                base_file = f"{v.name.split('+refine')[0].replace(':', '_')}.npz"
                                saved = next((np.load(d / tag / base_file) for d in [save_dir, *a.reuse_dir]
                                              if (d / tag / base_file).exists()), None)
                            if saved is not None:  # the same draw: same poison, same clean rows
                                assert np.array_equal(saved["pois"], pois) and np.array_equal(saved["val_rows"], val_rows)
                                keep, step_of, score_at = saved["keep"], saved["step_of"], saved["score_at"]
                                steps = int(step_of.max()) + 1
                            else:
                                keep, step_of, score_at, steps = grow(score_fn, n, k, len(fit_pos), v, y_pois, log_fn)
                            rounds = 0
                            if v.refine:
                                keep, rounds = refine(score_fn, keep, k, v.refine, seed=s)
                            correct = test_correct(keep)
                            acc = float(correct.mean())
                            row = key | dict(variant=v.name, acc=acc, prec=float(pois[~keep].mean()),
                                             n_pois_kept=int(pois[keep].sum()), steps=steps, refine_rounds=rounds,
                                             seconds=round(time.time() - t0, 1))
                            w.writerow(row)
                            fh.flush()
                            np.savez_compressed(save_dir / tag / f"{v.name.replace(':', '_')}.npz", keep=keep, step_of=step_of,
                                                score_at=score_at, pois=pois, y_given=y_pois, val_rows=val_rows,
                                                test_correct=correct)
                            print(f"{tag} {v.name}: acc {100 * acc:.1f} prec {row['prec']:.3f} steps {steps}"
                                  f"{f' refine {rounds}' if v.refine else ''} ({row['seconds']:.0f} s)", flush=True)
    fh.close()


if __name__ == "__main__":
    main()
