#!/usr/bin/env python
"""Beam search over the order in which context rows join the trusted set.

The greedy growth of scripts/iterative_filter.py moves, at every step, the m context rows with
the lowest CE of their given label (TabPFN with the trusted set T as its context) into T, until
k = R% of the context is left out. Here every beam (one trusted set) branches into X children
per step:
  m = 1   child j adds the row with the j-th lowest CE (j < X)
  m > 1   child 0 adds the m lowest-CE rows; child j >= 1 adds m rows drawn without replacement
          with probability proportional to p(given label)^(1/tau) (Gumbel-top-m): a perturbation
          of the greedy batch that stays among the plausible rows
Each child costs one TabPFN pass with V_T + its accepted rows as the context, queried on its
remaining candidates (their CE: its next choices) and on the clean holdout, if any. The K best
children by the criterion survive (children with the same trusted set are merged). At the end
the best beam's accepted rows are kept; the final training set is all of V + the kept rows.

Criteria (lower is better):
  hold     CE of a clean holdout (20% of V, never in T) with T as the context; V_T = the other 80%
  holdent  mean predictive entropy on the holdout's features (its labels unused); same split
  path     the summed CE, at acceptance, of every row the beam accepted (a path likelihood); V_T = V
  entP     mean predictive entropy of T on its remaining candidates; V_T = V
  vcv      5-fold cross-validated CE of V: each fold of V predicted with the beam's accepted rows +
           the other 4 folds as the context (5 extra passes per child); V_T = V, nothing held out

Variant spec: beam:K<k>:X<x>:(m<m> | p<pct>):<criterion>[:prior][:tau<t>] (p<pct>: m = ceil(pct% of
the context); prior: T's posterior re-weighted from T's class frequencies to V's before scoring, as
in iterative_filter.py; tau default 0.1).
K1:X1 is the greedy growth itself, on V_T.

Writes one CSV row per (draw, variant) like iterative_filter.py (plus the number of TabPFN
passes) and <out dir>/runs/<draw>/<variant>.npz (kept rows, acceptance step, test correctness).

    python scripts/beam_filter.py --gpu 3 --datasets heloc --sizes 200 \\
        --variants beam:K1:X1:m50:hold beam:K4:X4:m50:hold --out results/iterative_filter/e18/rows_heloc.csv
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

CRITERIA = ("hold", "holdent", "path", "entP", "vcv")
VCV_FOLDS = 5
FIELDS = ["dataset", "attack", "form", "R", "V", "seed", "n", "k", "variant", "acc", "prec", "n_pois_kept",
          "steps", "passes", "seconds"]
HOLD_FRAC = 0.2


@dataclass(frozen=True)
class BeamSpec:
    name: str
    K: int
    X: int
    m: int                 # rows per step; 0 = given as a share of the context (pct)
    criterion: str
    prior: bool = False
    tau: float = 0.1
    pct: float = 0.0

    def step(self, n: int) -> int:
        return self.m if self.m else max(1, math.ceil(self.pct / 100.0 * n))

    @property
    def uses_holdout(self) -> bool:
        return self.criterion in ("hold", "holdent")


def parse_beam(spec: str) -> BeamSpec:
    head, *opts = spec.split(":")
    if head != "beam":
        raise ValueError(f"bad beam spec {spec!r}")
    K = X = m = None
    crit, prior, tau, pct = None, False, 0.1, 0.0
    for o in opts:
        if o in CRITERIA:
            crit = o
        elif o == "prior":
            prior = True
        elif o.startswith("tau"):
            tau = float(o[3:])
        elif o.startswith("p") and o[1:].replace(".", "", 1).isdigit():
            pct, m = float(o[1:]), 0
        elif o[:1] in ("K", "X", "m") and o[1:].isdigit():
            K, X, m = (int(o[1:]) if o[0] == c else val for c, val in zip("KXm", (K, X, m)))
        else:
            raise ValueError(f"bad option {o!r} in {spec!r}")
    if None in (K, X, m, crit) or min(K, X) < 1 or (m < 1 and pct <= 0) or tau <= 0:
        raise ValueError(f"beam spec needs K, X >= 1, a step (m<m> or p<pct>) and a criterion: {spec!r}")
    return BeamSpec(spec, K, X, m, crit, prior, tau, pct)


def branches(s: np.ndarray, m: int, X: int, rng: np.random.Generator, tau: float) -> list[np.ndarray]:
    """The X candidate batches of one beam, as positions into its candidate list (s = their CE)."""
    order = np.argsort(s, kind="stable")
    if m == 1:
        return [order[j:j + 1] for j in range(min(X, len(s)))]
    out = [order[:m]]
    for _ in range(1, X):  # Gumbel-top-m on log p / tau, log p = -CE
        out.append(np.argsort(-(-s / tau + rng.gumbel(size=len(s))), kind="stable")[:m])
    return out


PassFn = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, dict]]


def beam_search(pass_fn: PassFn, n: int, k: int, spec: BeamSpec, rng: np.random.Generator,
                log_fn: Callable[[int, int], None] | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """pass_fn(accepted rows) -> (remaining rows, their CE, {criterion: value}). Returns (kept mask,
    step at which each kept row joined T or -1, info)."""
    P, s, parts = pass_fn(np.zeros(0, dtype=int))
    beams = [dict(acc=np.zeros(0, dtype=int), path=0.0, P=P, s=s, parts=parts, order=[])]
    steps, passes = 0, 1
    while len(beams[0]["acc"]) < n - k:
        m = min(spec.step(n), n - k - len(beams[0]["acc"]))
        cand = {}
        for b in beams:
            for sel in branches(b["s"], m, spec.X, rng, spec.tau):
                acc = np.union1d(b["acc"], b["P"][sel])
                path = b["path"] + float(b["s"][sel].sum())
                key = acc.tobytes()
                if key not in cand or path < cand[key]["path"]:
                    cand[key] = dict(acc=acc, path=path, order=b["order"] + [b["P"][sel]])
        kids = []
        for c in cand.values():
            P, s, parts = pass_fn(c["acc"])
            passes += 1
            kids.append(c | dict(P=P, s=s, parts=parts))
        crit = (lambda c: c["path"]) if spec.criterion == "path" else (lambda c: c["parts"][spec.criterion])
        kids.sort(key=crit)
        beams = kids[:spec.K]
        steps += 1
        if log_fn is not None:
            log_fn(steps, passes)
    best = beams[0]
    keep = np.zeros(n, dtype=bool)
    keep[best["acc"]] = True
    step_of = np.full(n, -1, dtype=np.int32)
    for t, batch in enumerate(best["order"]):
        step_of[batch] = t
    crit_val = best["path"] if spec.criterion == "path" else best["parts"][spec.criterion]
    return keep, step_of, dict(steps=steps, passes=passes, criterion=crit_val)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="3")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[200])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--attacks", nargs="+", default=["hub-flip-local"])
    ap.add_argument("--R", type=float, default=20.0)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    specs = [parse_beam(v) for v in a.variants]

    import torch
    import gate_context as G
    from eval_vfilter_attacks import CONTEXT, TEST
    from tabfm_experiments.config import Seeds
    from tabfm_experiments.multiclass_attacks import poison
    from tabfm_experiments.runtime import build_tabpfn_v2
    from tabfm_experiments.sampling import k_from_percent

    device = G.device_from_gpu(a.gpu)
    save_dir = a.out.parent / "runs"
    done = set()
    if a.out.exists():
        done = {(r["dataset"], r["attack"], float(r["R"]), int(r["V"]), int(r["seed"]), r["variant"]) for r in csv.DictReader(open(a.out))}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    new = not a.out.exists()
    fh = open(a.out, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new:
        w.writeheader()
    for ds in a.datasets:
        X_file, y_file = G.load(Path(CONTEXT[ds]), device)
        Xt, yt = G.load(Path(TEST[ds]), device)
        info = G.DATASETS_DIR / ds / "splits" / "split_info.json"
        K = int((json.loads(info.read_text()).get("n_classes") if info.exists() else None) or len(np.unique(y_file)))
        clf = build_tabpfn_v2(device, Seeds.model_seed, recompute_layers=True, n_classes=K)
        idx = lambda r: torch.as_tensor(np.asarray(r), device=device)  # noqa: E731

        def probs(X_ctx, y_ctx, Xq, yq):
            return G.evaluate(clf, X_ctx, y_ctx, Xq, torch.as_tensor(np.asarray(yq), device=device), None, None,
                              need_grad=False, batch_size=None)[2]

        for V in a.sizes:
            for s in a.seeds:
                for attack in a.attacks:
                    todo = [sp for sp in specs if (ds, attack, a.R, V, s, sp.name) not in done]
                    if not todo:
                        continue
                    val_rows = G.draw_validation(y_file, V, s)
                    ctx_rows = np.setdiff1d(np.arange(len(y_file)), val_rows)
                    Xv, Xc = X_file[idx(val_rows)], X_file[idx(ctx_rows)]
                    yv_np = y_file[val_rows].astype(int)
                    yv = torch.tensor(yv_np, dtype=torch.float32, device=device)
                    n = len(ctx_rows)
                    k = k_from_percent(n, a.R)
                    rows, y_pois = poison(attack, Xc, y_file[ctx_rows], K, a.R, run_id=0, row_seed=Seeds.row_seed)
                    y_pois = np.asarray(y_pois).astype(int)
                    pois = np.zeros(n, dtype=bool)
                    pois[rows] = True
                    yp = torch.tensor(y_pois, dtype=torch.float32, device=device)
                    prior_v = np.bincount(yv_np, minlength=K) / len(yv_np)
                    tag = f"{ds}_{attack}_stratified_R{a.R:g}_V{V}s{s}"
                    (save_dir / tag).mkdir(parents=True, exist_ok=True)
                    for sp in todo:
                        t0 = time.time()
                        if sp.uses_holdout:
                            fit_pos, hold_pos = G.split_holdout(yv_np, HOLD_FRAC, s)
                        else:
                            fit_pos, hold_pos = np.arange(len(val_rows)), np.zeros(0, dtype=int)
                        Xf, yf_np = Xv[idx(fit_pos)], yv_np[fit_pos]
                        yf = torch.tensor(yf_np, dtype=torch.float32, device=device)
                        Xh, yh_np = Xv[idx(hold_pos)], yv_np[hold_pos]

                        def pass_fn(acc, sp=sp, Xf=Xf, yf=yf, yf_np=yf_np, Xh=Xh, yh_np=yh_np):
                            P = np.setdiff1d(np.arange(n), acc)
                            p = probs(torch.cat([Xc[idx(acc)], Xf]), torch.cat([yp[idx(acc)], yf]),
                                      torch.cat([Xc[idx(P)], Xh]), np.concatenate([y_pois[P], yh_np]))
                            pc, ph = p[:len(P)], p[len(P):]
                            if sp.prior:  # from T's class frequencies to V's
                                f_T = np.bincount(np.concatenate([y_pois[acc], yf_np]), minlength=K) / (len(acc) + len(yf_np))
                                pc = pc * (prior_v / np.maximum(f_T, 1e-12))[None, :]
                                pc = pc / pc.sum(1, keepdims=True)
                            ce = -np.log(np.clip(pc[np.arange(len(P)), y_pois[P]], 1e-12, None))
                            ent = lambda q: float(-(q * np.log(np.clip(q, 1e-12, None))).sum(1).mean()) if len(q) else 0.0  # noqa: E731
                            parts = {"entP": ent(p[:len(P)]), "holdent": ent(ph),
                                     "hold": float(-np.log(np.clip(ph[np.arange(len(yh_np)), yh_np], 1e-12, None)).mean()) if len(yh_np) else 0.0}
                            if sp.criterion == "vcv":  # each fold of V predicted by the accepted rows + the other folds
                                fold = np.random.default_rng(0).permutation(len(yf_np)) % VCV_FOLDS
                                tot = 0.0
                                for f in range(VCV_FOLDS):
                                    tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
                                    q = probs(torch.cat([Xc[idx(acc)], Xf[idx(tr)]]), torch.cat([yp[idx(acc)], yf[idx(tr)]]),
                                              Xf[idx(te)], yf_np[te])
                                    tot += float(-np.log(np.clip(q[np.arange(len(te)), yf_np[te]], 1e-12, None)).sum())
                                parts["vcv"] = tot / len(yf_np)
                            return P, ce, parts

                        last = [time.time()]

                        def log_fn(step, passes, sp=sp):
                            if time.time() - last[0] >= 300:
                                last[0] = time.time()
                                print(f"  {tag} {sp.name}: step {step}, {passes} passes ({time.time() - t0:.0f} s)", flush=True)

                        keep, step_of, info_b = beam_search(pass_fn, n, k, sp, np.random.default_rng(s), log_fn)
                        kk = idx(np.flatnonzero(keep))
                        correct = probs(torch.cat([Xc[kk], Xv]), torch.cat([yp[kk], yv]), Xt, yt).argmax(1) == yt
                        row = dict(dataset=ds, attack=attack, form="stratified", R=a.R, V=V, seed=s, n=n, k=k, variant=sp.name,
                                   acc=float(correct.mean()), prec=float(pois[~keep].mean()), n_pois_kept=int(pois[keep].sum()),
                                   steps=info_b["steps"], passes=info_b["passes"], seconds=round(time.time() - t0, 1))
                        w.writerow(row)
                        fh.flush()
                        np.savez_compressed(save_dir / tag / f"{sp.name.replace(':', '_')}.npz", keep=keep, step_of=step_of,
                                            pois=pois, y_given=y_pois, val_rows=val_rows, test_correct=correct)
                        print(f"{tag} {sp.name}: acc {100 * row['acc']:.1f} prec {row['prec']:.3f} steps {row['steps']} "
                              f"passes {row['passes']} ({row['seconds']:.0f} s)", flush=True)
    fh.close()


if __name__ == "__main__":
    main()
