#!/usr/bin/env python
"""Summarise an iterative-filter experiment (scripts/iterative_filter.py) against the V-filter.

Reads every rows_*.csv of results/iterative_filter/<exp> (several experiments are merged, the
first one listed winning; a draw's reference rows, written by each process that ran the draw,
are deduplicated) and, for significance, the per-test-row
correctness saved with each run (runs/<draw>/<variant>.npz, base_<reference>.npz). Runs saved
before that was recorded are re-scored once from their saved kept rows (--gpu): TabPFN with
V + the kept rows as its context on the test set, as iterative_filter.py scores them.

Writes results/tables/iterative_filter/<exp>.md:
  per draw    V only, the V-filter + V (oneshot), every variant (accuracy, precision of the
              removed rows), oracle + V; best non-oracle in bold
  per variant mean accuracy and precision over the draws it ran on (and the V-filter's on the
              same draws), draws above-below the V-filter in accuracy and in precision, and
              how many of the accuracy gaps are significant (95% paired bootstrap over test rows)

    python scripts/analyze_iterative_filter.py --exp e1 [e2 ...] [--gpu 1]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

REFS = ("v_only", "oracle_plus_v", "poisoned")
DATASETS = ("gas-drift", "url_unique", "heloc", "students_dropout", "credit-approval")


def bootstrap_ci(ca: np.ndarray, cb: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    d = ca.astype(float) - cb.astype(float)
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(1)
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


class Rescorer:
    """Test correctness of V + the kept rows for runs saved without it (one TabPFN pass each)."""

    def __init__(self, gpu: str):
        import torch
        import gate_context as G
        from eval_vfilter_attacks import CONTEXT, TEST
        from tabfm_experiments.config import Seeds
        from tabfm_experiments.runtime import build_tabpfn_v2
        self.torch, self.G, self.CONTEXT, self.TEST = torch, G, CONTEXT, TEST
        self.Seeds, self.build = Seeds, build_tabpfn_v2
        self.device = G.device_from_gpu(gpu)
        self.cache = {}

    def data(self, ds: str):
        if ds not in self.cache:
            G, torch = self.G, self.torch
            X_file, y_file = G.load(Path(self.CONTEXT[ds]), self.device)
            Xt, yt = G.load(Path(self.TEST[ds]), self.device)
            info = G.DATASETS_DIR / ds / "splits" / "split_info.json"
            K = int((json.loads(info.read_text()).get("n_classes") if info.exists() else None) or len(np.unique(y_file)))
            clf = self.build(self.device, self.Seeds.model_seed, recompute_layers=True, n_classes=K)
            self.cache[ds] = (X_file, y_file, Xt, yt, torch.as_tensor(yt, device=self.device), clf)
        return self.cache[ds]

    def correct(self, ds: str, z) -> np.ndarray:
        torch, G = self.torch, self.G
        X_file, y_file, Xt, yt, yt_t, clf = self.data(ds)
        val = z["val_rows"]
        ctx = np.setdiff1d(np.arange(len(y_file)), val)
        kk = torch.as_tensor(ctx[z["keep"]], device=self.device)
        vv = torch.as_tensor(val, device=self.device)
        yk = torch.tensor(z["y_given"][z["keep"]], dtype=torch.float32, device=self.device)
        yv = torch.tensor(y_file[val], dtype=torch.float32, device=self.device)
        p = G.evaluate(clf, torch.cat([X_file[kk], X_file[vv]]), torch.cat([yk, yv]), Xt, yt_t, None, None,
                       need_grad=False, batch_size=None)[2]
        return p.argmax(1) == yt


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", nargs="+", required=True)
    ap.add_argument("--gpu", default=None, help="re-score runs saved without per-test-row correctness")
    ap.add_argument("--root", type=Path, default=Path("results/iterative_filter"))
    a = ap.parse_args(argv)
    exp_dirs = [a.root / e for e in a.exp]
    rows = {}
    for f in [f for d in exp_dirs for f in sorted(d.glob("rows_*.csv"))]:
        for r in csv.DictReader(open(f)):
            key = (r["dataset"], r["attack"], r["form"], float(r["R"]), int(r["V"]), int(r["seed"]), r["variant"])
            rows.setdefault(key, r)
    draws = defaultdict(dict)
    for key, r in rows.items():
        draws[key[:6]][key[6]] = r
    variants = sorted({v for d in draws.values() for v in d if v not in REFS and v != "oneshot"},
                      key=lambda v: (v.split("+")[0], v))
    rescorer = Rescorer(a.gpu) if a.gpu is not None else None

    def correct_of(draw, variant):
        ds, attack, form, R, V, s = draw
        tag = f"{ds}_{attack}_{form}_R{R:g}_V{V}s{s}"
        name = f"base_{variant}.npz" if variant in REFS else f"{variant.replace(':', '_')}.npz"
        f = next((d / "runs" / tag / name for d in exp_dirs if (d / "runs" / tag / name).exists()), None)
        if f is not None:
            z = np.load(f)
            if "test_correct" in z.files:
                return z["test_correct"]
            if rescorer is not None and "keep" in z.files:
                c = rescorer.correct(ds, z)
                np.savez_compressed(f, **{k: z[k] for k in z.files}, test_correct=c)
                return c
        return None

    order = sorted(draws, key=lambda d: (d[1], d[2], d[3], DATASETS.index(d[0]) if d[0] in DATASETS else 9, d[0], d[4], d[5]))
    L = [f"# Iterative filter, experiment {'+'.join(a.exp)}", "",
         "Test accuracy (%) of TabPFN with V + the kept rows as its context; precision of the removed rows in "
         "brackets. oneshot = the V-filter (V + filtered). Best non-oracle in bold. Runs: scripts/iterative_filter.py.", ""]
    head = ["dataset", "attack", "form", "R", "V", "seed", "V only", "V-filter + V"] + variants + ["oracle + V"]
    L += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    stats = {v: {"acc": [], "acc_os": [], "prec": [], "prec_os": [], "win": 0, "loss": 0, "sig_win": 0, "sig_loss": 0,
                 "pwin": 0, "ploss": 0, "n_sig": 0} for v in variants}
    for d in order:
        r = draws[d]
        if "oneshot" not in r:
            continue
        f = lambda v: 100 * float(r[v]["acc"]) if v in r else None  # noqa: E731
        cand = {v: f(v) for v in ["v_only", "oneshot"] + variants if v in r}
        best = max(cand.values())
        cells = []
        for v in ["v_only", "oneshot"] + variants + ["oracle_plus_v"]:
            if v not in r:
                cells.append("")
                continue
            txt = f"{f(v):.1f}"
            if v not in REFS:
                txt += f" ({float(r[v]['prec']):.2f})"
            if v != "oracle_plus_v" and abs(f(v) - best) < 1e-9:
                txt = f"**{txt}**"
            cells.append(txt)
        L.append("| " + " | ".join(str(x) for x in (d[0], d[1], d[2], f"{d[3]:g}", d[4], d[5])) + " | " + " | ".join(cells) + " |")
        c_os = correct_of(d, "oneshot")
        for v in variants:
            if v not in r:
                continue
            st = stats[v]
            st["acc"].append(f(v))
            st["acc_os"].append(f("oneshot"))
            st["prec"].append(float(r[v]["prec"]))
            st["prec_os"].append(float(r["oneshot"]["prec"]))
            st["win"] += f(v) > f("oneshot") + 1e-9
            st["loss"] += f(v) < f("oneshot") - 1e-9
            st["pwin"] += float(r[v]["prec"]) > float(r["oneshot"]["prec"]) + 1e-9
            st["ploss"] += float(r[v]["prec"]) < float(r["oneshot"]["prec"]) - 1e-9
            c_v = correct_of(d, v)
            if c_v is not None and c_os is not None:
                lo, hi = bootstrap_ci(c_v, c_os)
                st["sig_win"] += lo > 0
                st["sig_loss"] += hi < 0
                st["n_sig"] += 1
    L += ["", "Per variant, over the draws it ran on (V-filter on the same draws in brackets):", "",
          "| variant | draws | mean acc | mean precision | acc above-below V-filter | of them significant "
          "(tested draws) | precision above-below |", "|---|---|---|---|---|---|---|"]
    for v in variants:
        st = stats[v]
        if not st["acc"]:
            continue
        L.append(f"| {v} | {len(st['acc'])} | {np.mean(st['acc']):.1f} ({np.mean(st['acc_os']):.1f}) | "
                 f"{np.mean(st['prec']):.3f} ({np.mean(st['prec_os']):.3f}) | {st['win']}-{st['loss']} | "
                 f"{st['sig_win']}-{st['sig_loss']} ({st['n_sig']}) | {st['pwin']}-{st['ploss']} |")
    out = Path("results/tables/iterative_filter") / f"{'+'.join(a.exp)}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
