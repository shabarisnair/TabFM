#!/usr/bin/env python
"""Analyse the generalisation sweep of the best top-k configuration (results/gate_topk_gen,
run_gate_topk_gen_sweep.sh) against the soft gates on the same draws (results/gate_valsplit).

For every draw (dataset, R, V, validation seed) with a vote (vote.json):
  top-k vote     the configuration's output: the vote of its restarts (vote_topk.py)
  top-k best     the restart with the lowest held-out CE (the rule the vote replaced)
  top-k mean     the mean over the restarts
  soft gates     the soft-gate run on the same draw (same validation rows, holdout and
                 poison), at the iteration its held-out accuracy picks (valtest_acc, the
                 rule used for the soft gates so far; its gates' probabilities are saved);
                 the held-out CE rule (valtest_ce) is listed too
  references     the poisoned context, the V clean rows alone, the poisoned context + the
                 V clean rows, the oracle gate
  detection      precision@R: the vote mask's switched-off rows / the soft gates' k lowest
  bootstrap      paired bootstrap over test rows (2000 resamples): 95% intervals for the
                 test-accuracy differences (top-k vote) - (soft gates) and
                 (top-k vote) - (V clean rows alone); per dataset and overall, the mean over
                 draws with test rows resampled within each draw (pooled_interval)
Test numbers are reported, never used to choose anything. Every draw checks that the
soft-gate run's test labels and clean-rows reference match the top-k draw's.

Also a ranking of the three ways to use the V clean rows (the V clean rows alone as the
context, soft gates, the top-k vote): listed from the highest to the lowest test accuracy,
with ">" between two neighbours when the paired bootstrap interval of their difference
excludes 0, "≈" when it does not, and "=" for equal accuracy.

Writes results/tables/gate_topk_gen/summary_R<R>.txt, draws_R<R>.csv and ordering_R<R>.md.

    python scripts/analyze_gate_topk_gen.py [--R 20] [--extra-votes <dir> ...]

--extra-votes adds vote folders outside --root (e.g. a provisional vote of fewer
restarts); a draw voted by fewer restarts than the others is marked with *.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from analyze_gate_valsplit import bootstrap_diff, pooled_interval  # noqa: E402

DATASETS = ("gas-drift", "url_unique", "heloc", "students_dropout", "credit-approval")


def correct(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (p.argmax(1) == y).astype(float)


def soft_run(dataset: str, R: int, V: int, seed: int, root: Path) -> Path | None:
    hits = sorted(root.glob(f"{dataset}/*_r{R:03d}/gate-valsplit_{dataset}_*_R{R}_V{V}s{seed}_H20_*/test_metrics.json"))
    return hits[0].parent if hits else None


def load_draw(vote_dir: Path, soft_root: Path) -> dict:
    v = json.loads((vote_dir / "vote.json").read_text())
    vz = np.load(vote_dir / "vote.npz")
    y = vz["y_test"].astype(int)
    tp0 = np.load(Path(v["restart_runs"][0]) / "test_probs.npz")  # the references' probabilities
    assert np.array_equal(tp0["y_test"].astype(int), y), vote_dir
    ds, R, V, s = v["dataset"], int(v["row_percent"]), int(v["val_size"]), int(v["val_seed"])
    row = {"dataset": ds, "R": R, "V": V, "seed": s, "n_restarts": v["n_restarts"], "n_test": len(y),
           "poisoned": v["references"]["poisoned_no_gate"]["accuracy"],
           "clean_V": v["references"]["val_as_context"]["accuracy"],
           "poisoned_plus_V": v["references"]["poisoned_plus_val"]["accuracy"],
           "oracle": v["references"]["poisoned_oracle_gate"]["accuracy"],
           "vote": v["vote"]["test"]["accuracy"], "vote_removed": v["vote"]["test_removed"]["accuracy"],
           "vote_prec": v["vote"]["precision_at_R"], "vote_hold_ce": v["vote"]["valtest"]["ce"],
           "best": v["best_restart_by_valtest_ce"]["test"]["accuracy"],
           "best_prec": v["best_restart_by_valtest_ce"]["precision_at_R"],
           "best_hold_ce": v["best_restart_by_valtest_ce"]["valtest_ce"],
           "mean": v["restart_means"]["test_accuracy"], "mean_prec": v["restart_means"]["precision_at_R"],
           "mean_hold_ce": v["restart_means"]["valtest_ce"]}
    c_vote, c_clean = correct(vz["test_probs"], y), correct(tp0["val_as_context"], y)
    assert abs(c_vote.mean() - row["vote"]) < 1e-9 and abs(c_clean.mean() - row["clean_V"]) < 1e-9, vote_dir
    row["d_vote_clean"] = c_vote - c_clean
    row["boot_vote_clean"] = bootstrap_diff(vz["test_probs"], tp0["val_as_context"], y)

    sd = soft_run(ds, R, V, s, soft_root)
    if sd is not None:
        tm = json.loads((sd / "test_metrics.json").read_text())
        tps = np.load(sd / "test_probs.npz")
        assert np.array_equal(tps["y_test"].astype(int), y), sd
        rules = tm["selection_rules"]
        acc_rule, ce_rule = rules["valtest_acc"], rules["valtest_ce"]
        # the saved best gates are the held-out-accuracy rule's
        assert tm["best_iteration"] == acc_rule["iteration"], sd
        assert abs(correct(tps["poisoned_best_gate_soft"], y).mean() - acc_rule["test_soft"]["accuracy"]) < 1e-9, sd
        # same draw -> the same clean-rows reference (up to a borderline row or two across machines)
        assert abs(tm["metrics"]["val_as_context"]["accuracy"] - row["clean_V"]) <= 0.003, (sd, "different draw?")
        log = {int(r["iter"]): r for r in csv.DictReader(open(sd / "log.csv"))}
        row |= {"soft": acc_rule["test_soft"]["accuracy"], "soft_removed": acc_rule["test_hard"]["accuracy"],
                "soft_iter": acc_rule["iteration"],
                "soft_prec": float(log[acc_rule["iteration"]]["precision_at_R"]),
                "soft_hold_ce": acc_rule["valtest"]["ce"],
                "soft_ce_rule": ce_rule["test_soft"]["accuracy"],
                "soft_ce_rule_prec": float(log[ce_rule["iteration"]]["precision_at_R"])}
        row["d_vote_soft"] = c_vote - correct(tps["poisoned_best_gate_soft"], y)
        row["boot_vote_soft"] = bootstrap_diff(vz["test_probs"], tps["poisoned_best_gate_soft"], y)
        row["boot_soft_clean"] = bootstrap_diff(tps["poisoned_best_gate_soft"], tp0["val_as_context"], y)
    return row


METHODS = (("V only", "clean_V"), ("soft", "soft"), ("top-k", "vote"))
PAIRS = {frozenset({"top-k", "soft"}): "boot_vote_soft", frozenset({"top-k", "V only"}): "boot_vote_clean",
         frozenset({"soft", "V only"}): "boot_soft_clean"}


def ordering(r: dict) -> str:
    """The three methods from the highest test accuracy to the lowest: '>' = the paired
    bootstrap interval of the neighbours' difference excludes 0, '≈' = it does not, '=' = equal."""
    ms = sorted(METHODS, key=lambda m: -round(r[m[1]], 9))
    out = ms[0][0]
    for (a, ka), (b, kb) in zip(ms, ms[1:]):
        if round(r[ka], 9) == round(r[kb], 9):
            sym = "="
        else:
            lo, hi = r[PAIRS[frozenset({a, b})]][1:]
            sym = ">" if lo > 0 or hi < 0 else "≈"
        out += f" {sym} {b}"
    return out


def ordering_table(rows: list[dict], R: int) -> str:
    rows = [r for r in rows if "soft" in r]
    L = [f"Test accuracy (%) at R = {R}%, validation seed 0. Ordering: highest to lowest test accuracy among the "
         "V clean rows alone (V only), the soft gates and the top-k vote; '>' = significant (95% paired bootstrap "
         "over test rows), '≈' = not significant, '=' = equal. * = vote of fewer restarts.", "",
         "| dataset | V | poisoned | V only | soft | top-k | oracle | soft prec | top-k prec | ordering |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    full = max(r["n_restarts"] for r in rows)
    for r in rows:
        star = "*" if r["n_restarts"] < full else ""
        L.append(f"| {r['dataset']} | {r['V']}{star} | {100 * r['poisoned']:.1f} | {100 * r['clean_V']:.1f} | "
                 f"{100 * r['soft']:.1f} | {100 * r['vote']:.1f} | {100 * r['oracle']:.1f} | {r['soft_prec']:.2f} | "
                 f"{r['vote_prec']:.2f} | {ordering(r)} |")
    first = {name: 0 for name, _ in METHODS}
    ranks = {name: [] for name, _ in METHODS}
    for r in rows:
        ms = sorted(METHODS, key=lambda m: -round(r[m[1]], 9))
        best = round(r[ms[0][1]], 9)
        for name, key in METHODS:
            first[name] += round(r[key], 9) == best  # ties share the first place
            ranks[name].append(1 + sum(round(r[k2], 9) > round(r[key], 9) for _, k2 in METHODS))
    # each interval is of (first - second): significant wins of the first, then of the second
    wins = {(a, b): (sum(r[key][1] > 0 for r in rows), sum(r[key][2] < 0 for r in rows))
            for (a, b), key in ((("top-k", "soft"), "boot_vote_soft"), (("top-k", "V only"), "boot_vote_clean"),
                                (("soft", "V only"), "boot_soft_clean"))}
    L += ["", f"Over the {len(rows)} draws: highest test accuracy (ties shared) "
          + ", ".join(f"{name} {first[name]}" for name, _ in METHODS)
          + "; mean rank " + ", ".join(f"{name} {np.mean(ranks[name]):.2f}" for name, _ in METHODS) + ".",
          "Significant pairwise wins: " + "; ".join(f"{a} vs {b}: {wa}-{wb}" for (a, b), (wa, wb) in wins.items())
          + "."]
    return "\n".join(L)


def fmt_ci(b) -> str:
    mark = "+" if b[1] > 0 else ("-" if b[2] < 0 else " ")
    return f"{100 * b[0]:+5.1f} [{100 * b[1]:+5.1f},{100 * b[2]:+5.1f}]{mark}"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("results/gate_topk_gen"))
    ap.add_argument("--soft-root", type=Path, default=Path("results/gate_valsplit"))
    ap.add_argument("--R", type=int, default=20)
    ap.add_argument("--extra-votes", type=Path, nargs="*", default=[])
    ap.add_argument("--out", type=Path, default=Path("results/tables/gate_topk_gen"))
    a = ap.parse_args(argv)

    vote_dirs = sorted(p.parent for p in a.root.glob(f"*/*_r{a.R:03d}/gate-topk-vote*/vote.json"))
    vote_dirs += [d for d in a.extra_votes if (d / "vote.json").exists()]
    rows = [load_draw(d, a.soft_root) for d in vote_dirs]
    rows = [r for r in rows if r["R"] == a.R]
    if not rows:
        raise SystemExit(f"no votes at R={a.R} under {a.root}")
    full = max(r["n_restarts"] for r in rows)
    rows.sort(key=lambda r: (DATASETS.index(r["dataset"]) if r["dataset"] in DATASETS else 99, r["V"], r["seed"]))

    L = [f"Best top-k configuration (alpha 0.05, lr 5, 300 updates, vote of {full} restarts) vs soft gates, "
         f"R = {a.R}%, test accuracy (%). Soft gates: held-out-accuracy iteration. clean V = the V clean rows "
         "alone. Intervals: paired bootstrap over test rows; + / - = the interval excludes 0. "
         "* = vote of fewer restarts.",
         ""]
    head = (f"  {'V':>4s} | {'poisoned':>8s} {'clean V':>7s} {'soft':>6s} {'top-k':>6s} {'best':>6s} {'mean':>6s} "
            f"{'oracle':>6s} | {'top-k - soft':>21s} {'top-k - clean V':>21s} | {'prec soft':>9s} {'top-k':>6s}")
    for ds in [d for d in DATASETS if any(r["dataset"] == d for r in rows)]:
        g = [r for r in rows if r["dataset"] == ds]
        L += [f"{ds}  (top-k: the vote; best: the restart with the lowest held-out CE; mean: over restarts)", head]
        for r in g:
            soft = f"{100 * r['soft']:6.1f}" if "soft" in r else f"{'-':>6s}"
            vs = fmt_ci(r["boot_vote_soft"]) if "boot_vote_soft" in r else f"{'-':>21s}"
            ps = f"{r['soft_prec']:9.2f}" if "soft_prec" in r else f"{'-':>9s}"
            star = "*" if r["n_restarts"] < full else " "
            L.append(f"  {r['V']:>4d}{star}| {100 * r['poisoned']:8.1f} {100 * r['clean_V']:7.1f} {soft} "
                     f"{100 * r['vote']:6.1f} {100 * r['best']:6.1f} {100 * r['mean']:6.1f} {100 * r['oracle']:6.1f} | "
                     f"{vs} {fmt_ci(r['boot_vote_clean'])} | {ps} {r['vote_prec']:6.2f}")
        both = [r for r in g if "d_vote_soft" in r]
        if both:
            m = pooled_interval([r["d_vote_soft"] for r in both])
            L.append(f"  mean over V: top-k - soft {fmt_ci(m)}, "
                     f"top-k - clean V {fmt_ci(pooled_interval([r['d_vote_clean'] for r in g]))}; "
                     f"precision soft {np.mean([r['soft_prec'] for r in both]):.2f}, "
                     f"top-k {np.mean([r['vote_prec'] for r in both]):.2f}")
        L.append("")

    both = [r for r in rows if "d_vote_soft" in r]
    sig = lambda key, s: sum(1 for r in rows if key in r and (r[key][1] > 0 if s > 0 else r[key][2] < 0))  # noqa: E731
    L += [f"Overall, {len(rows)} draws ({len(both)} with a soft-gate run):",
          f"  top-k vote - soft gates      mean {fmt_ci(pooled_interval([r['d_vote_soft'] for r in both]))}; "
          f"significantly better in {sig('boot_vote_soft', 1)}, worse in {sig('boot_vote_soft', -1)} draws",
          f"  top-k vote - clean V alone   mean {fmt_ci(pooled_interval([r['d_vote_clean'] for r in rows]))}; "
          f"significantly better in {sig('boot_vote_clean', 1)}, worse in {sig('boot_vote_clean', -1)} draws",
          f"  test accuracy, mean over draws: poisoned {100 * np.mean([r['poisoned'] for r in rows]):.1f}, "
          f"clean V {100 * np.mean([r['clean_V'] for r in rows]):.1f}, "
          f"soft {100 * np.mean([r['soft'] for r in both]):.1f} (over its {len(both)} draws), "
          f"top-k vote {100 * np.mean([r['vote'] for r in rows]):.1f}, "
          f"best restart {100 * np.mean([r['best'] for r in rows]):.1f}, "
          f"restart mean {100 * np.mean([r['mean'] for r in rows]):.1f}, "
          f"oracle {100 * np.mean([r['oracle'] for r in rows]):.1f}",
          f"  precision@R, mean over draws: soft {np.mean([r['soft_prec'] for r in both]):.2f}, "
          f"top-k vote {np.mean([r['vote_prec'] for r in rows]):.2f}, best restart "
          f"{np.mean([r['best_prec'] for r in rows]):.2f}, restart mean {np.mean([r['mean_prec'] for r in rows]):.2f}",
          f"  vote vs best restart (test): vote higher in {sum(r['vote'] > r['best'] for r in rows)}, "
          f"lower in {sum(r['vote'] < r['best'] for r in rows)} of {len(rows)} draws; held-out CE: vote lower in "
          f"{sum(r['vote_hold_ce'] < r['best_hold_ce'] for r in rows)} (vs the best restart, the minimum of its "
          f"restarts on the same rows) and lower than the restart mean in "
          f"{sum(r['vote_hold_ce'] < r['mean_hold_ce'] for r in rows)}"]
    text = "\n".join(L)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"summary_R{a.R}.txt").write_text(text + "\n")
    order = ordering_table(rows, a.R)
    (a.out / f"ordering_R{a.R}.md").write_text(order + "\n")
    keys = list(dict.fromkeys(k for r in rows for k in r if not k.startswith(("d_", "boot_"))))
    with open(a.out / f"draws_R{a.R}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(keys + ["vote_minus_soft", "vote_minus_soft_lo", "vote_minus_soft_hi",
                           "vote_minus_clean", "vote_minus_clean_lo", "vote_minus_clean_hi"])
        for r in rows:
            w.writerow([r.get(k, "") for k in keys] + list(r.get("boot_vote_soft", ("", "", "")))
                       + list(r["boot_vote_clean"]))
    print(text)
    print("\n" + order)


if __name__ == "__main__":
    main()
