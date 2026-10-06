#!/usr/bin/env python
"""Tables for the test-agnostic hub label-flip sweep.

Reads ``<root>/<dataset>/r<NNN>/summary.json`` written by hub_flip_attention.py and
emits a long CSV in the SAME schema as results/tables/main/main_results.csv, so
``plot_results.py --csv <this file> --attack hub-flip`` draws the metric-vs-R curves
without any change.

The attack is deterministic (n_runs=1), so every std is 0 by construction. R=0 is the
clean baseline that every delta is measured against.

Several --root paths are merged into one table set. The sweep was run in two places
(the attention-recording runs went to /hdd1 for space), and a dataset lives under
exactly one of them, so merging is a plain union -- an overlap is an error, not a
silent pick.

    python scripts/make_hub_tables.py
    python scripts/make_hub_tables.py --root results/hub_flip /hdd1/ssn899/TabFM/results/hub_flip_attention
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

METRICS = ["ce", "roc_auc", "accuracy", "balanced_accuracy", "mcc"]
SHORT = {"ce": "CE", "roc_auc": "ROC-AUC", "accuracy": "accuracy",
         "balanced_accuracy": "bal. acc", "mcc": "MCC"}
WEAK = {"mcc": 0.15}              # |clean| below this -> the relative % is noise


def load(roots: list[Path]) -> dict:
    out: dict[str, dict[int, dict]] = {}
    seen: dict[str, Path] = {}
    for root in roots:
        if not root.is_dir():
            raise SystemExit(f"--root not a directory: {root}")
        for ds in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
            cells = {}
            for d in sorted(ds.glob("r*")):
                f = d / "summary.json"
                if not f.exists():
                    continue
                s = json.loads(f.read_text())
                runs = s.get("metrics_per_run") or []
                if not runs:
                    continue
                cells[int(round(s["row_percent"]))] = {
                    "summary": s, "m": runs[0], "n_runs": len(runs)}
            if not cells:
                continue
            if ds.name in seen:
                raise SystemExit(f"{ds.name} appears under both {seen[ds.name]} and {root}; "
                                 "pass only the root you want for it")
            seen[ds.name] = root
            out[ds.name] = cells
    return out


def fmt_rel(v, weak):
    return "-" if v is None else f"{'~' if weak else ''}{v:+.1f}%"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, nargs="+",
                    default=[Path("results/hub_flip"),
                             Path("/hdd1/ssn899/TabFM/results/hub_flip_attention")],
                    help="one or more sweep roots; datasets are unioned across them")
    ap.add_argument("--out", type=Path, default=Path("results/tables/hub_flip"))
    ap.add_argument("--metrics", nargs="+", default=METRICS)
    a = ap.parse_args(argv)

    data = load(a.root)
    if not data:
        raise SystemExit("no summary.json found under " + ", ".join(map(str, a.root)))
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []

    for ds, cells in data.items():
        if 0 not in cells:
            print(f"[skip] {ds}: no R=0 clean baseline, cannot form deltas")
            continue
        clean = cells[0]["m"]
        rs = [r for r in sorted(cells) if r != 0]
        s0 = cells[0]["summary"]
        expected = set(range(2, max(rs) + 1, 2)) if rs else set()
        missing = sorted(expected - set(rs))
        head = (f"{ds}   hub-flip (test-agnostic, {s0['hub_metric']} k={s0['hub_k']})   "
                f"context {s0['n_context']}x{s0['n_features']}, test {s0['n_test']}, "
                f"n_runs={cells[0]['n_runs']} (deterministic)")
        lines = [head]
        if missing:
            lines.append(f"  MISSING R: {', '.join(str(m) for m in missing)}")
        cols = f"  {'R%':>4s} {'k flipped':>9s}"
        for m in a.metrics:
            cols += f"  {SHORT[m]:>9s} {'delta':>9s} {'rel':>9s}"
        lines += ["", cols, "  " + "-" * (len(cols) - 2)]
        base = f"  {'0':>4s} {'0':>9s}"
        for m in a.metrics:
            base += f"  {clean[m]:>9.4f} {'-':>9s} {'-':>9s}"
        lines += [base + "   <- clean baseline", "  " + "-" * (len(cols) - 2)]

        for r in rs:
            c = cells[r]
            mm, s = c["m"], c["summary"]
            line = f"  {r:>4d} {s['k_flipped']:>9d}"
            for m in a.metrics:
                dv = mm[m] - clean[m]
                weak = abs(clean[m]) < WEAK.get(m, 0.0)
                rel = None if abs(clean[m]) < 1e-12 else 100.0 * dv / abs(clean[m])
                line += f"  {mm[m]:>9.4f} {dv:>+9.4f} {fmt_rel(rel, weak):>9s}"
                rows.append({"dataset": ds, "attack": "hub-flip", "row_percent": r,
                             "metric": m, "clean": clean[m], "poisoned_mean": mm[m],
                             "poisoned_std": 0.0, "delta_mean": dv, "delta_std": 0.0,
                             "relative_pct": rel, "relative_unstable": weak,
                             "n_runs": c["n_runs"], "k_flipped": s["k_flipped"]})
            lines.append(line)
        text = "\n".join(lines)
        print("\n" + text + "\n")
        (a.out / f"{ds}.txt").write_text(text + "\n")

    with open(a.out / "hub_flip_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.out / 'hub_flip_results.csv'} ({len(rows)} rows) and per-dataset tables")
    return rows


if __name__ == "__main__":
    main()
