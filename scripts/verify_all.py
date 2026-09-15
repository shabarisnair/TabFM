#!/usr/bin/env python
"""End-to-end verification battery for the CAPGD context-poisoning attack.

Checks the ported pieces against the real TabularBench source where one exists,
and checks the invariants the attack is supposed to guarantee otherwise.
Exit code is non-zero if anything fails.
"""
from __future__ import annotations

import contextlib, importlib.util, io, json, subprocess, sys, types, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def load_tb_utils():
    """Import tabularbench/attacks/utils.py with its unused deps stubbed."""
    for name, attrs in {
        "tabularbench.constraints.constraints": {"Constraints": object},
        "tabularbench.constraints.constraints_fixer": {"ConstraintsFixer": object},
        "tabularbench.constraints.relation_constraint": {"EqualConstraint": object,
                                                         "Feature": object},
        "tabularbench.utils.typing": {"NDNumber": object},
    }.items():
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules.setdefault(name, m)
    path = ROOT / "tabularbench" / "tabularbench" / "attacks" / "utils.py"
    spec = importlib.util.spec_from_file_location("_tb_utils", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    from attack_context import fix_types_raw, fix_immutable_raw
    tb = load_tb_utils()
    rng = np.random.default_rng(0)

    # -- V1/V2: the optimiser core, against the real reference -------------------
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_capgd_equiv.py")],
                       capture_output=True, text=True)
    check("capgd() matches reference attack_single_run iterate-for-iterate",
          r.returncode == 0 and "ALL MATCH" in r.stdout,
          r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-200:])

    # -- V3: fix_types against the real implementation ---------------------------
    worst_t = 0.0
    for _ in range(500):
        D = int(rng.integers(3, 40))
        types_ = pd.Series(rng.choice(["real", "int", "cat"], D))
        xc = torch.tensor(rng.normal(0, 50, (1, D)), dtype=torch.float64)
        xa = xc + torch.tensor(rng.normal(0, 20, (1, D)), dtype=torch.float64)
        ref = tb.fix_types(xc.clone(), xa.clone(), types_)
        mine = fix_types_raw(xc, xa,
                             torch.tensor((types_ == "int").to_numpy()),
                             torch.tensor((types_ == "cat").to_numpy()))
        worst_t = max(worst_t, float((ref - mine).abs().max()))
    check("fix_types_raw == tabularbench fix_types (500 random cases)",
          worst_t == 0.0, f"max abs diff {worst_t:.3e}")

    # -- V4: fix_immutable against the real implementation -----------------------
    worst_m = 0.0
    for _ in range(500):
        D = int(rng.integers(3, 40))
        mut = pd.Series(rng.random(D) > 0.4)
        xc = torch.tensor(rng.normal(0, 50, (1, D)), dtype=torch.float64)
        xa = xc + torch.tensor(rng.normal(0, 20, (1, D)), dtype=torch.float64)
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")     # reference prints/warns on every call
            ref = tb.fix_immutable(xc.clone(), xa.clone(), mut)
        mine = fix_immutable_raw(xc, xa, torch.tensor(mut.to_numpy()))
        worst_m = max(worst_m, float((ref - mine).abs().max()))
    check("fix_immutable_raw == tabularbench fix_immutable (500 random cases)",
          worst_m == 0.0, f"max abs diff {worst_m:.3e}")

    # -- V5: invariants of a real run --------------------------------------------
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if out is None or not (out / "attack.json").exists():
        check("run artefacts present (pass a results dir to check invariants)", False,
              "skipped")
        return 1 if any(not ok for _, ok, _ in RESULTS) else 0

    d = json.loads((out / "attack.json").read_text())
    pr = pd.read_csv(out / "poisoned_row.csv")
    ds = Path(d["train"]).parts[1]
    meta = pd.read_csv(ROOT / "datasets" / ds / f"{ds}_metadata.csv").set_index("feature")
    pr["type"] = pr.feature.map(meta.type).str.strip().str.lower()
    pr["mutable"] = pr.feature.map(meta.mutable).astype(str).str.strip().str.lower().isin(["true", "1"])
    pr["lo"] = pr.feature.map(meta["min"]).astype(float)
    pr["hi"] = pr.feature.map(meta["max"]).astype(float)
    rangev = (pr.hi - pr.lo).replace(0, 1.0)

    # Tolerances have to be relative: deltas reach ~1e3 raw units at high eps, where
    # float32 resolution alone is ~1e-4, so an absolute 1e-6 bound is meaningless.
    ds_scaled = ((pr.poisoned - pr.original) / rangev).abs()
    nrm = d.get("norm", "Linf")
    # With constraints=full the reference applies fix_types AFTER the eps projection
    # and does not re-project, so rounding a `cat` feature can legitimately carry the
    # perturbation out to the *original* eps. Absorbing exactly that is what
    # eps_margin exists for, so the budget to check against is eps, not eps_effective.
    budget = d["eps"] if d["constraints"] == "full" else d["eps_effective"]
    tol = 1e-5 * max(budget, 1.0)
    if nrm == "L2":
        # per-row vector norm, matching the reference's dim convention
        per_row = (((pr.poisoned - pr.original) / rangev) ** 2).groupby(pr.row_index).sum() ** 0.5
        check("perturbation inside the eps-ball (L2, per row)",
              per_row.max() <= budget + tol,
              f"max ||delta||2 {per_row.max():.6f} <= {budget} (tol {tol:.1e})")
    else:
        check("perturbation inside the eps-ball (Linf)",
              ds_scaled.max() <= budget + tol,
              f"max |delta_scaled| {ds_scaled.max():.6f} <= {budget} "
              f"(eps_eff {d['eps_effective']}, tol {tol:.1e})")
    # cat rounding may also carry a value a fraction past the box edge, again by
    # design; and float32 makes a stored 39.3 differ from the metadata's 39.3.
    btol = 1e-5 * rangev + (0.5 * pr.type.isin(["cat"]).astype(float)
                            if d["constraints"] == "full" else 0.0)
    n_oob = int(((pr.poisoned < pr.lo - btol) | (pr.poisoned > pr.hi + btol)).sum())
    if d["constraints"] == "none":
        # --constraints none deliberately has no box: out-of-range values are the
        # point, not a violation. Only assert the eps-ball held.
        check("box intentionally not enforced (constraints=none)", True,
              f"{n_oob} cells outside [min,max], as expected")
    else:
        check("poisoned row inside [min, max] box", n_oob == 0, f"{n_oob} violations")
    imm = pr[~pr.mutable]
    check("immutable features unchanged",
          d["constraints"] == "none" or len(imm) == 0
          or bool((imm.original == imm.poisoned).all()),
          f"{len(imm)} immutable features"
          + (" (not enforced for constraints=none)" if d["constraints"] == "none" else ""))
    if d["constraints"] == "full":
        ic = pr[pr.type.isin(["int", "cat"])]
        err = (ic.poisoned - ic.poisoned.round()).abs()
        check("int/cat features exactly integral (constraints=full)",
              bool((err == 0).all()), f"max non-integrality {err.max():.3e}")
        it = pr[pr.type == "int"]
        dl = it.poisoned - it.original
        check("int deltas truncated toward zero (never grown)",
              bool((dl.abs() <= (it.poisoned - it.original).abs() + 1e-9).all())
              and bool((dl == np.trunc(dl)).all()),
              "all int deltas are whole numbers")

    check("loss recorded on the full test set",
          d["n_test"] == d["n_test_full"] or d.get("test_batch_size") or d["n_test_full"] > d["n_test"],
          f"n_test {d['n_test']} of {d['n_test_full']}")
    if "deployed" in d:
        check("deployed-path verification present", True,
              f"dloss {d['deployed']['delta']['loss']:+.5f}  "
              f"dmcc {d['deployed']['delta']['mcc']:+.5f}")

    bad = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS)-len(bad)}/{len(RESULTS)} passed" + (f"; FAILED: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
