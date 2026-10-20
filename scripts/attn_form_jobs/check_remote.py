#!/usr/bin/env python
"""Checks that the second machine computes exactly what the main machine computes before it runs its share of the
attention-form study (remote_gpu<N>.sh call this first and stop if anything fails):

1. inputs: the 6 datasets' split files and the R = 30% V = 500 draws are byte-identical to the main machine's
   (inputs.sha256);
2. model: the checkpoint TabPFN loads here (build_tabpfn_v2's model_path, tabpfn-v2-classifier-finetuned-zk73skhh.ckpt
   in the TabPFN cache) is the main machine's file;
3. code: tabpfn is imported from this repo's TabPFN/src, with the row-weight hooks;
4. numbers: 3 updates of every configuration on is_this_a_good_customer give the main machine's held-out log loss at
   the start and after the first update (results/attn_form/preflight_reference, to 0.01 + 1%; later updates are only
   reported, since a large step can amplify GPU rounding).

    python scripts/attn_form_jobs/check_remote.py --device cuda:0 [--skip-numbers]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CKPT_SHA256 = "cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def numbers(device: str) -> list[str]:
    import attn_form_study as S
    ref_dir = REPO / "results" / "attn_form" / "preflight_reference"
    refs = sorted(ref_dir.glob("*.json"))
    if not refs:
        return [f"missing {ref_dir} (copy it from the main machine)"]
    root = Path(tempfile.mkdtemp(prefix="attn_form_check_"))
    os.environ["ATTN_FORM_SMOKE_ROOT"] = str(root)
    S.smoke(device)
    bad = []
    for ref in refs:
        got = root / "runs" / "is_this_a_good_customer_R30_V500_s0" / ref.name
        a = [e["hold_ce"] for e in json.loads(got.read_text())["evals"]] if got.exists() else []
        b = [e["hold_ce"] for e in json.loads(ref.read_text())["evals"]]
        off = [abs(x - y) > 0.01 + 0.01 * abs(y) for x, y in zip(a, b)]
        line = f"{ref.stem}: {[round(x, 4) for x in a]} here vs {[round(y, 4) for y in b]} on the main machine"
        if len(a) < 2 or any(off[:2]):
            bad.append(line)
        elif any(off):
            print(f"  (later updates drift, allowed) {line}")
    print(f"  numbers: {len(refs) - len(bad)} of {len(refs)} configurations match the main machine")
    return bad


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-numbers", action="store_true")
    a = ap.parse_args(argv)
    os.chdir(REPO)
    sys.path.insert(0, str(REPO / "scripts"))
    bad = []
    for line in (HERE / "inputs.sha256").read_text().splitlines():
        want, path = line.split(maxsplit=1)
        p = Path(path)
        if not p.exists():
            bad.append(f"missing {p}")
        elif sha256(p) != want:
            bad.append(f"differs from the main machine's: {p}")
    print(f"  inputs: {'ok' if not bad else 'NOT ok'} (datasets' split files and draws)")
    import tabpfn
    import tabpfn.preprocessing.steps.context_row_weights  # noqa: F401 - the row-weight hooks
    if not Path(tabpfn.__file__).resolve().is_relative_to(REPO / "TabPFN" / "src"):
        bad.append(f"tabpfn is imported from {tabpfn.__file__}, not this repo's TabPFN/src")
    from tabfm_experiments.config import Seeds
    from tabfm_experiments.runtime import build_tabpfn_v2
    ckpt = Path(build_tabpfn_v2("cpu", Seeds.model_seed, n_classes=2).model_path)
    if not ckpt.exists():
        bad.append(f"missing the model checkpoint {ckpt} (copy ~/.cache/tabpfn/{ckpt.name} from the main machine)")
    elif sha256(ckpt) != CKPT_SHA256:
        bad.append(f"the model checkpoint {ckpt} differs from the main machine's")
    print(f"  model: {ckpt}  code: tabpfn from {Path(tabpfn.__file__).parent}")
    if not bad and not a.skip_numbers:
        bad += numbers(a.device)
    for line in bad:
        print("  PROBLEM:", line)
    print("  all checks passed" if not bad else "  checks failed: not starting")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
