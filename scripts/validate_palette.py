#!/usr/bin/env python
"""Python port of the dataviz skill's scripts/validate_palette.js (node is not installed here).

Same six checks, same thresholds, same Machado-Oliveira-Fernandes (2009) severity-1.0
CVD transforms, so the verdicts match what the JS validator would print.

    python scripts/validate_palette.py "#2a78d6,#eb6834,#1baf7a" --mode light --pairs all

``--pairs all`` is the scatter / bubble / choropleth / SMALL-MULTIPLES pairlist; the
default ``adjacent`` is for stacks, bars and single-panel lines.

Exit code 1 on any hard FAIL. A CVD result in the 6-8 band is a floor (legal only with
secondary encoding); sub-3:1 contrast is "relief" (obliges visible labels or a table
view) -- neither hard-fails, matching the JS.
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys

BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}   # OKLCH L
CHROMA_FLOOR = 0.10
CVD_TARGET, CVD_FLOOR = 8.0, 6.0        # OKLab dE x100, min(protan, deutan)
NORMAL_FLOOR = 15.0                     # hard gate, unsimulated vision
CONTRAST_MIN = 3.0
DEFAULT_SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}

MACHADO = {
    "protan": ((0.152286, 1.052583, -0.204868), (0.114503, 0.786281, 0.099216),
               (-0.003882, -0.048116, 1.051998)),
    "deutan": ((0.367322, 0.860646, -0.227968), (0.280085, 0.672501, 0.047413),
               (-0.011820, 0.042940, 0.968881)),
    "tritan": ((1.255528, -0.076749, -0.178779), (-0.078411, 0.930809, 0.147602),
               (0.004733, 0.691367, 0.303900)),
}


def hex2srgb(h):
    h = h.strip().lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]


def s2lin(c): return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
def lin(h): return [s2lin(c) for c in hex2srgb(h)]


def rel_lum(h):
    r, g, b = lin(h)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    hi, lo = sorted([rel_lum(a), rel_lum(b)], reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def oklab_from_lin(rgb):
    r, g, b = rgb
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


def oklch(h):
    L, a, b = oklab_from_lin(lin(h))
    return L, math.hypot(a, b)


def simulate(h, kind):
    r, g, b = lin(h)
    M = MACHADO[kind]
    return [min(1.0, max(0.0, M[i][0] * r + M[i][1] * g + M[i][2] * b)) for i in range(3)]


def delta_e(h1, h2, kind=None):
    a = oklab_from_lin(simulate(h1, kind) if kind else lin(h1))
    b = oklab_from_lin(simulate(h2, kind) if kind else lin(h2))
    return 100 * math.dist(a, b)


def validate(palette, mode="light", surface=None, pairs="adjacent"):
    surface = surface or DEFAULT_SURFACE[mode]
    lo, hi = BAND[mode]
    report, ok = [], True

    offband = [(c, round(oklch(c)[0], 3)) for c in palette if not lo <= oklch(c)[0] <= hi]
    ok &= not offband
    report.append(("Lightness band", "pass" if not offband else "fail",
                   f"outside band: {offband}" if offband
                   else f"all {len(palette)} inside L {lo}-{hi}"))

    lowc = [(c, round(oklch(c)[1], 3)) for c in palette if oklch(c)[1] < CHROMA_FLOOR]
    ok &= not lowc
    report.append(("Chroma floor", "pass" if not lowc else "fail",
                   f"below floor (reads gray): {lowc}" if lowc
                   else f"all {len(palette)} >= {CHROMA_FLOOR}"))

    n = len(palette)
    pairlist = (list(itertools.combinations(range(n), 2)) if pairs == "all"
                else [(i, i + 1) for i in range(n - 1)])
    label = "all-pairs" if pairs == "all" else "adjacent"

    worst = None
    for kind in ("protan", "deutan"):
        for i, j in pairlist:
            d = delta_e(palette[i], palette[j], kind)
            if worst is None or d < worst[0]:
                worst = (d, kind, palette[i], palette[j])
    tri = min((delta_e(palette[i], palette[j], "tritan") for i, j in pairlist), default=99)
    wd = worst[0] if worst else 99
    state = "pass" if wd >= CVD_TARGET else "floor" if wd >= CVD_FLOOR else "fail"
    ok &= state != "fail"
    report.append(("CVD separation", state,
                   f"worst {label} {worst[3]}<->{worst[2]} dE {wd:.1f} ({worst[1]}) "
                   f"· tritan {tri:.1f}" if worst else "n/a"))

    nworst = min(((delta_e(palette[i], palette[j]), palette[i], palette[j])
                  for i, j in pairlist), default=None, key=lambda t: t[0])
    nd = nworst[0] if nworst else 99
    nstate = "pass" if nd >= NORMAL_FLOOR else "fail"
    ok &= nstate == "pass"
    report.append(("Normal-vision floor", nstate,
                   f"worst {label} {nworst[2]}<->{nworst[1]} dE {nd:.1f} (normal)"
                   + ("" if nd >= NORMAL_FLOOR else f" — below {NORMAL_FLOOR:.0f}")
                   if nworst else "n/a"))

    low = [(c, round(contrast(c, surface), 2)) for c in palette
           if contrast(c, surface) < CONTRAST_MIN]
    report.append(("Contrast vs surface", "relief" if low else "pass",
                   f"below {CONTRAST_MIN}:1 — needs visible labels or a table view: {low}"
                   if low else f"all {len(palette)} >= {CONTRAST_MIN}:1"))
    return ok, report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("palette", help="comma-separated hex values")
    ap.add_argument("--mode", default="light", choices=["light", "dark"])
    ap.add_argument("--surface", default=None)
    ap.add_argument("--pairs", default="adjacent", choices=["adjacent", "all"])
    a = ap.parse_args(argv)

    pal = [c.strip() for c in a.palette.split(",") if c.strip()]
    ok, report = validate(pal, a.mode, a.surface, a.pairs)
    print(f"palette: {', '.join(pal)}   mode={a.mode}  pairs={a.pairs}  "
          f"surface={a.surface or DEFAULT_SURFACE[a.mode]}\n")
    for name, state, detail in report:
        mark = {"pass": "PASS", "fail": "FAIL", "floor": "WARN", "relief": "WARN"}[state]
        print(f"  {mark:<5s} {name:<22s} {detail}")
    print("\n" + ("ALL CHECKS PASS" if ok else "HAS HARD FAILURES"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
