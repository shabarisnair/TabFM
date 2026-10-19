#!/usr/bin/env python
"""Job files for scripts/schedule_study.py: every (seed, c, R, dataset) draw x schedule.

    python scripts/make_sched_jobs.py --out results/sched/jobs_p2.jsonl --seeds 0 1 2 --specs fixed:frac0.1:eprior ...
"""
import argparse
import json

DATASETS = ["gas-drift", "coil2000_insurance_policies", "url_unique", "heloc", "students_dropout", "credit-approval"]

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--specs", nargs="+", required=True)
ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
ap.add_argument("--cs", nargs="+", type=int, default=[1, 5])
ap.add_argument("--Rs", nargs="+", type=float, default=[20.0, 10.0, 30.0])
ap.add_argument("--datasets", nargs="+", default=DATASETS)
a = ap.parse_args()
n = 0
with open(a.out, "w") as f:
    for seed in a.seeds:
        for c in a.cs:
            for R in a.Rs:
                for ds in a.datasets:
                    for sp in a.specs:
                        f.write(json.dumps({"dataset": ds, "R": R, "c": c, "seed": seed, "spec": sp}) + "\n")
                        n += 1
print(a.out, n, "jobs")
