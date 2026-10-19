# Larger-V poisoning benchmark: runbook

**Question.** With V = 50 to 500 clean rows instead of a handful, do the attention-weight learners (top-k M1
and soft A3) clean a poisoned TabPFN context better than the NaN gates (soft and top-k) and the references?

**Code:**
- `scripts/bench_largev.py`: draws, import, jobs, status and the GPU job lists;
- `scripts/report_bench_largev.py`: tables with 95% intervals over the samplings of V, and a TabArena-style Elo.

Results go to `results/bench_largev/` (git-ignored), and tables to `results/tables/bench_largev/`.

## Protocol

- **Instances:** 6 datasets × V ∈ {50, 100, 200, 300, 400, 500} × R ∈ {10, 20, 30}% × one sampling of V
  (validation seed 0), which is 108 instances.
  - The draws for seeds 1-4 are built too, so samplings can be added later (`--seeds 0 1 2 ...` for `plan`,
    `status` and the report).
  - The datasets and files are the tiny-V benchmark's (`docs/bench_tinyv.md`): gas-drift, heloc,
    ecommerce_shipping, students_dropout, is_this_a_good_customer (1400-row train file) and coil2000.
- **The draw** (`attn_learn.make_draw`, the soft gate's draw of `results/gate_valsplit`):
  - V clean rows, class-stratified, from the clean train context. One seed gives nested draws: the V = 200
    draw contains the V = 100 one.
  - 20% of V is held out, class-stratified.
  - The rest of the context is poisoned with hub-flip-local at R%: k = R% of its rows get a wrong label.
  - The draws were built once and are checked against the soft gate's saved draws where those exist
    (gas-drift, heloc and students at R = 10 and 20%, V 50-500, seeds 0-2).
- **Learners** fit on the other 80% of V. Each keeps the iteration with the lowest log loss on the held-out
  20% (ties: the earlier one).
- **Deployment:** every entry's context includes all of V (V + the cleaned rows). The learners are also
  scored without V (suffix `_nov`).
- **Metrics:** test accuracy, balanced accuracy, ROC-AUC (binary: class 1; multi-class: one-vs-rest macro),
  log loss; the precision of the removed rows.

## Methods

| job | entry (with V) | what | settings |
|---|---|---|---|
| `refs` | `v_only`, `poisoned_plus_v`, `oracle_plus_v`, `vfilter` | V alone; poisoned + V; poisoned rows removed + V; V-filter (all of V scores the given labels, the k least plausible removed) + V | none |
| `attn_topk` | `attn_topk` | attention weights, hard top-k (M1) | PGD-L-inf lr 0.2, probe 0.05, V-filter start; 200 updates, held-out loss every 10 |
| `attn_soft` | `attn_soft` (hard), `attn_soft_softw` (soft weights) | attention weights, log w ∈ [-1000, 0] (A3) | Adam lr 0.1, V-filter start, no prior; 200 updates, every 10 |
| `nan_soft` | `nan_soft` (hard), `nan_soft_softg` (soft gates) | the soft feature-input (NaN) gate | g ∈ [0, 1] from 1, PGD-L-inf lr 0.05; 200 updates, every 5 |
| `nan_topk_r0`-`r3` | `nan_topk` (vote), `nan_topk_gated` | the top-k NaN gate, 4 restarts; their chosen masks vote | N(0, 1) scores, gradient at the relaxed point 0.05, L-inf lr 5; 200 updates, every 5 |

These are the settings each method was tuned to in its own study:
- **M1 and A3:** `results/tables/attn_learn/findings.md`.
- **The soft NaN gate:** `results/gate_valsplit`.
- **The top-k NaN gate:** `results/tables/gate_topk/night_log.md`.

Two changes from those studies:
- The NaN gates get 200 updates, the attention learners' budget; their studies ran 300.
- All learners choose their iteration the same way, by held-out log loss. The soft gate's study had chosen it
  by held-out accuracy.

**Hard deployment:** a learner's k = R% lowest-weight (or lowest-gated) rows are removed. For the top-k gate,
the vote's k rows are removed; `nan_topk_gated` instead keeps those rows with their features set to missing.

## Already done (imported, not re-run)

`import-attn` copies the attention study's runs (`results/attn_learn/confirm`). Their draws, configurations,
holdout and selection are identical to this benchmark's, and the import checks the poisoned rows. It covers:
- `refs` and `attn_topk` for heloc, coil2000 and students_dropout × V ∈ {50, 500} × R ∈ {10, 20, 30}%, seed 0
  (18 instances);
- `attn_soft` for heloc and students_dropout on the same grid (12 instances).

The study also has seeds 1-2; `import-attn --seeds 0 1 2` brings them in if samplings are added.

**Not reused:**
- The soft NaN gate's earlier runs (`results/gate_valsplit`: gas-drift, heloc, students; R = 10 and 20%;
  V 50-500; seeds 0-2): they were scored without V in the context, they chose the iteration by held-out
  accuracy, and the gates at the log-loss choice were not saved.
- The top-k vote runs (`results/gate_topk_gen`: R = 20%, seed 0): they were scored without V.

## Running it

```
# once, on one machine (done 2026-10-19); then copy results/bench_largev/draws/ to every machine
python scripts/bench_largev.py draws --device cuda:0
python scripts/bench_largev.py import-attn

# the 8 GPU scripts (one per GPU), and progress
GPU=0 bash scripts/bench_largev_jobs/gpu_00.sh      # ... gpu_07.sh; WORKERS=2 per GPU by default
python scripts/bench_largev.py status

# tables (partial results are fine)
python scripts/report_bench_largev.py
```

- **Job lists:** `python scripts/bench_largev.py plan --scripts N` packs every unfinished job into N lists.
  It runs longest first and keeps an instance's four top-k restarts together, since their vote needs all four.
- **Workers:** each GPU script runs 2 workers, then one worker alone to retry what failed (for example out of
  memory), then any missing top-k vote.
  - Finished jobs are skipped, so a script can simply be started again.
  - Workers on one machine share jobs through lock files.
- **Machines without a shared folder:**
  - each machine needs the same `results/bench_largev/draws/`;
  - afterwards, copy every machine's `results/bench_largev/runs/` into one (one file per instance and job, so
    the copies never collide).
- **Environment:** the attention jobs need this repository's TabPFN copy (`pip install -e TabPFN`); a worker
  without it skips them and says so.
- **GPU memory:** gas-drift and coil2000 jobs take about 45 GB each, hence 2 workers per 94 GB H100.

## Cost

The cost model:
- Seconds per update are measured from the tiny-V benchmark's attention and NaN-gate jobs (3 per H100,
  divided by the 2.8 jobs per GPU measured there).
- They are scaled by the context size squared.
- Jobs count 4 (refs) or 215 (attention, and each NaN gate run) update-equivalents.

816 jobs, about 72 H100-hours (±50%): 9 per GPU script, so about 9 hours on 8 GPUs.
`scripts/bench_largev_jobs/split.md` has the per-script estimate. The four top-k restarts are 59% of the
total, and gas-drift and coil2000 three quarters.
