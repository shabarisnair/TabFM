# Reading List: Designing a Robustness Mechanism for TabPFN's In-Context Set

**Goal.** TabPFN uses `X_train`/`Y_train` as its ICL context. Some unknown R% of those rows are
corrupted (X-perturbation or Y-flip). Build a mechanism that detects and cleans / downweights
them so inference degrades gracefully. **Not** distribution-shift robustness.

Companion to `literature_survey_poisoning_tfm.md` (attacks) and `paper_notes/` (per-paper notes).
Curated for *design inspiration*, not exhaustive.

---

## 1. Must-cite: the nearest prior work (differentiate from these)

| Paper | Why it matters to you |
|---|---|
| [Noise Immunity in In-Context Tabular Learning](https://arxiv.org/abs/2604.04868) — Hu & Ghelichi, Apr 2026 | **Closest paper that exists.** Same setting (TabPFN, label+feature noise in context, attention internals). But purely *descriptive* — the downweighting it reports is emergent model behaviour, **not** an added mechanism. Your method must beat this implicit robustness. |
| [On the Robustness of TFMs: Test-Time Attacks and In-Context Defenses](https://arxiv.org/abs/2506.02978) — Djilani et al., SaTML 2026 | Proposes **AICL**, a defense that edits the context. But defends *test-time evasion*; threat model explicitly assumes the context is clean. Your direction is the complement. |
| [A Mechanistic Study of TFMs](https://arxiv.org/abs/2605.21288) — Biloš et al., May 2026 | Hand-designed context poisons (hub/boundary label poison) on TabPFNv2/TabICLv2/Mitra. Attacks only, no defense. **Also the source of the L9-attention result you'll build on.** |
| [In-Context Learning with Noisy Labels](https://arxiv.org/abs/2411.19581) — Kang et al., Nov 2024 | **Formulates your exact problem** (corrupted labels in ICL demos) and proposes a method — but for **LLMs on text**. The problem is addressed for LLM ICL and open for tabular ICL: that's your novelty sentence. Expect "why not just port this?" in review. |

---

## 2. Detection scores to adapt (classical, non-ICL)

Each is a per-row score you can compute with TabPFN standing in for the base model.

| Paper | Signal | Fit for TabPFN |
|---|---|---|
| [Confident Learning / cleanlab](https://arxiv.org/abs/1911.00068) — Northcutt et al., JAIR 2021 | Confident joint from out-of-sample predicted probabilities | **Best fit.** K-fold probabilities cost K *forward passes*, not K trainings. Near drop-in. |
| [Deep k-NN Defense](https://arxiv.org/abs/1909.13374) — Peri et al., ECCV-W 2020 | Label disagrees with k-NN plurality in feature space | Use `get_embeddings()`, or better: TabPFN's own row-attention as the neighbour weighting. |
| [Sever](https://arxiv.org/abs/1803.02815) — Diakonikolas et al., ICML 2019 | SVD of per-sample **gradients**, remove top-ε outliers | Blocked as-is (no fitted params) → adapt to **input gradients** via `differentiable_input=True`. Assumes known ε, no clean set: matches your setting best theoretically. |
| [SPECTRE](https://arxiv.org/abs/2104.11315) — Hayase et al., ICML 2021 | Robust covariance whitening + QUantum Entropy scoring | Run on TabPFN embeddings. Caveat: built for *backdoors* (coherent subspace); weaker for scattered label flips. |
| [AUM](https://arxiv.org/abs/2001.10528) — Pleiss et al., NeurIPS 2020 | Margin trajectory over training epochs | **Doesn't port** (no epochs). But steal the *threshold trick*: insert deliberately-flipped indicator rows to calibrate the cut. |
| [Δ-Influence](https://arxiv.org/abs/2411.13731) — TMLR 2026 | "Influence collapse" after data transforms | Influence works better than naive IF for poisons. TabPFN makes **exact LOO** free — see §4. |
| [Threshold KNN-Shapley](https://arxiv.org/abs/2308.15709) — NeurIPS 2023 | Closed-form data value; flag large-negative | Beats Data Shapley at mislabel detection. ⚠ needs a utility set — use self-consistency instead of a clean set. |
| [Meta-Sift](https://arxiv.org/abs/2210.06516) — Zeng et al., 2022/23 | Bilevel optimisation: fit on a suspected-clean subset, score the rest by prediction loss; iterate. Reports **100% precision** on the sifted subset | **Strongest structural fit after CL.** It is the paper that attacks the "defender has clean data" assumption — i.e. exactly your constraint. Key insight: its bilevel inner loop is expensive *because it requires training*; for TabPFN the inner step is **one forward pass** (put the suspected-clean rows in the context, score the rest). The obstacle that makes Meta-Sift costly for everyone else is free for you. Natural fit with EM-style iteration. ⚠ Check whether it needs a known poison rate — the abstract doesn't say. |
| [Certified Defenses for Data Poisoning](https://arxiv.org/abs/1706.03691) — Steinhardt et al., NeurIPS 2017 | Upper bound on any "outlier-removal + ERM" defense | Tells you when your defense *can't* work. Good for an honest limitations section. |

---

## 3. ICL-specific inspiration (LLM side — the closest methodological analogues)

- [DemoShapley](https://arxiv.org/abs/2410.07523) — Shapley valuation of demonstrations; explicitly
  detects label-noise demos. The ICL-native version of data valuation.
- [ICLPoison](https://aclanthology.org/2025.findings-naacl.91/) — NAACL Findings 2025. Optimized
  demonstration poisoning for LLMs; useful for the *attacker* side of your evaluation.
- [Understanding ICL of Linear Models Through an Adversarial Lens](https://arxiv.org/abs/2411.05189) —
  Anwar et al., TMLR 2025. A *single* perturbed context example can hijack a linear transformer;
  adversarial training helps. Theory for why context corruption is potent.
- [Robust In-Context RL Under Reward Poisoning (AT-DPT)](https://arxiv.org/abs/2506.06891) — ICML 2026.
  The cleanest existing "poisoned context → adversarially-trained defense" template.
- [FCert](https://arxiv.org/abs/2404.08631) — IEEE S&P 2024. **Certified** robustness to poisoned
  few-shot support sets (vision/language). If you want a certification angle, start here.

---

## 4. TabPFN internals — what you can actually hook into

| Resource | What it gives you |
|---|---|
| [A Mechanistic Study of TFMs](https://arxiv.org/abs/2605.21288) | **L9 row-attention ≈ weighted vote over context labels (Pearson r = 0.89).** This is the hook for both detection *and* downweighting. |
| [Where Computation Lives Inside TabPFN](https://arxiv.org/abs/2606.12917) — Gupta et al., Jun 2026 | Causal head localisation. Note their **contrastive activation steering failed to transfer** — a warning about naive internals interventions. |
| [Interpretable ML for TabPFN](https://arxiv.org/abs/2403.10923) — Rundel et al., xAI 2024 | ⚠ **Check this before claiming novelty.** Claims ICL "facilitates estimation of Shapley values by avoiding approximate retraining" and enables LOCO; covers **data valuation** too. May partially pre-empt the "exact LOO is free" argument. |
| `classifier.get_embeddings(X, data_source="train"\|"test")` | Representation access → Deep k-NN / SPECTRE directly available. |
| `AlongColumnAttention` (architectures/tabpfn_v2.py) | Row/sample attention. **No `sample_weight` API exists** → downweighting = patch this (add `log w_i` to key logits). |
| `_flatten_multiclass_targets` | ⚠ **Soft labels are blocked**: computes `(y > unique_ys).sum(-1)`, a step function. Any `y > 0` → class 1, and no gradient flows through `y`. Rules out soft-label downweighting. |

---

## 5. Baselines you must beat — and the headroom problem

**The biggest risk to the project**: TabPFN is already very robust to *random* label noise, so your
method may show no gain there.

- [TabPFN: One Model to Rule Them All?](https://arxiv.org/abs/2505.20003) — Zhang et al. At ρ=0.3
  random flips, TabPFN "nearly matches the optimal classifier given ≥5K samples."
- [Noise Immunity](https://arxiv.org/abs/2604.04868) — ROC-AUC flat up to 35% flips.
- [TabICLv2](https://arxiv.org/abs/2602.11139) — degrades less than tree baselines at 0–50% flips.

**Implication:** target **adversarial / optimized** corruption (CAPGD X-perturbations, GA-selected
label flips), where none of these results apply. Our own sweep showed optimized poisoning collapses
url_unique at 1.3% of context while random noise at 30% does nothing — that contrast *is* the
motivation section.

---

## 6. Evaluation design

- [Mislabeled examples detection as probing ML models](https://arxiv.org/abs/2410.15772) — George et al.,
  **TMLR 2024.** Unifies the field into 4 building blocks, **emphasis on tabular / non-deep classifiers.**
  The single best methodological template for your experiments.
- [Benchmarking ML methods for identification of mislabeled data](https://www.researchsquare.com/article/rs-4011683/v1) —
  AI Review 2025. 34 methods reviewed, 20 benchmarked on tabular. Two findings to design around:
  detection is best at **20–30% noise**; **ensembles beat any single method**.
- [Calibration improves detection of mislabeled examples](https://arxiv.org/abs/2511.02738) — Nov 2025.
  Calibration is the key lever. ⚠ But [High Performance, Low Reliability](https://arxiv.org/abs/2605.28554)
  found TFMs have *worse* conformal coverage than GBDTs — measure, don't assume.
- [GBDT label-noise detection](https://arxiv.org/abs/2409.08647) — tabular-native baselines (incl. a
  "Gradients" method) to compare against.
- **Adaptive attackers — do not skip:** [Bypassing Backdoor Detection Algorithms](https://arxiv.org/abs/1905.13409)
  (Tan & Shokri) defeats spectral signatures/activation clustering; [Breaking the De-Pois Defense](https://arxiv.org/abs/2204.01090).
  Any defense claim needs an attacker that knows the defense — which your CAPGD code enables.
- **No-clean-set constraint:** see Meta-Sift in §2 — it formalises why "defender has a clean base set"
  is an unsafe assumption, which is the framing your evaluation should adopt.

---

## 7. Adjacent / optional

- [uLEAD-TabPFN](https://arxiv.org/abs/2604.20255) — TabPFN as a dependency-based anomaly detector.
  Candidate X-space poison detector; **structurally blind to label flips** (no `y`). See `paper_notes/ulead-tabpfn.md`.
- [Model-Aware Data Cleaning for TFMs](https://arxiv.org/abs/2604.25154) — RL-sequenced cleaning for
  TFM contexts. Benign data quality, never tested adversarially — an easy defense baseline to break.
- [VIP-COP](https://arxiv.org/abs/2605.12904) — black-box row/feature valuation for TFM contexts.
  Deliberately internals-free; the natural contrast to an attention-based method.
- Robust statistics background: [Recent Advances in Algorithmic High-Dimensional Robust Statistics](https://arxiv.org/abs/1911.05911)
  — where "known contamination fraction ε, no clean set" is the standard assumption.

---

## Suggested first build

1. TabPFN K-fold (or exact LOO) probabilities on the noisy context.
2. Three scores: confident-learning margin · attention-weighted-neighbour disagreement · exact LOO influence.
3. Ensemble; cut at known R% (or keep as P(poisoned)).
4. Downweight by patching `AlongColumnAttention`; baselines = hard removal, relabelling, oracle ceiling.
5. Report detection PR-AUC **and** downstream recovery vs. clean-data ceiling, under *optimized* poison.
