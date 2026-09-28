# Paper Notes: uLEAD-TabPFN

**Title:** uLEAD-TabPFN: Uncertainty-aware Dependency-based Anomaly Detection with TabPFN
**Authors:** Sha Lu, Jixue Liu, Stefan Peters, Thuc Duy Le, Lin Liu, Jiuyong Li (Adelaide University); Yongzheng Xie (Queen's University Belfast)
**arXiv:** 2604.20255v1 (Apr 22, 2026)
**Status:** Preprint (uses a placeholder ACM template — "Conference acronym 'XX", 2018 copyright, dummy DOI; venue unconfirmed).
**Local file:** `papers/ulead-tabpfn.pdf`

> Framing note: this paper uses TabPFN as an anomaly **detector**, not as an object being
> attacked. It is the inverse of the context-poisoning POC — but directly relevant, because it
> is essentially a TabPFN-driven machine for scoring how anomalous a row is relative to a
> reference set, i.e. exactly the primitive an internals-based *defense* against context
> poisoning would need. See §6.

---

## 1. Problem

Semi-supervised **tabular anomaly detection**: given a training set of only *normal* rows
`X_train = {x_i}`, learn a score `s(x)` where higher = more anomalous. No anomaly labels at
train time. Score used for ranking or thresholding.

Weaknesses of prior art the paper targets:

- **Proximity-based** methods (kNN, LOF, isolation forest, CBLOF): flag low-density points.
  Degrade in high dimensions (distances stop being informative) and miss anomalies that
  violate *inter-feature relationships* rather than sitting far away.
- **Dependency-based** methods (DepAD, ADMIRE, DAGnosis): principled — an anomaly breaks the
  normal *conditional* structure. But prior versions (i) model dependencies in the **raw**
  feature space (unstable in high-d), (ii) need **task-specific model training** (fragile with
  limited data), and (iii) treat every dependency deviation as equally suspicious, **ignoring
  heteroscedastic noise** in normal data (some regions naturally vary more).

---

## 2. Proposed method

Three components (Figure 1):

### (a) Representative Context Set (RCS)
A fixed compact subset `C ⊆ X_train`, budget `|C| ≤ c` (they use **c = 500**), used as
TabPFN's in-context reference during both training and inference (avoids feeding the whole
training set, which is expensive).
- Clustering-based: **K-means, k = 100**; budget allocated uniformly across clusters.
- Within each cluster, sample **both central points (near centroid) and dispersed points (far
  from centroid)**, ~half each → covers typical *and* boundary regions of the normal manifold.
- Built once, frozen thereafter.

### (b) Dependency-aligned latent modeling  *(the core)*
- **Linear encoder** `z_i = W_enc x_i + b_enc` maps rows to a latent space `R^p` (`p ≤ d`).
  Two jobs: align the representation with conditional dependency, and reduce dimensionality
  for high-d data.
- **Frozen TabPFN** as conditional predictor: for each latent dim `j`, predict the conditional
  mean `μ̂_ij = E[Z_j | Z_{-j} = z_{i,-j}, C]` — "given the other latent dims + RCS, what should
  dim j be?" Done via in-context inference, **no fine-tuning**.
- **Dependency loss** (Eq. 2) trains the encoder so normal data has strong, consistent
  conditional dependencies:
  `L_dep = (1/Np) Σ_i Σ_j (z_ij − μ̂_ij)^2`
- **Reconstruction regularizer** (linear decoder, Huber penalty, Eq. 3–4) prevents the encoder
  from collapsing to a degenerate representation.
- Final objective (Eq. 5): `L_rl = λ_dep L_dep + λ_rec L_rec`, with **λ_dep = 1, λ_rec = 0.3**.
- **TabPFN stays frozen** — only the linear encoder/decoder (and the variance net below) are
  trained. This is the source of the robustness/efficiency claim: inherits TabPFN's pretrained
  conditional modeling without dataset-specific dependency-model training.

### (c) Uncertainty-aware scoring  *(the "u")*
Motivation (Figure 2): a raw deviation `z_ij − μ̂_ij` is insufficient under heteroscedastic
normal noise — a small deviation in a normally-stable region is more suspicious than a large
deviation in a normally-noisy region.
- **Variance network** (small MLP, 64→32→1, ReLU, dropout 0.1) predicts conditional residual
  variance `σ̂²_ij` from the other latent dims.
- Trained by Gaussian NLL (Eq. 6): penalizes large errors while discouraging trivially
  inflated variances.
- **Anomaly score** (Eq. 7) = variance-normalized mean squared deviation:
  `s(x) = (1/p) Σ_j (z_j − μ̂_j)^2 / σ̂²_j`

**Training:** two-stage — dependency model (encoder) first, then variance networks on the
resulting residuals.

**Complexity:** train and inference scale **linearly** in `p` and `|C|` (both fixed, small) →
efficient even in high-d; no test-time fine-tuning.

---

## 3. Evaluation

- **Benchmark:** ADBench — **57** tabular anomaly-detection datasets, standard semi-supervised
  protocol. Grouped by dimensionality: **Low** (`d ≤ 20`, 24 sets), **Medium** (`20 < d ≤ 100`,
  18 sets), **High** (`d > 100`, 15 sets).
- **Baselines:** 26 methods — proximity (kNN, LOF, CBLOF, ECOD, COPOD, IForest, PCA, OCSVM),
  deep (DROCC, GOAD, ICL, SLAD, DIF), generative/diffusion (VAE, GAN, DDPM, DAGMM, DAEE), and
  foundation-model-based (**FoMo-0D**, another PFN-based zero-shot detector), plus DAGnosis/DDAE.
- **Metrics:** ROC-AUC and PR-AUC, mean over **5 seeds**, reported with rank.
- **Hyperparameters:** fixed across all datasets (encoder 20 epochs, Adam, batch 1024; latent
  `p = d` for `d ≤ 100`, else 100; RCS budget 500). Minimal tuning is a stated selling point.

---

## 4. Results

### Main result (Table 1) — by dimensionality (ROC-AUC, rank in parens)

| Regime | uLEAD-TabPFN | Rank |
|---|---|---|
| High (`d > 100`) | 73.98 | **1** |
| Medium | 80.13 | **1** |
| Low | 85.51 | mid-pack |
| All | 80.78 | ~2 |

- Wins on **medium & high** dimensions (where proximity methods fail); only competitive on
  low-d (where simple methods already suffice).
- Reported gains: **+1.5% ROC-AUC / +2.8% PR-AUC over best baseline**; **~+16% / +20% over the
  *average* baseline.**
- Wilcoxon signed-rank: **17/25** significant ROC-AUC wins overall; **20/20+** significant wins
  in the high-d group.

### Complementarity (Figure 3, §4.3) — the more interesting finding
- Per-dataset ROC-AUC Spearman correlation with other methods is **low** (avg **0.67** vs
  **0.83+** among proximity methods); forms its own cluster in hierarchical clustering.
- I.e. it catches anomalies others miss, not incremental improvement.
- Examples: InternetAds — baselines 53–76%, uLEAD **~90%**; Wilt — baselines 32–75%, uLEAD
  **97%**.

### Ablations (Table 2)
| Variant | Effect |
|---|---|
| **uLEAD-Original** (dependencies in raw space, no encoder) | Largest drop: 71.98 vs 83.21 ROC-AUC (`d ≤ 100`). Latent alignment is the critical piece. |
| **uLEAD-CART** (CART instead of TabPFN) | Big drop: 76.90 vs 80.78 (All). TabPFN-as-predictor matters. |
| **LEAD-TabPFN** (no uncertainty scoring) | Smaller but consistent drop, esp. PR-AUC → uncertainty mainly adds *stability/robustness*, not raw gain. |

### Sensitivity (Figure 4)
Performance rises quickly with latent dim `p`, plateaus around `p ≈ 40`. Not sensitive to exact
value.

### Case study — Wilt (Figure 5, §4.6)
- ROC-AUC **97.5%** vs baselines <75%.
- UMAP: normal/anomalous overlap in **both** raw and latent space (no geometric separation) →
  why proximity/density fail.
- But **conditional structure** sharpens in latent space: mean abs feature-pair correlation
  0.239 → **0.531**; conditional `R²` under frozen TabPFN 0.239 → **0.991**. Anomalies caught as
  dependency violations, not distance outliers.
- Score is **interpretable**: per-latent-dim contributions map back through the linear encoder
  to feature-level attribution, no post-hoc explainer.

---

## 5. Takeaways

- **Novelty:** a *frozen* TabPFN as an in-context conditional-mean estimator inside a learned
  latent space, plus heteroscedastic variance normalization of residuals. Sidesteps both raw-
  space modeling (unstable in high-d) and dependency-model training (fragile).
- **Where it wins:** concentrated in medium/high dimensions; the headline contribution is
  **complementarity**, not dominance (raw margin over best baseline is modest, ~1.5%).
- **Caveats:** preprint template (venue unconfirmed); "+1.5% over best" is small — read the
  complementarity result as the real point.

---

## 6. Relevance to the context-poisoning project

1. **Ready-made poison-detection primitive.** The open "internals-based defense" idea (score
   context rows, prune suspicious ones) needs exactly a per-row normality score relative to a
   reference set. uLEAD-TabPFN *is* that: a TabPFN-driven dependency-violation score. A poison
   row optimized for maximal influence is plausibly a conditional-dependency outlier, so
   uLEAD-style scoring over the context + dropping high-scorers is a concrete defense to test
   against the CAPGD attack. Makes the "adaptive attacker vs internals-based defense" story
   buildable.
2. **Also an attack target for the surgical variant.** If a defender filters the context with
   dependency-based scoring, the poison rows must be *low-anomaly under uLEAD* **and**
   high-influence — a clean-label constraint. That is the adaptive-attacker evaluation a
   credible defense paper would need.

See also: `literature_survey_poisoning_tfm.md` (§ candidate defenses / TFM-as-detector).
