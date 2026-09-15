# Literature Survey: Noise, Poisoning & Adversarial Attacks on Tabular Foundation Models (2025–2026)

Scope: papers from Jan 2025 – Sep 2026 relevant to (a) noise/corruption in the in-context
training samples of TFMs, (b) adversarial attacks on TFMs by attacking the train/context or
the test input, and (c) adversarial attacks on TFMs adapted for time-series forecasting.
Each entry notes what part of the pipeline is attacked and whether perturbations are
optimised or hand-designed/random.

Verification: claims were checked against each paper's abstract, and against full text where
retrievable. Entries marked **[abstract only]** could not be fully verified beyond the abstract.

---

## Bottom line

- **No published work optimises perturbations of the ICL training rows (the context) to
  degrade a TFM.** This is the gap the POC targets.
- The closest tabular work either (i) attacks **test inputs only** and explicitly excludes
  context poisoning (Djilani et al., SaTML 2026), (ii) applies **hand-designed** context
  perturbations with small effect (mechanistic study, up to −3.7 pp for label poisoning), or
  (iii) injects **random** noise, to which TabPFN is robust.
- On time series, one systematic adversarial study includes TabPFN-TS. Because TabPFN-TS
  encodes past timesteps as its ICL training rows, that paper is implicitly the nearest thing
  to an optimised *context* attack on a TFM — but framed as a test-time attack on a TS model.

---

## 1. Noise / corruption in the ICL training samples

| Paper | Authors | Date / Venue | Into the context | Optimised? | Key finding |
|---|---|---|---|---|---|
| [Noise Immunity in In-Context Tabular Learning: An Empirical Robustness Analysis of TabPFN's Attention Mechanisms](https://arxiv.org/abs/2604.04868) | J. Hu, M. Ghelichi | Apr 2026 | Random uncorrelated / nonlinearly-correlated features; mislabeled targets | Random | ROC-AUC stays high, attention stays sharp; informative features stay top-ranked. Descriptive, no optimised noise. |
| [TabPFN: One Model to Rule Them All?](https://arxiv.org/abs/2505.20003) | Q. Zhang, Y. S. Tan, Q. Tian, P. Li | May 2025 (v2 Nov 2025) | Random symmetric label flips ρ∈{0.1..0.4} in the training set | Random | "Even with noise levels as high as ρ=0.3, TabPFN (noisy) nearly matches the optimal classifier given a sufficiently large training set (≥5K samples)." Compared vs Bayes, kNN, LDA. |
| [A Mechanistic Study of Tabular Foundation Models](https://arxiv.org/abs/2605.21288) | M. Biloš, J. T. Wilson, A. Schneider, Y. Nevmyvaka | May 2026 | 8 hand-designed perturbations on the context (labels & features) | Hand-designed (mechanism-derived) | Label poisoning is weak: hub poison −3.7/−3.3/−2.8 pp (TabPFNv2/TabICLv2/Mitra), boundary poison −1.7 pp, centroid injection ~0. Feature transforms hurt more: rank warp −8.0/−10.1/−0.4 pp, SVD burial −8.3/−5.0/−8.9 pp. **Ridge/XGBoost/MLP refit on the same poisoned context absorb it comparably.** |
| [LimiX: Unleashing Structured-Data Modeling Capability for Generalist Intelligence](https://arxiv.org/abs/2509.03505) | X. Zhang et al. (38 authors) | Sep 2025 (v2 Nov 2025) | Uninformative features, outliers (robustness §7.4) | Random | Secondary sources report LimiX holds up with up to ~90% uninformative features while TabICL/CatBoost degrade. **[§7.4 numbers not verified from source]** |

**Takeaway:** random context noise is largely harmless to TabPFN; the only *targeted* context
poisoning (mechanistic study) is heuristic, small, and no worse than for refit baselines.

---

## 2. Adversarial attacks on TFMs

### 2a. Attacks on test inputs (evasion)

| Paper | Authors | Date / Venue | Models / data | Attack | Notes |
|---|---|---|---|---|---|
| [On the Robustness of Tabular Foundation Models: Test-Time Attacks and In-Context Defenses](https://arxiv.org/abs/2506.02978) | M. Djilani, T. Simonetto, K. Tit, F. Tambon, S. Ghamizi, M. Cordy, M. Papadakis | v1 Jun 2025, v2 Apr 2026 · **SaTML 2026** | TabPFNv2, TabICL · URL, LCLD, WIDS | CAPGD, MOEVA, CAA; ε=0.5 attack / 0.3 defense; `n_estimators=1` | **The primary reference for the POC.** Test-input perturbations sharply cut accuracy; TFMs generate evasion transferable to RF/XGBoost. Defenses: adversarial fine-tuning + AICL (swap context for adversarial rows, no weight updates). Reports WIDS as 186 features (one-hot-encoded categoricals). |

Threat-model quotes (full text):

> "As we do not tackle **backdoor or poisoning attacks**, we suppose that neither the Tabular FM
> under attack itself, **nor its context**, is corrupted at any point, only the X_infer are affected."

> "…we leave to future work the study of **backdoor and poisoning attacks against tabular FMs**."

Nearby test-time / shift work:

| Paper | Authors | Date / Venue | What | Perturbs |
|---|---|---|---|---|
| [When Tabular Foundation Models Meet Strategic Tabular Data: A Prior Alignment Approach](https://arxiv.org/abs/2605.19662) | X. Lv et al. | May 2026 · **ICML 2026** | Strategic classification (agents change own features), not an attack; proposes SPN | Test features |
| [Are Tabular Foundation Models Robust to Realistic Query Distribution Shifts in Microbiome Data?](https://arxiv.org/abs/2606.24995) | G. Perciballi, A. Fall, F. Granese, E. Prifti, J.-D. Zucker | Jun 2026 | Sparsification hurts TFMs more than RF | Query rows only **[abstract only]** |
| [Empirical Evaluation of Out-Of-Distribution Performance of Tabular Foundation Models](https://arxiv.org/abs/2607.26000) | M. Loza, D. Chushig-Muzo, E. Milara, L. Bote-Curiel, L. Estrada-Petrocelli, F. Grijalva | Jul 2026 | 9 TFMs (TabPFNv2→v3, TabICL/v2, Mitra, LimiX, TabFM) all degrade under natural shift (gaps 0.003–0.060) | Test/query (natural shift) |
| [Beyond IID: How General Are Tabular Foundation Models, Really?](https://arxiv.org/abs/2606.30410) | L. Purucker, A. Tschalzev, N. Erickson, G. Blayer, D. Holzmüller, A. Arazi, A. Pfefferle, M. Tajjar, G. Varoquaux, F. Hutter | Jun 2026 | BeyondArena: tree/deep models still win on non-IID, large, high-dim data | Natural (temporal/grouped) |

### 2b. Attacks on the training rows / context (poisoning / backdoors)

**None found for TFMs.** No optimised poisoning or backdoor attack on a TFM context exists in
the surveyed literature. The mechanistic study (§1) is the only work that deliberately perturbs
the context, and it is heuristic. This is the open gap.

### 2c. "Robust"/defense/context-optimization work that is NOT an input attack

| Paper | Authors | Date / Venue | What |
|---|---|---|---|
| [Robust Tabular Foundation Models (RTFM)](https://arxiv.org/abs/2512.03307) | M. Peroni, F. Le, V. Sheinin | Dec 2025 · **AAAI 2026** | "Adversarial" = min-max over the *synthetic data generator* during pretraining (hard datasets), **not** input perturbation. +6% mean normalised AUC on TabPFN v2. |
| [Towards Fair In-Context Learning with Tabular Foundation Models](https://arxiv.org/abs/2505.09503) | P. Kenfack, S. E. Kahou, U. Aïvodji | May 2025 · **TMLR** | Deliberately edits the context (decorrelation, group-balancing, uncertainty selection) to reduce bias on TabPFNv2/TabICL/TabDPT. Shows the context is a real lever (for mitigation). |
| [VIP-COP: Context Optimization for Tabular Foundation Models](https://arxiv.org/abs/2605.12904) | Y. Chen, X. Ding, L. Akoglu | May 2026 | Black-box (KernelSHAP) selection of most-valuable context rows/features; tested under added noise. Essentially the inverse of poisoning. |

### 2d. Privacy of context rows

| Paper | Authors | Date / Venue | Finding |
|---|---|---|---|
| [Risk In Context: Benchmarking Privacy Leakage of Foundation Models in Synthetic Tabular Data Generation](https://arxiv.org/abs/2507.17066) | J. Byun, X. Lin, J. Ward, G. Cheng | Jul 2025 | TabPFN v2 (as generator), GPT-4o-mini, LLaMA 3.3 leak membership; prompt tweaks reduce leakage. |
| [TabPATE: Differentially Private Tabular In-Context Learning Without Public Data](https://arxiv.org/abs/2606.31474) | D. Wahdany, M. Jagielski, J. C. Cresswell, A. Dziedzic, F. Boenisch | Jun 2026 | Membership inference succeeds against tabular ICL; PATE-style defense reduces it to near-random. |

---

## 3. Adversarial attacks on TFMs for time-series forecasting

| Paper | Authors | Date / Venue | Models | Attack | Finding |
|---|---|---|---|---|---|
| [Are Time-Series Foundation Models Deployment-Ready? A Systematic Study of Adversarial Robustness Across Domains](https://arxiv.org/abs/2505.19397) | J. Zhang, Z. Zhang, S. Zheng, X. Wen, J. Li, J. Bian | v1 May 2025, v2 Dec 2025 | TimesFM, TimeMoE, UniTS, Moirai, Chronos, **TabPFN-TS** | White-box PGD (300 it.); black-box SimBA, ZOO; budget ‖δ‖₀≤rL, ‖δ‖∞≤ε·var(x) on the **final rL timesteps of input history**; targets: scale/flip/drift/offset | Most TSFMs brittle; shape/trend attacks > local; horizon-proximal points most vulnerable; longer context → more vulnerable; weak cross-model transfer. **TabPFN-TS "particularly high sensitivity."** Latent adversarial training cuts worst-case error 4–10× (white-box). |

**Why this is close to the POC (inference, not stated in the paper):** TabPFN-TS encodes past
timesteps as its ICL *training rows* (targets included) and the forecast horizon as *test rows*.
So perturbing the input history in TabPFN-TS is effectively perturbing context rows / labels —
the nearest published optimised context attack on a TFM, but framed as a test-time TS attack.

Other TS work (none attacks a TFM context):

| Paper | Authors | Date / Venue | Notes |
|---|---|---|---|
| [GITCO: Gated Inference-Time Context Optimization in TSFMs](https://arxiv.org/abs/2606.05332) | M. Pandey, D. Kumar, M. Mandal, S. Deshpande | Jun 2026 | "Context poisoning" here = *benign* anomalous patches in TimesFM 2.5; a **defense** (+1.95% MASE). |
| [Keep the Lights On, Keep the Lengths in Check: Plug-In Adversarial Detection for Time-Series LLMs in Energy Forecasting](https://arxiv.org/abs/2512.12154) | H. Ma, R. Sun, M. Xue, X. Yuan, C. Rudolph, S. Nepal, L. Liu | Dec 2025 | Detection on TimeGPT/TimesFM/TimeLLM via consistency on shortened inputs. |
| [Adversarial Vulnerabilities in Large Language Models for Time Series Forecasting](https://arxiv.org/abs/2412.08099) | F. Liu, S. Jiang, L. Miranda-Moreno, S. Choi, L. Sun | v1 Dec 2024 · **AISTATS 2025** | Gradient-free black-box attacks on LLMTime (GPT-3.5/4, LLaMA, Mistral), TimeGPT, TimeLLM. |
| [Temporally Unified Adversarial Perturbations for Time Series Forecasting](https://arxiv.org/abs/2602.11940) | R. Su, Y. Bao, X. Zhang | Feb 2026 | Attacks "4 SOTA forecasters"; whether any are foundation models unstated. **[abstract only]** |
| [BadTime: An Effective Backdoor Attack on Multivariate Long-Term Time Series Forecasting](https://arxiv.org/abs/2508.04189) | K. Xiang et al. | Aug 2025 (v2 Nov 2025) | Backdoor on task-specific long-term forecasters, not TSFMs. |
| [Beyond Immediate Activation: Temporally Decoupled Backdoor Attacks on Time Series Forecasting (TDBA)](https://arxiv.org/abs/2601.04247) | Z. Liu, X. Liu, S. Xu, Y. Qiao, Y. Zhang, X. Cai | Jan 2026 | Backdoor on task-specific MTS forecasters, not TSFMs. |
| [TimeGuard: Channel-wise Pool Training for Backdoor Defense in Time Series Forecasting](https://arxiv.org/abs/2605.22365) | Q. D. Nguyen, S. Liang, Y. Li, F. Huo, D. Tao | May 2026 · **ICML 2026** | Backdoor **defense**; task-specific forecasters, no TSFMs. |

---

## 4. Related work outside TFMs (useful for framing / contrast)

| Paper | Authors | Date / Venue | Relevance |
|---|---|---|---|
| [Understanding In-Context Learning of Linear Models in Transformers Through an Adversarial Lens](https://arxiv.org/abs/2411.05189) | U. Anwar, J. von Oswald, L. Kirsch, D. Krueger, S. Frei | v1 Nov 2024, v2 Aug 2025 · **TMLR 2025 (Featured)** | Linear transformers can be *hijacked to arbitrary output by perturbing a single in-context example*; GPT-2-style too. Targeted, single-query — contrast with the POC's untargeted, 1000-query result where one row did nothing. |
| [Adversarially Pretrained Transformers May Be Universally Robust In-Context Learners](https://arxiv.org/abs/2505.14042) | S. Kumano, H. Kera, T. Yamasaki | May 2025 (v3 Mar 2026) · **ICLR 2026** | Robustness from adversarial pretraining, using *clean* demonstrations. |
| [Robust In-Context Reinforcement Learning Under Reward Poisoning Attacks (AT-DPT)](https://arxiv.org/abs/2506.06891) | P. Sasnauskas, Y. Yalın, G. Radanović | Jun 2025 (v3 Jun 2026) · **ICML 2026** | Poisoned in-context rewards for Decision-Pretrained Transformers; adversarial training defense. Closest "poisoned context → learned defense" analogue. |
| [Data Poisoning for In-Context Learning (ICLPoison)](https://aclanthology.org/2025.findings-naacl.91/) | P. He, H. Xu, Y. Xing, H. Liu, M. Yamada, J. Tang | **NAACL Findings 2025** | LLM analogue of optimised demonstration poisoning (discrete text perturbations on hidden states). |
| [TabAttackBench: A Benchmark for Adversarial Attacks on Tabular Data](https://arxiv.org/abs/2505.21027) | Z. He, C. Ouyang, L. Wen, C. Liu, C. Moreira | May 2025 · *Expert Systems with Applications* 2026 | Test-time attacks on LR/MLP/TabTransformer/FT-Transformer. **No TFM included.** |
| [Insights on Adversarial Attacks for Tabular Machine Learning via a Systematic Literature Review](https://arxiv.org/abs/2506.15506) | S. Dyrmishi, M. Djilani, T. Simonetto, S. Ghamizi, M. Cordy | Jun 2025 · under review ACM Comput. Surv. | Survey of tabular adversarial attacks; notes the field is scattered. |
| [Constrained Adaptive Attack (CAA)](https://arxiv.org/abs/2406.00775) | Simonetto, Ghamizi, Cordy et al. | 2024 | Origin of the CAPGD/CAA methods reused by 2506.02978 and by this POC. |

---

## 5. Open problems (targets for the POC)

1. **Optimised poisoning of TFM contexts** — gradient-based, budgeted, with realistic
   constraints. Not done anywhere; explicitly left to future work by Djilani et al.
2. **Scaling of poisoning with number of poisoned rows** and the degradation threshold.
3. **Constrained vs unconstrained** context poisoning — unmeasured.
4. **Transfer of poisoned contexts** across TFMs and to refit baselines.
5. **Backdoors in TFM contexts** — trigger feature-pattern in the query flips predictions.
6. **Context poisoning in TabPFN-TS as label poisoning** — 2505.19397 does this implicitly,
   only as a test-time TS attack.

---

## Coverage notes

- The LIACS BSc thesis (Scholten, 2025) is omitted per prior instruction.
- Very recent (last few weeks) submissions may not be indexed yet.
- Entries marked **[abstract only]** / **[not verified]** need a full-text pass before citing.
- Model/version families referenced: TabPFN v1/v2/v2.5/v3, TabICL/v2, Mitra, LimiX, TabDPT,
  TabFM (Google); TabPFN-TS, Chronos, TimesFM, Moirai, TimeMoE, UniTS for time series.
