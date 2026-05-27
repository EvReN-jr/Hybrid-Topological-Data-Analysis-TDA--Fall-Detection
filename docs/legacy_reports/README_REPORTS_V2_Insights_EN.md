# V15 EXPERIMENT RESULTS: FINDINGS & INSIGHTS
---

## Table of Contents
1. [Overview](#overview)
2. [Dataset Comparison](#dataset-comparison)
3. [Model Performance Analysis](#model-performance-analysis)
4. [Threshold Optimization Analysis](#threshold-optimization-analysis)
5. [Personal vs General Model](#personal-vs-general-model)
6. [Naive Baseline Analysis](#naive-baseline-analysis)
7. [TDA Parameter Analysis](#tda-parameter-analysis)
8. [Subject-Level Observations](#subject-level-observations)
9. [Summary & Recommendations](#summary--recommendations)

---

## Overview

V15 is a comprehensive experiment series conducted across three fall detection datasets using TDA (Topological Data Analysis)-based feature extraction and Optuna-driven hyperparameter optimization. Strong results were achieved on all datasets through a three-layer evaluation protocol (Naive → General → Personal) combined with per-subject threshold sweep analysis.

| Dataset | Best F2 (Optuna) | General G-F2 (LogReg) | General G-F2 (SVM) | Personal P-F2 (LogReg) | Personal P-F2 (SVM) |
|---------|-----------------|----------------------|--------------------|------------------------|---------------------|
| MobiFall | 0.9746 | 0.9049±0.0393 | 0.8918±0.0327 | 0.7325±0.0295 | 0.5981±0.0479 |
| SisFall | 0.9733 | 0.9709±0.0108 | 0.9700±0.0111 | 0.9405±0.0080 | 0.9196±0.0082 |
| FAD_40Hz | 0.9860 | 0.9429±0.0363 | 0.9592±0.0234 | 0.9168±0.0454 | 0.8724±0.0696 |

> **Reference:** Summary Comparison table — [README_V15_REPORT.md → Summary Comparison](#)

---

## Dataset Comparison

### 1. Best Optuna Performance: FAD_40Hz

FAD_40Hz achieved the highest F2 score (0.9860) during Optuna optimization, reflecting the advantages of a large sample size (9481 × 24) and relatively low-noise sensor data. The SVM model delivered the most stable result across all configurations at the General layer, with G-F2 = 0.9592±0.0234.

| Criterion | MobiFall | SisFall | FAD_40Hz |
|-----------|----------|---------|----------|
| Sample count | 1808 | 7738 | 9481 |
| Subject count | 9 | 24 | 13 |
| Best Optuna F2 | 0.9746 | 0.9733 | **0.9860** |
| General LogReg G-F2 | 0.9049 | **0.9709** | 0.9429 |
| General SVM G-F2 | 0.8918 | 0.9700 | **0.9592** |
| Personal LogReg P-F2 | 0.7325 | **0.9405** | 0.9168 |
| Personal SVM P-F2 | 0.5981 | 0.9196 | 0.8724 |

**Insight:** Although FAD_40Hz yields the highest Optuna F2, when the General and Personal layers are considered together, **SisFall is the most consistent and well-rounded dataset**. This points to a more balanced class distribution and a larger subject pool in SisFall.

---

### 2. MobiFall: Severe Drop in Personalized Performance

MobiFall shows a substantial performance degradation at the Personal layer compared to General. SVM in particular yields a very low P-F2 = 0.5981.

| Model | G-F2 (General) | P-F2 (Personal) | Difference |
|-------|---------------|-----------------|------------|
| LogReg | 0.9049±0.0393 | 0.7325±0.0295 | **−0.1724** |
| SVM | 0.8918±0.0327 | 0.5981±0.0479 | **−0.2937** |

**Insight:** MobiFall has relatively few samples per subject (~200 on average), which means the narrowed grid used for Personal model training is insufficient. The Personal model provides no benefit — and actively hurts — on this dataset.

> **Reference:** [README_V15_REPORT.md → Dataset 1: MobiFall → Personal Model Performance Summary](#)

---

## Model Performance Analysis

### LogReg vs SVM: Overall Comparison

| Dataset | LogReg G-F2 | SVM G-F2 | Difference | Superior Model |
|---------|------------|---------|------------|----------------|
| MobiFall | 0.9049±0.0393 | 0.8918±0.0327 | +0.0131 | **LogReg** |
| SisFall | 0.9709±0.0108 | 0.9700±0.0111 | +0.0009 | ≈ Tied |
| FAD_40Hz | 0.9429±0.0363 | 0.9592±0.0234 | −0.0163 | **SVM** |

**Insight:** Neither model consistently outperforms the other across all datasets. However, SVM's lower variance on FAD_40Hz (±0.0234 vs ±0.0363) suggests it is more reliable on larger datasets.

### Precision vs Recall Balance

G-REC is systematically higher than G-PRE across all datasets, which is expected given the F2 score's emphasis on recall.

| Dataset | Model | G-REC | G-PRE (approx.) | Difference (REC−PRE) |
|---------|-------|-------|-----------------|----------------------|
| MobiFall | LogReg | 0.9574 | ~0.73 | ~+0.23 |
| MobiFall | SVM | 0.9543 | ~0.72 | ~+0.23 |
| SisFall | LogReg | 0.9836 | ~0.93 | ~+0.05 |
| SisFall | SVM | 0.9742 | ~0.95 | ~−0.02 |
| FAD_40Hz | LogReg | 0.9929 | ~0.83 | ~+0.16 |
| FAD_40Hz | SVM | 0.9878 | ~0.89 | ~+0.10 |

**Insight:** Precision-recall balance is much better on SisFall and FAD_40Hz. MobiFall continues to exhibit low precision, indicating a persistently high false positive rate.

> **Reference:** [README_V15_REPORT.md → General Model Performance Summary](#)

---

## Threshold Optimization Analysis

One of V15's key contributions is per-subject threshold sweep analysis. The table below summarizes mean F2 gains:

| Dataset | Model | Mean ΔF2 | Max ΔF2 | Max ΔF2 Subject |
|---------|-------|----------|---------|-----------------|
| MobiFall | LogReg | +0.0348 | +0.0615 | Sub 8 |
| MobiFall | SVM | **+0.0474** | +0.1013 | Sub 9 |
| SisFall | LogReg | +0.0079 | +0.0527 | Sub 29 |
| SisFall | SVM | +0.0042 | +0.0169 | Sub 29 |
| FAD_40Hz | LogReg | +0.0250 | **+0.1165** | Sub 11 |
| FAD_40Hz | SVM | +0.0121 | +0.0691 | Sub 11 |

> **Reference:** [README_V15_REPORT.md → Threshold Sweep Summary tables](#)

**Insight 1 — High gains on MobiFall:** Threshold optimization yields significantly larger F2 improvements on MobiFall than on other datasets (mean +0.0474 for SVM). This indicates that the default thresholds for MobiFall (ranging widely from 0.37 to 0.77) are suboptimal and highly subject-dependent.

**Insight 2 — Minimal gains on SisFall:** Threshold optimization provides only marginal gains on SisFall (+0.0042–+0.0079). This suggests the default thresholds are already near-optimal for SisFall; model decisions are less sensitive to the threshold value on this dataset.

**Insight 3 — FAD_40Hz Sub 11 anomaly:** Sub 11 on FAD_40Hz shows an extreme gain for LogReg (ΔF2 = +0.1165). The very poor performance at the default threshold (DEF_F2 = 0.8046) suggests this subject's class characteristics differ substantially from the rest of the cohort.

### Optimal Threshold Trends

| Dataset | Model | Dominant Direction | Interpretation |
|---------|-------|--------------------|----------------|
| MobiFall | LogReg | Upward (0.80+) | More selective decision boundary needed |
| MobiFall | SVM | Upward (0.60–0.65) | Moderately elevated threshold |
| SisFall | LogReg | Minor shifts | Default is near-optimal |
| SisFall | SVM | Minor shifts | Default is near-optimal |
| FAD_40Hz | LogReg | Upward (0.65–0.75) | Higher threshold generally better |
| FAD_40Hz | SVM | Mixed shifts | Subject-specific tuning effective |

---

## Personal vs General Model

V15's three-layer evaluation reveals how much benefit the Personal model provides across different data scenarios.

### P-F2 / G-F2 Ratio (Personalization Gain)

| Dataset | Model | G-F2 | P-F2 | Ratio (P/G) | Assessment |
|---------|-------|------|------|-------------|------------|
| MobiFall | LogReg | 0.9049 | 0.7325 | 0.81 | ❌ Personalization hurts |
| MobiFall | SVM | 0.8918 | 0.5981 | 0.67 | ❌ Personalization severely hurts |
| SisFall | LogReg | 0.9709 | 0.9405 | 0.97 | ✅ Acceptable, minor loss |
| SisFall | SVM | 0.9700 | 0.9196 | 0.95 | ✅ Acceptable |
| FAD_40Hz | LogReg | 0.9429 | 0.9168 | 0.97 | ✅ Acceptable, minor loss |
| FAD_40Hz | SVM | 0.9592 | 0.8724 | 0.91 | ⚠️ Notable degradation |

> **Reference:** [README_V15_REPORT.md → Personal & General Model Performance Summary](#)

**Insight:** The personalized model is only beneficial when sufficient per-subject training data is available (SisFall and, to a lesser extent, FAD_40Hz). On low-sample datasets like MobiFall, the Personal model significantly underperforms the General model. This finding highlights the need to define a minimum personal data threshold for any clinical deployment scenario.

---

## Naive Baseline Analysis

The Naive model represents the performance of repeated training with global parameters and no subject-specific grid search.

### Naive F2 vs General G-F2

| Dataset | Model | Naive F2 | General G-F2 | Difference |
|---------|-------|----------|-------------|------------|
| MobiFall | LogReg | 0.8857 | 0.9049 | +0.0192 |
| MobiFall | SVM | 0.8573 | 0.8918 | +0.0345 |
| SisFall | LogReg | 0.9713 | 0.9709 | −0.0004 |
| SisFall | SVM | 0.9715 | 0.9700 | −0.0015 |
| FAD_40Hz | LogReg | 0.9502 | 0.9429 | −0.0073 |
| FAD_40Hz | SVM | 0.9603 | 0.9592 | −0.0011 |

**Insight 1:** For MobiFall, LOGO cross-validation (General) clearly outperforms the Naive baseline. This reflects high subject-level variance in MobiFall; leaving each subject out during evaluation provides a more realistic performance estimate.

**Insight 2:** For SisFall and FAD_40Hz, Naive and General performance are nearly identical. The Naive baseline is a sufficiently reliable performance estimate for production purposes on these two datasets.

---

## TDA Parameter Analysis

Comparing the optimal parameters selected by Optuna for each dataset:

| Parameter | MobiFall | SisFall | FAD_40Hz |
|-----------|----------|---------|----------|
| `win_sec` | **5.0** | 1.0 | 1.0 |
| `complex_type` | SparseRips | Alpha | Rips |
| `dim` | **5** | **5** | 3 |
| `delay` | 2 | 4 | 2 |
| `stride_factor` | 2 | **4** | 1 |
| `metric` | chebyshev | — | manhattan |
| `eps_percentile` | **80** | — | 40 |

> **Reference:** [README_V15_REPORT.md → Optimal TDA Parameters tables](#)

**Insight 1 — win_sec:** MobiFall required the longest window (5.0 s), suggesting that fall patterns in this dataset need a broader temporal context to be captured. SisFall and FAD_40Hz both favor a short 1.0 s window, enabling faster fall detection.

**Insight 2 — Diversity in complex_type:** Each dataset selected a different simplicial complex type (SparseRips / Alpha / Rips), confirming that no single complex type is universally superior across fall detection scenarios.

**Insight 3 — stride_factor:** SisFall uses stride_factor=4, meaning feature extraction is sparse. As the largest dataset (7738 samples), SisFall benefits from sparser sampling for computational efficiency without sacrificing accuracy.

**Insight 4 — dim:** MobiFall and SisFall selected dim=5 (maximum), while FAD_40Hz selected dim=3. Higher-dimensional topological features help discriminate fall events in smaller and noisier datasets (MobiFall), while simpler topology is sufficient for large, higher-quality datasets (FAD_40Hz).

---

## Subject-Level Observations

### Worst-Performing Subjects

The table below lists subjects with the lowest G-F2 scores:

| Dataset | Model | SUB | G-F2 | P-F2 | Likely Cause |
|---------|-------|-----|------|------|--------------|
| MobiFall | LogReg | 8 | 0.8164 | 0.6765 | Low precision (0.5698), high FP rate |
| MobiFall | SVM | 8 | 0.8122 | 0.4749 | Similar issue, SVM performs worse |
| SisFall | LogReg | 29 | 0.8861 | 0.9211 | Low G-REC (0.8702) |
| SisFall | SVM | 29 | 0.8989 | 0.9129 | Low G-REC (0.8827) |
| FAD_40Hz | LogReg | 11 | 0.7516 | 0.7001 | Very low precision (0.3788) |
| FAD_40Hz | SVM | 11 | 0.8417 | 0.5399 | Precision: 0.5195 |

**Insight:** Sub 11 (FAD_40Hz) and Sub 8 (MobiFall) consistently underperform across all models. These subjects likely exhibit atypical movement patterns or recording quality issues. Data inspection and potentially subject-specific modeling approaches are warranted.

### Subjects with the Largest Threshold Sweep Gains

| Dataset | Model | SUB | DEF_F2 | OPT_F2 | ΔF2 |
|---------|-------|-----|--------|--------|-----|
| FAD_40Hz | LogReg | 11 | 0.8046 | 0.9211 | **+0.1165** |
| MobiFall | SVM | 9 | 0.8824 | 0.9836 | +0.1013 |
| MobiFall | SVM | 5 | 0.8955 | 0.9836 | +0.0881 |
| FAD_40Hz | LogReg | 12 | 0.9216 | 0.9958 | +0.0742 |
| FAD_40Hz | SVM | 11 | 0.8642 | 0.9333 | +0.0691 |

**Insight:** Large ΔF2 values indicate that the default threshold (~0.50–0.55) is highly suboptimal for these subjects. Their decision boundaries differ markedly from the rest of the cohort, and subject-specific threshold calibration could yield substantial gains in a production setting.

---

## Summary & Recommendations

### Key Findings

| Finding | Detail | Reference Table |
|---------|--------|-----------------|
| **Best dataset (Optuna)** | FAD_40Hz: F2 = 0.9860 | Dataset Overview |
| **Most consistent dataset** | SisFall: high performance at every evaluation layer | General & Personal Summary |
| **Personal model failure** | MobiFall SVM P-F2 = 0.5981 (67% of G-F2) | Personal Summary |
| **Threshold optimization most impactful** | MobiFall SVM: mean +0.0474 | THR-Sweep Summary |
| **Threshold optimization least impactful** | SisFall SVM: mean +0.0042 | THR-Sweep Summary |
| **Most problematic subject** | FAD_40Hz Sub 11: G-F2 = 0.7516 (LogReg) | Per-Subject Results |
| **Optimal window size** | 1.0 s (SisFall & FAD_40Hz), 5.0 s (MobiFall) | Optimal TDA Parameters |
| **Preferred complex type** | Dataset-dependent: SparseRips / Alpha / Rips | Optimal TDA Parameters |

### Recommendations

1. **Revise the Personal Model for MobiFall:** The Personal model is actively harmful for low-sample subjects. A minimum sample count threshold should be established, or data augmentation techniques should be explored before enabling personal model training on this dataset.

2. **Subject-Specific Threshold Calibration:** Particularly for MobiFall and FAD_40Hz, using an individually optimized threshold per subject yields an average gain of +0.035–0.047 in F2. This should be evaluated as a deployment strategy.

3. **Data Inspection for Sub 11 (FAD_40Hz) and Sub 8 (MobiFall):** The persistently low and inconsistent performance of these subjects points to data quality issues or anomalous movement patterns. Manual inspection of the raw recordings is recommended.

4. **SisFall Parameters as Transfer Candidates:** SisFall demonstrated the most balanced performance across all evaluation layers. Its optimal parameters (Alpha complex, win_sec=1.0, dim=5) are strong candidates as a starting point for future experiment versions.

5. **Prefer SVM for FAD_40Hz:** On FAD_40Hz, SVM delivers both higher G-F2 (0.9592 vs 0.9429) and lower variance compared to LogReg. SVM is recommended as the production model for this dataset.

---

*This document references tables and results from the V15 experiment report ([README_V15_REPORT.md]).*
