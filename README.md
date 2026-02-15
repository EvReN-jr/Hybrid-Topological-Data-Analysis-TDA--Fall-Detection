# Hybrid TDA-Based Fall Detection Framework
## F2-Optimized, Subject-Independent and Subject-Specific Evaluation

---

## Abstract

This project presents a hybrid Topological Data Analysis (TDA)-based fall detection framework optimized using the F2-score to prioritize recall. The system integrates delay embedding, landmark subsampling, persistent homology (Alpha, Rips, and Sparse Rips complexes), and statistical features into a unified 24-dimensional representation. Hyperparameters are optimized using Optuna, and evaluation is conducted using both subject-independent (Leave-One-Subject-Out) and subject-specific validation strategies. The framework is evaluated on three benchmark datasets: MobiFall, SisFall, and FAD_40Hz.

---

# 1. Introduction

Fall detection remains a critical problem in wearable sensing systems, particularly in healthcare monitoring for elderly populations. Traditional machine learning approaches rely on handcrafted statistical features, which may fail to capture the intrinsic geometric structure of motion signals.

Topological Data Analysis (TDA) offers a powerful alternative by modeling the geometric and topological properties of time-series signals. This study proposes a hybrid TDA-based fall detection pipeline optimized for safety-oriented performance using the F2-score, where recall is emphasized over precision.

---

# 2. Datasets

The framework is evaluated on:

- MobiFall  
  https://www.kaggle.com/datasets/kmknation/mobifall-dataset-v20/data  

- SisFall  

- FAD_40Hz  
  https://www.kaggle.com/datasets/sankalpsinghvishen/derived-fallalld-dataset/data  

All datasets are stored in SQLite format and processed under a unified protocol.

---

# 3. Methodology

## 3.1 Signal Preprocessing

### Resampling

All signals are resampled to:

TARGET_FS = 50 Hz

This ensures consistent temporal resolution across datasets.

### Signal Magnitude Vector (SMV)

For each tri-axial sensor group:

SMV = sqrt(x² + y² + z²)

This reduces orientation dependency and enhances fall peak localization.

### Window Extraction

win_sec ∈ [1.0, 5.0] (step = 0.5)

The window is centered around fall peaks for fall trials and extracted sequentially for ADL trials.

---

# 4. Topological Feature Extraction

## 4.1 Delay Embedding

If enabled, delay embedding reconstructs the phase space.

dim ∈ [2, 5]  
delay ∈ [1, 5]  
stride_factor ∈ {1, 2, 4}

Purpose:
- Transform 1D signal into a geometric manifold
- Reveal nonlinear temporal structure

## 4.2 Landmark Selection

LIMIT_POINTS = 150  
sampling_method = maxmin

MaxMin sampling selects geometrically diverse representative points.

## 4.3 Simplicial Complex Construction

The following complexes are considered:

- Alpha Complex
- Rips Complex
- Sparse Rips Complex

For Rips-based methods:

metric ∈ {euclidean, cosine, manhattan, chebyshev}  
eps_percentile ∈ {20, 40, 60, 80}

The percentile-based epsilon ensures adaptive scale selection.

## 4.4 Extracted Features

Persistent homology is computed for:

H0 (connected components)  
H1 (loops)

For each dimension, the following statistics are extracted:

- Lifetime count ratio
- Persistence entropy
- Maximum lifetime
- Mean lifetime
- Standard deviation
- Median
- 25th and 75th percentiles
- Quadratic and cubic weighted lifetime sums

TDA features = 20  
Acceleration statistics = 4  

Total features = 24

---

# 5. Optimization Strategy

Hyperparameter search is conducted using Optuna.

N_TRIALS = 200  
Objective = F2-score (β = 2)

F2 = (5 × Precision × Recall) / (4 × Precision + Recall)

This prioritizes recall, which is critical in fall detection where false negatives are more harmful than false positives.

---

# 6. Validation Protocol

## 6.1 Subject-Independent Evaluation

Leave-One-Subject-Out (LOSO):

Train on N−1 subjects  
Test on unseen subject  

Ensures generalization capability.

## 6.2 Subject-Specific Evaluation

Training = 60%  
Testing  = 40%

Demonstrates personalized model performance.

## 6.3 Threshold Optimization

Decision thresholds are selected using Precision-Recall curves to maximize F2-score instead of using the default 0.5 probability.

---

# 7. Optimal Hyperparameters per Dataset

## MobiFall

Best F2 = 0.9749

win_sec = 2.5  
complex_type = SparseRips  
dim = 3  
delay = 2  
stride_factor = 1  
metric = euclidean  
eps_percentile = 40  

## SisFall

Best F2 = 0.9726

win_sec = 1.0  
complex_type = Alpha  
dim = 5  
delay = 5  
stride_factor = 4  

## FAD_40Hz

Best F2 = 0.9855

win_sec = 1.5  
complex_type = SparseRips  
dim = 4  
delay = 4  
stride_factor = 4  
metric = cosine  
eps_percentile = 20  

---

# 8. Original Experimental Results

## MobiFall (General Model – Selected Subjects)

Logistic Regression (F2):

Subject 2  → 0.8209  
Subject 4  → 0.8219  
Subject 10 → 0.8462  
Subject 11 → 0.8571  

SVM (Best Cases – F2):

Subject 8  → 0.8955  
Subject 10 → 0.8955  
Subject 11 → 0.8955  

## SisFall

F2 > 0.98

Subject 1  → 0.9973  
Subject 17 → 0.9973  
Subject 20 → 0.9918  

## FAD_40Hz

Subject 3  → 0.9904  
Subject 8  → 0.9589  
Subject 12 → 0.9711  

---

# 9. Key Findings

- Sparse Rips frequently outperformed Alpha in higher-dimensional embeddings.
- Shorter windows were optimal for SisFall.
- Cosine distance improved robustness in FAD_40Hz.
- F2-based threshold optimization significantly increased recall stability.
- Subject-independent validation is essential for real-world deployment.

---
