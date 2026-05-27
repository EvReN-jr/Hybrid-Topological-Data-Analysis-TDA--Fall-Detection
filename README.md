# Hybrid Topological Data Analysis for Wearable Fall Detection

**M.Sc. Thesis — Istanbul Technical University, Graduate School (Mathematical Engineering)**

- **Title (EN):** Topological Data Analysis of Wearable Sensor Signals: Development and Evaluation of Recall-Oriented Subject-General and Subject-Personalized Models
- **Title (TR):** Giyilebilir Sensör Sinyalleriyle Topolojik Veri Analizi: Düşme Vakalarını Kaçırmamayı Önceleyen Genel ve Kişisel Modellerin Geliştirilmesi ve Değerlendirilmesi
- **Author:** Kenan Evren Boyabatlı
- **Advisor:** Prof. Dr. Atabey Kaygun
- **Date:** June 2026

---

## Overview

This project detects human falls from short windows of wearable inertial (accelerometer/gyroscope)
signals using **persistent homology**. Each window is mapped to a phase-space point cloud via a
**delay embedding** (Takens), summarised with persistent homology in dimensions 0 and 1, and
reduced to a compact **24-dimensional feature vector** (10 H0 statistics + 10 H1 statistics + 4
signal-level descriptors). Simple classifiers (logistic regression, SVM) are trained on these
features, with the topological hyperparameters selected by **Bayesian optimisation (TPE / Optuna)**.

Because a missed fall is far costlier than a false alarm, every model is built **recall-oriented**:
it is tuned and thresholded with the recall-weighted **F₂** score. The study evaluates two modelling
regimes and compares them:

- **Subject-General (LOSO):** one model trained across subjects, applied to an unseen subject.
- **Subject-Personalized:** a separate model trained per subject (plus a cheap, retraining-free
  per-subject threshold personalisation).

### Headline results (General / LOSO protocol, mean F₂)

| Dataset    | LR     | SVM    |
|------------|--------|--------|
| MobiFall   | 0.9049 | 0.8918 |
| SisFall    | 0.9709 | 0.9700 |
| FAD\_40Hz  | 0.9429 | 0.9592 |

Recall stays above 0.95 across all dataset–classifier combinations. A feature-block ablation
(see `results/ablation/`) shows that the discriminative topological signal is concentrated in
**H0**, that the signal-vs-topology balance is dataset-dependent, that the **full** vector is always
best, and that the block ordering is **classifier-invariant**.

---

## Repository structure

```
.
├── thesis/              LaTeX source + figures + compiled tez.pdf
├── code/
│   ├── pipeline/        Main TDA + Optuna + LOSO/Personal protocol (uhem_big_optuna_v13.py),
│   │                    feature-block ablation (ablation_run.py), SLURM job script
│   ├── data_preparation/ Dataset parsing & merging into ML-ready stores
│   ├── feature_extraction/ TDA feature pipelines (V1/V2/V3) + raw signal statistics
│   ├── visualization/   Figure-generation scripts
│   └── archive/         Earlier / superseded experiment scripts (kept for provenance)
├── results/
│   ├── Results_V15.db   Consolidated results: 24-dim feature matrices (at optimal λ) +
│   │                    subject labels, 200 Optuna trials/dataset, per-subject metrics
│   ├── study_v15.db     Optuna study database
│   ├── Results_V15_Report.txt
│   ├── ablation/        Feature-block ablation outputs (LR & SVM, JSON + logs)
│   ├── archive/         Earlier result/study databases
│   └── cluster_logs/    UHeM SLURM run logs
├── notes/               Project context, advisor feedback, session log, status doc
└── docs/                Dataset sources, DB schema, thesis proposal
```

### `Results_V15.db` feature layout

Each `features_<dataset>_server` table holds the 24-dim vector at the optimal configuration:
`feat_0..9` = H0 persistence statistics, `feat_10..19` = H1 persistence statistics,
`feat_20..23` = raw-signal descriptors (max, std, mean, range), plus `label` and `subject`.

---

## Datasets (not included — public)

The raw datasets and the derived ML-ready stores are **not committed** (≈25 GB). They are public:

| Dataset  | Sensor placement | Source |
|----------|------------------|--------|
| MobiFall v2.0 | Trouser pocket (phone) | Kaggle: `kmknation/mobifall-dataset-v20` |
| SisFall  | Waist | Sucerquia et al., *Sensors* 2017 |
| FallAllD (FAD\_40Hz derived) | Waist | Kaggle: `sankalpsinghvishen/derived-fallalld-dataset`; Saleh et al., *IEEE Sensors J.* 2021 |

Full provenance: `docs/Data_Sources.txt`.

### Large files excluded from the repository

These are reproducible from the code + raw datasets and exceed GitHub's 100 MB file limit:

| File / type | Size | Notes |
|-------------|------|-------|
| `DataBases.rar` | 5.8 GB | Archive of all ML-ready databases |
| `Data/archive_pckl/FallAllD.json` | 4.0 GB | Raw FallAllD dump |
| `DataFirstTouch/*_ML_Ready*.db` | 0.1–4.7 GB | Per-dataset ML-ready SQLite stores |
| `TDA_Features_Extraction_V*/*.csv` | up to 433 MB | Extracted feature matrices |
| `RAW_Stats/*_Raw_Stats.csv` | up to 109 MB | Raw signal statistics |
| `Failed_1/` | 1.4 GB | Earlier experiment outputs (scripts kept in `code/archive/`) |
| Python `venv/` | — | Virtual environment |

---

## Reproducing the results

1. **Get the data:** download MobiFall, SisFall, FallAllD (see `docs/Data_Sources.txt`).
2. **Prepare:** `code/data_preparation/` — parse and merge each dataset into ML-ready stores.
3. **Extract features:** `code/feature_extraction/all_pipeline_V3.py` (+ `raw_stats.py`).
4. **Train & evaluate:** `code/pipeline/uhem_big_optuna_v13.py` — Bayesian search over the
   topological hyperparameters, then the Naive / General (LOSO) / Personal protocols with the
   F₂-driven threshold sweep. Produces `Results_V15.db`.
5. **Ablation:** `python code/pipeline/ablation_run.py LogReg` and `... SVM` (reads
   `Results_V15.db`; reproduces the thesis headline F₂ and the feature-block ablation).
6. **Thesis:** in `thesis/`, run `pdflatex tez.tex && biber tez && pdflatex tez.tex && pdflatex tez.tex`.

### Requirements

- **Python:** numpy, pandas, scikit-learn, gudhi, optuna, scipy
- **LaTeX:** a TeX Live distribution with `biber` (APA `biblatex`)
- Heavy feature extraction was run on the UHeM SLURM cluster (allocation 4025462026); see
  `code/pipeline/run_tez.slurm`.
