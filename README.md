# Separability Metric: A Data Quality Screening Framework

This repository contains the public implementation of the **Tao Index**, our proposed separability-based data quality metric, as presented in our paper accepted by the *Journal of Intelligent Manufacturing*.  

The Tao Index integrates **local/topological-style features** (distance, density, intrinsic dimension, label smoothness) with **global divergence metrics** (MMD, KS distance) to provide a realistic assessment of dataset separability.

---

## Features
- **Tao Index Implementation**  
  Includes all five components (local and global) with a trained regressor for separability scoring.
- **Baseline Indices**  
  Implements common unsupervised clustering metrics (Dunn, Silhouette, Davies–Bouldin) with normalization for comparability.
- **Benchmark Experiments**  
  Reproducible pipelines for:
  - **SECOM (binary classification)**  
  - **SPF (Steel Plate Faults, multi-class)**  

> ⚠️ Proprietary datasets (Cevher, Nissan, Cold Roll) and related code are **not included** for confidentiality reasons.

---

## Repository Structure
Separability_metric/
├── src/tao_index/ # Core metric implementation
├── experiments/ # Benchmark experiment scripts
│ ├── secom_binary_eval.py
│ ├── spf_multiclass_eval.py
├── Plots/ # Saved figures from experiments
├── requirements.txt # Dependencies
└── README.md

2. Running Experiments

Run benchmark experiments directly from the experiments/ folder:

SECOM (binary classification):

python experiments/secom_binary_eval.py


SPF (Steel Plate Faults, multi-class classification):

python experiments/spf_multiclass_eval.py


Each script will:

Load the dataset

Train a classifier with predefined “best” parameters

Compute ROC-AUC, PR-AUC, F1 score, and Accuracy

Output results in the console

3. Using Tao Index Components

You can import the metric directly for custom datasets:

from src.tao_index.components import compute_tao_index

score = compute_tao_index(X, y)  # X: features, y: labels
print("Separability Score:", score)

Results
Benchmark Separability Scores

Benchmark Metrics Comparison

These figures summarize the Tao Index results alongside baseline metrics for the SECOM and SPF benchmarks.
The Tao Index consistently aligns with the estimated ground truth of dataset separability, outperforming other indices in robustness and interpretability.

<img src="Plots/separability_scores.pdf" alt="Separability Scores" width="600"/>
<img src="Plots/mse_performance.pdf" alt="Performance" width="600"/>
