# Jane Street • Real‑Time Market Data Forecasting  
_Kaggle Code Competition  (Oct‑2024 → Jul‑2025)_

Forecasting **`responder_6`**—an anonymised market‑micro‑structure return—using 79 real‑world features streamed from Jane Street’s production trading flow.  
The task reflects genuine quant‑trading hurdles: fat‑tailed returns, non‑stationary signals, and frequent regime shifts.

---

## 1 Competition Snapshot
| Item | Detail |
|------|--------|
| **Dataset** | 79 anonymised features · 9 responders · ≈ 4.5 M rows / phase |
| **Objective** | Predict `responder_6` one step ahead in real time |
| **Metric** | Sample‑weighted, zero‑mean 𝐑² |
| **Rules** | Kaggle notebook submissions ≤ 8 h (training) / 9 h (forecasting) runtime; external public data permitted |

---

## 2 Modelling Approach

| Layer | Choices & Rationale |
|-------|---------------------|
| **Feature Engineering** | Minimal. Used raw 79 features; optional one‑day lags for all responders. Extensive engineering risked amplifying concept drift and exceeding memory limits. |
| **Validation Scheme** | Chronological split—last 100 days held out—to mimic leaderboard evaluation faithfully. |
| **Custom Metric** | Implemented the official weighted‑R² inside XGBoost and PyTorch training loops for early stopping and model selection. |
| **Experiment Tracking** | MLflow logs hyper‑parameters, metrics, artefacts (feature importances, residual plots, models). |

### 2.1 Gradient‑Boosted Trees (XGBoost)
* Histogram optimiser (`tree_method="hist"`).  
* Key hyper‑parameters (Optuna‑tuned):  

  | `max_depth` | `learning_rate` | `subsample` | `colsample_bytree` | `gamma` |
  |-------------|-----------------|-------------|--------------------|---------|
  | 4 | 0.10 | 0.71 | 0.73 | 0.26 |

* Trained for 75 rounds with early stopping on validation weighted‑R².

### 2.2 Two‑Layer LSTM
* **Input** : standard‑scaled features (+ optional lag columns).  
* **Architecture** : `[LSTM (32 hidden) × 2 → FC 1]`, dropout 0.4, seq‑length 1.  
* **Loss** : sample‑weighted MSE.  
* **Optimiser** : Adam (lr 5 × 10⁻⁵, weight_decay 1 × 10⁻³) with early stopping on training weighted‑R².

---

## 3 Submission Scores

| Notebook / Version | Model | Public LB | Private LB |
|--------------------|-------|-----------|------------|
| **Jane Street RTDF GBDT – Submission NB 3 v1** | XGBoost (75 trees) | **0.006171** | 0.006171 |
| Best Model LSTM NN v1 | Two‑Layer LSTM | 0.004329 | 0.004329 |
| JS LSTM 1.0 with Lags v2 | Two‑Layer LSTM (+ lag inputs) | –0.000567 | –0.000567 |
| Jane Street RTDF GBDT – Submission NB 2 v1 | XGBoost (early hyper‑param set) | 0.005870 | 0.005870 |
| Baseline `submission.parquet` | Simple tree baseline | 0.002646 | 0.002646 |
| Jane Street RTDF Submission by A.F. v2 | First public attempt | 0.001192 | 0.001192 |

> *Scores shown are those displayed on the public and private leaderboards immediately after each submission.*

---

## 4 Key Learnings
* **Trees outperform RNNs** when sequence context is short and feature interactions dominate.  
* **Temporal CV matters**: random folds overstated performance by >30 %.  
* **Custom metrics inside training loops** reduce leakage between offline validation and Kaggle evaluation.  
* **Memory is scarce** in notebook environments; efficient parquet partition loading (Polars) and on‑the‑fly down‑casting are essential.  

---

## 5 Next Directions
1. Blend XGBoost and LSTM predictions for regime diversification.  
2. Introduce rolling‑window hyper‑parameter tuning to adapt to drift.  
3. Explore symbol embeddings to capture cross‑sectional relationships.  
4. Prune low‑impact features (via SHAP) to streamline inference.

---

## 6 Acknowledgements
Many thanks to **Jane Street** for releasing a realistic dataset and to the Kaggle community for vibrant discussion threads.

*Last updated : 20 Jul 2025*
