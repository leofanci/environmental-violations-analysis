# Predicting and Analyzing Environmental Violations Using Machine Learning

## Overview
This project cleans, analyzes, and models environmental inspection data to **predict the likelihood of air pollution violations** at facilities. It covers the full workflow—from **data preparation** and **EDA** to **machine learning** and **causal inference**—to support environmental compliance and policy decisions.

---

## Workflow

1. **Data Cleaning**
   - Merge **`unclean.csv`** and **`outcome.csv`** on (`registry_id`, `year`).
   - Standardize `state` values (trim, uppercase, normalize variations).
   - Remove duplicates.
   - Identify and handle implausible values.
   - Save cleaned dataset to **`output/cleaned.csv`**.

2. **EDA**
   - Generate descriptive statistics by state.
   - Produce correlation heatmaps, boxplots, and pairplots.
   - Consider outlier detection (Isolation Forest) and document handling decisions.

3. **Modeling**
   - Split data (80% train / 20% test).
   - Build pipelines with imputers and encoders.
   - **Models:**
     - Logistic Regression with L1/L2 regularization.
     - Random Forest with tuned tree count and feature settings.
     - XGBoost with tuned estimators and learning rate.
   - Tune hyperparameters via **GridSearchCV**.
   - Evaluate metrics such as **Accuracy** (optionally **ROC‑AUC**, **Precision**, **Recall**).

4. **Predictions**
   - Load **`predict.csv`**.
   - Apply the best-performing model for operational use.
   - Save results to **`output/predictions.csv`**.

5. **Causal Analysis**
   - Design a randomized controlled trial.
   - Estimate treatment effects with and without county clustering.
   - Communicate effect sizes and uncertainty in plain language.

---

## Handling Missing Data & Outliers
- **Outcome (`violation`)**: drop rows with missing outcomes for supervised training.
- **Control variables**: impute where appropriate in model pipelines.
- **Outliers**: detect conservatively; keep if they represent genuine variation.

---

## Outputs
- **`output/cleaned.csv`** — cleaned and merged dataset.
- **`output/predictions.csv`** — predicted violations for new inspection data.
- **`output/SWEETVIZ_REPORT.html`** — automated EDA report.
- **Plots** — correlation heatmap, boxplots, pairplots, model tuning curves.

---

## Technology Stack
- **Python**
- **Libraries**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `statsmodels`, `scikit-learn`, `xgboost`, `sweetviz`

---

## Recommendations
- For **operational targeting**, Random Forest may be preferred for better calibration with observed violation rates.
- Experimental evidence supports adopting the **air‑scrubber technology**, but a **cost‑benefit analysis** is recommended to assess long‑term feasibility and operational impact.
