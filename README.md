# Financial Risk Modeling: Loan Default Prediction

**Author:** Daniel W. Livingston, Senior Data Scientist

## Overview
This repository contains an end-to-end machine learning pipeline designed to predict financial loan defaults using **Azure Machine Learning SDK v2** and **Azure AutoML**. The project emphasizes not only predictive accuracy but also regulatory compliance and business-driven decision-making.

The champion model, a `VotingEnsemble`, was built to identify high-risk accounts prior to charge-off, allowing for early intervention strategies while adhering to fair lending laws.

## Repository Structure
*   `notebooks/Notebook.ipynb`: Feature engineering, Azure AutoML orchestration, scoring, threshold analysis, and SHAP.
*   `notebooks/validate_locally.py`: Out-of-sample validation of the model and cutoff (runs locally, no Azure needed).
*   `notebooks/make_charts.py`: Draws the charts in `images/` from the challenge's `train.csv`, in light and dark versions.
*   `presentations/Purpose_Data_Challenge.pdf`: The original executive summary as submitted. Its threshold metrics were measured on training data; see [Out-of-Sample Validation](#5-out-of-sample-validation) for corrected figures.
*   `images/`: Feature importances, SHAP values, and the out-of-sample precision/recall threshold curve.

## Key Technical Highlights

### 1. Regulatory Compliance (ECOA) & Feature Engineering
To support compliance with the **Equal Credit Opportunity Act (ECOA)**, age and date of birth were removed from the champion model's features. Additional transformations included:
*   Consolidating active financial obligations (`total_open_accounts`).
*   Correcting a negative-dependents anomaly (`-1` recoded to `0`) and aligning column types to the registered model's schema.

A random-forest baseline on the no-age features ranks the two credit scores first, followed by credit utilization, debt-to-income and monthly income:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="images/Feature_Importance-dark.png">
  <img alt="Random-forest feature importances on the no-age features: Score 2 and Score 1 near 0.19, credit utilization and debt-to-income near 0.14, monthly income 0.12, and the account counts below 0.07" src="images/Feature_Importance.png">
</picture>

### 2. Azure AutoML Orchestration
The pipeline points AutoML directly at Azure Blob Storage via `MLTable`, avoiding local memory limits.
*   **Metric:** Optimized for `auc_weighted` with k-fold cross-validation on a ~90/10 target class imbalance (the notebook's no-age job is configured for 30 folds).
*   **Guardrails:** Compute costs capped with timeouts, trial limits, and concurrency limits (120 minutes and 50 trials for the no-age job; an earlier exploratory job used 5-fold CV, 60 minutes, and 15 trials).
*   **Champion Model:** A `VotingEnsemble` outperforming standalone XGBoost and LightGBM models, registered in MLflow as `Purpose_Challenger` (the no-age variant used for final scoring).

### 3. Business-Driven Threshold Optimization
The financial loss of a missed charge-off (false negative) outweighs the friction of a proactive account review (false positive), so the decision cutoff is chosen to maximize F2, which weighs recall four times as heavily as precision.

The original analysis set the cutoff at **0.15** using a precision/recall curve computed on the training data. Because the challenge's test set has no labels, that curve scored the same rows the model was trained on, which overstates performance. Section 5 re-measures it properly.

### 4. Global Interpretability
SHAP (via a `KernelExplainer` wrapper) was applied to the ensemble to identify the portfolio's main drivers of risk. Lower monthly income and lower credit scores were the dominant upward drivers of default risk; credit utilization and debt-to-income moved predictions far less.

The AutoML model stays in the Azure ML workspace, so the chart below repeats the notebook's SHAP setup (a 100-row background, 300 sampled customers) on the local stand-in ensemble from section 5. Its top five drivers come out in the same order as the notebook's run on the AutoML champion.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="images/SHAP-dark.png">
  <img alt="SHAP beeswarm for 300 customers: low monthly income and low scores push the predicted default probability up by as much as 0.20; credit utilization, debt-to-income and the account counts stay mostly within 0.05 of zero" src="images/SHAP.png">
</picture>

### 5. Out-of-Sample Validation
`notebooks/validate_locally.py` rebuilds the no-age features and a comparable soft-voting ensemble (LightGBM + XGBoost + logistic regression), then measures performance on customers the model has not seen. Default rate: 10.2% of 20,839 customers.

| Evaluation | Cutoff | Defaults caught | Precision | AUC |
|------------|--------|-----------------|-----------|-----|
| Training data (original method) | 0.15 | 79.8% | 32.7% | 0.90 |
| 10-fold cross-validation | 0.15 | 56.5% | 23.3% | 0.76 |
| Held-out 20%, original cutoff | 0.15 | 58.0% | 23.1% | 0.76 |
| **Held-out 20%, cutoff chosen on the other 80%** | **0.10** | **74.6%** | **19.8%** | **0.76** |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="images/Precision_recall_curve-dark.png">
  <img alt="Precision, recall, F1 and F2 against the decision cutoff on 10-fold cross-validated scores: F2 is nearly flat from 0.08 to 0.10; at 0.10, 74% of defaults are caught at 20% precision with 38% of accounts flagged; the original 0.15 cutoff catches 56%" src="images/Precision_recall_curve.png">
</picture>

**Takeaways:**
*   The original "75–80% of defaults caught" figure was a training-data result. At the same 0.15 cutoff, the model catches about 57% of defaults on unseen customers.
*   Choosing the F2 cutoff properly (on separate data) lands at **0.10**, which catches about **75% of defaults on a true holdout**, at the cost of lower precision: roughly 38% of accounts would be flagged for review. Whether that review volume is acceptable is a business decision about the cost of an account review versus a charge-off.
*   Out-of-sample AUC is about **0.76**, a solid but realistic level of separation for application-time credit data.

## Technologies Used
*   **Cloud & MLOps:** Azure Machine Learning SDK v2, Azure AutoML, MLflow
*   **Core Data Stack:** Python, Pandas, NumPy, Scikit-Learn, LightGBM, XGBoost
*   **Interpretability & Visualization:** SHAP, Matplotlib, Seaborn
