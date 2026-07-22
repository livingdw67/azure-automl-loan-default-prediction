# Financial Risk Modeling: Loan Default Prediction

**Author:** Daniel W. Livingston, Senior Data Scientist

## Overview
This repository contains an end-to-end machine learning pipeline designed to predict financial loan defaults using **Azure Machine Learning SDK v2** and **Azure AutoML**. The project emphasizes not only predictive accuracy but also strict regulatory compliance and business-driven decision-making. 

The champion model, an optimized `VotingEnsemble`, was built to identify high-risk accounts prior to charge-off, allowing for early intervention strategies while adhering to fair lending laws.

## Repository Structure
*   `notebooks/`: Contains the core `.ipynb` execution environment used to engineer features, orchestrate the Azure AutoML cloud runs, and evaluate the final model.
*   `presentations/`: Contains the final executive summary (`Final_Presentation.pdf`) detailing model selection, business thresholds, and macro-drivers of risk.
*   `images/`: Stores the SHAP output, Precision/Recall curves, and Azure ML Studio UI screenshots referenced in the presentation.

## Key Technical Highlights

### 1. Regulatory Compliance (ECOA) & Feature Engineering
To ensure algorithmic fairness and strict compliance with the **Equal Credit Opportunity Act (ECOA)**, age and date-of-birth attributes were programmatically scrubbed from the dataset during the feature engineering phase. Additional transformations included:
*   Consolidating active financial obligations (`total_open_accounts`).
*   Imputing and casting non-numeric artifacts (e.g., handling negative dependent anomalies and coercing string artifacts to integers).

### 2. Azure AutoML Orchestration
The pipeline bypasses local memory constraints by pointing compute clusters directly to Azure Blob Storage via `MLTable`. 
*   **Metric:** Optimized for `auc_weighted` via 10-Fold Cross-Validation to handle a ~90/10 target class imbalance.
*   **Guardrails:** Cloud compute costs were strictly capped using a 60-minute timeout, a 15-trial limit, and maximum concurrent run constraints.
*   **Champion Model:** A `VotingEnsemble` outperforming standalone XGBoost and LightGBM models, registered to the MLflow workspace as `Purpose_Champion`.

### 3. Business-Driven Threshold Optimization
Evaluating Precision and Recall in isolation is insufficient for real-world financial risk. To align the model with business realities—where the financial loss of a missed charge-off (false negative) outweighs the friction of a proactive account review (false positive)—the probability threshold was optimized.
*   **Operational Cutoff:** Set to **0.15**, aligning with the absolute peak of the F2 curve to prioritize Recall. 
*   **Impact:** Successfully captures 75% to 80% of actual defaults before they occur.

### 4. Global Interpretability
To bypass the "black-box" nature of ensemble pipelines, **SHAP (SHapley Additive exPlanations)** was applied via a `KernelExplainer` wrapper to extract the portfolio's macro-drivers of risk. Lower credit scores and higher credit utilization were successfully validated as the dominant upward drivers of default risk.

## Technologies Used
*   **Cloud & MLOps:** Azure Machine Learning SDK v2, Azure AutoML, MLflow
*   **Core Data Stack:** Python, Pandas, NumPy, Scikit-Learn
*   **Interpretability & Visualization:** SHAP, Matplotlib, Seaborn
