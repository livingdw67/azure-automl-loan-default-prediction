"""Raw applicant rows in, default probability and review flag out."""

import pandas as pd

from loan_lib.features import engineer_features


def score(model, cutoff, applicants):
    features = engineer_features(applicants)[list(model.feature_name_)]
    probability = model.predict_proba(features)[:, 1]
    return pd.DataFrame({"probability": probability, "flagged": probability >= cutoff}, index=applicants.index)


def mean_abs_contributions(model, applicants):
    """Mean absolute SHAP value per feature (LightGBM's built-in TreeSHAP), on the log-odds scale."""
    features = engineer_features(applicants)[list(model.feature_name_)]
    contributions = model.predict(features, pred_contrib=True)[:, :-1]  # last column is the base value
    return (pd.Series(abs(contributions).mean(axis=0), index=features.columns, name="mean_abs_shap")
            .sort_values(ascending=False))
