"""The tuned LightGBM model, and the stand-in for the Azure AutoML ensemble it is compared with."""

from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

FIXED_PARAMS = {"objective": "binary", "subsample_freq": 1, "verbose": -1, "random_state": 42}


def lgbm(params):
    """LightGBM with Optuna's settings on top of the fixed ones."""
    return LGBMClassifier(**{**FIXED_PARAMS, **params})


def stand_in_ensemble():
    """Soft-voting LightGBM + XGBoost + logistic regression, like the AutoML VotingEnsemble champion.

    The AutoML model stays in the Azure ML workspace, so this is what the README's out-of-sample
    numbers were measured on.
    """
    return VotingClassifier(voting="soft", estimators=[
        ("lgbm", LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=42)),
        ("xgb", XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, eval_metric="aucpr", random_state=42)),
        ("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
    ])
