import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "databricks" / "src"))

from loan_lib.evaluate import best_f2_cutoff, metrics, paired_bootstrap  # noqa: E402
from loan_lib.features import INPUTS, engineer_features  # noqa: E402
from loan_lib.models import lgbm  # noqa: E402
from loan_lib.scoring import mean_abs_contributions, score  # noqa: E402
from loan_lib.tuning import tune  # noqa: E402


@pytest.fixture(scope="module")
def applicants():
    """Synthetic applicants shaped like the challenge data, with default driven by score and income."""
    rng = np.random.default_rng(0)
    n = 1500
    df = pd.DataFrame({
        "id": np.arange(n), "date_of_birth": "1990-01-01",
        "number_dependants": rng.integers(-1, 4, n), "credit_utilization": rng.uniform(0, 1, n),
        "debt_to_income_ratio": rng.uniform(0, 1, n), "monthly_income": rng.integers(1000, 9000, n),
        "number_open_credit_lines": rng.integers(0, 12, n), "number_open_loans": rng.integers(0, 4, n),
        "number_90_days_past_due": rng.integers(0, 3, n), "number_charged_off": rng.integers(0, 2, n),
        "score1": rng.integers(450, 850, n), "score2": rng.integers(450, 850, n),
    })
    risk = -0.012 * (df["score1"] - 650) - 0.0004 * (df["monthly_income"] - 5000) + rng.normal(0, 1, n)
    df["target"] = (risk > np.quantile(risk, 0.88)).astype(int)
    return df


def test_features_leave_out_age_and_fix_dependants(applicants):
    features = engineer_features(applicants)
    assert "date_of_birth" not in features and "id" not in features and "target" not in features
    assert features["number_dependants"].min() == 0
    assert (features["total_open_accounts"] == applicants["number_open_credit_lines"] + applicants["number_open_loans"]).all()
    assert list(features.columns[:len(INPUTS)]) == INPUTS


def test_f2_cutoff_sits_at_or_below_the_f1_cutoff():
    from sklearn.metrics import f1_score

    from loan_lib.evaluate import CUTOFFS

    y = np.array([0] * 90 + [1] * 10)
    proba = np.r_[np.linspace(0, 0.3, 90), np.linspace(0.2, 0.5, 10)]
    f1_cutoff = CUTOFFS[int(np.argmax([f1_score(y, proba >= c) for c in CUTOFFS]))]
    cutoff = best_f2_cutoff(y, proba)
    assert cutoff <= f1_cutoff
    assert metrics(y, proba, cutoff)["recall"] >= metrics(y, proba, f1_cutoff)["recall"]


def test_bootstrap_of_a_model_against_itself_is_centred_on_zero(applicants):
    y = applicants["target"].to_numpy()
    proba = np.random.default_rng(1).uniform(size=len(y))
    result = paired_bootstrap(y, proba, proba, n=200)
    assert result["difference"] == 0 and result["ci_low"] == 0 and result["ci_high"] == 0


def test_bootstrap_detects_a_clearly_better_model(applicants):
    y = applicants["target"].to_numpy()
    rng = np.random.default_rng(2)
    good, noise = y + rng.normal(0, 0.5, len(y)), rng.uniform(size=len(y))
    assert paired_bootstrap(y, good, noise, n=200)["ci_low"] > 0


def test_tuning_finds_settings_that_score(applicants):
    X, y = engineer_features(applicants), applicants["target"]
    study = tune(X, y, n_trials=4)
    assert 0.5 < study.best_value <= 1
    model = lgbm(study.best_params).fit(X, y)
    scored = score(model, 0.2, applicants.drop(columns=["target"]))
    assert scored.index.equals(applicants.index) and scored["probability"].between(0, 1).all()
    assert set(mean_abs_contributions(model, applicants).index) == set(model.feature_name_)
