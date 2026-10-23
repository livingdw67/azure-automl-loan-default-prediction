"""Optuna search over LightGBM settings, scored by cross-validated AUC on the development data only.

The holdout never enters the search: Optuna sees only the 80% development split, and the holdout
is scored once, after the settings and the cutoff are fixed.
"""

import numpy as np
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from loan_lib.models import lgbm


def search_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 4, 64, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 300, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
    }


def cv_auc(params, X, y, folds=5, seed=42, trial=None):
    """Mean AUC over stratified folds. Given a trial, reports after each fold so Optuna can stop a weak one."""
    scores = []
    for step, (fit, val) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y)):
        model = lgbm(params).fit(X.iloc[fit], y.iloc[fit])
        scores.append(roc_auc_score(y.iloc[val], model.predict_proba(X.iloc[val])[:, 1]))
        if trial is not None:
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return float(np.mean(scores))


def tune(X, y, n_trials=60, timeout=None, seed=42, callbacks=None):
    """TPE search for the best cross-validated AUC. After 8 full trials, a trial whose running AUC is
    below the median of earlier trials at the same fold stops there (median pruning)."""
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
    study.optimize(lambda trial: cv_auc(search_space(trial), X, y, seed=seed, trial=trial),
                   n_trials=n_trials, timeout=timeout, callbacks=callbacks)
    return study
