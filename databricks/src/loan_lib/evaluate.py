"""Holdout metrics, the F2 cutoff, and a paired bootstrap for comparing two models on the same customers."""

import numpy as np
from sklearn.metrics import average_precision_score, fbeta_score, precision_score, recall_score, roc_auc_score

CUTOFFS = np.round(np.arange(0.05, 0.60, 0.01), 2)


def best_f2_cutoff(y, proba):
    """Cutoff with the best F2 (recall weighted four times precision): a missed charge-off costs more than a review."""
    return float(CUTOFFS[int(np.argmax([fbeta_score(y, proba >= c, beta=2) for c in CUTOFFS]))])


def metrics(y, proba, cutoff):
    flagged = proba >= cutoff
    return {
        "auc": roc_auc_score(y, proba), "pr_auc": average_precision_score(y, proba), "cutoff": cutoff,
        "recall": recall_score(y, flagged), "precision": precision_score(y, flagged),
        "f2": fbeta_score(y, flagged, beta=2), "flagged_share": float(np.mean(flagged)),
    }


def paired_bootstrap(y, proba_a, proba_b, metric=roc_auc_score, n=2000, seed=42):
    """How much better model A scores than model B on the same customers, with a 95% interval.

    Why: both models score the same holdout, so a plain comparison of two AUCs mixes the models'
    difference with the luck of which customers landed in the holdout. Resampling customers, and
    scoring both models on each resample, keeps the pairing and isolates the difference.
    Plain meaning: if the interval stays above 0, A's edge is unlikely to be holdout luck; if it
    straddles 0, the data can't tell the two models apart.
    Assumptions: customers are independent of each other, and the holdout looks like the applicants
    the model will score. The interval is the 2.5th to 97.5th percentile of the resampled differences.
    """
    rng = np.random.default_rng(seed)
    y, a, b = np.asarray(y), np.asarray(proba_a), np.asarray(proba_b)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():  # a resample needs both classes to have an AUC
            diffs.append(metric(y[idx], a[idx]) - metric(y[idx], b[idx]))
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {"difference": float(metric(y, a) - metric(y, b)), "ci_low": float(low), "ci_high": float(high)}
