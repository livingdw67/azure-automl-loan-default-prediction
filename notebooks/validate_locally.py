"""Out-of-sample check of the loan default model, run locally without Azure.

The challenge's test.csv has no labels, so the original notebook chose the 0.15
cutoff and measured default capture on the training data the model was fit on.
This script re-creates the no-age feature set and a soft-voting ensemble similar to
the AutoML champion, then measures performance on data the model has not seen.

Usage (train.csv from the data challenge, not included in this repo):
    python validate_locally.py path/to/train.csv
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, fbeta_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

# Features, the ensemble and the cutoff search are shared with the Databricks pipeline
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "databricks" / "src"))
from loan_lib.evaluate import best_f2_cutoff  # noqa: E402
from loan_lib.features import engineer_features  # noqa: E402
from loan_lib.models import stand_in_ensemble as ensemble  # noqa: E402

CUTOFF = 0.15


def report(label, y, proba, cutoff):
    pred = proba >= cutoff
    print(f"{label:<44} cutoff {cutoff:.2f} | captured {recall_score(y, pred):.1%} of defaults | "
          f"precision {precision_score(y, pred):.1%} | F2 {fbeta_score(y, pred, beta=2):.3f} | "
          f"AUC {roc_auc_score(y, proba):.3f} | PR-AUC {average_precision_score(y, proba):.3f}")


def main(path):
    raw = pd.read_csv(path)
    X, y = engineer_features(raw), raw["target"]
    print(f"{len(raw):,} customers | default rate {y.mean():.1%}\n")

    # 1. How the original number was measured: score the same rows the model was trained on
    in_sample = ensemble().fit(X, y).predict_proba(X)[:, 1]
    report("In-sample (training data, original method)", y, in_sample, CUTOFF)

    # 2. 10-fold cross-validated predictions: every row scored by a model that never saw it
    oof = cross_val_predict(ensemble(), X, y, cv=StratifiedKFold(10, shuffle=True, random_state=42),
                            method="predict_proba")[:, 1]
    report("10-fold cross-validated", y, oof, CUTOFF)

    # 3. Held-out 20%: choose the F2 cutoff on the other 80% only, then score the holdout once
    X_dev, X_hold, y_dev, y_hold = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    dev_oof = cross_val_predict(ensemble(), X_dev, y_dev, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                                method="predict_proba")[:, 1]
    cutoff = best_f2_cutoff(y_dev, dev_oof)
    hold = ensemble().fit(X_dev, y_dev).predict_proba(X_hold)[:, 1]
    report(f"Held-out 20% (cutoff chosen on other 80%)", y_hold, hold, cutoff)
    report("Held-out 20% at the original 0.15 cutoff", y_hold, hold, CUTOFF)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "train.csv")
