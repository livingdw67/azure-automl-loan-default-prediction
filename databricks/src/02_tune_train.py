# Databricks notebook source
# MAGIC %md
# MAGIC # 02 Tune and train: Optuna-tuned LightGBM, checked once on a holdout
# MAGIC
# MAGIC 1. Split the 20,839 customers 80/20 (stratified, seed 42): the same holdout as
# MAGIC    `notebooks/validate_locally.py`, so the results line up with the README's out-of-sample table.
# MAGIC 2. Optuna searches LightGBM settings by 5-fold cross-validated AUC on the 80% only. Every trial is a
# MAGIC    nested MLflow run, and trials that trail the median stop after a fold or two.
# MAGIC 3. The review cutoff maximises F2 on out-of-fold predictions of the 80%.
# MAGIC 4. The holdout is scored once, for the tuned model and for the stand-in of the Azure AutoML ensemble,
# MAGIC    both fit on the same 80%. A paired bootstrap says whether the difference is more than holdout luck.
# MAGIC 5. The model refit on all customers is registered as `loan_default@challenger`. It becomes
# MAGIC    `@champion` if there is none yet, or if it beats the champion on the holdout (interval above 0).

# COMMAND ----------

# MAGIC %pip install lightgbm==4.5.0 xgboost==2.1.4 optuna==4.9.0 --quiet

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "loans")
dbutils.widgets.text("n_trials", "60")
dbutils.widgets.text("timeout_minutes", "20")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
n_trials = int(dbutils.widgets.get("n_trials"))
timeout = 60 * int(dbutils.widgets.get("timeout_minutes"))

# COMMAND ----------

from datetime import datetime, timezone
from pathlib import Path

import lightgbm
import mlflow
import optuna
import pandas as pd
import sklearn
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

import loan_lib
from loan_lib.evaluate import best_f2_cutoff, metrics, paired_bootstrap
from loan_lib.features import INPUTS, TARGET, engineer_features
from loan_lib.models import lgbm, stand_in_ensemble
from loan_lib.scoring import mean_abs_contributions
from loan_lib.serving import LoanDefaultModel
from loan_lib.tuning import tune

mlflow.set_registry_uri("databricks-uc")
user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/loan-default")
model_name = f"{catalog}.{schema}.loan_default"
holdout_table = f"{catalog}.{schema}.holdout_predictions"
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Sorted by id, the CSV's row order, so the split matches validate_locally.py row for row
raw = spark.table(f"{catalog}.{schema}.raw_train").toPandas().sort_values("id").reset_index(drop=True)
X, y = engineer_features(raw), raw[TARGET]
X_dev, X_hold, y_dev, y_hold = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
print(f"{len(X_dev):,} development and {len(X_hold):,} holdout customers; default rate {y.mean():.1%}")

# COMMAND ----------


def log_trial(study, trial):
    """Each trial as a nested run: its settings, how many folds it ran, and its cross-validated AUC."""
    with mlflow.start_run(run_name=f"trial {trial.number}", nested=True):
        mlflow.log_params(trial.params)
        mlflow.set_tag("state", trial.state.name.lower())
        mlflow.log_metric("folds_run", len(trial.intermediate_values))
        if trial.value is not None:
            mlflow.log_metric("cv_auc", trial.value)
        elif trial.intermediate_values:
            mlflow.log_metric("cv_auc_when_stopped", list(trial.intermediate_values.values())[-1])


with mlflow.start_run(run_name="optuna-lightgbm") as run:
    study = tune(X_dev, y_dev, n_trials=n_trials, timeout=timeout, callbacks=[log_trial])
    params = study.best_params
    states = pd.Series([t.state.name.lower() for t in study.trials]).value_counts()
    mlflow.log_params({**params, "trials": len(study.trials), "trials_pruned": int(states.get("pruned", 0)),
                       "dev_rows": len(X_dev), "holdout_rows": len(X_hold), "features": ",".join(X.columns),
                       "lightgbm": lightgbm.__version__, "optuna": optuna.__version__})
    mlflow.log_metric("cv_auc", study.best_value)
    print(f"{len(study.trials)} trials ({int(states.get('pruned', 0))} stopped early), best CV AUC {study.best_value:.4f}")

    # Both models: cutoff from out-of-fold predictions on the 80%, then one look at the holdout
    models, probas, results = {}, {}, {}
    for name, make in [("tuned", lambda: lgbm(params)), ("stand_in", stand_in_ensemble)]:
        oof = cross_val_predict(make(), X_dev, y_dev, cv=cv, method="predict_proba")[:, 1]
        models[name] = make().fit(X_dev, y_dev)
        probas[name] = models[name].predict_proba(X_hold)[:, 1]
        results[name] = metrics(y_hold, probas[name], best_f2_cutoff(y_dev, oof))
        mlflow.log_metrics({f"holdout_{k}_{name}": v for k, v in results[name].items()})
    display(pd.DataFrame(results).T.round(3))

    comparison = {m: paired_bootstrap(y_hold, probas["tuned"], probas["stand_in"], metric=f)
                  for m, f in [("auc", roc_auc_score), ("pr_auc", average_precision_score)]}
    for m, r in comparison.items():
        mlflow.log_metrics({f"{m}_vs_stand_in_{k}": v for k, v in r.items()})
        print(f"Tuned minus stand-in, holdout {m}: {r['difference']:+.4f} (95% interval {r['ci_low']:+.4f} to {r['ci_high']:+.4f})")

    shap = mean_abs_contributions(models["tuned"], raw.loc[X_hold.index])
    mlflow.log_table(shap.rename_axis("feature").reset_index(), "shap_importance.json")
    display(shap.round(3).to_frame())

    # Stored so a later challenger can be compared with this run on the same customers
    holdout_rows = pd.DataFrame({"run_id": run.info.run_id, "id": raw.loc[X_hold.index, "id"].to_numpy(),
                                 "target": y_hold.to_numpy(), "probability": probas["tuned"],
                                 "logged_at": datetime.now(timezone.utc)})
    spark.createDataFrame(holdout_rows).write.mode("append").saveAsTable(holdout_table)

    # The registered model: same settings and cutoff, refit on every customer
    cutoff = results["tuned"]["cutoff"]
    wrapped = LoanDefaultModel(lgbm(params).fit(X, y), cutoff)
    sample = raw[INPUTS].head(20)
    info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=wrapped,
        code_paths=[str(Path(loan_lib.__file__).parent)],
        signature=infer_signature(sample, wrapped.predict(None, sample)),
        pip_requirements=[f"lightgbm=={lightgbm.__version__}", f"scikit-learn=={sklearn.__version__}",
                          f"pandas=={pd.__version__}"],
        registered_model_name=model_name,
    )

# COMMAND ----------

client = MlflowClient()
version = info.registered_model_version
client.set_registered_model_alias(model_name, "challenger", version)
try:
    champion = client.get_model_version_by_alias(model_name, "champion")
except MlflowException:
    champion = None

if champion is None:
    promote, reason = True, "there was no champion yet"
else:
    prior = spark.table(holdout_table).where(f"run_id = '{champion.run_id}'").select("id", "probability").toPandas()
    both = holdout_rows.merge(prior, on="id", suffixes=("", "_champion"))
    if both.empty:
        promote, reason = False, f"champion v{champion.version} has no stored holdout predictions to compare with"
    else:
        edge = paired_bootstrap(both["target"], both["probability"], both["probability_champion"])
        promote = edge["ci_low"] > 0
        reason = (f"holdout AUC {edge['difference']:+.4f} vs champion v{champion.version} "
                  f"(95% interval {edge['ci_low']:+.4f} to {edge['ci_high']:+.4f})")

if promote:
    client.set_registered_model_alias(model_name, "champion", version)
print(f"Registered {model_name} v{version} as challenger; {'promoted to' if promote else 'not'} champion: {reason}")
