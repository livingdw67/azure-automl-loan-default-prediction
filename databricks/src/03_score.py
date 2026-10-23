# Databricks notebook source
# MAGIC %md
# MAGIC # 03 Score: the champion model on the challenge's test customers
# MAGIC
# MAGIC Scores `raw_test` (1,000 customers, no default flag) with `loan_default@champion` and writes
# MAGIC `predictions`: the probability of default, whether the account is flagged for review at the
# MAGIC model's F2 cutoff, and the model version that scored it.

# COMMAND ----------

# MAGIC %pip install lightgbm==4.5.0 --quiet

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "loans")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

from datetime import datetime, timezone

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from loan_lib.features import INPUTS

mlflow.set_registry_uri("databricks-uc")
model_name = f"{catalog}.{schema}.loan_default"
version = MlflowClient().get_model_version_by_alias(model_name, "champion").version
model = mlflow.pyfunc.load_model(f"models:/{model_name}@champion")

test = spark.table(f"{catalog}.{schema}.raw_test").toPandas().sort_values("id").reset_index(drop=True)
scored = model.predict(test[INPUTS]).reset_index(drop=True)
out = pd.concat([test[["id"]], scored], axis=1).assign(model_version=int(version), scored_at=datetime.now(timezone.utc))

(spark.createDataFrame(out).write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{catalog}.{schema}.predictions"))
print(f"Scored {len(out):,} customers with {model_name} v{version}: {out['flagged'].mean():.1%} flagged for review")
display(spark.table(f"{catalog}.{schema}.predictions").orderBy("probability", ascending=False).limit(20))
