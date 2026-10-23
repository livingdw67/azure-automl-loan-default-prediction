# Databricks notebook source
# MAGIC %md
# MAGIC # 01 Load: the challenge's train and test files into Delta tables
# MAGIC
# MAGIC Reads `train.csv` and `test.csv` from the `raw_files` volume and writes `raw_train` (20,839
# MAGIC customers with the default flag) and `raw_test` (1,000 customers without it). The files are
# MAGIC uploaded to the volume once (see the README); they stay out of this public repo.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "loans")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.raw_files")

for name, expected in [("train", 20839), ("test", 1000)]:
    df = spark.read.csv(f"/Volumes/{catalog}/{schema}/raw_files/{name}.csv", header=True, inferSchema=True)
    rows = df.count()
    if rows != expected:
        raise ValueError(f"{name}.csv has {rows:,} rows, expected {expected:,}: is it the challenge file?")
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{catalog}.{schema}.raw_{name}")
    print(f"raw_{name}: {rows:,} customers")
