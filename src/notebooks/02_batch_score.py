# Databricks notebook source
# MAGIC %md
# MAGIC # Batch deployment and scoring
# MAGIC
# MAGIC This notebook resolves the movable UC model alias, requests a future horizon for every
# MAGIC store, performs batch inference, and publishes a Delta table for downstream SQL, dashboards,
# MAGIC or Genie consumers.

# COMMAND ----------

# MAGIC %pip install statsforecast==2.0.3

# COMMAND ----------

# Restart after changing the notebook-scoped environment so MLflow can import the
# packaged StatsForecast model when this notebook is run interactively.
dbutils.library.restartPython()

# COMMAND ----------

import logging
import re

logging.getLogger("pyspark.sql.connect.logging").setLevel(logging.ERROR)

# A DAB run receives deploy-time values from the upstream configure task. Direct,
# interactive runs use portable defaults based on the current user and catalog.
CURRENT_USER, CURRENT_CATALOG = spark.sql(
    "SELECT current_user(), current_catalog()"
).first()
USER_SLUG = re.sub(r"[^a-zA-Z0-9_]", "_", CURRENT_USER.split("@")[0])


def configured(key, default):
    try:
        return dbutils.jobs.taskValues.get(taskKey="configure", key=key, debugValue=default)
    except Exception:
        return default


CATALOG = configured("catalog", CURRENT_CATALOG)
SCHEMA = configured("schema", f"forecasting_demo_{USER_SLUG}")
HORIZON = int(configured("forecast_horizon", 30))
REGISTERED_MODEL_NAME = configured("registered_model_name", "store_revenue_forecaster")
MODEL_NAME = f"{CATALOG}.{SCHEMA}.{REGISTERED_MODEL_NAME}"
MODEL_ALIAS = "Champion"

TRAINING_TABLE = f"{CATALOG}.{SCHEMA}.store_revenue_training"
PREDICTIONS_TABLE = f"{CATALOG}.{SCHEMA}.store_revenue_batch_forecasts"

# COMMAND ----------

import json

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F

mlflow.set_registry_uri("databricks-uc")
model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

history = spark.table(TRAINING_TABLE).select("store_id", "ds")
max_ds = history.agg(F.max("ds").alias("max_ds")).first()["max_ds"]
store_ids = [row["store_id"] for row in history.select("store_id").distinct().collect()]

future_dates = pd.date_range(pd.Timestamp(max_ds) + pd.Timedelta(days=1), periods=HORIZON, freq="D")
requests_pdf = pd.DataFrame(
    [(str(store_id), future_date) for store_id in sorted(store_ids) for future_date in future_dates],
    columns=["store_id", "ds"],
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Resolve and score the registered champion

# COMMAND ----------

loaded_model = mlflow.pyfunc.load_model(model_uri)
predictions_pdf = loaded_model.predict(requests_pdf)
scored_pdf = requests_pdf.copy()
scored_pdf["prediction"] = predictions_pdf["prediction"].to_numpy(dtype=float)

client = MlflowClient(registry_uri="databricks-uc")
model_version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
scored_pdf["model_name"] = MODEL_NAME
scored_pdf["model_version"] = str(model_version.version)
scored_pdf["model_alias"] = MODEL_ALIAS

scored = spark.createDataFrame(scored_pdf).withColumn("scored_at", F.current_timestamp())
scored.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(PREDICTIONS_TABLE)

display(scored.orderBy("store_id", "ds"))

dbutils.notebook.exit(
    json.dumps(
        {
            "model_uri": model_uri,
            "model_version": str(model_version.version),
            "rows_scored": int(len(scored_pdf)),
            "predictions_table": PREDICTIONS_TABLE,
            "forecast_start": str(future_dates.min().date()),
            "forecast_end": str(future_dates.max().date()),
        }
    )
)
