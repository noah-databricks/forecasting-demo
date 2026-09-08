# Databricks notebook source
# MAGIC %md
# MAGIC # Compare forecasting models and register the champion
# MAGIC
# MAGIC This notebook runs a time-based holdout, logs every candidate to MLflow, persists the
# MAGIC comparison, retrains the winner on all history, and registers a batch-forecastable pyfunc
# MAGIC model in Unity Catalog.

# COMMAND ----------

# MAGIC %pip install statsforecast==2.0.3

# COMMAND ----------

# Restart after changing the notebook-scoped environment so the package is importable
# when this notebook is run interactively on serverless compute.
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
EXPERIMENT_PATH = configured(
    "experiment_path", f"/Users/{CURRENT_USER}/forecasting-demo/model-comparison"
)
REGISTERED_MODEL_NAME = configured("registered_model_name", "store_revenue_forecaster")
MODEL_NAME = f"{CATALOG}.{SCHEMA}.{REGISTERED_MODEL_NAME}"
MODEL_ALIAS = "Champion"

TRAINING_TABLE = f"{CATALOG}.{SCHEMA}.store_revenue_training"
METRICS_TABLE = f"{CATALOG}.{SCHEMA}.forecast_model_metrics"
BACKTEST_TABLE = f"{CATALOG}.{SCHEMA}.forecast_backtest_predictions"

# COMMAND ----------

import json
import os
import tempfile

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, Naive, SeasonalNaive

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_PATH)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Time-based holdout

# COMMAND ----------

history_pdf = (
    spark.table(TRAINING_TABLE)
    .select("ds", "store_id", "revenue")
    .orderBy("store_id", "ds")
    .toPandas()
)
history_pdf["ds"] = pd.to_datetime(history_pdf["ds"])
sf_pdf = history_pdf.rename(columns={"store_id": "unique_id", "revenue": "y"})

max_ds = sf_pdf["ds"].max()
cutoff = max_ds - pd.Timedelta(days=HORIZON)
train_pdf = sf_pdf[sf_pdf["ds"] <= cutoff].copy()
test_pdf = sf_pdf[sf_pdf["ds"] > cutoff].copy()

assert len(test_pdf) == HORIZON * sf_pdf["unique_id"].nunique()

candidates = [
    Naive(alias="Naive"),
    SeasonalNaive(season_length=7, alias="SeasonalNaive7"),
    SeasonalNaive(season_length=365, alias="SeasonalNaive365"),
    AutoETS(season_length=7, alias="AutoETS7"),
    AutoARIMA(season_length=7, alias="AutoARIMA7"),
]

# COMMAND ----------
# MAGIC %md
# MAGIC ## Train, compare, and log candidates

# COMMAND ----------

def metric_values(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = actual - predicted
    denominator = np.abs(actual) + np.abs(predicted)
    return {
        "smape": float(np.mean(np.where(denominator == 0, 0.0, 200.0 * np.abs(error) / denominator))),
        "wape": float(100.0 * np.abs(error).sum() / np.abs(actual).sum()),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
    }


with mlflow.start_run(run_name="store-revenue-model-comparison") as parent_run:
    mlflow.set_tags(
        {
            "problem_type": "multi_series_forecasting",
            "frequency": "daily",
            "source_table": TRAINING_TABLE,
            "selection_metric": "smape",
        }
    )
    mlflow.log_params(
        {
            "forecast_horizon": HORIZON,
            "series_count": int(sf_pdf["unique_id"].nunique()),
            "training_rows": len(train_pdf),
            "test_rows": len(test_pdf),
        }
    )

    forecaster = StatsForecast(models=candidates, freq="D", n_jobs=-1)
    forecast_pdf = forecaster.forecast(df=train_pdf, h=HORIZON)
    scored_pdf = test_pdf.merge(forecast_pdf, on=["unique_id", "ds"], how="inner")

    metric_rows = []
    candidate_names = [model.alias for model in candidates]
    for candidate_name in candidate_names:
        metrics = metric_values(scored_pdf["y"], scored_pdf[candidate_name])
        metric_rows.append({"model": candidate_name, **metrics})
        with mlflow.start_run(run_name=candidate_name, nested=True):
            mlflow.log_param("algorithm", candidate_name)
            mlflow.log_param("forecast_horizon", HORIZON)
            mlflow.log_metrics(metrics)

    metrics_pdf = pd.DataFrame(metric_rows).sort_values("smape").reset_index(drop=True)
    best_model_name = str(metrics_pdf.iloc[0]["model"])
    mlflow.log_metric("best_holdout_smape", float(metrics_pdf.iloc[0]["smape"]))
    mlflow.log_param("selected_model", best_model_name)
    mlflow.log_table(metrics_pdf, "model_comparison.json")

    plot = metrics_pdf.plot.bar(
        x="model", y="smape", legend=False, figsize=(10, 4), title="Holdout SMAPE by model"
    )
    plot.set_ylabel("SMAPE (%)")
    plot.figure.tight_layout()
    with tempfile.TemporaryDirectory() as temp_dir:
        plot_path = os.path.join(temp_dir, "model_comparison.png")
        plot.figure.savefig(plot_path, dpi=150)
        mlflow.log_artifact(plot_path, artifact_path="plots")

    metrics_spark = spark.createDataFrame(metrics_pdf)
    metrics_spark.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(METRICS_TABLE)

    backtest_pdf = scored_pdf.melt(
        id_vars=["unique_id", "ds", "y"],
        value_vars=candidate_names,
        var_name="model",
        value_name="prediction",
    ).rename(columns={"unique_id": "store_id", "y": "actual"})
    spark.createDataFrame(backtest_pdf).write.mode("overwrite").option(
        "overwriteSchema", "true"
    ).saveAsTable(BACKTEST_TABLE)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Package the selected forecaster for reusable batch inference

# COMMAND ----------

def build_candidate(name):
    builders = {
        "Naive": lambda: Naive(alias="prediction"),
        "SeasonalNaive7": lambda: SeasonalNaive(season_length=7, alias="prediction"),
        "SeasonalNaive365": lambda: SeasonalNaive(season_length=365, alias="prediction"),
        "AutoETS7": lambda: AutoETS(season_length=7, alias="prediction"),
        "AutoARIMA7": lambda: AutoARIMA(season_length=7, alias="prediction"),
    }
    return builders[name]()


class MultiStoreForecaster(mlflow.pyfunc.PythonModel):
    """Forecast requested future dates for each store using the selected StatsForecast model."""

    def __init__(self, selected_model):
        self.selected_model = selected_model

    def load_context(self, context):
        self.history = pd.read_parquet(context.artifacts["history"])
        self.history["ds"] = pd.to_datetime(self.history["ds"])

    def predict(self, context, model_input, params=None):
        requests = model_input.copy().reset_index(drop=True)
        requests["ds"] = pd.to_datetime(requests["ds"])
        requests["store_id"] = requests["store_id"].astype(str)
        output = pd.Series(index=requests.index, dtype=float)

        for store_id, positions in requests.groupby("store_id").groups.items():
            store_history = self.history[self.history["store_id"].astype(str) == store_id].copy()
            if store_history.empty:
                raise ValueError(f"Unknown store_id: {store_id}")
            store_requests = requests.loc[list(positions)]
            last_observed = store_history["ds"].max()
            if store_requests["ds"].min() <= last_observed:
                raise ValueError("All requested dates must be later than the model training history")
            horizon = int((store_requests["ds"].max() - last_observed).days)
            expected_dates = pd.date_range(last_observed + pd.Timedelta(days=1), periods=horizon, freq="D")

            series = store_history.rename(columns={"store_id": "unique_id", "revenue": "y"})[
                ["unique_id", "ds", "y"]
            ]
            series["unique_id"] = series["unique_id"].astype(str)
            fitted = StatsForecast(models=[build_candidate(self.selected_model)], freq="D", n_jobs=1)
            forecast = fitted.forecast(df=series, h=horizon)
            forecast["ds"] = expected_dates
            lookup = forecast.set_index("ds")["prediction"]
            output.loc[list(positions)] = store_requests["ds"].map(lookup).to_numpy()

        return pd.DataFrame({"prediction": output.astype(float)})


with tempfile.TemporaryDirectory() as temp_dir:
    history_path = os.path.join(temp_dir, "history.parquet")
    # Spark Connect may attach a non-JSON PlanMetrics object to pandas attrs. It is
    # execution metadata, not model data, so remove it before PyArrow serialization.
    history_artifact_pdf = history_pdf[["store_id", "ds", "revenue"]].copy()
    history_artifact_pdf.attrs = {}
    history_artifact_pdf.to_parquet(history_path, index=False)

    example_input = pd.DataFrame(
        {
            "store_id": [str(history_pdf["store_id"].iloc[0])],
            "ds": [history_pdf["ds"].max() + pd.Timedelta(days=1)],
        }
    )
    example_output = pd.DataFrame({"prediction": [float(history_pdf["revenue"].iloc[-1])]})
    signature = infer_signature(example_input, example_output)

    with mlflow.start_run(run_name=f"register-{best_model_name}") as registration_run:
        mlflow.log_param("selected_model", best_model_name)
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=MultiStoreForecaster(best_model_name),
            artifacts={"history": history_path},
            registered_model_name=MODEL_NAME,
            signature=signature,
            input_example=example_input,
            pip_requirements=[
                "mlflow>=2.18,<4",
                "statsforecast==2.0.3",
                "pandas>=2,<3",
                "pyarrow>=15,<22",
            ],
        )

client = MlflowClient(registry_uri="databricks-uc")
version = str(model_info.registered_model_version)
client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, version)
client.set_model_version_tag(MODEL_NAME, version, "selected_model", best_model_name)
client.set_model_version_tag(MODEL_NAME, version, "holdout_metric", "smape")

display(spark.table(METRICS_TABLE).orderBy("smape"))

dbutils.notebook.exit(
    json.dumps(
        {
            "registered_model": MODEL_NAME,
            "model_version": version,
            "alias": MODEL_ALIAS,
            "selected_model": best_model_name,
            "metrics_table": METRICS_TABLE,
            "backtest_table": BACKTEST_TABLE,
        }
    )
)
