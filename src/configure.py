"""Publish deploy-time DAB configuration for downstream notebook tasks."""

import argparse

from databricks.sdk.runtime import dbutils


parser = argparse.ArgumentParser()
parser.add_argument("--catalog", required=True)
parser.add_argument("--schema", required=True)
parser.add_argument("--source-catalog", default="samples")
parser.add_argument("--source-schema", default="tpcds_sf1")
parser.add_argument("--forecast-horizon", type=int, default=30)
parser.add_argument("--experiment-path", required=True)
parser.add_argument("--registered-model-name", default="store_revenue_forecaster")
args = parser.parse_args()

for key, value in {
    "catalog": args.catalog,
    "schema": args.schema,
    "source_catalog": args.source_catalog,
    "source_schema": args.source_schema,
    "forecast_horizon": args.forecast_horizon,
    "experiment_path": args.experiment_path,
    "registered_model_name": args.registered_model_name,
}.items():
    dbutils.jobs.taskValues.set(key=key, value=value)
