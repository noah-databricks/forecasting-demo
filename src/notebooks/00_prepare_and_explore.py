# Databricks notebook source
# MAGIC %md
# MAGIC # Store revenue forecasting: preparation and exploratory analysis
# MAGIC
# MAGIC This notebook converts the TPC-DS retail facts into a clean, regular multi-series dataset.
# MAGIC It also records the data-quality evidence and seasonality that justify a forecasting model.

# COMMAND ----------

import logging
import re

# Spark Connect logs the effective serverless usage policy at WARNING level. It is
# informational, so keep the notebook output focused on the analysis.
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
SOURCE_CATALOG = configured("source_catalog", "samples")
SOURCE_SCHEMA = configured("source_schema", "tpcds_sf1")
HORIZON = int(configured("forecast_horizon", 30))

TRAINING_TABLE = f"{CATALOG}.{SCHEMA}.store_revenue_training"
EDA_TABLE = f"{CATALOG}.{SCHEMA}.store_revenue_eda_summary"
SERIES_PROFILE_TABLE = f"{CATALOG}.{SCHEMA}.store_revenue_series_profile"

# COMMAND ----------
# MAGIC %md
# MAGIC ## Build a governed training table
# MAGIC
# MAGIC A date and store identify each series point. The final horizon is labeled `test`, the
# MAGIC preceding horizon is `validate`, and the remainder is `train`. This split column can also
# MAGIC be selected in the serverless AutoML forecasting UI.

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{CATALOG}`.`{SCHEMA}`")
spark.sql(
    f"""
    CREATE OR REPLACE TABLE {TRAINING_TABLE}
    COMMENT 'Daily TPC-DS net-paid revenue by store, prepared for multi-series forecasting'
    AS
    WITH daily AS (
      SELECT
        d.d_date AS ds,
        CAST(s.ss_store_sk AS STRING) AS store_id,
        CAST(SUM(s.ss_net_paid) AS DOUBLE) AS revenue
      FROM `{SOURCE_CATALOG}`.`{SOURCE_SCHEMA}`.store_sales AS s
      INNER JOIN `{SOURCE_CATALOG}`.`{SOURCE_SCHEMA}`.date_dim AS d
        ON s.ss_sold_date_sk = d.d_date_sk
      WHERE s.ss_store_sk IS NOT NULL
        AND s.ss_net_paid IS NOT NULL
      GROUP BY d.d_date, s.ss_store_sk
    ), date_bounds AS (
      SELECT MIN(ds) AS min_ds, MAX(ds) AS max_ds FROM daily
    ), stores AS (
      SELECT DISTINCT store_id FROM daily
    ), date_spine AS (
      SELECT EXPLODE(SEQUENCE(min_ds, max_ds, INTERVAL 1 DAY)) AS ds
      FROM date_bounds
    ), regularized AS (
      SELECT
        spine.ds,
        stores.store_id,
        COALESCE(
          daily.revenue,
          LAST(daily.revenue, true) OVER (
            PARTITION BY stores.store_id
            ORDER BY spine.ds
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
          ),
          0.0
        ) AS revenue,
        daily.revenue IS NULL AS was_imputed
      FROM date_spine AS spine
      CROSS JOIN stores
      LEFT JOIN daily
        ON spine.ds = daily.ds AND stores.store_id = daily.store_id
    ), bounds AS (
      SELECT MAX(ds) AS max_ds FROM daily
    )
    SELECT
      regularized.*,
      CASE
        WHEN ds > date_sub(max_ds, {HORIZON}) THEN 'test'
        WHEN ds > date_sub(max_ds, {2 * HORIZON}) THEN 'validate'
        ELSE 'train'
      END AS data_split
    FROM regularized
    CROSS JOIN bounds
    """
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Data quality and coverage

# COMMAND ----------

summary = spark.sql(
    f"""
    SELECT
      MIN(ds) AS min_date,
      MAX(ds) AS max_date,
      COUNT(*) AS observations,
      COUNT(DISTINCT ds) AS dates,
      COUNT(DISTINCT store_id) AS stores,
      (datediff(MAX(ds), MIN(ds)) + 1) * COUNT(DISTINCT store_id) - COUNT(*) AS missing_store_dates,
      SUM(CASE WHEN was_imputed THEN 1 ELSE 0 END) AS imputed_targets,
      SUM(CASE WHEN revenue IS NULL THEN 1 ELSE 0 END) AS null_targets,
      SUM(CASE WHEN revenue < 0 THEN 1 ELSE 0 END) AS negative_targets,
      COUNT(*) - COUNT(DISTINCT named_struct('ds', ds, 'store_id', store_id)) AS duplicate_keys,
      MIN(revenue) AS min_revenue,
      percentile_approx(revenue, 0.01) AS p01_revenue,
      AVG(revenue) AS avg_revenue,
      percentile_approx(revenue, 0.99) AS p99_revenue,
      MAX(revenue) AS max_revenue
    FROM {TRAINING_TABLE}
    """
)
display(summary)

summary.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(EDA_TABLE)

series_profile = spark.sql(
    f"""
    SELECT
      store_id,
      MIN(ds) AS min_date,
      MAX(ds) AS max_date,
      COUNT(*) AS observations,
      SUM(CASE WHEN data_split = 'train' THEN 1 ELSE 0 END) AS train_rows,
      SUM(CASE WHEN data_split = 'validate' THEN 1 ELSE 0 END) AS validate_rows,
      SUM(CASE WHEN data_split = 'test' THEN 1 ELSE 0 END) AS test_rows
    FROM {TRAINING_TABLE}
    GROUP BY store_id
    ORDER BY store_id
    """
)
display(series_profile)
series_profile.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    SERIES_PROFILE_TABLE
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Seasonal structure

# COMMAND ----------

monthly = spark.sql(
    f"""
    SELECT
      month(ds) AS month,
      AVG(revenue) AS average_daily_store_revenue,
      percentile_approx(revenue, 0.5) AS median_daily_store_revenue
    FROM {TRAINING_TABLE}
    GROUP BY month(ds)
    ORDER BY month
    """
)
display(monthly)

monthly_pdf = monthly.toPandas()
ax = monthly_pdf.plot(
    x="month",
    y="average_daily_store_revenue",
    kind="bar",
    title="Average daily store revenue by month",
    legend=False,
    figsize=(11, 4),
)
ax.set_ylabel("Revenue")
ax.figure.tight_layout()
display(ax.figure)

weekday = spark.sql(
    f"""
    SELECT
      dayofweek(ds) AS day_of_week,
      AVG(revenue) AS average_daily_store_revenue
    FROM {TRAINING_TABLE}
    GROUP BY dayofweek(ds)
    ORDER BY dayofweek(ds)
    """
)
display(weekday)

weekday_pdf = weekday.toPandas()
weekday_ax = weekday_pdf.plot(
    x="day_of_week",
    y="average_daily_store_revenue",
    kind="bar",
    title="Average daily store revenue by day of week",
    legend=False,
    figsize=(9, 4),
)
weekday_ax.set_ylabel("Revenue")
weekday_ax.figure.tight_layout()
display(weekday_ax.figure)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Recent series

# COMMAND ----------

recent = spark.sql(
    f"""
    SELECT ds, store_id, revenue
    FROM {TRAINING_TABLE}
    WHERE ds >= date_sub((SELECT MAX(ds) FROM {TRAINING_TABLE}), 365)
    ORDER BY ds, store_id
    """
)
display(recent)

recent_pdf = recent.toPandas()
recent_pdf["ds"] = recent_pdf["ds"].astype("datetime64[ns]")
recent_pivot = recent_pdf.pivot(index="ds", columns="store_id", values="revenue")
recent_ax = recent_pivot.plot(
    figsize=(12, 5),
    title="Daily store revenue — most recent year",
    alpha=0.8,
)
recent_ax.set_ylabel("Revenue")
recent_ax.figure.tight_layout()
display(recent_ax.figure)
