# Forecasting ML lifecycle

An end-to-end Databricks Asset Bundle for a data-science forecasting demonstration using
`samples.tpcds_sf1.store_sales`:

- exploratory analysis and governed training-table preparation;
- time-based comparison of naive, seasonal-naive, ETS, and ARIMA forecasts;
- MLflow parent/child runs with metrics, artifacts, and backtest predictions;
- champion registration in the Unity Catalog model registry with a movable alias;
- independently rerunnable batch scoring into a governed Delta table.

All DAB job tasks use serverless compute. Databricks' separate **Forecasting (serverless) with
AutoML** Public Preview is currently started from the UI, so the automated bundle compares the same
family of open-source statistical models directly. The UC model is registered by MLflow during the training task; the bundle provider's
`registered_models` resource targets the legacy workspace registry and cannot accept a three-part UC
model name.

## Deploy and run

Prerequisites: Databricks CLI authentication to the customer's workspace, read access to the
`samples` catalog, and create privileges in a Unity Catalog catalog chosen for the demo.

```bash
databricks bundle validate --strict -t dev --var="catalog=<customer_catalog>"
databricks bundle deploy -t dev --var="catalog=<customer_catalog>"
databricks bundle run forecasting_lifecycle -t dev --var="catalog=<customer_catalog>"
```

Run only the deployed batch inference job after a champion exists:

```bash
databricks bundle run forecasting_batch_inference -t dev --var="catalog=<customer_catalog>"
```

The schema defaults to `forecasting_demo_<deployer>` and can be overridden along with the
forecast horizon or source dataset:

```bash
databricks bundle deploy -t dev \
  --var="catalog=<customer_catalog>,schema=<customer_schema>,forecast_horizon=30"
```

No notebook widgets are used. Jobs receive DAB variables through a small serverless configuration
task. When a notebook is run directly, it uses the current UC catalog and a schema derived from the
signed-in user.

The tested workspace runs, model metrics, and batch-output checks are recorded in
[`docs/VALIDATION.md`](docs/VALIDATION.md).
