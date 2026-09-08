# Validation record

Validated on 2026-09-08 in an AWS Databricks workspace using serverless Jobs compute.

## Bundle and serverless jobs

- `databricks bundle validate --strict -t dev --var="catalog=<target_catalog>"`: passed.
- Deployment: passed.
- End-to-end lifecycle run: all serverless tasks passed.
- Independent batch run: configuration and scoring tasks passed.
- The deploy-time catalog/schema values reached both notebooks through task values; no notebook
  widgets or base parameters were used.

## Data checks

- 6 store series, 1,827 dates, and 10,962 regular store/date observations.
- 24 source gaps were forward-filled and retained as `was_imputed = true`.
- 0 remaining missing store-dates, duplicate keys, null targets, or negative targets.

## Model comparison and registry

| Rank | Model | SMAPE | WAPE | RMSE |
|---:|---|---:|---:|---:|
| 1 | AutoETS7 | 13.713 | 13.622 | 139,538.22 |
| 2 | AutoARIMA7 | 16.778 | 16.300 | 171,328.61 |
| 3 | SeasonalNaive7 | 18.210 | 18.407 | 189,109.30 |
| 4 | SeasonalNaive365 | 19.221 | 19.531 | 202,015.10 |
| 5 | Naive | 20.505 | 20.264 | 210,504.34 |

- UC model: `<target_catalog>.<target_schema>.store_revenue_forecaster`
- A new version was registered and resolved through the alias.
- Alias: `Champion`
- Selected algorithm: `AutoETS7`

## Batch output

- Table: `<target_catalog>.<target_schema>.store_revenue_batch_forecasts`
- 180 rows: 6 stores × 30 future dates.
- Forecast range: 2003-01-03 through 2003-02-01.
- The resolved model version was recorded on every output row.
- Null predictions: 0.

## Separate Genie Code presenter-script validation

The optional presenter script was tested separately in the Databricks notebook editor through Genie
Code. The prompt correctly elicited a multi-series forecasting workflow and a time-based split plan.
Genie Code identified missing-date checks, negative/outlier profiling, per-store split coverage, and
weekday seasonality as gaps; these were incorporated into the final notebook.

One recommendation was deliberately rejected after execution testing: pandas/Spark DataFrame
`cache()` maps to `PERSIST`, which is unsupported on serverless Spark Connect. The final notebook
therefore recomputes the small summary query.
