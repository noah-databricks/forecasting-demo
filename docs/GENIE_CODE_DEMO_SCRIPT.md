# Genie Code forecasting demo: from a blank notebook to an ML lifecycle

## Demo premise

This repository is the **gold-standard reference implementation**: a tested, portable Databricks
Asset Bundle covering exploration, experiment tracking, model registration, and batch deployment.

The live demo does not ask Genie Code to review or repair that implementation. It starts in a blank
notebook with two sample tables and a business outcome. The audience watches Genie Code create as
much of the same end-to-end lifecycle as possible from natural-language prompts. At the end, compare
the generated work with the gold-standard DAB and hand over the reference project.

Replace `<target_catalog>` with a writable Unity Catalog catalog. Use a new schema such as
`forecasting_demo_<your_name>` for `<target_schema>`.

## Before the demo

1. Create a blank Python notebook in a new, empty workspace folder.
2. Connect it to serverless compute and open Genie Code.
3. Confirm access to `samples.tpcds_sf1.store_sales` and `samples.tpcds_sf1.date_dim`.
4. Confirm create privileges in `<target_catalog>`.
5. Keep the reference DAB closed until the final comparison.

Use a disposable schema if Genie Code is allowed to execute changes automatically. The important
phrase in every prompt is **implement and run**: the demo should produce notebook cells, MLflow runs,
UC assets, predictions, and deployment files—not an answer explaining how one might do those things.

## Opening narration

> We have two governed sample tables and a forecasting objective. There is no prepared feature
> table, experiment, model, or deployment. I am going to ask Genie Code to act as the data scientist
> and take this as far through the ML lifecycle as it can, using serverless compute. Afterward, we
> will compare what it created with a tested gold-standard implementation.

## Act 1 — Frame the problem, explore the data, and create the training asset

Prompt Genie Code in the blank notebook:

> Act as the lead data scientist for this forecasting use case. We want a daily net-paid revenue
> forecast, using `ss_net_paid` as the business target, for
> each store using `samples.tpcds_sf1.store_sales` and
> `samples.tpcds_sf1.date_dim`. Work in `<target_catalog>.<target_schema>` and use serverless compute.
> Start by investigating the source tables, deciding on the forecasting grain, target, identifiers,
> and evaluation strategy. Reserve the latest 30 days as test and the preceding 30 days as
> validation. Then implement and run a complete exploratory analysis. Produce a
> governed, model-ready training table; address irregular dates without hiding imputation; avoid
> temporal leakage; create useful data-quality checks and visualizations; and explain the resulting
> modeling assumptions. Do not use notebook widgets and do not modify the source tables. Continue
> through execution until you can give me the actual table name, date range, series count, split
> sizes, data-quality results, and a conclusion on whether the data is suitable for forecasting.

What the audience should see:

- Genie Code inspecting the governed source data and choosing a multi-series formulation.
- Executable SQL/Python and markdown appearing in the notebook.
- A regularized training asset with explicit provenance and imputation evidence.
- Time-based train/validation/test logic, quality results, and seasonal visualizations.
- A data-suitability conclusion grounded in executed results.

Presenter transition:

> Genie Code has moved from a business question to a governed training asset and an evaluation
> design. Now we will ask it to conduct the experiment—not simply recommend an algorithm.

## Act 2 — Conduct a comparative forecasting experiment in MLflow

> Continue as the data scientist and implement the model-development phase. Establish credible
> baselines and compare several appropriate statistical forecasting approaches on the same future
> holdout for every store. Include models capable of representing weekly and longer seasonal
> behavior. Choose and explain suitable forecasting metrics, with SMAPE as the primary selection
> metric. Use MLflow to make the work reproducible: track the overall experiment and each candidate,
> including parameters, metrics, predictions, tables, and useful plots. Persist model-comparison and
> backtest results in `<target_catalog>.<target_schema>`. Install and pin any packages needed for a
> direct interactive serverless run. Implement and execute the experiment, then show the measured
> ranking, forecast-versus-actual evidence, and the selected champion. Do not stop at a plan or code
> sample.

What the audience should see:

- Multiple baselines and challenger models evaluated on an identical temporal holdout.
- A ranked metric table and actual-versus-predicted evidence.
- An MLflow parent experiment with comparable candidate runs and artifacts.
- A champion selected from measured results rather than intuition.

Optional managed AutoML prompt:

> Determine whether managed forecasting AutoML can be launched programmatically with serverless
> compute in this workspace. If it is available, run it against the same governed table, identifiers,
> horizon, split, and primary metric and include the result in the comparison. If this workspace
> exposes the workflow only through the AutoML UI, explain the exact UI selections and keep the
> code-first experiment on serverless compute; do not silently switch to classic compute.

## Act 3 — Turn the champion into a governed model product

> Take the measured champion through the model-management lifecycle. Retrain it appropriately,
> package it as a reusable MLflow model with a clear future-dates input contract, and make the model
> self-contained enough for a separate batch process to load it. Include a signature, input example,
> pinned dependencies, training provenance, and any artifacts required at inference time. Test the
> logged artifact by loading it back and scoring future dates before promotion. Register the
> validated model in Unity Catalog under `<target_catalog>.<target_schema>` and use a movable alias
> named `Champion` rather than coupling consumers to a numeric version. Implement and execute the
> lifecycle, then show the run, registered version, alias, signature, dependencies, lineage, and
> load-back validation result.

What the audience should see:

- An inference contract, signature, example, dependency environment, and training lineage.
- Load-back validation before registration or promotion.
- A UC model version promoted through the `Champion` alias.

Presenter transition:

> We now have a governed model product, not merely a Python object left in notebook memory. The next
> prompt asks Genie Code to create an independently executable consumer of that model.

## Act 4 — Build and execute batch deployment

> Create a separate, independently runnable batch-inference notebook for the registered champion.
> It must run interactively and as a serverless job without relying on state from the training
> notebook. Resolve the model through its Unity Catalog `Champion` alias, create the next 30 daily
> requests for every known store, score them, and publish an audited Delta prediction table in
> `<target_catalog>.<target_schema>`. Include enough model and scoring metadata to reproduce which
> model produced each row. Add a useful forecast visualization and executable assertions for row
> count, unique store/date keys, nulls, and non-finite predictions. Implement and run the notebook,
> then report the output table, model version, forecast range, and validation results.

What the audience should see:

- A clean session loading the model through an alias, with its own pinned dependencies.
- A governed prediction table containing model/version/scoring metadata.
- A forecast visualization and executed deployment-quality assertions.

## Act 5 — Operationalize the lifecycle as a customer-portable DAB

> Turn everything created in this session into a production-quality, customer-portable Databricks
> Asset Bundle. Preserve separate notebooks for exploration, training/registration, and batch
> inference. Create one serverless job for the full ordered lifecycle and another independently
> runnable serverless batch job. Parameterize the writable catalog, schema, source location,
> forecast horizon, and model name without notebook widgets. Make direct interactive runs and job
> runs both work, including their dependency environments. Do not embed this workspace's host,
> profile, user identity, catalog, or schema in source. Include clear README commands for a customer
> to validate, deploy, run the lifecycle, and rerun batch inference. Implement all bundle and source
> files in the current workspace folder, run strict DAB validation, and show the project tree,
> resource graph, validation result, and generic hand-off commands. Do not just describe the bundle.

What the audience should see:

- Source-controlled notebooks and resource definitions created from the completed analysis.
- Serverless task environments and correct task ordering.
- Portable deployment configuration with no presenter-specific identity.
- A full lifecycle job and an independently schedulable batch job.
- Successful strict bundle validation.

## Optional single-prompt version

Use this if the demo must fit into one interaction. The staged version is more legible for an
audience because it exposes the evidence at each lifecycle boundary.

> Starting from these two governed tables, act as the lead data scientist and implement this use
> case end to end: `samples.tpcds_sf1.store_sales` and
> `samples.tpcds_sf1.date_dim` should produce daily per-store net-paid revenue forecasts using
> `ss_net_paid` as the target in
> `<target_catalog>.<target_schema>`. Use serverless compute. Discover and validate the data, create a
> leakage-safe multi-series training asset with the latest 30 days as test and the preceding 30 days
> as validation, run exploratory analysis, compare credible forecasting
> baselines and seasonal models on a common future holdout, track the experiment and artifacts in
> MLflow, select and validate a champion, package it with a signature and pinned dependencies,
> register and promote it in Unity Catalog, and create and run independent audited batch inference
> for the next 30 days. Then operationalize the notebooks as a portable DAB with a full lifecycle
> job and an independently runnable batch job. Do not use notebook widgets or embed workspace/user
> details. Implement and run each safe phase rather than returning instructions, and finish with the
> measured results, created assets, strict bundle-validation output, and generic customer commands.

## Gold-standard reveal and comparison

Only after the scratch build, open this reference DAB. Use the following checklist to compare the
generated result without turning the live session into a repair exercise.

| Lifecycle boundary | Gold-standard evidence |
|---|---|
| Data contract | Daily `ds` × `store_id`, numeric target, complete spine, explicit `was_imputed` |
| Evaluation | Leakage-safe temporal splits and identical holdout across candidates |
| Experiment | Baselines plus seasonal challengers; SMAPE, WAPE, RMSE, MAE; nested MLflow runs |
| Evidence | Governed metrics/backtest tables and ranking/forecast plots |
| Model product | Pyfunc contract, signature, input example, pinned environment, bundled history |
| Governance | UC registration, lineage, validated version, movable `Champion` alias |
| Deployment | Independent batch load, audited output, row/key/null/finite assertions |
| Operations | Portable DAB, serverless environments, full lifecycle and batch-only jobs |
| Portability | Required customer catalog, deployer-scoped schema, no widgets or embedded identity |

Reference hand-off commands:

```bash
databricks bundle validate --strict -t dev --var="catalog=<customer_catalog>"
databricks bundle deploy -t dev --var="catalog=<customer_catalog>"
databricks bundle run forecasting_lifecycle -t dev --var="catalog=<customer_catalog>"
databricks bundle run forecasting_batch_inference -t dev --var="catalog=<customer_catalog>"
```

## Closing narration

> Genie Code started from a business outcome and governed source data, then created analysis,
> experiments, a registered model, batch deployment, and deployment-as-code. The gold-standard DAB
> gives the customer a tested reference for the same lifecycle. Together they show both the speed of
> getting from zero to a serious ML solution and the concrete engineering bar for productionization.
