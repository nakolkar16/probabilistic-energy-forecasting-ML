import json
import sys
import yaml
from pathlib import Path
from typing import Any

import great_expectations as gx
import pandas as pd

def load_params():
    with open(Path("params.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_pandas_datasource(context: Any):
    """Return a pandas datasource across GX API versions."""
    if hasattr(context, "sources"):  # GX 0.18.x style
        return context.sources.add_or_update_pandas(name="residual_load_source")
    return context.data_sources.add_or_update_pandas(name="residual_load_source")


def _build_batch_request(data_asset: Any, df: pd.DataFrame):
    """Build a batch request across GX API versions."""
    try:
        # GX 0.18.x style
        return data_asset.build_batch_request(dataframe=df)
    except TypeError:
        # GX 1.x style
        return data_asset.build_batch_request(options={"dataframe": df})


def _get_or_create_validator(context: Any, batch_request: Any, suite_name: str):
    """Load suite if it exists; otherwise create it."""
    try:
        return context.get_validator(
            batch_request=batch_request,
            expectation_suite_name=suite_name,
        )
    except Exception:
        return context.get_validator(
            batch_request=batch_request,
            create_expectation_suite_with_name=suite_name,
        )


def run_validation(input_path: str, output_path: str, metrics_path: str) -> bool:
    """Run GX validation and write metrics. Returns True if all passed."""

    # Load data
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df):,} rows from {input_path}")

    # Set up GX context and run checkpoint
    context = gx.get_context()

    datasource = _get_pandas_datasource(context)
    data_asset = datasource.add_dataframe_asset(name="residual_load_hourly")
    batch_request = _build_batch_request(data_asset, df)

    suite_name = "residual_load_quality_suite"
    checkpoint_name = "residual_load_checkpoint"

    # Guardrail check for ordering (more reliable than GX increasing expectation for tz-aware datetimes).
    if "timestamp" in df.columns and not df["timestamp"].is_monotonic_increasing:
        raise ValueError("Data is not sorted by timestamp.")

    # Initialize/refresh suite expectations, then run validation.
    validator = _get_or_create_validator(context, batch_request, suite_name)
    validator.expectation_suite.expectations = []

    validator.expect_column_to_exist("timestamp")
    validator.expect_column_to_exist("end_timestamp")
    validator.expect_column_to_exist("residual_load_mwh")

    validator.expect_column_values_to_not_be_null("timestamp")
    validator.expect_column_values_to_not_be_null("end_timestamp")
    validator.expect_column_values_to_not_be_null("residual_load_mwh")

    validator.expect_column_values_to_be_unique("timestamp")
    validator.expect_column_pair_values_a_to_be_greater_than_b(
        column_A="end_timestamp",
        column_B="timestamp",
    )
    try:
        validator.save_expectation_suite()
    except Exception:
        if hasattr(context, "suites") and hasattr(context.suites, "add_or_update"):
            context.suites.add_or_update(
                validator.get_expectation_suite(
                    discard_failed_expectations=False,
                    discard_result_format_kwargs=False,
                    discard_include_config_kwargs=False,
                    discard_catch_exceptions_kwargs=False,
                )
            )
        else:
            raise

    # Run validation. Use checkpoint when available (GX 0.18), otherwise direct validate (GX 1.x).
    if hasattr(context, "add_or_update_checkpoint"):
        checkpoint = context.add_or_update_checkpoint(
            name=checkpoint_name,
            validations=[
                {
                    "batch_request": batch_request,
                    "expectation_suite_name": suite_name,
                }
            ],
        )
        result = checkpoint.run()
        run_results = list(result.run_results.values())
        validation_result = run_results[0]["validation_result"]
        overall_success = result.success
    else:
        validation_result = validator.validate()
        overall_success = validation_result.success

    stats = validation_result.statistics
    metrics = {
        "total_expectations": stats["evaluated_expectations"],
        "successful_expectations": stats["successful_expectations"],
        "unsuccessful_expectations": stats["unsuccessful_expectations"],
        "success_percent": stats["success_percent"],
        "row_count": len(df),
        "overall_success": overall_success,
    }

    # Write metrics for DVC tracking
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Validation: {metrics['successful_expectations']}/{metrics['total_expectations']} "
          f"expectations passed ({metrics['success_percent']:.1f}%)")

    if overall_success:
        # Write validated data downstream
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"Validated data written to {output_path}")
    else:
        print("VALIDATION FAILED. Inspect data docs for details.")
        # Print failed expectations for quick debugging
        for r in validation_result.results:
            if not r.success:
                print(f"  FAILED: {r.expectation_config.expectation_type} "
                      f"on '{r.expectation_config.kwargs.get('column', 'table')}'")

    return overall_success


if __name__ == "__main__":
    params = load_params()
    v = params["validation"]

    success = run_validation(
        input_path=v["input_path"],
        output_path=v["output_path"],
        metrics_path=v["metrics_path"],
    )

    if v.get("build_data_docs", True):
        context = gx.get_context()
        if hasattr(context, "build_data_docs"):
            context.build_data_docs()

    if not success:
        sys.exit(1)
