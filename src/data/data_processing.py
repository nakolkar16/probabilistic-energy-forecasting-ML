import logging
from pathlib import Path
import pandas as pd
import yaml

from src.utils.reproducibility import set_global_seed

from src.data.column_schema import to_canonical
from src.data.load_data import load_consumption, load_generation

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_params():
    with open(Path("params.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_timestamp_series(series: pd.Series, source_tz: str, target_tz: str) -> pd.Series:
    if series.dt.tz is None:
        return series.dt.tz_localize(source_tz, ambiguous="infer", nonexistent="shift_forward").dt.tz_convert(target_tz)
    return series.dt.tz_convert(target_tz)


def clean_consumption(df, source_tz: str, target_tz: str, step_hours: int):
    """
    Cleans the consumption DataFrame.
    Args:
        df: Raw consumption DataFrame.
    Returns:
        Cleaned DataFrame.
    """

    logging.info("Cleaning consumption data.")
    df_clean = df.copy()
    # Parse datetime columns
    for col in ["Start date", "End date"]:
        df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
    # Canonical modeling layer: rename once right after parsing.
    df_clean = to_canonical(df_clean)

    # Convert numeric columns
    date_cols = ["timestamp", "end_timestamp"]
    numeric_cols = [col for col in df_clean.columns if col not in date_cols]
    df_clean[numeric_cols] = df_clean[numeric_cols].apply(pd.to_numeric, errors="coerce")
    # Localize to UTC
    df_clean["timestamp"] = normalize_timestamp_series(df_clean["timestamp"], source_tz, target_tz)
    df_clean["end_timestamp"] = df_clean["timestamp"] + pd.Timedelta(hours=step_hours)
    
    logging.info("Consumption data cleaned.")
    return df_clean

def clean_generation(df, source_tz: str, target_tz: str, step_hours: int):
    """
    Cleans the generation DataFrame.
    Args:
        df: Raw generation DataFrame.
    Returns:
        Cleaned DataFrame.
    """

    logging.info("Cleaning generation data.")
    df_clean = df.copy()
    # Parse datetime columns
    for col in ["Start date", "End date"]:
        df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
    # Canonical modeling layer: rename once right after parsing.
    df_clean = to_canonical(df_clean)
    # Convert numeric columns
    date_cols = ["timestamp", "end_timestamp"]
    numeric_cols = [col for col in df_clean.columns if col not in date_cols]
    df_clean[numeric_cols] = df_clean[numeric_cols].apply(pd.to_numeric, errors="coerce")
    # Localize to UTC
    df_clean["timestamp"] = normalize_timestamp_series(df_clean["timestamp"], source_tz, target_tz)
    df_clean["end_timestamp"] = df_clean["timestamp"] + pd.Timedelta(hours=step_hours)

    logging.info("Generation data cleaned.")
    return df_clean

def validate_hourly_utc_continuity(
    df: pd.DataFrame,
    time_col: str = "timestamp",
    end_time_col: str = "end_timestamp",
    step_hours: int = 1,
    target_tz: str = "UTC",
) -> None:
    timestamps = pd.to_datetime(df[time_col], utc=False)
    if timestamps.dt.tz is None:
        raise ValueError(f"{time_col} must be timezone-aware after merge.")

    actual_tz = str(timestamps.dt.tz)
    if target_tz.upper() == "UTC" and actual_tz not in {"UTC", "datetime64[ns, UTC]"}:
        raise ValueError(f"{time_col} must be UTC after merge, got {actual_tz}.")

    if timestamps.duplicated().any():
        duplicate_count = int(timestamps.duplicated().sum())
        raise ValueError(f"{time_col} has {duplicate_count} duplicate rows after merge.")

    sorted_ts = timestamps.sort_values().reset_index(drop=True)
    expected_step = pd.Timedelta(hours=step_hours)
    deltas = sorted_ts.diff().dropna()
    bad_steps = deltas[deltas != expected_step]
    if not bad_steps.empty:
        first_bad_idx = int(bad_steps.index[0])
        raise ValueError(
            f"{time_col} is not continuous hourly UTC after merge. "
            f"First bad step: {sorted_ts.iloc[first_bad_idx - 1]} -> {sorted_ts.iloc[first_bad_idx]} "
            f"({bad_steps.iloc[0]})."
        )

    expected_end = timestamps + expected_step
    end_timestamps = pd.to_datetime(df[end_time_col], utc=False)
    if not end_timestamps.equals(expected_end):
        mismatch_count = int((end_timestamps != expected_end).sum())
        raise ValueError(f"{end_time_col} does not equal {time_col} + {step_hours} hour(s) for {mismatch_count} rows.")


def merge_datasets(
    df_consumption: pd.DataFrame,
    df_generation: pd.DataFrame,
    step_hours: int = 1,
    target_tz: str = "UTC",
) -> pd.DataFrame:
    """
    Merges consumption and generation DataFrames.
    Args:
        df_consumption: Cleaned consumption DataFrame.
        df_generation: Cleaned generation DataFrame.
    Returns:
        Merged DataFrame.
    """

    logging.info("Merging datasets.")

    # Avoid pandas _x/_y suffixes by explicitly prefixing overlapping non-key columns.
    key_cols = {"timestamp", "end_timestamp"}
    overlapping_cols = sorted((set(df_consumption.columns) & set(df_generation.columns)) - key_cols)
    if overlapping_cols:
        logging.info(
            "Overlapping feature columns found (%d): %s",
            len(overlapping_cols),
            overlapping_cols,
        )
        df_consumption = df_consumption.rename(columns={col: f"cons_{col}" for col in overlapping_cols})
        df_generation = df_generation.rename(columns={col: f"gen_{col}" for col in overlapping_cols})

    df_merged = df_consumption.merge(
        df_generation,
        on=["timestamp", "end_timestamp"],
        how="inner"
    )
    df_merged = df_merged.sort_values("timestamp").reset_index(drop=True)
    validate_hourly_utc_continuity(df_merged, step_hours=step_hours, target_tz=target_tz)

    logging.info("Datasets merged successfully.")
    return df_merged

def process_data(output_path: str, source_tz: str, target_tz: str, step_hours: int, seed: int):
    """
    Processes the data from loading to saving.
    Args:
        output_path: Path to save the processed data.
    """
    logging.info("Starting data processing.")
    set_global_seed(seed)
    
    # Load data
    consumption = load_consumption()
    generation = load_generation()
    # Clean data
    df_consumption_clean = clean_consumption(consumption, source_tz, target_tz, step_hours)
    df_generation_clean = clean_generation(generation, source_tz, target_tz, step_hours)
    # Merge
    df_merged = merge_datasets(df_consumption_clean, df_generation_clean, step_hours=step_hours, target_tz=target_tz)
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_parquet(output_path, index=False)
    logging.info(f"Processed data saved to {output_path}.")

if __name__ == "__main__":
    params = load_params()
    dp = params["data_processing"]
    seed = int(params["global"]["seed"])

    process_data(
        output_path=dp["output_path"],
        source_tz=dp["source_tz"],
        target_tz=dp["target_tz"],
        step_hours=int(dp["step_hours"]),
        seed=seed,
    )
