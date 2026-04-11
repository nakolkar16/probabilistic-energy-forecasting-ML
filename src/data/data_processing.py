import logging
from pathlib import Path

import pandas as pd

from src.data.ge_validation import validate_with_ge
from src.data.load_data import load_consumption, load_generation

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def validate_data(df: pd.DataFrame, key_columns: list[str]) -> None:
#     """
#     Validates the DataFrame for key structural integrity checks.
    
#     Args:
#         df: The DataFrame to validate.
#         key_columns: List of column names that must be present and valid.
    
#     Raises:
#         ValueError: If any validation fails.
#     """
#     logging.info("Starting data validation.")
    
#     # Check if key columns exist
#     missing_cols = [col for col in key_columns if col not in df.columns]
#     if missing_cols:
#         raise ValueError(f"Missing key columns: {missing_cols}")
    
#     # Check for nulls in key columns
#     null_counts = df[key_columns].isnull().sum()
#     if null_counts.sum() > 0:
#         raise ValueError(f"Null values found in key columns: {null_counts[null_counts > 0].to_dict()}")
    
#     # Check for duplicate timestamps (assuming 'timestamp' is key)
#     if 'timestamp' in df.columns:
#         dup_count = df['timestamp'].duplicated().sum()
#         if dup_count > 0:
#             raise ValueError(f"Duplicate timestamps found: {dup_count}")
    
#     # Check if sorted by timestamp
#     if 'timestamp' in df.columns and not df['timestamp'].is_monotonic_increasing:
#         raise ValueError("Data is not sorted by timestamp.")
    
#     # Skip missing hours check due to DST gaps (as per notebook)
    
#     logging.info("Data validation passed.")

def clean_consumption(df: pd.DataFrame) -> pd.DataFrame:
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
    
    # Rename columns
    df_clean = df_clean.rename(columns={
        "Start date": "timestamp",
        "End date": "end_timestamp",
        "Residual load [MWh] Calculated resolutions": "residual_load_mwh"
    })
    
    # Convert numeric columns
    df_clean["residual_load_mwh"] = pd.to_numeric(
        df_clean["residual_load_mwh"].astype(str).str.replace(",", "", regex=False), errors="coerce"
    )
    df_clean["grid load [MWh] Calculated resolutions"] = pd.to_numeric(
        df_clean["grid load [MWh] Calculated resolutions"].astype(str).str.replace(",", "", regex=False), errors="coerce"
    )
    
    # Localize to UTC
    df_clean['timestamp'] = df_clean['timestamp'].dt.tz_localize("Europe/Berlin", ambiguous="infer", nonexistent="shift_forward").dt.tz_convert('UTC')
    df_clean["end_timestamp"] = df_clean["timestamp"] + pd.Timedelta(hours=1)
    
    logging.info("Consumption data cleaned.")
    return df_clean

def clean_generation(df: pd.DataFrame) -> pd.DataFrame:
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
    
    # Rename columns
    df_clean = df_clean.rename(columns={
        "Start date": "timestamp",
        "End date": "end_timestamp",
    })
    
    # Convert numeric columns
    date_cols = ["timestamp", "end_timestamp"]
    numeric_cols = [col for col in df_clean.columns if col not in date_cols]
    df_clean[numeric_cols] = df_clean[numeric_cols].apply(
        lambda col: pd.to_numeric(col.astype(str).str.replace(",", "", regex=False), errors="coerce")
    )
    
    # Localize to UTC
    df_clean['timestamp'] = df_clean['timestamp'].dt.tz_localize("Europe/Berlin", ambiguous="infer", nonexistent="shift_forward").dt.tz_convert('UTC')
    df_clean["end_timestamp"] = df_clean["timestamp"] + pd.Timedelta(hours=1)
    
    logging.info("Generation data cleaned.")
    return df_clean

def merge_datasets(df_consumption: pd.DataFrame, df_generation: pd.DataFrame) -> pd.DataFrame:
    """
    Merges consumption and generation DataFrames.
    
    Args:
        df_consumption: Cleaned consumption DataFrame.
        df_generation: Cleaned generation DataFrame.
    
    Returns:
        Merged DataFrame.
    """
    logging.info("Merging datasets.")
    df_merged = df_consumption.merge(
        df_generation,
        on=["timestamp", "end_timestamp"],
        how="inner"
    )
    df_merged = df_merged.sort_values("timestamp").reset_index(drop=True)
    
    # Skip missing hours validation due to DST (as per notebook)
    
    logging.info("Datasets merged successfully.")
    return df_merged

def calculate_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates derived features for the DataFrame.
    
    Args:
        df: Merged DataFrame.
    
    Returns:
        DataFrame with added features.
    """
    logging.info("Calculating derived features.")
    renewable_cols = [
        "Wind offshore [MWh] Calculated resolutions",
        "Wind onshore [MWh] Calculated resolutions",
        "Photovoltaics [MWh] Calculated resolutions",
    ]
    df["calculated_renewable_generation_mwh"] = df[renewable_cols].sum(axis=1)
    df["calculated_residual_load_mwh"] = (
        df["grid load [MWh] Calculated resolutions"] - df["calculated_renewable_generation_mwh"]
    )
    logging.info("Derived features calculated.")
    return df

def process_data(output_path: str | Path) -> None:
    """
    Processes the data from loading to saving.
    
    Args:
        output_path: Path to save the processed data.
    """
    logging.info("Starting data processing.")
    
    # Load data
    consumption = load_consumption()
    generation = load_generation()
    
    # Clean data
    df_consumption_clean = clean_consumption(consumption)
    df_generation_clean = clean_generation(generation)
    
    # # Validate
    # validate_data(df_consumption_clean, ["timestamp", "end_timestamp", "residual_load_mwh"])
    # validate_data(df_generation_clean, ["timestamp", "end_timestamp"])
    # validate_with_ge(
    #     df_consumption_clean,
    #     ["timestamp", "end_timestamp", "residual_load_mwh"],
    #     "consumption",
    # )
    # validate_with_ge(df_generation_clean, ["timestamp", "end_timestamp"], "generation")
    
    # Merge
    df_merged = merge_datasets(df_consumption_clean, df_generation_clean)
    
    # Calculate features
    df_final = calculate_derived_features(df_merged)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(output_path, index=False)
    logging.info(f"Processed data saved to {output_path}.")

if __name__ == "__main__":
    output_path = "data/processed/merged_data.parquet"
    process_data(output_path)
