from __future__ import annotations
from collections.abc import Iterable

import pandas as pd


# Raw SMARD headers are preserved at ingestion for source traceability.
RAW_TO_CANONICAL = {
    "Start date": "timestamp",
    "End date": "end_timestamp",
    "grid load [MWh] Calculated resolutions": "grid_load_mwh",
    "Grid load incl. hydro pumped storage [MWh] Calculated resolutions": "grid_load_incl_pumped_storage_mwh",
    "Hydro pumped storage [MWh] Calculated resolutions": "hydro_pumped_storage_mwh",
    "Residual load [MWh] Calculated resolutions": "residual_load_mwh",
    "Biomass [MWh] Calculated resolutions": "biomass_mwh",
    "Hydropower [MWh] Calculated resolutions": "hydropower_mwh",
    "Wind offshore [MWh] Calculated resolutions": "wind_offshore_mwh",
    "Wind onshore [MWh] Calculated resolutions": "wind_onshore_mwh",
    "Photovoltaics [MWh] Calculated resolutions": "photovoltaics_mwh",
    "Other renewable [MWh] Calculated resolutions": "other_renewable_mwh",
    "Nuclear [MWh] Calculated resolutions": "nuclear_mwh",
    "Lignite [MWh] Calculated resolutions": "lignite_mwh",
    "Hard coal [MWh] Calculated resolutions": "hard_coal_mwh",
    "Fossil gas [MWh] Calculated resolutions": "fossil_gas_mwh",
    "Other conventional [MWh] Calculated resolutions": "other_conventional_mwh",
}


CANONICAL_TO_LABEL = {
    "timestamp": "Timestamp",
    "end_timestamp": "End Timestamp",
    "grid_load_mwh": "Grid Load (MWh)",
    "grid_load_incl_pumped_storage_mwh": "Grid Load incl. Pumped Storage (MWh)",
    "hydro_pumped_storage_mwh": "Hydro Pumped Storage (MWh)",
    "residual_load_mwh": "Residual Load (MWh)",
    "biomass_mwh": "Biomass (MWh)",
    "hydropower_mwh": "Hydropower (MWh)",
    "wind_offshore_mwh": "Wind Offshore (MWh)",
    "wind_onshore_mwh": "Wind Onshore (MWh)",
    "photovoltaics_mwh": "Photovoltaics (MWh)",
    "other_renewable_mwh": "Other Renewable (MWh)",
    "nuclear_mwh": "Nuclear (MWh)",
    "lignite_mwh": "Lignite (MWh)",
    "hard_coal_mwh": "Hard Coal (MWh)",
    "fossil_gas_mwh": "Fossil Gas (MWh)",
    "other_conventional_mwh": "Other Conventional (MWh)",
    "calculated_renewable_generation_mwh": "Calculated Renewable Generation (MWh)",
    "calculated_residual_load_mwh": "Calculated Residual Load (MWh)",
    "residual_share_pct": "Residual Load Share (%)",
    "monthly_grid_load": "Monthly Grid Load (MWh)",
    "monthly_residual_load": "Monthly Residual Load (MWh)",
}


def to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with raw SMARD headers mapped to canonical names."""
    return df.rename(columns=RAW_TO_CANONICAL)


def label_for(column_name: str) -> str:
    """Return a human-friendly label for charting."""
    return CANONICAL_TO_LABEL.get(column_name, column_name)


def labels_for(column_names: Iterable[str]) -> list[str]:
    """Map canonical names to presentation labels in order."""
    return [label_for(name) for name in column_names]
