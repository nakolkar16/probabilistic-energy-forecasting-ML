"""Reproducibility utilities."""
import hashlib
import json
import random
from typing import Any

import numpy as np
import pandas as pd


def set_global_seed(seed: int = 42) -> None:
    """Set common random seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)


def hash_config(config: dict[str, Any]) -> str:
    """Generate a deterministic hash of the experiment configuration."""
    config_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:12]


def hash_dataframe(df: pd.DataFrame) -> str:
    """Generate a deterministic hash of a pandas DataFrame."""
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()[:12]