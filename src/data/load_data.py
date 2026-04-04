from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def _find_file(pattern: str) -> Path:
    matches = list(DATA_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file found matching: {pattern} in {DATA_DIR}")
    if len(matches) > 1:
        print(f"Multiple files found for {pattern}. Using: {matches[0].name}")
    return matches[0]


def load_consumption() -> pd.DataFrame:
    file_path = _find_file("Actual_consumption*.csv")
    df = pd.read_csv(file_path, sep=";")
    return df


def load_generation() -> pd.DataFrame:
    file_path = _find_file("Actual_generation*.csv")
    df = pd.read_csv(file_path, sep=";")
    return df