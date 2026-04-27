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


def _infer_number_locale(file_path: Path, sep: str = ";") -> tuple[str, str]:
    sample = pd.read_csv(file_path, sep=sep, nrows=200, dtype=str)
    date_cols = {"Start date", "End date"}
    numeric_tokens = []
    for col in sample.columns:
        if col in date_cols:
            continue
        values = sample[col].dropna().astype(str).str.strip()
        numeric_tokens.extend(v for v in values if v and v != "-")

    for token in numeric_tokens:
        if "," in token and "." in token:
            if token.rfind(",") > token.rfind("."):
                return ",", "."
            return ".", ","

    comma_decimal_count = sum("," in token for token in numeric_tokens)
    dot_decimal_count = sum("." in token for token in numeric_tokens)
    if comma_decimal_count and not dot_decimal_count:
        return ",", "."
    return ".", ","


def _read_smard_csv(file_path: Path) -> pd.DataFrame:
    decimal, thousands = _infer_number_locale(file_path)
    return pd.read_csv(
        file_path,
        sep=";",
        decimal=decimal,
        thousands=thousands,
        na_values=["-"],
    )


def load_consumption() -> pd.DataFrame:
    file_path = _find_file("Actual_consumption*.csv")
    df = _read_smard_csv(file_path)
    return df


def load_generation() -> pd.DataFrame:
    file_path = _find_file("Actual_generation*.csv")
    df = _read_smard_csv(file_path)
    return df
