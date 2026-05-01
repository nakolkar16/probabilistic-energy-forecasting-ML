from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRUSTED_MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"


def trusted_model_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate

    resolved = candidate.resolve()
    trusted_dir = TRUSTED_MODEL_DIR.resolve()

    if resolved.suffix != ".joblib":
        raise ValueError(f"Model artifact must be a .joblib file: {path}")

    if not resolved.is_relative_to(trusted_dir):
        raise ValueError(f"Model artifact must live under {trusted_dir}: {path}")

    return resolved
