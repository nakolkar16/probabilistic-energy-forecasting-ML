from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import shutil
import joblib
import numpy as np
import pandas as pd
import yaml


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)


def load_params() -> dict[str, Any]:
    with open(Path("params.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _path_cfg(cfg: dict[str, Any]) -> dict[str, Path]:
    infer_cfg = cfg.get("inference", {})
    ds_cfg = cfg.get("dataset", {})
    return {
        "input_path": Path(infer_cfg.get("input_path", ds_cfg.get("test_path", "data/modeling/test.parquet"))),
        "output_dir": Path(infer_cfg.get("output_dir", "artifacts/predictions/models")),
        "manifest_path": Path(infer_cfg.get("manifest_path", "artifacts/predictions/prediction_manifest.csv")),
        "legacy_output_path": Path(infer_cfg.get("output_path", "artifacts/predictions/champion_predictions.parquet")),
        "metadata_path": Path(
            infer_cfg.get("feature_metadata_path", ds_cfg.get("feature_metadata_path", "data/modeling/feature_metadata.json"))
        ),
        "stage_dir": Path("artifacts/stages/07_predict"),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _resolve_model(model_map: dict[Any, Any], q: float) -> Any:
    if q in model_map:
        return model_map[q]
    for k, v in model_map.items():
        try:
            if np.isclose(float(k), q):
                return v
        except Exception:
            continue
    raise KeyError(f"No model found for quantile={q}")


def _enforce_quantile_order(pred_map: dict[float, np.ndarray], quantiles: list[float]) -> dict[float, np.ndarray]:
    ordered = np.sort(np.column_stack([pred_map[q] for q in quantiles]), axis=1)
    return {q: ordered[:, i] for i, q in enumerate(quantiles)}


def _resolve_model_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    infer_cfg = cfg.get("inference", {})
    model_paths_cfg = infer_cfg.get("model_paths", {})
    if isinstance(model_paths_cfg, dict) and model_paths_cfg:
        return {str(name): Path(path) for name, path in model_paths_cfg.items()}

    fallback_model = infer_cfg.get("model_path", "artifacts/models/champion.joblib")
    return {"champion": Path(fallback_model)}


def predict() -> None:
    params = load_params()
    cfg = params["model_training"]
    paths = _path_cfg(cfg)
    target_col = str(cfg["target_col"])
    time_col = str(cfg["time_col"])
    primary_model = str(cfg.get("inference", {}).get("primary_model", "champion"))
    model_paths = _resolve_model_paths(cfg)

    with open(paths["metadata_path"], "r", encoding="utf-8") as f:
        metadata = json.load(f)
    target_lower = float(metadata.get("target_lower_bound", -np.inf))
    target_upper = float(metadata.get("target_upper_bound", np.inf))

    df = pd.read_parquet(paths["input_path"]).copy()
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    quantile_cols_all: list[dict[str, Any]] = []
    champion_output: Path | None = None

    for model_name, model_path in model_paths.items():
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found for '{model_name}': {model_path}")

        bundle = joblib.load(model_path)
        feature_cols = list(bundle["feature_cols"])
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required features for model '{model_name}': {missing}")

        quantiles = sorted(float(q) for q in bundle["quantiles"])
        q_low, q_high = quantiles[0], quantiles[-1]
        qhat = float(bundle.get("qhat", 0.0))

        X = df[feature_cols]
        pred_map: dict[float, np.ndarray] = {}
        for q in quantiles:
            model = _resolve_model(bundle["models"], q)
            pred_map[q] = np.asarray(model.predict(X), dtype=float)

        pred_map = _enforce_quantile_order(pred_map, quantiles)
        pred_map[q_low] = pred_map[q_low] - qhat
        pred_map[q_high] = pred_map[q_high] + qhat
        pred_map = _enforce_quantile_order(pred_map, quantiles)
        pred_map = {q: np.clip(pred, target_lower, target_upper) for q, pred in pred_map.items()}

        out_df = pd.DataFrame(index=df.index)
        if time_col in df.columns:
            out_df[time_col] = df[time_col]
        if target_col in df.columns:
            out_df[target_col] = df[target_col]

        for q in quantiles:
            q_label = int(round(q * 100))
            col_name = f"pred_q{q_label}"
            out_df[col_name] = pred_map[q]
            quantile_cols_all.append({"model_name": model_name, "quantile": q, "col": col_name})

        out_df["pred_interval_width"] = pred_map[q_high] - pred_map[q_low]
        out_df["model_name"] = model_name
        out_df["model_family"] = str(bundle.get("family", "unknown"))

        output_path = paths["output_dir"] / f"{model_name}_predictions.parquet"
        out_df.to_parquet(output_path, index=False)

        if model_name == primary_model:
            champion_output = output_path

        manifest_rows.append(
            {
                "model_name": model_name,
                "model_path": str(model_path),
                "prediction_path": str(output_path),
                "model_family": str(bundle.get("family", "unknown")),
                "quantiles": ",".join(str(q) for q in quantiles),
                "qhat": qhat,
                "rows_scored": int(len(out_df)),
            }
        )

    stage_dir = paths["stage_dir"]
    stage_dir.mkdir(parents=True, exist_ok=True)
    manifest_df = pd.DataFrame(manifest_rows).sort_values("model_name").reset_index(drop=True)
    paths["manifest_path"].parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(paths["manifest_path"], index=False)

    if champion_output is None and manifest_rows:
        champion_output = Path(manifest_rows[0]["prediction_path"])
    if champion_output is not None:
        paths["legacy_output_path"].parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(champion_output, paths["legacy_output_path"])

    summary = {
        "rows_scored": int(len(df)),
        "n_models_scored": int(len(manifest_rows)),
        "model_names": [r["model_name"] for r in manifest_rows],
        "primary_model": primary_model,
        "input_path": str(paths["input_path"]),
        "output_dir": str(paths["output_dir"]),
        "manifest_path": str(paths["manifest_path"]),
        "legacy_output_path": str(paths["legacy_output_path"]),
    }
    _write_json(stage_dir / "metrics.json", summary)
    pd.DataFrame(quantile_cols_all).drop_duplicates().to_csv(stage_dir / "prediction_columns.csv", index=False)
    manifest_df.to_csv(stage_dir / "model_manifest.csv", index=False)

    LOGGER.info("Predictions written for %s model(s) into %s", len(manifest_rows), paths["output_dir"])


if __name__ == "__main__":
    predict()
