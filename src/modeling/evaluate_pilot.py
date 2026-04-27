from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)


def load_params() -> dict[str, Any]:
    with open(Path("params.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _path_cfg(cfg: dict[str, Any]) -> dict[str, Path]:
    infer_cfg = cfg.get("inference", {})
    return {
        "pred_dir": Path(infer_cfg.get("output_dir", "artifacts/predictions/models")),
        "manifest_path": Path(infer_cfg.get("manifest_path", "artifacts/predictions/prediction_manifest.csv")),
        "metrics_path": Path(infer_cfg.get("pilot_metrics_path", "artifacts/metrics/pilot_evaluation_metrics.json")),
        "stage_dir": Path("artifacts/stages/08_evaluate_pilot"),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    residual = y_true - y_pred
    return float(np.where(residual >= 0, q * residual, (q - 1) * residual).mean())


def _resolve_prediction_files(paths: dict[str, Path]) -> list[tuple[str, Path]]:
    if paths["manifest_path"].exists():
        manifest_df = pd.read_csv(paths["manifest_path"])
        rows: list[tuple[str, Path]] = []
        for _, rec in manifest_df.iterrows():
            model_name = str(rec["model_name"])
            pred_path = Path(str(rec["prediction_path"]))
            rows.append((model_name, pred_path))
        return rows

    rows = []
    for p in sorted(paths["pred_dir"].glob("*_predictions.parquet")):
        name = p.stem.replace("_predictions", "")
        rows.append((name, p))
    return rows


def _manifest_metrics(paths: dict[str, Path]) -> dict[str, dict[str, float]]:
    if not paths["manifest_path"].exists():
        return {}

    manifest_df = pd.read_csv(paths["manifest_path"])
    metrics: dict[str, dict[str, float]] = {}
    for _, rec in manifest_df.iterrows():
        model_name = str(rec["model_name"])
        model_metrics: dict[str, float] = {}
        for col in ["raw_crossing_rate", "repaired_crossing_rate"]:
            if col in manifest_df.columns and pd.notna(rec[col]):
                model_metrics[col] = float(rec[col])
        metrics[model_name] = model_metrics
    return metrics


def evaluate_pilot() -> None:
    params = load_params()
    cfg = params["model_training"]
    backtest_cfg = cfg.get("backtest", {})
    paths = _path_cfg(cfg)

    target_col = str(cfg["target_col"])
    quantiles = sorted(float(q) for q in cfg["quantiles"])
    q_low, q_med, q_high = quantiles[0], quantiles[len(quantiles) // 2], quantiles[-1]
    nominal_coverage = q_high - q_low
    coverage_tolerance = float(backtest_cfg.get("coverage_tolerance", 0.05))
    max_crossing_rate = float(backtest_cfg.get("max_crossing_rate", 0.0))
    min_fold_coverage = float(backtest_cfg.get("min_fold_coverage", nominal_coverage - coverage_tolerance))

    pred_files = _resolve_prediction_files(paths)
    if not pred_files:
        raise ValueError("No prediction files found for pilot evaluation.")
    manifest_metrics = _manifest_metrics(paths)

    quantile_rows_all: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for model_name, pred_path in pred_files:
        if not pred_path.exists():
            raise FileNotFoundError(f"Prediction file not found for model '{model_name}': {pred_path}")
        pred_df = pd.read_parquet(pred_path).copy()
        if target_col not in pred_df.columns:
            raise ValueError(f"Predictions file '{pred_path}' must include '{target_col}' for pilot evaluation.")

        quantile_cols = {q: f"pred_q{int(round(q * 100))}" for q in quantiles}
        missing_pred_cols = [c for c in quantile_cols.values() if c not in pred_df.columns]
        if missing_pred_cols:
            raise ValueError(f"Missing prediction columns in '{pred_path}': {missing_pred_cols}")

        y_true = pred_df[target_col].to_numpy()
        model_quantile_rows: list[dict[str, Any]] = []
        for q in quantiles:
            pred = pred_df[quantile_cols[q]].to_numpy()
            model_quantile_rows.append(
                {
                    "model_name": model_name,
                    "quantile": q,
                    "pinball_loss": pinball_loss(y_true, pred, q),
                    "mae": float(mean_absolute_error(y_true, pred)),
                    "rmse": rmse(y_true, pred),
                    "r2": float(r2_score(y_true, pred)),
                }
            )
        model_quantile_df = pd.DataFrame(model_quantile_rows).sort_values("quantile").reset_index(drop=True)
        quantile_rows_all.extend(model_quantile_rows)

        low_pred = pred_df[quantile_cols[q_low]].to_numpy()
        high_pred = pred_df[quantile_cols[q_high]].to_numpy()
        q_matrix = np.column_stack([pred_df[quantile_cols[q]].to_numpy() for q in quantiles])
        repaired_crossing_rate = float((np.diff(q_matrix, axis=1) < 0).any(axis=1).mean())
        crossing_rate = manifest_metrics.get(model_name, {}).get("raw_crossing_rate", repaired_crossing_rate)
        coverage = float(((low_pred <= y_true) & (y_true <= high_pred)).mean())
        interval_width = float((high_pred - low_pred).mean())
        coverage_gap = float(coverage - nominal_coverage)

        p50_row = model_quantile_df[model_quantile_df["quantile"] == q_med].iloc[0]
        summary = {
            "model_name": model_name,
            "prediction_path": str(pred_path),
            "rows_evaluated": int(len(pred_df)),
            "pinball_q10": float(model_quantile_df.iloc[0]["pinball_loss"]),
            "pinball_q50": float(p50_row["pinball_loss"]),
            "pinball_q90": float(model_quantile_df.iloc[-1]["pinball_loss"]),
            "pinball_mean": float(model_quantile_df["pinball_loss"].mean()),
            "mae_p50": float(p50_row["mae"]),
            "rmse_p50": float(p50_row["rmse"]),
            "r2_p50": float(p50_row["r2"]),
            "coverage": coverage,
            "coverage_gap": coverage_gap,
            "interval_width": interval_width,
            "crossing_rate": crossing_rate,
            "raw_crossing_rate": crossing_rate,
            "repaired_crossing_rate": repaired_crossing_rate,
            "coverage_gate_pass": bool(abs(coverage_gap) <= coverage_tolerance),
            "crossing_gate_pass": bool(crossing_rate <= max_crossing_rate),
            "min_coverage_gate_pass": bool(coverage >= min_fold_coverage),
        }
        summary["overall_gate_pass"] = bool(
            summary["coverage_gate_pass"] and summary["crossing_gate_pass"] and summary["min_coverage_gate_pass"]
        )
        summary_rows.append(summary)

    quantile_metrics_df = pd.DataFrame(quantile_rows_all).sort_values(["model_name", "quantile"]).reset_index(drop=True)
    summary_df = pd.DataFrame(summary_rows).sort_values("pinball_mean").reset_index(drop=True)
    best_row = summary_df.iloc[0]

    paths["metrics_path"].parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_models_evaluated": int(len(summary_df)),
        "best_model_by_pinball_mean": str(best_row["model_name"]),
        "best_pinball_mean": float(best_row["pinball_mean"]),
        "models": summary_df.to_dict(orient="records"),
    }
    _write_json(paths["metrics_path"], payload)

    stage_dir = paths["stage_dir"]
    stage_dir.mkdir(parents=True, exist_ok=True)
    quantile_metrics_df.to_csv(stage_dir / "quantile_metrics_by_model.csv", index=False)
    summary_df.to_csv(stage_dir / "model_summary_metrics.csv", index=False)
    _write_json(stage_dir / "metrics.json", payload)
    pd.DataFrame(
        [
            {"metric": "coverage_tolerance", "value": coverage_tolerance},
            {"metric": "max_crossing_rate", "value": max_crossing_rate},
            {"metric": "min_fold_coverage", "value": min_fold_coverage},
        ]
    ).to_csv(stage_dir / "gates.csv", index=False)

    LOGGER.info(
        "Pilot evaluation complete | models=%s | best=%s (pinball_mean=%.3f)",
        len(summary_df),
        best_row["model_name"],
        best_row["pinball_mean"],
    )


if __name__ == "__main__":
    evaluate_pilot()
