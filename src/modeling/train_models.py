from __future__ import annotations

import json
import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor
from scipy.stats import ttest_rel, wilcoxon
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.modeling.quantile_repair import quantile_crossing_rate, repair_quantile_order
from src.utils.reproducibility import set_global_seed

try:
    import mlflow
except Exception:  # pragma: no cover - optional runtime dependency in some environments
    mlflow = None

try:
    import optuna
    from optuna.trial import TrialState
except Exception:  # pragma: no cover - optional runtime dependency in some environments
    optuna = None
    TrialState = None

try:
    import shap
except Exception:  # pragma: no cover - optional runtime dependency in some environments
    shap = None

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)


def load_params() -> dict[str, Any]:
    with open(Path("params.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dataset_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    ds_cfg = cfg.get("dataset", {})
    return {
        "train_path": Path(ds_cfg.get("train_path", "data/modeling/train.parquet")),
        "test_path": Path(ds_cfg.get("test_path", "data/modeling/test.parquet")),
        "metadata_path": Path(ds_cfg.get("feature_metadata_path", "data/modeling/feature_metadata.json")),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_builtin(payload), f, indent=2)


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [_to_builtin(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    residual = y_true - y_pred
    return float(np.where(residual >= 0, q * residual, (q - 1) * residual).mean())


def interval_metrics(
    pred_map: dict[float, np.ndarray], y_true: np.ndarray, q_low: float, q_high: float
) -> dict[str, float]:
    coverage = float(((pred_map[q_low] <= y_true) & (y_true <= pred_map[q_high])).mean())
    width = float((pred_map[q_high] - pred_map[q_low]).mean())
    return {"coverage": coverage, "interval_width": width}


def split_proper_calibration(df: pd.DataFrame, frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) < 3:
        return df.iloc[:-1].copy(), df.iloc[-1:].copy()
    calib_size = int(round(len(df) * frac))
    calib_size = min(max(calib_size, 1), len(df) - 1)
    return df.iloc[:-calib_size].copy(), df.iloc[-calib_size:].copy()


def split_train_earlystop_calibration(
    df: pd.DataFrame,
    holdout_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    proper_train, calib = split_proper_calibration(df, holdout_frac)
    if len(proper_train) < 3:
        return proper_train, calib, calib
    model_train, early_stop = split_proper_calibration(proper_train, holdout_frac)
    return model_train, early_stop, calib


def conformal_qhat(y_true: np.ndarray, pred_low: np.ndarray, pred_high: np.ndarray, alpha: float) -> float:
    scores = np.maximum(pred_low - y_true, y_true - pred_high)
    if len(scores) == 0:
        return 0.0
    n = len(scores)
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    try:
        q_val = np.quantile(scores, q_level, method="higher")
    except TypeError:  # numpy compatibility fallback
        q_val = np.quantile(scores, q_level, interpolation="higher")
    return float(max(0.0, q_val))


def apply_conformal_interval(
    pred_map: dict[float, np.ndarray], quantiles: list[float], q_low: float, q_high: float, qhat: float
) -> dict[float, np.ndarray]:
    adjusted = {q: np.asarray(pred_map[q]).copy() for q in quantiles}
    adjusted[q_low] -= qhat
    adjusted[q_high] += qhat
    return adjusted


def summarize_fold_metrics(fold_df: pd.DataFrame) -> dict[str, float]:
    metric_cols = [
        "pinball_q10",
        "pinball_q50",
        "pinball_q90",
        "pinball_mean",
        "mae_p50",
        "rmse_p50",
        "r2_p50",
        "coverage",
        "interval_width",
        "coverage_gap",
        "crossing_rate",
        "raw_crossing_rate",
        "calibrated_crossing_rate",
        "repair_rate",
        "output_crossing_rate",
    ]
    out: dict[str, float] = {}
    for col in metric_cols:
        if col not in fold_df.columns:
            continue
        out[f"{col}_mean"] = float(fold_df[col].mean())
        out[f"{col}_std"] = float(fold_df[col].std(ddof=0))
    out["coverage_min"] = float(fold_df["coverage"].min())
    return out


def _coverage_lower_bound(ctx: dict[str, Any]) -> float:
    return float(ctx["nominal_coverage"]) - float(ctx["coverage_tolerance"])


def metrics_gate_pass(summary_dict: dict[str, float], ctx: dict[str, Any]) -> bool:
    crossing_metric = float(summary_dict.get("output_crossing_rate_mean", summary_dict["crossing_rate_mean"]))
    return bool(
        summary_dict["coverage_mean"] >= _coverage_lower_bound(ctx)
        and crossing_metric <= float(ctx["max_crossing_rate"])
        and summary_dict.get("coverage_min", 0.0) >= float(ctx["min_fold_coverage"])
    )


def hard_sanity_gate(summary_dict: dict[str, float], ctx: dict[str, Any]) -> bool:
    required = [
        "pinball_mean_mean",
        "coverage_mean",
        "coverage_min",
        "crossing_rate_mean",
        "interval_width_mean",
    ]
    if any(not np.isfinite(float(summary_dict.get(col, np.nan))) for col in required):
        return False
    crossing_metric = float(summary_dict.get("output_crossing_rate_mean", summary_dict["crossing_rate_mean"]))
    return bool(
        summary_dict["coverage_min"] >= float(ctx["baseline_min_coverage"])
        and crossing_metric <= float(ctx["baseline_max_crossing_rate"])
        and summary_dict["interval_width_mean"] > 0
    )


def compare_to_baseline(summary_dict: dict[str, float], baseline_ref: dict[str, float], ctx: dict[str, Any]) -> tuple[float, bool]:
    improvement = (
        (baseline_ref["pinball_mean"] - summary_dict["pinball_mean_mean"])
        / max(abs(baseline_ref["pinball_mean"]), 1e-6)
    )
    return float(improvement), metrics_gate_pass(summary_dict, ctx)


def tuning_objective_score(summary_dict: dict[str, float], ctx: dict[str, Any]) -> float:
    base = float(summary_dict["pinball_mean_mean"])
    coverage_excess = max(0.0, _coverage_lower_bound(ctx) - float(summary_dict["coverage_mean"]))
    overcoverage_width_penalty = (
        0.05
        * float(summary_dict.get("interval_width_mean", 0.0))
        * max(0.0, float(summary_dict["coverage_gap_mean"]))
    )
    crossing_metric = float(summary_dict.get("output_crossing_rate_mean", summary_dict["crossing_rate_mean"]))
    crossing_excess = max(0.0, crossing_metric - float(ctx["max_crossing_rate"]))
    return float(base + 3000.0 * coverage_excess + 3000.0 * crossing_excess + overcoverage_width_penalty)


def candidate_selection_score(summary_dict: dict[str, float], ctx: dict[str, Any]) -> float:
    base = float(summary_dict["pinball_mean_mean"])
    coverage_excess = max(0.0, _coverage_lower_bound(ctx) - float(summary_dict["coverage_mean"]))
    overcoverage_width_penalty = (
        0.05
        * float(summary_dict.get("interval_width_mean", 0.0))
        * max(0.0, float(summary_dict["coverage_gap_mean"]))
    )
    crossing_metric = float(summary_dict.get("output_crossing_rate_mean", summary_dict["crossing_rate_mean"]))
    crossing_excess = max(0.0, crossing_metric - float(ctx["max_crossing_rate"]))
    return float(base + 3000.0 * coverage_excess + 1000.0 * crossing_excess + overcoverage_width_penalty)


def paired_significance(
    fold_df_a: pd.DataFrame, fold_df_b: pd.DataFrame, metric_col: str = "pinball_mean"
) -> dict[str, float]:
    merged = fold_df_a[["fold", metric_col]].merge(
        fold_df_b[["fold", metric_col]], on="fold", suffixes=("_a", "_b")
    )
    if len(merged) < 2:
        return {
            "n_folds": int(len(merged)),
            "metric": metric_col,
            "mean_delta_b_minus_a": np.nan,
            "ttest_stat": np.nan,
            "ttest_pvalue": np.nan,
            "wilcoxon_stat": np.nan,
            "wilcoxon_pvalue": np.nan,
        }

    a = merged[f"{metric_col}_a"].to_numpy()
    b = merged[f"{metric_col}_b"].to_numpy()
    delta = b - a
    t_stat, t_p = ttest_rel(a, b, nan_policy="omit")
    try:
        w_stat, w_p = wilcoxon(delta, alternative="greater")
    except ValueError:
        w_stat, w_p = (np.nan, np.nan)

    return {
        "n_folds": int(len(merged)),
        "metric": metric_col,
        "mean_delta_b_minus_a": float(np.mean(delta)),
        "ttest_stat": float(t_stat),
        "ttest_pvalue": float(t_p),
        "wilcoxon_stat": float(w_stat) if np.isfinite(w_stat) else np.nan,
        "wilcoxon_pvalue": float(w_p) if np.isfinite(w_p) else np.nan,
    }


def select_train_window(df: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    if max_rows is None or int(max_rows) >= len(df):
        return df.copy()
    return df.iloc[-int(max_rows) :].copy()


def make_quantile_model(cfg: dict[str, Any], q: float, alpha_override: float | None = None) -> Pipeline:
    default_alpha = float(cfg["quantile_regression"]["alpha"])
    alpha_value = default_alpha if alpha_override is None else float(alpha_override)
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                QuantileRegressor(
                    quantile=q,
                    alpha=alpha_value,
                    solver=str(cfg["quantile_regression"]["solver"]),
                ),
            ),
        ]
    )


def get_lgbm_params(
    cfg: dict[str, Any], seed: int, param_overrides: dict[str, Any] | None = None
) -> tuple[str, dict[str, Any]]:
    lgbm_cfg_local = dict(cfg.get("lightgbm", {}))
    objective = str(lgbm_cfg_local.get("objective", "quantile")).lower()
    params_local = {
        "learning_rate": float(lgbm_cfg_local.get("learning_rate", 0.05)),
        "n_estimators": int(lgbm_cfg_local.get("n_estimators", 300)),
        "num_leaves": int(lgbm_cfg_local.get("num_leaves", 31)),
        "min_child_samples": int(lgbm_cfg_local.get("min_child_samples", 20)),
        "subsample": float(lgbm_cfg_local.get("subsample", 0.9)),
        "colsample_bytree": float(lgbm_cfg_local.get("colsample_bytree", 0.9)),
        "reg_alpha": float(lgbm_cfg_local.get("reg_alpha", 0.0)),
        "reg_lambda": float(lgbm_cfg_local.get("reg_lambda", 0.0)),
        "random_state": int(lgbm_cfg_local.get("random_state", seed)),
        "n_jobs": int(lgbm_cfg_local.get("n_jobs", -1)),
        "verbosity": -1,
    }
    if param_overrides:
        params_local.update(param_overrides)
    params_local["n_estimators"] = int(params_local["n_estimators"])
    params_local["num_leaves"] = int(params_local["num_leaves"])
    params_local["min_child_samples"] = int(params_local["min_child_samples"])
    return objective, params_local


def fit_lgbm_quantile(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    q: float,
    objective: str,
    model_params: dict[str, Any],
    early_stopping_rounds: int,
    X_valid: pd.DataFrame | None = None,
    y_valid: np.ndarray | None = None,
) -> LGBMRegressor:
    model = LGBMRegressor(objective=objective, alpha=q, **model_params)
    fit_kwargs: dict[str, Any] = {}
    if X_valid is not None and y_valid is not None and int(early_stopping_rounds) > 0:
        fit_kwargs["eval_set"] = [(X_valid, y_valid)]
        fit_kwargs["eval_metric"] = "quantile"
        fit_kwargs["callbacks"] = [
            lgb.early_stopping(stopping_rounds=int(early_stopping_rounds), verbose=False)
        ]
    model.fit(X_train, y_train, **fit_kwargs)
    return model


def _safe_n_splits(n_splits: int, n_rows: int) -> int:
    return max(2, min(int(n_splits), max(2, n_rows - 1)))


def run_quantile_backtest(
    source_df: pd.DataFrame,
    feature_cols: list[str],
    n_splits: int,
    cfg: dict[str, Any],
    ctx: dict[str, Any],
    quantile_alpha: float | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    n_splits = _safe_n_splits(n_splits, len(source_df))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_rows: list[dict[str, float]] = []
    calib_frac = float(cfg["split"].get("val_fraction_within_train", 0.2))

    target_col = str(ctx["target_col"])
    quantiles = list(ctx["quantiles"])
    q_low, q_med, q_high = float(ctx["q_low"]), float(ctx["q_med"]), float(ctx["q_high"])
    nominal_coverage = float(ctx["nominal_coverage"])
    for fold, (tr_idx, te_idx) in enumerate(tscv.split(source_df), start=1):
        fold_train = source_df.iloc[tr_idx].copy()
        fold_test = source_df.iloc[te_idx].copy()
        model_train, early_stop, calib = split_train_earlystop_calibration(fold_train, calib_frac)

        X_train = model_train[feature_cols]
        y_train_local = model_train[target_col]
        X_valid = early_stop[feature_cols]
        y_valid = early_stop[target_col].to_numpy()
        X_cal = calib[feature_cols]
        y_cal = calib[target_col].to_numpy()
        X_test = fold_test[feature_cols]
        y_test_local = fold_test[target_col].to_numpy()

        pred_cal: dict[float, np.ndarray] = {}
        pred_test: dict[float, np.ndarray] = {}
        for q in quantiles:
            model = make_quantile_model(cfg, q, alpha_override=quantile_alpha)
            model.fit(X_train, y_train_local)
            pred_cal[q] = model.predict(X_cal)
            pred_test[q] = model.predict(X_test)

        qhat = conformal_qhat(y_cal, pred_cal[q_low], pred_cal[q_high], alpha=1.0 - nominal_coverage)
        raw_crossing = quantile_crossing_rate(pred_test, quantiles)
        pred_test = apply_conformal_interval(pred_test, quantiles, q_low, q_high, qhat)
        calibrated_crossing = quantile_crossing_rate(pred_test, quantiles)
        pred_test, repair_rate = repair_quantile_order(pred_test, quantiles)
        output_crossing = quantile_crossing_rate(pred_test, quantiles)

        interval = interval_metrics(pred_test, y_test_local, q_low, q_high)
        pin_q10 = pinball_loss(y_test_local, pred_test[q_low], q_low)
        pin_q50 = pinball_loss(y_test_local, pred_test[q_med], q_med)
        pin_q90 = pinball_loss(y_test_local, pred_test[q_high], q_high)

        fold_rows.append(
            {
                "fold": fold,
                "pinball_q10": pin_q10,
                "pinball_q50": pin_q50,
                "pinball_q90": pin_q90,
                "pinball_mean": float(np.mean([pin_q10, pin_q50, pin_q90])),
                "mae_p50": float(mean_absolute_error(y_test_local, pred_test[q_med])),
                "rmse_p50": rmse(y_test_local, pred_test[q_med]),
                "r2_p50": float(r2_score(y_test_local, pred_test[q_med])),
                "coverage": interval["coverage"],
                "interval_width": interval["interval_width"],
                "coverage_gap": float(interval["coverage"] - nominal_coverage),
                "crossing_rate": output_crossing,
                "raw_crossing_rate": raw_crossing,
                "calibrated_crossing_rate": calibrated_crossing,
                "repair_rate": repair_rate,
                "output_crossing_rate": output_crossing,
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    return fold_df, summarize_fold_metrics(fold_df)


def run_lgbm_backtest(
    source_df: pd.DataFrame,
    feature_cols: list[str],
    n_splits: int,
    cfg: dict[str, Any],
    ctx: dict[str, Any],
    param_overrides: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    n_splits = _safe_n_splits(n_splits, len(source_df))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_rows: list[dict[str, float]] = []
    calib_frac = float(cfg["split"].get("val_fraction_within_train", 0.2))
    early_stopping_rounds = int(cfg.get("lightgbm_tuning", {}).get("early_stopping_rounds", 0))

    target_col = str(ctx["target_col"])
    quantiles = list(ctx["quantiles"])
    q_low, q_med, q_high = float(ctx["q_low"]), float(ctx["q_med"]), float(ctx["q_high"])
    nominal_coverage = float(ctx["nominal_coverage"])
    seed = int(ctx["seed"])

    objective, model_params = get_lgbm_params(cfg, seed, param_overrides)

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(source_df), start=1):
        fold_train = source_df.iloc[tr_idx].copy()
        fold_test = source_df.iloc[te_idx].copy()
        model_train, early_stop, calib = split_train_earlystop_calibration(fold_train, calib_frac)

        X_train = model_train[feature_cols]
        y_train_local = model_train[target_col]
        X_valid = early_stop[feature_cols]
        y_valid = early_stop[target_col].to_numpy()
        X_cal = calib[feature_cols]
        y_cal = calib[target_col].to_numpy()
        X_test = fold_test[feature_cols]
        y_test_local = fold_test[target_col].to_numpy()

        pred_cal: dict[float, np.ndarray] = {}
        pred_test: dict[float, np.ndarray] = {}
        for q in quantiles:
            model = fit_lgbm_quantile(
                X_train=X_train,
                y_train=y_train_local,
                q=q,
                objective=objective,
                model_params=model_params,
                early_stopping_rounds=early_stopping_rounds,
                X_valid=X_valid,
                y_valid=y_valid,
            )
            pred_cal[q] = model.predict(X_cal)
            pred_test[q] = model.predict(X_test)

        qhat = conformal_qhat(y_cal, pred_cal[q_low], pred_cal[q_high], alpha=1.0 - nominal_coverage)
        raw_crossing = quantile_crossing_rate(pred_test, quantiles)
        pred_test = apply_conformal_interval(pred_test, quantiles, q_low, q_high, qhat)
        calibrated_crossing = quantile_crossing_rate(pred_test, quantiles)
        pred_test, repair_rate = repair_quantile_order(pred_test, quantiles)
        output_crossing = quantile_crossing_rate(pred_test, quantiles)

        interval = interval_metrics(pred_test, y_test_local, q_low, q_high)
        pin_q10 = pinball_loss(y_test_local, pred_test[q_low], q_low)
        pin_q50 = pinball_loss(y_test_local, pred_test[q_med], q_med)
        pin_q90 = pinball_loss(y_test_local, pred_test[q_high], q_high)

        fold_rows.append(
            {
                "fold": fold,
                "pinball_q10": pin_q10,
                "pinball_q50": pin_q50,
                "pinball_q90": pin_q90,
                "pinball_mean": float(np.mean([pin_q10, pin_q50, pin_q90])),
                "mae_p50": float(mean_absolute_error(y_test_local, pred_test[q_med])),
                "rmse_p50": rmse(y_test_local, pred_test[q_med]),
                "r2_p50": float(r2_score(y_test_local, pred_test[q_med])),
                "coverage": interval["coverage"],
                "interval_width": interval["interval_width"],
                "coverage_gap": float(interval["coverage"] - nominal_coverage),
                "crossing_rate": output_crossing,
                "raw_crossing_rate": raw_crossing,
                "calibrated_crossing_rate": calibrated_crossing,
                "repair_rate": repair_rate,
                "output_crossing_rate": output_crossing,
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    return fold_df, summarize_fold_metrics(fold_df)


def run_lgbm_single_quantile_cv(
    source_df: pd.DataFrame,
    feature_cols: list[str],
    n_splits: int,
    q: float,
    cfg: dict[str, Any],
    ctx: dict[str, Any],
    param_overrides: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, float]]:
    n_splits = _safe_n_splits(n_splits, len(source_df))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    calib_frac = float(cfg["split"].get("val_fraction_within_train", 0.2))
    target_col = str(ctx["target_col"])
    early_stopping_rounds = int(cfg.get("lightgbm_tuning", {}).get("early_stopping_rounds", 0))

    rows: list[dict[str, float]] = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(source_df), start=1):
        fold_train = source_df.iloc[train_idx].copy()
        fold_test = source_df.iloc[test_idx].copy()
        model_train, early_stop, _ = split_train_earlystop_calibration(fold_train, calib_frac)

        X_train_local = model_train[feature_cols]
        y_train_local = model_train[target_col]
        X_valid = early_stop[feature_cols]
        y_valid = early_stop[target_col].to_numpy()
        X_test_local = fold_test[feature_cols]
        y_test_local = fold_test[target_col].to_numpy()

        model = fit_lgbm_quantile(
            X_train=X_train_local,
            y_train=y_train_local,
            q=q,
            objective="quantile",
            model_params=param_overrides,
            early_stopping_rounds=early_stopping_rounds,
            X_valid=X_valid,
            y_valid=y_valid,
        )
        pred = model.predict(X_test_local)

        rows.append(
            {
                "fold": fold,
                "pinball_q50": pinball_loss(y_test_local, pred, q),
                "mae_p50": float(mean_absolute_error(y_test_local, pred)),
                "rmse_p50": rmse(y_test_local, pred),
                "r2_p50": float(r2_score(y_test_local, pred)),
            }
        )

    fold_df = pd.DataFrame(rows)
    summary = {
        "pinball_q50_mean": float(fold_df["pinball_q50"].mean()),
        "mae_p50_mean": float(fold_df["mae_p50"].mean()),
        "rmse_p50_mean": float(fold_df["rmse_p50"].mean()),
        "r2_p50_mean": float(fold_df["r2_p50"].mean()),
    }
    return fold_df, summary


def _mlflow_run(enabled: bool, tracking_uri: str, experiment: str, run_name: str):
    if not enabled or mlflow is None:
        return nullcontext()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    return mlflow.start_run(run_name=run_name)


def _mlflow_log_params(enabled: bool, params_dict: dict[str, Any], prefix: str | None = None) -> None:
    if not enabled or mlflow is None or mlflow.active_run() is None:
        return
    for key, value in params_dict.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, (dict, list, tuple, set)):
            mlflow.log_param(full_key, str(value))
        else:
            mlflow.log_param(full_key, value)


def _mlflow_log_metrics(enabled: bool, metrics_dict: dict[str, Any], prefix: str | None = None) -> None:
    if not enabled or mlflow is None or mlflow.active_run() is None:
        return
    for key, value in metrics_dict.items():
        metric_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, (bool, np.bool_)):
            metric_value = float(value)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            metric_value = float(value)
        else:
            continue
        if np.isfinite(metric_value):
            mlflow.log_metric(metric_key, metric_value)


def _mlflow_log_df(enabled: bool, df: pd.DataFrame, artifact_path: Path) -> None:
    if not enabled or mlflow is None or mlflow.active_run() is None:
        return
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(artifact_path, index=False)
    mlflow.log_artifact(str(artifact_path))


def _plot_feature_importance(
    out_path: Path,
    quantile_features: list[str],
    quantile_model_q50: Pipeline,
    lgbm_features: list[str],
    lgbm_model_q50: LGBMRegressor,
    quantile_name: str,
    lgbm_name: str,
) -> None:
    q_coef = np.abs(np.asarray(quantile_model_q50.named_steps["model"].coef_, dtype=float))
    q_df = (
        pd.DataFrame({"feature": quantile_features, "importance": q_coef})
        .sort_values("importance", ascending=False)
        .head(15)
    )
    lgbm_imp = np.asarray(lgbm_model_q50.feature_importances_, dtype=float)
    lgbm_df = (
        pd.DataFrame({"feature": lgbm_features, "importance": lgbm_imp})
        .sort_values("importance", ascending=False)
        .head(15)
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].barh(q_df["feature"][::-1], q_df["importance"][::-1], color="#1f77b4")
    axes[0].set_title(f"Quantile Winner (q50) | {quantile_name}")
    axes[0].set_xlabel("|coefficient|")

    axes[1].barh(lgbm_df["feature"][::-1], lgbm_df["importance"][::-1], color="#1f77b4")
    axes[1].set_title(f"LightGBM Winner (q50) | {lgbm_name}")
    axes[1].set_xlabel("feature importance")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _fit_quantile_bundle(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    cfg: dict[str, Any],
    ctx: dict[str, Any],
    alpha: float,
) -> dict[str, Any]:
    target_col = str(ctx["target_col"])
    quantiles = list(ctx["quantiles"])
    q_low, q_high = float(ctx["q_low"]), float(ctx["q_high"])
    nominal_coverage = float(ctx["nominal_coverage"])
    calib_frac = float(cfg["split"].get("val_fraction_within_train", 0.2))

    proper_train, calib = split_proper_calibration(train_df, calib_frac)
    X_proper = proper_train[feature_cols]
    y_proper = proper_train[target_col]
    X_cal = calib[feature_cols]
    y_cal = calib[target_col].to_numpy()
    X_full = train_df[feature_cols]
    y_full = train_df[target_col]

    models: dict[float, Pipeline] = {}
    pred_cal: dict[float, np.ndarray] = {}
    for q in quantiles:
        calibration_model = make_quantile_model(cfg, q, alpha_override=alpha)
        calibration_model.fit(X_proper, y_proper)
        pred_cal[q] = calibration_model.predict(X_cal)

        final_model = make_quantile_model(cfg, q, alpha_override=alpha)
        final_model.fit(X_full, y_full)
        models[q] = final_model

    qhat = conformal_qhat(y_cal, pred_cal[q_low], pred_cal[q_high], alpha=1.0 - nominal_coverage)

    return {
        "family": "quantile_regression",
        "feature_cols": feature_cols,
        "quantiles": quantiles,
        "alpha": float(alpha),
        "qhat": float(qhat),
        "interval_calibration": "split_conformal_qhat",
        "qhat_applied": True,
        "final_fit_rows": int(len(train_df)),
        "qhat_calibration_rows": int(len(calib)),
        "prediction_mutation_policy": "split-conformal qhat, then quantile-order repair; no clipping",
        "models": models,
    }


def _fit_lgbm_bundle(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    cfg: dict[str, Any],
    ctx: dict[str, Any],
    param_overrides: dict[str, Any],
) -> dict[str, Any]:
    target_col = str(ctx["target_col"])
    quantiles = list(ctx["quantiles"])
    q_low, q_high = float(ctx["q_low"]), float(ctx["q_high"])
    nominal_coverage = float(ctx["nominal_coverage"])
    calib_frac = float(cfg["split"].get("val_fraction_within_train", 0.2))
    seed = int(ctx["seed"])

    objective, model_params = get_lgbm_params(cfg, seed, param_overrides)
    model_train, early_stop, calib = split_train_earlystop_calibration(train_df, calib_frac)
    X_train = model_train[feature_cols]
    y_train = model_train[target_col]
    X_valid = early_stop[feature_cols]
    y_valid = early_stop[target_col].to_numpy()
    X_cal = calib[feature_cols]
    y_cal = calib[target_col].to_numpy()
    X_full = train_df[feature_cols]
    y_full = train_df[target_col]

    models: dict[float, LGBMRegressor] = {}
    pred_cal: dict[float, np.ndarray] = {}
    for q in quantiles:
        calibration_model = fit_lgbm_quantile(
            X_train=X_train,
            y_train=y_train,
            q=q,
            objective=objective,
            model_params=model_params,
            early_stopping_rounds=int(cfg.get("lightgbm_tuning", {}).get("early_stopping_rounds", 0)),
            X_valid=X_valid,
            y_valid=y_valid,
        )
        pred_cal[q] = calibration_model.predict(X_cal)

        final_model = fit_lgbm_quantile(
            X_train=X_full,
            y_train=y_full,
            q=q,
            objective=objective,
            model_params=model_params,
            early_stopping_rounds=0,
        )
        models[q] = final_model

    qhat = conformal_qhat(y_cal, pred_cal[q_low], pred_cal[q_high], alpha=1.0 - nominal_coverage)

    return {
        "family": "lightgbm_quantile",
        "feature_cols": feature_cols,
        "quantiles": quantiles,
        "params": model_params,
        "qhat": float(qhat),
        "interval_calibration": "split_conformal_qhat",
        "qhat_applied": True,
        "final_fit_rows": int(len(train_df)),
        "qhat_calibration_rows": int(len(calib)),
        "prediction_mutation_policy": "split-conformal qhat, then quantile-order repair; no clipping",
        "models": models,
    }


def _bundle_model_for_quantile(model_map: dict[Any, Any], q: float) -> Any:
    if q in model_map:
        return model_map[q]
    for key, model in model_map.items():
        try:
            if np.isclose(float(key), q):
                return model
        except Exception:
            continue
    raise KeyError(f"No model found for quantile={q}")


def evaluate_bundle_on_test(
    bundle: dict[str, Any],
    test_df: pd.DataFrame,
    ctx: dict[str, Any],
    model_name: str,
) -> tuple[pd.DataFrame, dict[str, float], dict[float, np.ndarray]]:
    target_col = str(ctx["target_col"])
    feature_cols = list(bundle["feature_cols"])
    quantiles = list(ctx["quantiles"])
    q_low, q_high = float(ctx["q_low"]), float(ctx["q_high"])
    nominal_coverage = float(ctx["nominal_coverage"])
    qhat = float(bundle.get("qhat", 0.0))

    X_test = test_df[feature_cols]
    y_test = test_df[target_col].to_numpy()
    pred_map: dict[float, np.ndarray] = {}
    for q in quantiles:
        model = _bundle_model_for_quantile(bundle["models"], q)
        pred_map[q] = np.asarray(model.predict(X_test), dtype=float)

    raw_crossing = quantile_crossing_rate(pred_map, quantiles)
    pred_map = apply_conformal_interval(pred_map, quantiles, q_low, q_high, qhat)
    calibrated_crossing = quantile_crossing_rate(pred_map, quantiles)
    pred_map, repair_rate = repair_quantile_order(pred_map, quantiles)
    output_crossing = quantile_crossing_rate(pred_map, quantiles)
    interval = interval_metrics(pred_map, y_test, q_low, q_high)

    rows: list[dict[str, Any]] = []
    for q in quantiles:
        pred = pred_map[q]
        rows.append(
            {
                "model_name": model_name,
                "quantile": q,
                "pinball_loss": pinball_loss(y_test, pred, q),
                "mae": float(mean_absolute_error(y_test, pred)),
                "rmse": rmse(y_test, pred),
                "r2": float(r2_score(y_test, pred)),
            }
        )

    return (
        pd.DataFrame(rows).sort_values("quantile").reset_index(drop=True),
        {
            "coverage": interval["coverage"],
            "coverage_gap": float(interval["coverage"] - nominal_coverage),
            "interval_width": interval["interval_width"],
            "crossing_rate": output_crossing,
            "raw_crossing_rate": raw_crossing,
            "calibrated_crossing_rate": calibrated_crossing,
            "repair_rate": repair_rate,
            "output_crossing_rate": output_crossing,
        },
        pred_map,
    )


def _tune_lgbm_candidate(
    candidate: str,
    feature_cols: list[str],
    tuning_source_df: pd.DataFrame,
    tune_splits: int,
    cfg: dict[str, Any],
    ctx: dict[str, Any],
    baseline_ref: dict[str, float],
    tuning_trials: int,
    tuning_random_state: int,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    tuned_best_params: dict[str, Any] = {}

    if bool(cfg.get("lightgbm_tuning", {}).get("enabled", True)) and optuna is not None:
        search_space = cfg.get("lightgbm_tuning", {}).get("search_space", {})

        def _bounds(name: str, low: float, high: float) -> tuple[float, float]:
            spec = search_space.get(name, {})
            return float(spec.get("min", low)), float(spec.get("max", high))

        def sample_optuna_params(trial: Any) -> dict[str, Any]:
            lr_min, lr_max = _bounds("learning_rate", 0.01, 0.2)
            est_min, est_max = _bounds("n_estimators", 100, 400)
            leaves_min, leaves_max = _bounds("num_leaves", 15, 127)
            mcs_min, mcs_max = _bounds("min_child_samples", 10, 80)
            subs_min, subs_max = _bounds("subsample", 0.6, 1.0)
            csbt_min, csbt_max = _bounds("colsample_bytree", 0.6, 1.0)
            ra_min, ra_max = _bounds("reg_alpha", 0.0, 5.0)
            rl_min, rl_max = _bounds("reg_lambda", 0.0, 5.0)
            return {
                "learning_rate": trial.suggest_float("learning_rate", lr_min, lr_max, log=True),
                "n_estimators": trial.suggest_int("n_estimators", int(est_min), int(est_max)),
                "num_leaves": trial.suggest_int("num_leaves", int(leaves_min), int(leaves_max)),
                "min_child_samples": trial.suggest_int("min_child_samples", int(mcs_min), int(mcs_max)),
                "subsample": trial.suggest_float("subsample", subs_min, subs_max),
                "colsample_bytree": trial.suggest_float("colsample_bytree", csbt_min, csbt_max),
                "reg_alpha": trial.suggest_float("reg_alpha", ra_min, ra_max),
                "reg_lambda": trial.suggest_float("reg_lambda", rl_min, rl_max),
            }

        def objective(trial: Any) -> float:
            params_try = sample_optuna_params(trial)
            _, summary = run_lgbm_backtest(
                source_df=tuning_source_df,
                feature_cols=feature_cols,
                n_splits=tune_splits,
                cfg=cfg,
                ctx=ctx,
                param_overrides=params_try,
            )
            improve_pct, gate_pass = compare_to_baseline(summary, baseline_ref, ctx)
            score = tuning_objective_score(summary, ctx)
            trial.set_user_attr("candidate", candidate)
            trial.set_user_attr("improvement_pct_vs_locked", float(improve_pct))
            trial.set_user_attr("gate_pass", bool(gate_pass))
            trial.set_user_attr("pinball_mean_mean", float(summary["pinball_mean_mean"]))
            trial.set_user_attr("coverage_mean", float(summary["coverage_mean"]))
            trial.set_user_attr("coverage_gap_mean", float(summary["coverage_gap_mean"]))
            trial.set_user_attr("coverage_min", float(summary.get("coverage_min", np.nan)))
            trial.set_user_attr("crossing_rate_mean", float(summary["crossing_rate_mean"]))
            trial.set_user_attr("raw_crossing_rate_mean", float(summary.get("raw_crossing_rate_mean", np.nan)))
            trial.set_user_attr("calibrated_crossing_rate_mean", float(summary.get("calibrated_crossing_rate_mean", np.nan)))
            trial.set_user_attr("repair_rate_mean", float(summary.get("repair_rate_mean", np.nan)))
            trial.set_user_attr("output_crossing_rate_mean", float(summary.get("output_crossing_rate_mean", np.nan)))
            trial.set_user_attr("r2_p50_mean", float(summary["r2_p50_mean"]))
            trial.report(score, step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()
            return score

        sampler = optuna.samplers.TPESampler(seed=tuning_random_state)
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=max(5, min(10, max(1, tuning_trials // 4)))
        )
        study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
        study.optimize(objective, n_trials=tuning_trials, show_progress_bar=False)

        for trial in study.trials:
            if TrialState is None or trial.state != TrialState.COMPLETE:
                continue
            records.append(
                {
                    "candidate": candidate,
                    "trial": int(trial.number),
                    "score": float(trial.value),
                    "gate_pass": bool(trial.user_attrs.get("gate_pass", False)),
                    "improvement_pct_vs_locked": float(
                        trial.user_attrs.get("improvement_pct_vs_locked", np.nan)
                    ),
                    "pinball_mean_mean": float(trial.user_attrs.get("pinball_mean_mean", np.nan)),
                    "r2_p50_mean": float(trial.user_attrs.get("r2_p50_mean", np.nan)),
                    "coverage_mean": float(trial.user_attrs.get("coverage_mean", np.nan)),
                    "coverage_gap_mean": float(trial.user_attrs.get("coverage_gap_mean", np.nan)),
                    "coverage_min": float(trial.user_attrs.get("coverage_min", np.nan)),
                    "crossing_rate_mean": float(trial.user_attrs.get("crossing_rate_mean", np.nan)),
                    "raw_crossing_rate_mean": float(trial.user_attrs.get("raw_crossing_rate_mean", np.nan)),
                    "calibrated_crossing_rate_mean": float(
                        trial.user_attrs.get("calibrated_crossing_rate_mean", np.nan)
                    ),
                    "repair_rate_mean": float(trial.user_attrs.get("repair_rate_mean", np.nan)),
                    "output_crossing_rate_mean": float(
                        trial.user_attrs.get("output_crossing_rate_mean", np.nan)
                    ),
                    **trial.params,
                }
            )

    tuning_df = pd.DataFrame(records)
    if not tuning_df.empty:
        tuning_df = tuning_df.sort_values(["score", "pinball_mean_mean"]).reset_index(drop=True)
        pass_df = tuning_df[tuning_df["gate_pass"]]
        best_trial = pass_df.iloc[0] if not pass_df.empty else tuning_df.iloc[0]
        tuned_best_params = {
            k: best_trial[k]
            for k in [
                "learning_rate",
                "n_estimators",
                "num_leaves",
                "min_child_samples",
                "subsample",
                "colsample_bytree",
                "reg_alpha",
                "reg_lambda",
            ]
            if k in best_trial
        }
        stage_metrics = {
            "candidate": candidate,
            "trials_completed": int(len(tuning_df)),
            "best_score": float(tuning_df.iloc[0]["score"]),
            "best_pinball_mean": float(tuning_df.iloc[0]["pinball_mean_mean"]),
            "best_gate_pass": bool(tuning_df.iloc[0]["gate_pass"]),
            "selected_trial": int(best_trial["trial"]),
            "selected_trial_gate_pass": bool(best_trial["gate_pass"]),
            "tune_rows": int(len(tuning_source_df)),
        }
        return tuned_best_params, tuning_df, stage_metrics

    _, default_summary = run_lgbm_backtest(
        source_df=tuning_source_df,
        feature_cols=feature_cols,
        n_splits=tune_splits,
        cfg=cfg,
        ctx=ctx,
        param_overrides=tuned_best_params,
    )
    stage_metrics = {
        "candidate": candidate,
        "trials_completed": 0,
        "best_score": float(tuning_objective_score(default_summary, ctx)),
        "best_pinball_mean": float(default_summary["pinball_mean_mean"]),
        "best_gate_pass": metrics_gate_pass(default_summary, ctx),
        "selected_trial": None,
        "selected_trial_gate_pass": metrics_gate_pass(default_summary, ctx),
        "tune_rows": int(len(tuning_source_df)),
    }
    return tuned_best_params, tuning_df, stage_metrics


class TrainingProgress:
    def __init__(self, stages: list[str]) -> None:
        self.stages = stages
        self.total = len(stages)
        self.current = 0

    def start(self, label: str, detail: str = "") -> None:
        self.current += 1
        suffix = f" | {detail}" if detail else ""
        LOGGER.info("")
        LOGGER.info("============================================================")
        LOGGER.info("[%s/%s] START %s%s", self.current, self.total, label, suffix)
        LOGGER.info("Remaining after this: %s", ", ".join(self.stages[self.current:]) or "none")
        LOGGER.info("============================================================")

    def done(self, label: str, detail: str = "") -> None:
        suffix = f" | {detail}" if detail else ""
        LOGGER.info("[%s/%s] DONE  %s%s", self.current, self.total, label, suffix)


def train_models() -> None:
    params = load_params()
    cfg = params["model_training"]
    seed = int(params["global"]["seed"])
    set_global_seed(seed)

    stage_root = Path("artifacts/stages")
    model_dir = Path("artifacts/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    stage_dirs = {
        "baseline": stage_root / "01_baseline_calibration",
        "lgbm_race": stage_root / "02_lgbm_feature_race",
        "lgbm_tune": stage_root / "03_lgbm_top2_tuning",
        "model_selection": stage_root / "04_model_selection",
        "test_report": stage_root / "05_test_report",
        "point_vs_p50": stage_root / "06_point_vs_p50_check",
    }
    for d in stage_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    data_paths = _dataset_paths(cfg)
    train_df = pd.read_parquet(data_paths["train_path"])
    test_df = pd.read_parquet(data_paths["test_path"])
    with open(data_paths["metadata_path"], "r", encoding="utf-8") as f:
        metadata = json.load(f)

    target_col = str(metadata["target_col"])
    quantiles = [float(q) for q in metadata["quantiles"]]
    q_low, q_med, q_high = quantiles[0], quantiles[len(quantiles) // 2], quantiles[-1]
    nominal_coverage = float(metadata["nominal_coverage"])
    feature_map = metadata["feature_map"]
    baseline_name = str(metadata.get("baseline_name", cfg.get("benchmark", {}).get("baseline", "baseline_quantile_cyclic")))
    lgbm_candidates = list(metadata.get("lgbm_candidates", []))

    if baseline_name not in feature_map:
        raise ValueError(f"Configured baseline '{baseline_name}' is not in feature metadata.")
    if not lgbm_candidates:
        raise ValueError("No LGBM candidates available for feature race.")

    backtest_cfg = cfg.get("backtest", {})
    coverage_tolerance = float(backtest_cfg.get("coverage_tolerance", 0.05))
    max_crossing_rate = float(backtest_cfg.get("max_crossing_rate", 0.0))
    min_fold_coverage = float(backtest_cfg.get("min_fold_coverage", nominal_coverage - coverage_tolerance))
    approval_cfg = cfg.get("approval", {})
    baseline_min_coverage = float(approval_cfg.get("baseline_min_coverage", min_fold_coverage))
    baseline_max_crossing_rate = float(approval_cfg.get("baseline_max_crossing_rate", max_crossing_rate))

    ctx = {
        "seed": seed,
        "target_col": target_col,
        "quantiles": quantiles,
        "q_low": q_low,
        "q_med": q_med,
        "q_high": q_high,
        "nominal_coverage": nominal_coverage,
        "coverage_tolerance": coverage_tolerance,
        "max_crossing_rate": max_crossing_rate,
        "min_fold_coverage": min_fold_coverage,
        "baseline_min_coverage": baseline_min_coverage,
        "baseline_max_crossing_rate": baseline_max_crossing_rate,
        "target_lower_bound": float(metadata["target_lower_bound"]),
        "target_upper_bound": float(metadata["target_upper_bound"]),
    }

    mlflow_cfg = cfg.get("mlflow", {})
    mlflow_enabled = bool(mlflow_cfg.get("enabled", True) and mlflow is not None)
    tracking_uri = str((Path(mlflow_cfg.get("tracking_uri", "mlruns"))).resolve())
    main_experiment = str(mlflow_cfg.get("lightgbm_tuning_experiment", "model_training"))

    LOGGER.info("Starting simplified model training pipeline")
    LOGGER.info("Rows -> train=%s, test=%s", len(train_df), len(test_df))
    progress = TrainingProgress(
        [
            "baseline calibration",
            "LGBM full feature race",
            "top-2 LGBM tuning",
            "final CV model selection",
            "fit and save frozen artifacts",
            "held-out test report",
            "point-vs-p50 check and summary",
        ]
    )

    with _mlflow_run(mlflow_enabled, tracking_uri, main_experiment, "dvc_train_models_simplified"):
        _mlflow_log_params(
            mlflow_enabled,
            {
                "seed": seed,
                "rows_train": len(train_df),
                "rows_test": len(test_df),
                "quantiles": ",".join(str(q) for q in quantiles),
                "baseline": baseline_name,
                "lgbm_candidates": ",".join(lgbm_candidates),
            },
            prefix="run",
        )

        # Stage 1: fixed baseline alpha tuning and calibration check.
        benchmark_tune_cfg = cfg.get("benchmark", {}).get("tune", {})
        baseline_splits = int(benchmark_tune_cfg.get("cv_splits", 3))
        baseline_default_alpha = float(cfg["quantile_regression"]["alpha"])
        baseline_tune_enabled = bool(benchmark_tune_cfg.get("enabled", True))
        alpha_grid = [float(a) for a in benchmark_tune_cfg.get("alpha_grid", [baseline_default_alpha])]
        alpha_search = alpha_grid if baseline_tune_enabled else [baseline_default_alpha]

        baseline_tuning_rows: list[dict[str, Any]] = []
        baseline_tuning_fold_map: dict[float, pd.DataFrame] = {}
        baseline_feature_cols = list(feature_map[baseline_name])
        progress.start("baseline calibration", f"baseline={baseline_name}, alphas={alpha_search}")
        LOGGER.info("Baseline feature columns: %s", baseline_feature_cols)
        for alpha in alpha_search:
            LOGGER.info("Baseline CV alpha=%s (%s folds)", alpha, baseline_splits)
            fold_df, summary = run_quantile_backtest(
                source_df=train_df,
                feature_cols=baseline_feature_cols,
                n_splits=baseline_splits,
                cfg=cfg,
                ctx=ctx,
                quantile_alpha=alpha,
            )
            baseline_tuning_fold_map[float(alpha)] = fold_df
            score = tuning_objective_score(summary, ctx)
            baseline_tuning_rows.append(
                {
                    "stage": "alpha_tune",
                    "candidate": baseline_name,
                    "alpha": float(alpha),
                    "score": score,
                    "gate_pass": metrics_gate_pass(summary, ctx),
                    **summary,
                }
            )

        baseline_tuning_df = (
            pd.DataFrame(baseline_tuning_rows)
            .sort_values(["score", "pinball_mean_mean", "coverage_gap_mean"])
            .reset_index(drop=True)
        )
        baseline_pass_df = baseline_tuning_df[baseline_tuning_df["gate_pass"]]
        baseline_best_row = baseline_pass_df.iloc[0] if not baseline_pass_df.empty else baseline_tuning_df.iloc[0]
        locked_baseline_alpha = float(baseline_best_row["alpha"])
        baseline_fold_df = baseline_tuning_fold_map[locked_baseline_alpha].copy()
        baseline_summary = summarize_fold_metrics(baseline_fold_df)
        baseline_ref = {
            "pinball_mean": float(baseline_summary["pinball_mean_mean"]),
            "coverage": float(baseline_summary["coverage_mean"]),
            "coverage_gap": float(baseline_summary["coverage_gap_mean"]),
            "crossing_rate": float(baseline_summary["crossing_rate_mean"]),
            "r2_p50": float(baseline_summary["r2_p50_mean"]),
            "coverage_min": float(baseline_summary["coverage_min"]),
        }
        baseline_stage_metrics = {
            "baseline_name": baseline_name,
            "baseline_alpha": locked_baseline_alpha,
            "pinball_mean": baseline_ref["pinball_mean"],
            "coverage": baseline_ref["coverage"],
            "coverage_gap": baseline_ref["coverage_gap"],
            "crossing_rate": baseline_ref["crossing_rate"],
            "gate_pass": metrics_gate_pass(baseline_summary, ctx),
            "hard_sanity_pass": hard_sanity_gate(baseline_summary, ctx),
        }
        baseline_tuning_df.to_csv(stage_dirs["baseline"] / "alpha_tuning.csv", index=False)
        baseline_fold_df.to_csv(stage_dirs["baseline"] / "baseline_backtest_folds.csv", index=False)
        _write_json(
            stage_dirs["baseline"] / "metrics.json",
            {
                "baseline_name": baseline_name,
                "baseline_alpha": locked_baseline_alpha,
                "baseline_ref": baseline_ref,
                "stage_metrics": baseline_stage_metrics,
            },
        )
        _mlflow_log_metrics(mlflow_enabled, baseline_stage_metrics, prefix="stage1_baseline")
        _mlflow_log_df(mlflow_enabled, baseline_tuning_df, stage_dirs["baseline"] / "alpha_tuning.csv")
        progress.done(
            "baseline calibration",
            f"alpha={locked_baseline_alpha}, coverage={baseline_ref['coverage']:.3f}, pinball={baseline_ref['pinball_mean']:.1f}",
        )

        # Stage 2: full probabilistic LGBM feature race. No q50 shortcut.
        progress.start("LGBM full feature race", f"candidates={len(lgbm_candidates)}")
        lgbm_tuning_cfg = cfg.get("lightgbm_tuning", {})
        race_splits = int(lgbm_tuning_cfg.get("race_cv_splits", 5))
        use_full_train_if_small = bool(lgbm_tuning_cfg.get("use_full_train_if_small", True))
        small_data_threshold = int(lgbm_tuning_cfg.get("small_data_threshold", 30000))
        race_fraction_cfg = lgbm_tuning_cfg.get("fraction", {})
        max_train_rows = race_fraction_cfg.get("max_train_rows", None)

        if use_full_train_if_small and len(train_df) <= small_data_threshold:
            race_source_df = train_df.copy()
        else:
            race_source_df = select_train_window(train_df, None if max_train_rows is None else int(max_train_rows))
        race_splits = _safe_n_splits(race_splits, len(race_source_df))
        _, race_params = get_lgbm_params(cfg, seed, None)

        race_rows: list[dict[str, Any]] = []
        race_fold_map: dict[str, pd.DataFrame] = {}
        for idx, candidate in enumerate(lgbm_candidates, start=1):
            LOGGER.info(
                "LGBM feature race candidate %s/%s: %s (%s features)",
                idx,
                len(lgbm_candidates),
                candidate,
                len(feature_map[candidate]),
            )
            fold_df, summary = run_lgbm_backtest(
                source_df=race_source_df,
                feature_cols=list(feature_map[candidate]),
                n_splits=race_splits,
                cfg=cfg,
                ctx=ctx,
                param_overrides=race_params,
            )
            race_fold_map[candidate] = fold_df
            score = candidate_selection_score(summary, ctx)
            race_rows.append(
                {
                    "candidate": candidate,
                    "score": score,
                    "gate_pass": metrics_gate_pass(summary, ctx),
                    **summary,
                }
            )

        race_df = (
            pd.DataFrame(race_rows)
            .assign(coverage_gap_abs=lambda d: d["coverage_gap_mean"].abs())
            .sort_values(["score", "pinball_mean_mean", "coverage_gap_abs", "crossing_rate_mean"])
            .reset_index(drop=True)
        )
        tune_top_k = max(1, int(lgbm_tuning_cfg.get("tune_top_k", 2)))
        top_candidates = race_df["candidate"].head(min(tune_top_k, len(race_df))).tolist()

        race_sig = {}
        if len(race_df) > 1:
            race_sig = {
                "comparison": f"{race_df.iloc[0]['candidate']} vs {race_df.iloc[1]['candidate']}",
                **paired_significance(
                    race_fold_map[str(race_df.iloc[0]["candidate"])],
                    race_fold_map[str(race_df.iloc[1]["candidate"])],
                    metric_col="pinball_mean",
                ),
            }

        race_metrics = {
            "top_candidates": top_candidates,
            "winner_candidate": str(race_df.iloc[0]["candidate"]),
            "winner_score": float(race_df.iloc[0]["score"]),
            "winner_pinball_mean": float(race_df.iloc[0]["pinball_mean_mean"]),
            "winner_gate_pass": bool(race_df.iloc[0]["gate_pass"]),
            "race_source_rows": int(len(race_source_df)),
            "race_splits": int(race_splits),
            "race_params": race_params,
        }
        race_df.to_csv(stage_dirs["lgbm_race"] / "feature_race.csv", index=False)
        if race_sig:
            pd.DataFrame([race_sig]).to_csv(stage_dirs["lgbm_race"] / "feature_race_significance.csv", index=False)
        _write_json(stage_dirs["lgbm_race"] / "metrics.json", {"stage_metrics": race_metrics, "significance": race_sig})
        _mlflow_log_metrics(mlflow_enabled, {k: v for k, v in race_metrics.items() if isinstance(v, (int, float, bool))}, prefix="stage2_lgbm_race")
        _mlflow_log_df(mlflow_enabled, race_df, stage_dirs["lgbm_race"] / "feature_race.csv")
        progress.done(
            "LGBM full feature race",
            f"top={top_candidates}, best_pinball={race_metrics['winner_pinball_mean']:.1f}",
        )

        # Stage 3: tune the top two LGBM feature sets with equal budget.
        progress.start("top-2 LGBM tuning", f"trials_per_candidate={lgbm_tuning_cfg.get('optuna_trials', 20)}")
        use_full_data_for_tuning = bool(lgbm_tuning_cfg.get("use_full_data_for_tuning", True))
        tuning_trials = int(lgbm_tuning_cfg.get("optuna_trials", 20))
        tuning_random_state = int(lgbm_tuning_cfg.get("random_state", seed))
        tune_splits_default = int(race_fraction_cfg.get("cv_splits", race_splits))
        if use_full_data_for_tuning:
            tuning_source_df = train_df.copy()
        else:
            tuning_source_df = select_train_window(train_df, None if max_train_rows is None else int(max_train_rows))
        tune_splits = _safe_n_splits(tune_splits_default, len(tuning_source_df))

        tuned_candidates: list[dict[str, Any]] = []
        tuning_frames: list[pd.DataFrame] = []
        for rank, candidate in enumerate(top_candidates, start=1):
            feature_cols = list(feature_map[candidate])
            LOGGER.info(
                "Tuning LGBM top candidate %s/%s: %s (%s trials, %s folds, %s features)",
                rank,
                len(top_candidates),
                candidate,
                tuning_trials,
                tune_splits,
                len(feature_cols),
            )
            tuned_params, tuning_df, stage_metrics = _tune_lgbm_candidate(
                candidate=candidate,
                feature_cols=feature_cols,
                tuning_source_df=tuning_source_df,
                tune_splits=tune_splits,
                cfg=cfg,
                ctx=ctx,
                baseline_ref=baseline_ref,
                tuning_trials=tuning_trials,
                tuning_random_state=tuning_random_state + rank - 1,
            )
            if not tuning_df.empty:
                tuning_frames.append(tuning_df.assign(tune_rank=rank))
            tuned_candidates.append(
                {
                    "rank": rank,
                    "model_name": f"challenger_{rank}",
                    "candidate": candidate,
                    "feature_cols": feature_cols,
                    "params": tuned_params,
                    "stage_metrics": stage_metrics,
                }
            )
            LOGGER.info(
                "Finished tuning %s | selected_trial=%s | best_pinball=%.1f | gate_pass=%s",
                candidate,
                stage_metrics.get("selected_trial"),
                float(stage_metrics.get("best_pinball_mean", np.nan)),
                stage_metrics.get("selected_trial_gate_pass"),
            )

        tuning_df_all = pd.concat(tuning_frames, ignore_index=True) if tuning_frames else pd.DataFrame()
        tuning_df_all.to_csv(stage_dirs["lgbm_tune"] / "tuning_trials_optuna.csv", index=False)
        _write_json(
            stage_dirs["lgbm_tune"] / "metrics.json",
            {
                "top_candidates": top_candidates,
                "tuned_candidates": tuned_candidates,
                "tune_splits": tune_splits,
                "trials_per_candidate": tuning_trials,
            },
        )
        if not tuning_df_all.empty:
            _mlflow_log_df(mlflow_enabled, tuning_df_all, stage_dirs["lgbm_tune"] / "tuning_trials_optuna.csv")
        progress.done("top-2 LGBM tuning", f"tuned={[c['candidate'] for c in tuned_candidates]}")

        # Stage 4: final train-only CV comparison: baseline vs both tuned challengers.
        progress.start("final CV model selection", f"models={1 + len(tuned_candidates)}, folds={backtest_cfg.get('n_splits', 5)}")
        final_splits = _safe_n_splits(int(backtest_cfg.get("n_splits", 5)), len(train_df))
        out_of_time_recent_splits = int(lgbm_tuning_cfg.get("out_of_time_recent_splits", 2))
        significance_alpha = float(lgbm_tuning_cfg.get("significance_alpha", 0.05))
        min_improvement_pct = float(lgbm_tuning_cfg.get("min_improvement_pct", 0.0))

        baseline_final_fold_df, baseline_final_summary = run_quantile_backtest(
            source_df=train_df,
            feature_cols=baseline_feature_cols,
            n_splits=final_splits,
            cfg=cfg,
            ctx=ctx,
            quantile_alpha=locked_baseline_alpha,
        )
        baseline_final_ref = {
            "pinball_mean": float(baseline_final_summary["pinball_mean_mean"]),
            "coverage": float(baseline_final_summary["coverage_mean"]),
            "coverage_gap": float(baseline_final_summary["coverage_gap_mean"]),
            "crossing_rate": float(baseline_final_summary["crossing_rate_mean"]),
            "r2_p50": float(baseline_final_summary["r2_p50_mean"]),
            "coverage_min": float(baseline_final_summary["coverage_min"]),
        }
        baseline_gate = metrics_gate_pass(baseline_final_summary, ctx)
        baseline_sanity_pass = hard_sanity_gate(baseline_final_summary, ctx)

        challenger_final_rows: list[dict[str, Any]] = []
        challenger_fold_map: dict[str, pd.DataFrame] = {}
        challenger_recent_map: dict[str, pd.DataFrame] = {}
        challenger_summaries: dict[str, dict[str, Any]] = {}
        for cand in tuned_candidates:
            model_name = str(cand["model_name"])
            LOGGER.info("Final CV for %s: %s", model_name, cand["candidate"])
            fold_df, summary = run_lgbm_backtest(
                source_df=train_df,
                feature_cols=list(cand["feature_cols"]),
                n_splits=final_splits,
                cfg=cfg,
                ctx=ctx,
                param_overrides=dict(cand["params"]),
            )
            challenger_fold_map[model_name] = fold_df
            recent_n = max(1, min(out_of_time_recent_splits, len(fold_df)))
            recent_fold_df = fold_df.sort_values("fold").tail(recent_n).reset_index(drop=True)
            challenger_recent_map[model_name] = recent_fold_df
            recent_summary = summarize_fold_metrics(recent_fold_df)
            improve_pct, challenger_gate = compare_to_baseline(summary, baseline_final_ref, ctx)
            recent_gate = metrics_gate_pass(recent_summary, ctx)
            sig = paired_significance(fold_df, baseline_final_fold_df, metric_col="pinball_mean")
            sig_pvalue = sig.get("ttest_pvalue", np.nan)
            significance_pass = bool(np.isfinite(sig_pvalue) and sig_pvalue <= significance_alpha)
            improvement_pass = bool(improve_pct >= min_improvement_pct)
            ref = {
                "pinball_mean": float(summary["pinball_mean_mean"]),
                "coverage": float(summary["coverage_mean"]),
                "coverage_gap": float(summary["coverage_gap_mean"]),
                "crossing_rate": float(summary["crossing_rate_mean"]),
                "r2_p50": float(summary["r2_p50_mean"]),
                "coverage_min": float(summary["coverage_min"]),
            }
            beats_baseline = bool(ref["pinball_mean"] < baseline_final_ref["pinball_mean"])
            promotion_eligible = bool(beats_baseline and challenger_gate and recent_gate and improvement_pass)
            challenger_summaries[model_name] = {
                "candidate": cand["candidate"],
                "summary": summary,
                "recent_summary": recent_summary,
                "significance": sig,
                "ref": ref,
                "promotion_eligible": promotion_eligible,
            }
            challenger_final_rows.append(
                {
                    "model": f"lightgbm_tuned_{cand['candidate']}",
                    "model_name": model_name,
                    "role": "challenger",
                    "candidate": cand["candidate"],
                    "promotion_gate_pass": challenger_gate,
                    "recent_gate_pass": recent_gate,
                    "significance_pass": significance_pass,
                    "improvement_pass": improvement_pass,
                    "challenger_beats_baseline": beats_baseline,
                    "promotion_eligible": promotion_eligible,
                    "improvement_pct_vs_baseline": improve_pct,
                    **ref,
                }
            )

        final_comparison_df = pd.DataFrame(
            [
                {
                    "model": f"baseline_{baseline_name}",
                    "model_name": "baseline",
                    "role": "baseline",
                    "strict_gate_pass": baseline_gate,
                    "hard_sanity_pass": baseline_sanity_pass,
                    **baseline_final_ref,
                }
            ]
            + challenger_final_rows
        )

        eligible_df = final_comparison_df[final_comparison_df.get("promotion_eligible", False) == True].copy()
        promote_challenger = bool(not eligible_df.empty)
        if promote_challenger:
            selected_row = eligible_df.sort_values("pinball_mean").iloc[0]
            selected_model_name = str(selected_row["model_name"])
            selected_candidate = str(selected_row["candidate"])
            champion_family = "lightgbm_quantile"
            selected_champion_gate_pass = True
            approval_status = "approved_champion"
            decision_reason = f"{selected_model_name} promoted: tuned {selected_candidate} passed gates and beat baseline."
        else:
            selected_model_name = "baseline"
            selected_candidate = baseline_name
            champion_family = "quantile_regression"
            selected_champion_gate_pass = bool(baseline_gate)
            approval_status = "fallback_only" if baseline_sanity_pass else "no_model_meets_quality_bar"
            reasons = []
            if not baseline_gate:
                reasons.append("baseline failed strict quality gate")
            if challenger_final_rows:
                reasons.append("no tuned challenger passed all promotion rules")
            decision_reason = "Baseline retained as fallback: " + "; ".join(reasons)

        final_decision_metrics = {
            "baseline_final_ref": baseline_final_ref,
            "baseline_final_summary": baseline_final_summary,
            "challengers": challenger_summaries,
            "baseline_strict_gate_pass": baseline_gate,
            "baseline_hard_sanity_pass": baseline_sanity_pass,
            "promote_challenger": promote_challenger,
            "selected_model_name": selected_model_name,
            "selected_candidate": selected_candidate,
            "champion_family": champion_family,
            "selected_champion_gate_pass": selected_champion_gate_pass,
            "approval_status": approval_status,
            "decision_reason": decision_reason,
            "gate_thresholds": {
                "max_crossing_rate": float(ctx["max_crossing_rate"]),
                "coverage_tolerance": float(ctx["coverage_tolerance"]),
                "coverage_lower_bound": _coverage_lower_bound(ctx),
                "min_fold_coverage": float(ctx["min_fold_coverage"]),
                "baseline_min_coverage": float(ctx["baseline_min_coverage"]),
                "baseline_max_crossing_rate": float(ctx["baseline_max_crossing_rate"]),
                "min_improvement_pct": min_improvement_pct,
                "significance_alpha": significance_alpha,
            },
        }
        final_comparison_df.to_csv(stage_dirs["model_selection"] / "selection_comparison.csv", index=False)
        baseline_final_fold_df.to_csv(stage_dirs["model_selection"] / "baseline_cv_folds.csv", index=False)
        for model_name, fold_df in challenger_fold_map.items():
            fold_df.to_csv(stage_dirs["model_selection"] / f"{model_name}_cv_folds.csv", index=False)
            challenger_recent_map[model_name].to_csv(
                stage_dirs["model_selection"] / f"{model_name}_recent_cv_folds.csv", index=False
            )
            pd.DataFrame([challenger_summaries[model_name]["significance"]]).to_csv(
                stage_dirs["model_selection"] / f"{model_name}_significance.csv", index=False
            )
        _write_json(stage_dirs["model_selection"] / "metrics.json", final_decision_metrics)
        progress.done("final CV model selection", f"status={approval_status}, selected={selected_model_name}")

        progress.start("fit and save frozen artifacts", f"selected={selected_model_name}")
        baseline_bundle = _fit_quantile_bundle(
            train_df=train_df,
            feature_cols=baseline_feature_cols,
            cfg=cfg,
            ctx=ctx,
            alpha=locked_baseline_alpha,
        )
        challenger_bundles: dict[str, dict[str, Any]] = {}
        for cand in tuned_candidates:
            challenger_bundles[str(cand["model_name"])] = _fit_lgbm_bundle(
                train_df=train_df,
                feature_cols=list(cand["feature_cols"]),
                cfg=cfg,
                ctx=ctx,
                param_overrides=dict(cand["params"]),
            )
        champion_bundle = challenger_bundles[selected_model_name] if promote_challenger else baseline_bundle

        joblib.dump(baseline_bundle, model_dir / "baseline_locked.joblib")
        for model_name, bundle in challenger_bundles.items():
            joblib.dump(bundle, model_dir / f"{model_name}_tuned.joblib")
        joblib.dump(champion_bundle, model_dir / "champion.joblib")

        champion_metadata = {
            "workflow": "simplified_fixed_baseline_top2_lgbm",
            "champion_family": champion_family,
            "approval_status": approval_status,
            "promote_challenger": promote_challenger,
            "selected_model_name": selected_model_name,
            "selected_candidate": selected_candidate,
            "baseline_name": baseline_name,
            "baseline_alpha": locked_baseline_alpha,
            "top_lgbm_candidates": top_candidates,
            "tuned_candidates": tuned_candidates,
            "interval_calibration": "split_conformal_qhat",
            "qhat_applied": True,
            "prediction_mutation_policy": "split-conformal qhat, then quantile-order repair; no clipping",
            "decision_reason": decision_reason,
        }
        _write_json(model_dir / "champion_metadata.json", champion_metadata)
        _mlflow_log_metrics(
            mlflow_enabled,
            {
                "baseline_pinball_mean_final": baseline_final_ref["pinball_mean"],
                "promote_challenger": promote_challenger,
                "selected_champion_gate_pass": selected_champion_gate_pass,
            },
            prefix="stage4_model_selection",
        )
        _mlflow_log_df(mlflow_enabled, final_comparison_df, stage_dirs["model_selection"] / "selection_comparison.csv")
        progress.done("fit and save frozen artifacts", f"artifacts_dir={model_dir}")

        # Stage 5: held-out test report after model selection and freezing.
        progress.start("held-out test report", f"models={len(challenger_bundles) + 2}")
        test_rows: list[dict[str, Any]] = []
        test_metric_frames: dict[str, pd.DataFrame] = {}
        test_intervals: dict[str, dict[str, float]] = {}
        bundles_for_report = {"baseline": baseline_bundle, **challenger_bundles, "champion": champion_bundle}
        for model_name, bundle in bundles_for_report.items():
            metrics_df, interval_metrics_out, _ = evaluate_bundle_on_test(
                bundle=bundle,
                test_df=test_df,
                ctx=ctx,
                model_name=model_name,
            )
            pinball_mean = float(metrics_df["pinball_loss"].mean())
            test_metric_frames[model_name] = metrics_df
            test_intervals[model_name] = interval_metrics_out
            test_rows.append(
                {
                    "model_name": model_name,
                    "model": selected_candidate if model_name == "champion" else model_name,
                    "pinball_mean_test": pinball_mean,
                    "coverage_test": interval_metrics_out["coverage"],
                    "coverage_gap_test": interval_metrics_out["coverage_gap"],
                    "interval_width_test": interval_metrics_out["interval_width"],
                    "crossing_rate_test": interval_metrics_out["crossing_rate"],
                    "raw_crossing_rate_test": interval_metrics_out["raw_crossing_rate"],
                    "calibrated_crossing_rate_test": interval_metrics_out["calibrated_crossing_rate"],
                    "repair_rate_test": interval_metrics_out["repair_rate"],
                    "output_crossing_rate_test": interval_metrics_out["output_crossing_rate"],
                }
            )

        test_comparison_df = pd.DataFrame(test_rows).sort_values("pinball_mean_test").reset_index(drop=True)

        shap_importance_df = pd.DataFrame()
        shap_sample_size = int(lgbm_tuning_cfg.get("shap_sample_size", 500))
        shap_model_name = selected_model_name if selected_model_name in challenger_bundles else next(iter(challenger_bundles), None)
        if shap_model_name and shap is not None and q_med in challenger_bundles[shap_model_name]["models"]:
            try:
                shap_features = list(challenger_bundles[shap_model_name]["feature_cols"])
                shap_X = test_df[shap_features].copy()
                if len(shap_X) > shap_sample_size:
                    shap_X = shap_X.sample(shap_sample_size, random_state=seed)
                explainer = shap.TreeExplainer(challenger_bundles[shap_model_name]["models"][q_med])
                shap_values = explainer.shap_values(shap_X)
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                shap_importance_df = (
                    pd.DataFrame(
                        {
                            "feature": shap_features,
                            "importance": np.mean(np.abs(np.asarray(shap_values)), axis=0),
                        }
                    )
                    .sort_values("importance", ascending=False)
                    .reset_index(drop=True)
                )
            except Exception as exc:  # pragma: no cover - optional analysis artifact
                LOGGER.warning("SHAP calculation failed: %s", exc)

        plot_model_name = shap_model_name or next(iter(challenger_bundles))
        _plot_feature_importance(
            out_path=stage_dirs["test_report"] / "feature_importance_winners.png",
            quantile_features=baseline_feature_cols,
            quantile_model_q50=baseline_bundle["models"][q_med],
            lgbm_features=list(challenger_bundles[plot_model_name]["feature_cols"]),
            lgbm_model_q50=challenger_bundles[plot_model_name]["models"][q_med],
            quantile_name=baseline_name,
            lgbm_name=plot_model_name,
        )

        for model_name, metrics_df in test_metric_frames.items():
            metrics_df.to_csv(stage_dirs["test_report"] / f"{model_name}_test_metrics.csv", index=False)
        test_comparison_df.to_csv(stage_dirs["test_report"] / "test_comparison.csv", index=False)
        if not shap_importance_df.empty:
            shap_importance_df.to_csv(stage_dirs["test_report"] / "shap_importance_q50.csv", index=False)
        _write_json(
            stage_dirs["test_report"] / "metrics.json",
            {
                "test_intervals": test_intervals,
                "test_pinball_mean": {
                    row["model_name"]: row["pinball_mean_test"] for row in test_rows
                },
                "test_used_for_selection": False,
            },
        )
        _mlflow_log_df(mlflow_enabled, test_comparison_df, stage_dirs["test_report"] / "test_comparison.csv")
        progress.done("held-out test report", f"best_test={test_comparison_df.iloc[0]['model_name']}")

        # Stage 6: point baseline vs champion p50.
        progress.start("point-vs-p50 check and summary")
        point_cfg = cfg.get("point_models", {}).get("linear_regression", {})
        point_feature_cols = baseline_feature_cols
        lr = LinearRegression(fit_intercept=bool(point_cfg.get("fit_intercept", True)))
        X_train_point = train_df[point_feature_cols]
        y_train_point = train_df[target_col]
        X_test_point = test_df[point_feature_cols]
        y_test_point = test_df[target_col].to_numpy()
        lr.fit(X_train_point, y_train_point)
        point_pred = lr.predict(X_test_point)

        point_metrics = {
            "pinball_q50": pinball_loss(y_test_point, point_pred, q_med),
            "mae": float(mean_absolute_error(y_test_point, point_pred)),
            "rmse": rmse(y_test_point, point_pred),
            "r2": float(r2_score(y_test_point, point_pred)),
        }
        champion_test_df = test_metric_frames["champion"]
        champion_p50_metrics_row = champion_test_df[champion_test_df["quantile"] == q_med].iloc[0]
        champion_p50_metrics = {
            "pinball_q50": float(champion_p50_metrics_row["pinball_loss"]),
            "mae": float(champion_p50_metrics_row["mae"]),
            "rmse": float(champion_p50_metrics_row["rmse"]),
            "r2": float(champion_p50_metrics_row["r2"]),
        }
        point_vs_p50_df = pd.DataFrame(
            [
                {"model": "linear_point_forecast", **point_metrics},
                {"model": "champion_p50", **champion_p50_metrics},
            ]
        )
        point_vs_p50_df.to_csv(stage_dirs["point_vs_p50"] / "point_vs_p50.csv", index=False)
        _write_json(
            stage_dirs["point_vs_p50"] / "metrics.json",
            {
                "point_features": point_feature_cols,
                "winner_by_mae": str(point_vs_p50_df.sort_values("mae").iloc[0]["model"]),
                "winner_by_rmse": str(point_vs_p50_df.sort_values("rmse").iloc[0]["model"]),
            },
        )
        _mlflow_log_df(mlflow_enabled, point_vs_p50_df, stage_dirs["point_vs_p50"] / "point_vs_p50.csv")

        challenger_pinball = {
            row["model_name"]: row["pinball_mean_test"]
            for row in test_rows
            if str(row["model_name"]).startswith("challenger_")
        }
        metrics_summary = {
            "workflow": "simplified_fixed_baseline_top2_lgbm",
            "fixed_baseline": {
                "name": baseline_name,
                "alpha": locked_baseline_alpha,
                "ref": baseline_ref,
            },
            "lgbm_feature_race": race_metrics,
            "lgbm_tuned_candidates": tuned_candidates,
            "test_report": {
                "baseline_pinball_mean": float(test_comparison_df.loc[test_comparison_df["model_name"] == "baseline", "pinball_mean_test"].iloc[0]),
                "challenger_pinball_mean": challenger_pinball,
                "champion_pinball_mean": float(test_comparison_df.loc[test_comparison_df["model_name"] == "champion", "pinball_mean_test"].iloc[0]),
                "test_used_for_selection": False,
            },
            "model_selection": {
                "champion_family": champion_family,
                "approval_status": approval_status,
                "promote_challenger": promote_challenger,
                "selected_model_name": selected_model_name,
                "selected_candidate": selected_candidate,
                "decision_reason": decision_reason,
            },
            "point_vs_p50": {
                "point_metrics": point_metrics,
                "champion_p50_metrics": champion_p50_metrics,
            },
        }
        _write_json(Path("artifacts/metrics/model_training_metrics.json"), metrics_summary)
        _mlflow_log_metrics(
            mlflow_enabled,
            {
                "final_promote_challenger": promote_challenger,
                "final_baseline_pinball_mean": baseline_final_ref["pinball_mean"],
                "final_selected_champion_gate_pass": selected_champion_gate_pass,
            },
            prefix="final",
        )
        progress.done("point-vs-p50 check and summary", "metrics written")

    LOGGER.info("Simplified model training pipeline complete. Status: %s", approval_status)


if __name__ == "__main__":
    train_models()
