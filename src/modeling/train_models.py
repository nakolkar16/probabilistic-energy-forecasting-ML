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


def enforce_quantile_order(pred_map: dict[float, np.ndarray], quantiles: list[float]) -> dict[float, np.ndarray]:
    ordered = np.sort(np.column_stack([pred_map[q] for q in quantiles]), axis=1)
    return {q: ordered[:, i] for i, q in enumerate(quantiles)}


def apply_domain_bounds(
    pred_map: dict[float, np.ndarray], quantiles: list[float], lower: float, upper: float
) -> dict[float, np.ndarray]:
    return {q: np.clip(np.asarray(pred_map[q]), lower, upper) for q in quantiles}


def interval_metrics(
    pred_map: dict[float, np.ndarray], y_true: np.ndarray, q_low: float, q_high: float
) -> dict[str, float]:
    coverage = float(((pred_map[q_low] <= y_true) & (y_true <= pred_map[q_high])).mean())
    width = float((pred_map[q_high] - pred_map[q_low]).mean())
    return {"coverage": coverage, "interval_width": width}


def quantile_crossing_rate(pred_map: dict[float, np.ndarray], quantiles: list[float]) -> float:
    q_matrix = np.column_stack([pred_map[q] for q in quantiles])
    return float((np.diff(q_matrix, axis=1) < 0).any(axis=1).mean())


def split_proper_calibration(df: pd.DataFrame, frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) < 3:
        return df.iloc[:-1].copy(), df.iloc[-1:].copy()
    calib_size = int(round(len(df) * frac))
    calib_size = min(max(calib_size, 1), len(df) - 1)
    return df.iloc[:-calib_size].copy(), df.iloc[-calib_size:].copy()


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
    return enforce_quantile_order(adjusted, quantiles)


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
    ]
    out: dict[str, float] = {}
    for col in metric_cols:
        out[f"{col}_mean"] = float(fold_df[col].mean())
        out[f"{col}_std"] = float(fold_df[col].std(ddof=0))
    out["coverage_min"] = float(fold_df["coverage"].min())
    return out


def metrics_gate_pass(summary_dict: dict[str, float], ctx: dict[str, Any]) -> bool:
    return bool(
        abs(summary_dict["coverage_gap_mean"]) <= float(ctx["coverage_tolerance"])
        and summary_dict["crossing_rate_mean"] <= float(ctx["max_crossing_rate"])
        and summary_dict.get("coverage_min", 0.0) >= float(ctx["min_fold_coverage"])
    )


def compare_to_baseline(summary_dict: dict[str, float], baseline_ref: dict[str, float], ctx: dict[str, Any]) -> tuple[float, bool]:
    improvement = (
        (baseline_ref["pinball_mean"] - summary_dict["pinball_mean_mean"])
        / max(abs(baseline_ref["pinball_mean"]), 1e-6)
    )
    return float(improvement), metrics_gate_pass(summary_dict, ctx)


def tuning_objective_score(summary_dict: dict[str, float], ctx: dict[str, Any]) -> float:
    base = float(summary_dict["pinball_mean_mean"])
    coverage_excess = max(0.0, abs(summary_dict["coverage_gap_mean"]) - float(ctx["coverage_tolerance"]))
    crossing_excess = max(0.0, summary_dict["crossing_rate_mean"] - float(ctx["max_crossing_rate"]))
    return float(base + 3000.0 * coverage_excess + 3000.0 * crossing_excess)


def candidate_selection_score(summary_dict: dict[str, float], ctx: dict[str, Any]) -> float:
    base = float(summary_dict["pinball_mean_mean"])
    coverage_excess = max(0.0, abs(summary_dict["coverage_gap_mean"]) - float(ctx["coverage_tolerance"]))
    crossing_excess = max(0.0, summary_dict["crossing_rate_mean"] - float(ctx["max_crossing_rate"]))
    return float(base + 3000.0 * coverage_excess + 1000.0 * crossing_excess)


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
    lower, upper = float(ctx["target_lower_bound"]), float(ctx["target_upper_bound"])

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(source_df), start=1):
        fold_train = source_df.iloc[tr_idx].copy()
        fold_test = source_df.iloc[te_idx].copy()
        proper_train, calib = split_proper_calibration(fold_train, calib_frac)

        X_train = proper_train[feature_cols]
        y_train_local = proper_train[target_col]
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

        pred_cal = enforce_quantile_order(pred_cal, quantiles)
        pred_test = enforce_quantile_order(pred_test, quantiles)
        qhat = conformal_qhat(y_cal, pred_cal[q_low], pred_cal[q_high], alpha=1.0 - nominal_coverage)
        pred_test = apply_conformal_interval(pred_test, quantiles, q_low, q_high, qhat)
        pred_test = apply_domain_bounds(pred_test, quantiles, lower, upper)

        interval = interval_metrics(pred_test, y_test_local, q_low, q_high)
        crossing = quantile_crossing_rate(pred_test, quantiles)
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
                "crossing_rate": crossing,
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
    lower, upper = float(ctx["target_lower_bound"]), float(ctx["target_upper_bound"])
    seed = int(ctx["seed"])

    objective, model_params = get_lgbm_params(cfg, seed, param_overrides)

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(source_df), start=1):
        fold_train = source_df.iloc[tr_idx].copy()
        fold_test = source_df.iloc[te_idx].copy()
        proper_train, calib = split_proper_calibration(fold_train, calib_frac)

        X_train = proper_train[feature_cols]
        y_train_local = proper_train[target_col]
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
                X_valid=X_cal,
                y_valid=y_cal,
            )
            pred_cal[q] = model.predict(X_cal)
            pred_test[q] = model.predict(X_test)

        pred_cal = enforce_quantile_order(pred_cal, quantiles)
        pred_test = enforce_quantile_order(pred_test, quantiles)
        qhat = conformal_qhat(y_cal, pred_cal[q_low], pred_cal[q_high], alpha=1.0 - nominal_coverage)
        pred_test = apply_conformal_interval(pred_test, quantiles, q_low, q_high, qhat)
        pred_test = apply_domain_bounds(pred_test, quantiles, lower, upper)

        interval = interval_metrics(pred_test, y_test_local, q_low, q_high)
        crossing = quantile_crossing_rate(pred_test, quantiles)
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
                "crossing_rate": crossing,
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    return fold_df, summarize_fold_metrics(fold_df)


def run_quantile_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    cfg: dict[str, Any],
    ctx: dict[str, Any],
    quantile_alpha: float | None = None,
) -> tuple[pd.DataFrame, dict[float, np.ndarray], dict[str, float], dict[float, Pipeline]]:
    calib_frac = float(cfg["split"].get("val_fraction_within_train", 0.2))
    proper_train, calib = split_proper_calibration(train_df, calib_frac)

    target_col = str(ctx["target_col"])
    quantiles = list(ctx["quantiles"])
    q_low, q_high = float(ctx["q_low"]), float(ctx["q_high"])
    nominal_coverage = float(ctx["nominal_coverage"])
    y_test = test_df[target_col].to_numpy()
    lower, upper = float(ctx["target_lower_bound"]), float(ctx["target_upper_bound"])

    X_train = proper_train[feature_cols]
    y_train_local = proper_train[target_col]
    X_cal = calib[feature_cols]
    y_cal = calib[target_col].to_numpy()
    X_test = test_df[feature_cols]

    pred_cal: dict[float, np.ndarray] = {}
    pred_test: dict[float, np.ndarray] = {}
    models: dict[float, Pipeline] = {}

    for q in quantiles:
        model = make_quantile_model(cfg, q, alpha_override=quantile_alpha)
        model.fit(X_train, y_train_local)
        models[q] = model
        pred_cal[q] = model.predict(X_cal)
        pred_test[q] = model.predict(X_test)

    pred_cal = enforce_quantile_order(pred_cal, quantiles)
    pred_test = enforce_quantile_order(pred_test, quantiles)
    qhat = conformal_qhat(y_cal, pred_cal[q_low], pred_cal[q_high], alpha=1.0 - nominal_coverage)
    pred_test = apply_conformal_interval(pred_test, quantiles, q_low, q_high, qhat)
    pred_test = apply_domain_bounds(pred_test, quantiles, lower, upper)

    interval = interval_metrics(pred_test, y_test, q_low, q_high)
    crossing = quantile_crossing_rate(pred_test, quantiles)

    rows: list[dict[str, float]] = []
    for q in quantiles:
        pred = pred_test[q]
        rows.append(
            {
                "quantile": q,
                "pinball_loss": pinball_loss(y_test, pred, q),
                "mae": float(mean_absolute_error(y_test, pred)),
                "rmse": rmse(y_test, pred),
                "r2": float(r2_score(y_test, pred)),
            }
        )

    return (
        pd.DataFrame(rows).sort_values("quantile").reset_index(drop=True),
        pred_test,
        {
            "coverage": interval["coverage"],
            "interval_width": interval["interval_width"],
            "crossing_rate": crossing,
            "coverage_gap": float(interval["coverage"] - nominal_coverage),
        },
        models,
    )


def run_lgbm_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    cfg: dict[str, Any],
    ctx: dict[str, Any],
    param_overrides: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[float, np.ndarray], dict[str, float], dict[float, LGBMRegressor]]:
    calib_frac = float(cfg["split"].get("val_fraction_within_train", 0.2))
    proper_train, calib = split_proper_calibration(train_df, calib_frac)
    target_col = str(ctx["target_col"])
    quantiles = list(ctx["quantiles"])
    q_low, q_high = float(ctx["q_low"]), float(ctx["q_high"])
    nominal_coverage = float(ctx["nominal_coverage"])
    y_test = test_df[target_col].to_numpy()
    lower, upper = float(ctx["target_lower_bound"]), float(ctx["target_upper_bound"])
    early_stopping_rounds = int(cfg.get("lightgbm_tuning", {}).get("early_stopping_rounds", 0))
    seed = int(ctx["seed"])

    objective, model_params = get_lgbm_params(cfg, seed, param_overrides)
    X_train = proper_train[feature_cols]
    y_train_local = proper_train[target_col]
    X_cal = calib[feature_cols]
    y_cal = calib[target_col].to_numpy()
    X_test = test_df[feature_cols]

    pred_cal: dict[float, np.ndarray] = {}
    pred_test: dict[float, np.ndarray] = {}
    models: dict[float, LGBMRegressor] = {}

    for q in quantiles:
        model = fit_lgbm_quantile(
            X_train=X_train,
            y_train=y_train_local,
            q=q,
            objective=objective,
            model_params=model_params,
            early_stopping_rounds=early_stopping_rounds,
            X_valid=X_cal,
            y_valid=y_cal,
        )
        models[q] = model
        pred_cal[q] = model.predict(X_cal)
        pred_test[q] = model.predict(X_test)

    pred_cal = enforce_quantile_order(pred_cal, quantiles)
    pred_test = enforce_quantile_order(pred_test, quantiles)
    qhat = conformal_qhat(y_cal, pred_cal[q_low], pred_cal[q_high], alpha=1.0 - nominal_coverage)
    pred_test = apply_conformal_interval(pred_test, quantiles, q_low, q_high, qhat)
    pred_test = apply_domain_bounds(pred_test, quantiles, lower, upper)

    interval = interval_metrics(pred_test, y_test, q_low, q_high)
    crossing = quantile_crossing_rate(pred_test, quantiles)
    rows: list[dict[str, float]] = []
    for q in quantiles:
        pred = pred_test[q]
        rows.append(
            {
                "quantile": q,
                "pinball_loss": pinball_loss(y_test, pred, q),
                "mae": float(mean_absolute_error(y_test, pred)),
                "rmse": rmse(y_test, pred),
                "r2": float(r2_score(y_test, pred)),
            }
        )

    return (
        pd.DataFrame(rows).sort_values("quantile").reset_index(drop=True),
        pred_test,
        {
            "coverage": interval["coverage"],
            "interval_width": interval["interval_width"],
            "crossing_rate": crossing,
            "coverage_gap": float(interval["coverage"] - nominal_coverage),
        },
        models,
    )


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
    lower, upper = float(ctx["target_lower_bound"]), float(ctx["target_upper_bound"])
    early_stopping_rounds = int(cfg.get("lightgbm_tuning", {}).get("early_stopping_rounds", 0))

    rows: list[dict[str, float]] = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(source_df), start=1):
        fold_train = source_df.iloc[train_idx].copy()
        fold_test = source_df.iloc[test_idx].copy()
        proper_train, calib = split_proper_calibration(fold_train, calib_frac)

        X_train_local = proper_train[feature_cols]
        y_train_local = proper_train[target_col]
        X_cal = calib[feature_cols]
        y_cal = calib[target_col].to_numpy()
        X_test_local = fold_test[feature_cols]
        y_test_local = fold_test[target_col].to_numpy()

        model = fit_lgbm_quantile(
            X_train=X_train_local,
            y_train=y_train_local,
            q=q,
            objective="quantile",
            model_params=param_overrides,
            early_stopping_rounds=early_stopping_rounds,
            X_valid=X_cal,
            y_valid=y_cal,
        )
        pred = np.clip(model.predict(X_test_local), lower, upper)

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
    lower, upper = float(ctx["target_lower_bound"]), float(ctx["target_upper_bound"])

    proper_train, calib = split_proper_calibration(train_df, calib_frac)
    X_train = proper_train[feature_cols]
    y_train = proper_train[target_col]
    X_cal = calib[feature_cols]
    y_cal = calib[target_col].to_numpy()

    models: dict[float, Pipeline] = {}
    pred_cal: dict[float, np.ndarray] = {}
    for q in quantiles:
        model = make_quantile_model(cfg, q, alpha_override=alpha)
        model.fit(X_train, y_train)
        models[q] = model
        pred_cal[q] = model.predict(X_cal)

    pred_cal = enforce_quantile_order(pred_cal, quantiles)
    qhat = conformal_qhat(y_cal, pred_cal[q_low], pred_cal[q_high], alpha=1.0 - nominal_coverage)
    pred_cal = apply_conformal_interval(pred_cal, quantiles, q_low, q_high, qhat)
    pred_cal = apply_domain_bounds(pred_cal, quantiles, lower, upper)

    return {
        "family": "quantile_regression",
        "feature_cols": feature_cols,
        "quantiles": quantiles,
        "alpha": float(alpha),
        "qhat": float(qhat),
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
    lower, upper = float(ctx["target_lower_bound"]), float(ctx["target_upper_bound"])
    early_stopping_rounds = int(cfg.get("lightgbm_tuning", {}).get("early_stopping_rounds", 0))
    seed = int(ctx["seed"])

    objective, model_params = get_lgbm_params(cfg, seed, param_overrides)
    proper_train, calib = split_proper_calibration(train_df, calib_frac)
    X_train = proper_train[feature_cols]
    y_train = proper_train[target_col]
    X_cal = calib[feature_cols]
    y_cal = calib[target_col].to_numpy()

    models: dict[float, LGBMRegressor] = {}
    pred_cal: dict[float, np.ndarray] = {}
    for q in quantiles:
        model = fit_lgbm_quantile(
            X_train=X_train,
            y_train=y_train,
            q=q,
            objective=objective,
            model_params=model_params,
            early_stopping_rounds=early_stopping_rounds,
            X_valid=X_cal,
            y_valid=y_cal,
        )
        models[q] = model
        pred_cal[q] = model.predict(X_cal)

    pred_cal = enforce_quantile_order(pred_cal, quantiles)
    qhat = conformal_qhat(y_cal, pred_cal[q_low], pred_cal[q_high], alpha=1.0 - nominal_coverage)
    pred_cal = apply_conformal_interval(pred_cal, quantiles, q_low, q_high, qhat)
    pred_cal = apply_domain_bounds(pred_cal, quantiles, lower, upper)

    return {
        "family": "lightgbm_quantile",
        "feature_cols": feature_cols,
        "quantiles": quantiles,
        "params": model_params,
        "qhat": float(qhat),
        "models": models,
    }


def train_models() -> None:
    params = load_params()
    cfg = params["model_training"]
    seed = int(params["global"]["seed"])
    set_global_seed(seed)

    stage_root = Path("artifacts/stages")
    model_dir = Path("artifacts/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    stage_dirs = {
        "baseline": stage_root / "01_baseline_screen_tune",
        "lgbm_race": stage_root / "02_lgbm_feature_screen",
        "lgbm_tune": stage_root / "03_lgbm_tuning",
        "test_check": stage_root / "04_test_check",
        "final_decision": stage_root / "05_final_decision",
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
    baseline_candidates = metadata.get("baseline_candidates", [])
    lgbm_candidates = metadata.get("lgbm_candidates", [])

    backtest_cfg = cfg.get("backtest", {})
    coverage_tolerance = float(backtest_cfg.get("coverage_tolerance", 0.05))
    max_crossing_rate = float(backtest_cfg.get("max_crossing_rate", 0.0))
    min_fold_coverage = float(backtest_cfg.get("min_fold_coverage", nominal_coverage - coverage_tolerance))

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
        "target_lower_bound": float(metadata["target_lower_bound"]),
        "target_upper_bound": float(metadata["target_upper_bound"]),
    }

    mlflow_cfg = cfg.get("mlflow", {})
    mlflow_enabled = bool(mlflow_cfg.get("enabled", True) and mlflow is not None)
    tracking_uri = str((Path(mlflow_cfg.get("tracking_uri", "mlruns"))).resolve())
    main_experiment = str(mlflow_cfg.get("lightgbm_tuning_experiment", "model_training"))

    LOGGER.info("Starting model training pipeline")
    LOGGER.info("Rows -> train=%s, test=%s", len(train_df), len(test_df))

    with _mlflow_run(mlflow_enabled, tracking_uri, main_experiment, "dvc_train_models"):
        _mlflow_log_params(
            mlflow_enabled,
            {
                "seed": seed,
                "rows_train": len(train_df),
                "rows_test": len(test_df),
                "quantiles": ",".join(str(q) for q in quantiles),
                "baseline_candidates": ",".join(baseline_candidates),
                "lgbm_candidates": ",".join(lgbm_candidates),
            },
            prefix="run",
        )

        # Stage 1: quantile baseline feature screen + alpha tuning
        benchmark_tune_cfg = cfg.get("benchmark", {}).get("tune", {})
        baseline_screen_splits = int(benchmark_tune_cfg.get("cv_splits", 3))
        baseline_default_alpha = float(cfg["quantile_regression"]["alpha"])
        baseline_tune_enabled = bool(benchmark_tune_cfg.get("enabled", True))
        alpha_grid = [float(a) for a in benchmark_tune_cfg.get("alpha_grid", [baseline_default_alpha])]

        baseline_screen_rows: list[dict[str, Any]] = []
        baseline_screen_fold_map: dict[str, pd.DataFrame] = {}
        for candidate in baseline_candidates:
            feature_cols = list(feature_map[candidate])
            fold_df, summary = run_quantile_backtest(
                source_df=train_df,
                feature_cols=feature_cols,
                n_splits=baseline_screen_splits,
                cfg=cfg,
                ctx=ctx,
                quantile_alpha=baseline_default_alpha,
            )
            baseline_screen_fold_map[candidate] = fold_df
            score = tuning_objective_score(summary, ctx)
            baseline_screen_rows.append(
                {
                    "stage": "feature_screen",
                    "candidate": candidate,
                    "alpha": baseline_default_alpha,
                    "score": score,
                    "gate_pass": metrics_gate_pass(summary, ctx),
                    **summary,
                }
            )

        baseline_screen_df = (
            pd.DataFrame(baseline_screen_rows)
            .sort_values(["score", "pinball_mean_mean", "coverage_gap_mean"])
            .reset_index(drop=True)
        )
        baseline_feature_winner = str(baseline_screen_df.iloc[0]["candidate"])

        baseline_sig = {}
        if len(baseline_screen_df) > 1:
            runner = str(baseline_screen_df.iloc[1]["candidate"])
            baseline_sig = {
                "comparison": f"{baseline_feature_winner} vs {runner}",
                **paired_significance(
                    baseline_screen_fold_map[baseline_feature_winner],
                    baseline_screen_fold_map[runner],
                    metric_col="pinball_mean",
                ),
            }

        baseline_tuning_rows: list[dict[str, Any]] = []
        baseline_tuning_fold_map: dict[float, pd.DataFrame] = {}
        alpha_search = alpha_grid if baseline_tune_enabled else [baseline_default_alpha]
        for alpha in alpha_search:
            fold_df, summary = run_quantile_backtest(
                source_df=train_df,
                feature_cols=list(feature_map[baseline_feature_winner]),
                n_splits=baseline_screen_splits,
                cfg=cfg,
                ctx=ctx,
                quantile_alpha=alpha,
            )
            baseline_tuning_fold_map[float(alpha)] = fold_df
            score = tuning_objective_score(summary, ctx)
            baseline_tuning_rows.append(
                {
                    "stage": "alpha_tune",
                    "candidate": baseline_feature_winner,
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
        baseline_best_row = (
            baseline_pass_df.iloc[0] if not baseline_pass_df.empty else baseline_tuning_df.iloc[0]
        )
        locked_baseline_name = str(baseline_best_row["candidate"])
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
            "feature_winner": locked_baseline_name,
            "feature_winner_alpha": locked_baseline_alpha,
            "pinball_mean": baseline_ref["pinball_mean"],
            "coverage": baseline_ref["coverage"],
            "coverage_gap": baseline_ref["coverage_gap"],
            "crossing_rate": baseline_ref["crossing_rate"],
            "gate_pass": metrics_gate_pass(baseline_summary, ctx),
        }

        baseline_screen_df.to_csv(stage_dirs["baseline"] / "feature_screen.csv", index=False)
        baseline_tuning_df.to_csv(stage_dirs["baseline"] / "alpha_tuning.csv", index=False)
        baseline_fold_df.to_csv(stage_dirs["baseline"] / "winner_backtest_folds.csv", index=False)
        if baseline_sig:
            pd.DataFrame([baseline_sig]).to_csv(
                stage_dirs["baseline"] / "feature_screen_significance.csv", index=False
            )
        _write_json(
            stage_dirs["baseline"] / "metrics.json",
            {
                "locked_baseline_name": locked_baseline_name,
                "locked_baseline_alpha": locked_baseline_alpha,
                "baseline_ref": baseline_ref,
                "stage_metrics": baseline_stage_metrics,
                "feature_screen_significance": baseline_sig,
            },
        )
        _mlflow_log_metrics(mlflow_enabled, baseline_stage_metrics, prefix="stage1_baseline")
        _mlflow_log_df(mlflow_enabled, baseline_screen_df, stage_dirs["baseline"] / "feature_screen.csv")
        _mlflow_log_df(mlflow_enabled, baseline_tuning_df, stage_dirs["baseline"] / "alpha_tuning.csv")

        # Stage 2: LGBM feature race (2-stage screening)
        lgbm_tuning_cfg = cfg.get("lightgbm_tuning", {})
        race_splits = int(lgbm_tuning_cfg.get("race_cv_splits", 5))
        top_k = int(lgbm_tuning_cfg.get("race_top_k", 4))

        use_full_train_if_small = bool(lgbm_tuning_cfg.get("use_full_train_if_small", True))
        small_data_threshold = int(lgbm_tuning_cfg.get("small_data_threshold", 30000))
        race_fraction_cfg = lgbm_tuning_cfg.get("fraction", {})
        max_train_rows = race_fraction_cfg.get("max_train_rows", None)

        if use_full_train_if_small and len(train_df) <= small_data_threshold:
            race_source_df = train_df.copy()
        else:
            race_source_df = select_train_window(train_df, None if max_train_rows is None else int(max_train_rows))
        race_splits = _safe_n_splits(race_splits, len(race_source_df))

        _, base_params = get_lgbm_params(cfg, seed, None)
        screen_params = {
            k: base_params[k]
            for k in [
                "learning_rate",
                "n_estimators",
                "num_leaves",
                "min_child_samples",
                "subsample",
                "colsample_bytree",
                "reg_alpha",
                "reg_lambda",
                "random_state",
                "n_jobs",
                "verbosity",
            ]
        }
        screen_params["n_estimators"] = int(max(120, base_params["n_estimators"] // 2))

        fast_rows: list[dict[str, Any]] = []
        for candidate in lgbm_candidates:
            fold_df, summary = run_lgbm_single_quantile_cv(
                source_df=race_source_df,
                feature_cols=list(feature_map[candidate]),
                n_splits=race_splits,
                q=q_med,
                cfg=cfg,
                ctx=ctx,
                param_overrides=screen_params,
            )
            fast_rows.append({"candidate": candidate, "score": summary["pinball_q50_mean"], **summary})

        fast_race_df = (
            pd.DataFrame(fast_rows)
            .sort_values(["score", "rmse_p50_mean", "mae_p50_mean"])
            .reset_index(drop=True)
        )
        prob_candidates = fast_race_df["candidate"].head(min(top_k, len(fast_race_df))).tolist()

        prob_rows: list[dict[str, Any]] = []
        stage2_fold_map: dict[str, pd.DataFrame] = {}
        for candidate in prob_candidates:
            fold_df, summary = run_lgbm_backtest(
                source_df=race_source_df,
                feature_cols=list(feature_map[candidate]),
                n_splits=race_splits,
                cfg=cfg,
                ctx=ctx,
                param_overrides=screen_params,
            )
            stage2_fold_map[candidate] = fold_df
            score = candidate_selection_score(summary, ctx)
            prob_rows.append(
                {
                    "candidate": candidate,
                    "score": score,
                    "gate_pass": metrics_gate_pass(summary, ctx),
                    **summary,
                }
            )

        prob_race_df = (
            pd.DataFrame(prob_rows)
            .assign(coverage_gap_abs=lambda d: d["coverage_gap_mean"].abs())
            .sort_values(
                ["score", "pinball_mean_mean", "coverage_gap_abs", "interval_width_mean", "crossing_rate_mean"]
            )
            .reset_index(drop=True)
        )

        winner_candidate = str(prob_race_df.iloc[0]["candidate"])
        winner_feature_cols = list(feature_map[winner_candidate])
        race_sig = {}
        if len(prob_race_df) > 1:
            runner = str(prob_race_df.iloc[1]["candidate"])
            race_sig = {
                "comparison": f"{winner_candidate} vs {runner}",
                **paired_significance(
                    stage2_fold_map[winner_candidate],
                    stage2_fold_map[runner],
                    metric_col="pinball_mean",
                ),
            }

        lgbm_race_metrics = {
            "winner_candidate": winner_candidate,
            "winner_score": float(prob_race_df.iloc[0]["score"]),
            "winner_pinball_mean": float(prob_race_df.iloc[0]["pinball_mean_mean"]),
            "winner_gate_pass": bool(prob_race_df.iloc[0]["gate_pass"]),
            "race_source_rows": int(len(race_source_df)),
            "race_splits": int(race_splits),
        }

        fast_race_df.to_csv(stage_dirs["lgbm_race"] / "quick_race_stage1.csv", index=False)
        prob_race_df.to_csv(stage_dirs["lgbm_race"] / "quick_race_stage2.csv", index=False)
        if race_sig:
            pd.DataFrame([race_sig]).to_csv(stage_dirs["lgbm_race"] / "quick_race_significance.csv", index=False)
        _write_json(
            stage_dirs["lgbm_race"] / "metrics.json",
            {
                "winner_candidate": winner_candidate,
                "winner_feature_cols": winner_feature_cols,
                "stage_metrics": lgbm_race_metrics,
                "significance": race_sig,
            },
        )
        _mlflow_log_metrics(mlflow_enabled, lgbm_race_metrics, prefix="stage2_lgbm_race")
        _mlflow_log_df(mlflow_enabled, fast_race_df, stage_dirs["lgbm_race"] / "quick_race_stage1.csv")
        _mlflow_log_df(mlflow_enabled, prob_race_df, stage_dirs["lgbm_race"] / "quick_race_stage2.csv")

        # Stage 3: LGBM tuning (Optuna)
        use_full_data_for_tuning = bool(lgbm_tuning_cfg.get("use_full_data_for_tuning", True))
        tuning_trials = int(lgbm_tuning_cfg.get("optuna_trials", 20))
        tuning_random_state = int(lgbm_tuning_cfg.get("random_state", seed))
        tune_splits_default = int(race_fraction_cfg.get("cv_splits", race_splits))

        if use_full_data_for_tuning:
            tuning_source_df = train_df.copy()
        else:
            tuning_source_df = select_train_window(
                train_df, None if max_train_rows is None else int(max_train_rows)
            )
        tune_splits = _safe_n_splits(tune_splits_default, len(tuning_source_df))

        tuned_best_params: dict[str, Any] = {}
        tuning_records: list[dict[str, Any]] = []
        if bool(lgbm_tuning_cfg.get("enabled", True)) and optuna is not None:
            search_space = lgbm_tuning_cfg.get("search_space", {})

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
                    feature_cols=winner_feature_cols,
                    n_splits=tune_splits,
                    cfg=cfg,
                    ctx=ctx,
                    param_overrides=params_try,
                )
                improve_pct, gate_pass = compare_to_baseline(summary, baseline_ref, ctx)
                score = tuning_objective_score(summary, ctx)
                trial.set_user_attr("improvement_pct_vs_locked", float(improve_pct))
                trial.set_user_attr("gate_pass", bool(gate_pass))
                trial.set_user_attr("pinball_mean_mean", float(summary["pinball_mean_mean"]))
                trial.set_user_attr("coverage_mean", float(summary["coverage_mean"]))
                trial.set_user_attr("coverage_gap_mean", float(summary["coverage_gap_mean"]))
                trial.set_user_attr("coverage_min", float(summary.get("coverage_min", np.nan)))
                trial.set_user_attr("crossing_rate_mean", float(summary["crossing_rate_mean"]))
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
                tuning_records.append(
                    {
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
                        **trial.params,
                    }
                )

            if tuning_records:
                tuning_df = pd.DataFrame(tuning_records).sort_values("score").reset_index(drop=True)
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
            else:
                tuning_df = pd.DataFrame()
                tuned_best_params = {}
        else:
            tuning_df = pd.DataFrame()
            tuned_best_params = {}

        if tuning_df.empty:
            _, default_summary = run_lgbm_backtest(
                source_df=tuning_source_df,
                feature_cols=winner_feature_cols,
                n_splits=tune_splits,
                cfg=cfg,
                ctx=ctx,
                param_overrides=tuned_best_params,
            )
            tuning_stage_metrics = {
                "trials_completed": 0,
                "best_score": float(tuning_objective_score(default_summary, ctx)),
                "best_pinball_mean": float(default_summary["pinball_mean_mean"]),
                "best_gate_pass": metrics_gate_pass(default_summary, ctx),
                "tune_rows": int(len(tuning_source_df)),
            }
        else:
            tuning_stage_metrics = {
                "trials_completed": int(len(tuning_df)),
                "best_score": float(tuning_df.iloc[0]["score"]),
                "best_pinball_mean": float(tuning_df.iloc[0]["pinball_mean_mean"]),
                "best_gate_pass": bool(tuning_df.iloc[0]["gate_pass"]),
                "tune_rows": int(len(tuning_source_df)),
            }

        tuning_df.to_csv(stage_dirs["lgbm_tune"] / "tuning_trials_optuna.csv", index=False)
        _write_json(
            stage_dirs["lgbm_tune"] / "metrics.json",
            {
                "winner_candidate": winner_candidate,
                "winner_feature_cols": winner_feature_cols,
                "tuned_best_params": tuned_best_params,
                "stage_metrics": tuning_stage_metrics,
                "tune_splits": tune_splits,
            },
        )
        _mlflow_log_metrics(mlflow_enabled, tuning_stage_metrics, prefix="stage3_lgbm_tuning")
        if not tuning_df.empty:
            _mlflow_log_df(mlflow_enabled, tuning_df, stage_dirs["lgbm_tune"] / "tuning_trials_optuna.csv")

        # Stage 4: test check baseline vs tuned winner
        baseline_test_df, _, baseline_test_interval, baseline_test_models = run_quantile_test(
            train_df=train_df,
            test_df=test_df,
            feature_cols=list(feature_map[locked_baseline_name]),
            cfg=cfg,
            ctx=ctx,
            quantile_alpha=locked_baseline_alpha,
        )
        winner_test_df, _, winner_test_interval, winner_test_models = run_lgbm_test(
            train_df=train_df,
            test_df=test_df,
            feature_cols=winner_feature_cols,
            cfg=cfg,
            ctx=ctx,
            param_overrides=tuned_best_params,
        )

        baseline_test_pinball_mean = float(baseline_test_df["pinball_loss"].mean())
        winner_test_pinball_mean = float(winner_test_df["pinball_loss"].mean())

        test_comparison_df = pd.DataFrame(
            [
                {
                    "model": f"baseline_{locked_baseline_name}",
                    "pinball_mean_test": baseline_test_pinball_mean,
                    "coverage_test": baseline_test_interval["coverage"],
                    "coverage_gap_test": baseline_test_interval["coverage_gap"],
                    "crossing_rate_test": baseline_test_interval["crossing_rate"],
                },
                {
                    "model": f"lgbm_tuned_{winner_candidate}",
                    "pinball_mean_test": winner_test_pinball_mean,
                    "coverage_test": winner_test_interval["coverage"],
                    "coverage_gap_test": winner_test_interval["coverage_gap"],
                    "crossing_rate_test": winner_test_interval["crossing_rate"],
                },
            ]
        )

        shap_importance_df = pd.DataFrame()
        shap_sample_size = int(lgbm_tuning_cfg.get("shap_sample_size", 500))
        if shap is not None and q_med in winner_test_models:
            try:
                shap_X = test_df[winner_feature_cols].copy()
                if len(shap_X) > shap_sample_size:
                    shap_X = shap_X.sample(shap_sample_size, random_state=seed)
                explainer = shap.TreeExplainer(winner_test_models[q_med])
                shap_values = explainer.shap_values(shap_X)
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                shap_importance_df = (
                    pd.DataFrame(
                        {
                            "feature": winner_feature_cols,
                            "importance": np.mean(np.abs(np.asarray(shap_values)), axis=0),
                        }
                    )
                    .sort_values("importance", ascending=False)
                    .reset_index(drop=True)
                )
            except Exception as exc:  # pragma: no cover - optional analysis artifact
                LOGGER.warning("SHAP calculation failed: %s", exc)

        _plot_feature_importance(
            out_path=stage_dirs["test_check"] / "feature_importance_winners.png",
            quantile_features=list(feature_map[locked_baseline_name]),
            quantile_model_q50=baseline_test_models[q_med],
            lgbm_features=winner_feature_cols,
            lgbm_model_q50=winner_test_models[q_med],
            quantile_name=locked_baseline_name,
            lgbm_name=winner_candidate,
        )

        baseline_test_df.to_csv(stage_dirs["test_check"] / "baseline_test_metrics.csv", index=False)
        winner_test_df.to_csv(stage_dirs["test_check"] / "winner_test_metrics.csv", index=False)
        test_comparison_df.to_csv(stage_dirs["test_check"] / "test_comparison.csv", index=False)
        if not shap_importance_df.empty:
            shap_importance_df.to_csv(stage_dirs["test_check"] / "shap_importance_q50.csv", index=False)
        _write_json(
            stage_dirs["test_check"] / "metrics.json",
            {
                "baseline_test_interval": baseline_test_interval,
                "winner_test_interval": winner_test_interval,
                "baseline_test_pinball_mean": baseline_test_pinball_mean,
                "winner_test_pinball_mean": winner_test_pinball_mean,
                "winner_is_better_on_test": winner_test_pinball_mean < baseline_test_pinball_mean,
            },
        )
        _mlflow_log_metrics(
            mlflow_enabled,
            {
                "baseline_pinball_mean_test": baseline_test_pinball_mean,
                "winner_pinball_mean_test": winner_test_pinball_mean,
                "baseline_coverage_test": baseline_test_interval["coverage"],
                "winner_coverage_test": winner_test_interval["coverage"],
            },
            prefix="stage4_test_check",
        )
        _mlflow_log_df(mlflow_enabled, test_comparison_df, stage_dirs["test_check"] / "test_comparison.csv")

        # Stage 5: final decision gate
        final_splits = int(cfg.get("backtest", {}).get("n_splits", 5))
        final_splits = _safe_n_splits(final_splits, len(train_df))
        out_of_time_recent_splits = int(lgbm_tuning_cfg.get("out_of_time_recent_splits", 2))
        significance_alpha = float(lgbm_tuning_cfg.get("significance_alpha", 0.05))
        min_improvement_pct = float(lgbm_tuning_cfg.get("min_improvement_pct", 0.0))

        baseline_final_fold_df, baseline_final_summary = run_quantile_backtest(
            source_df=train_df,
            feature_cols=list(feature_map[locked_baseline_name]),
            n_splits=final_splits,
            cfg=cfg,
            ctx=ctx,
            quantile_alpha=locked_baseline_alpha,
        )
        winner_final_fold_df, winner_final_summary = run_lgbm_backtest(
            source_df=train_df,
            feature_cols=winner_feature_cols,
            n_splits=final_splits,
            cfg=cfg,
            ctx=ctx,
            param_overrides=tuned_best_params,
        )

        baseline_final_ref = {
            "pinball_mean": float(baseline_final_summary["pinball_mean_mean"]),
            "coverage": float(baseline_final_summary["coverage_mean"]),
            "coverage_gap": float(baseline_final_summary["coverage_gap_mean"]),
            "crossing_rate": float(baseline_final_summary["crossing_rate_mean"]),
            "r2_p50": float(baseline_final_summary["r2_p50_mean"]),
            "coverage_min": float(baseline_final_summary["coverage_min"]),
        }
        winner_final_ref = {
            "pinball_mean": float(winner_final_summary["pinball_mean_mean"]),
            "coverage": float(winner_final_summary["coverage_mean"]),
            "coverage_gap": float(winner_final_summary["coverage_gap_mean"]),
            "crossing_rate": float(winner_final_summary["crossing_rate_mean"]),
            "r2_p50": float(winner_final_summary["r2_p50_mean"]),
            "coverage_min": float(winner_final_summary["coverage_min"]),
        }

        improve_pct, challenger_gate = compare_to_baseline(winner_final_summary, baseline_final_ref, ctx)
        recent_n = max(1, min(out_of_time_recent_splits, len(winner_final_fold_df)))
        winner_recent_fold_df = (
            winner_final_fold_df.sort_values("fold").tail(recent_n).reset_index(drop=True)
        )
        winner_recent_summary = summarize_fold_metrics(winner_recent_fold_df)
        recent_gate = metrics_gate_pass(winner_recent_summary, ctx)

        final_sig = paired_significance(
            winner_final_fold_df, baseline_final_fold_df, metric_col="pinball_mean"
        )
        sig_pvalue = final_sig.get("ttest_pvalue", np.nan)
        significance_pass = bool(np.isfinite(sig_pvalue) and sig_pvalue <= significance_alpha)
        improvement_pass = bool(improve_pct >= min_improvement_pct)
        challenger_beats_baseline = bool(
            winner_final_ref["pinball_mean"] < baseline_final_ref["pinball_mean"]
        )

        promote_challenger = bool(
            challenger_beats_baseline
            and challenger_gate
            and recent_gate
            and significance_pass
            and improvement_pass
        )
        champion_family = "lightgbm_quantile" if promote_challenger else "quantile_regression"

        baseline_bundle = _fit_quantile_bundle(
            train_df=train_df,
            feature_cols=list(feature_map[locked_baseline_name]),
            cfg=cfg,
            ctx=ctx,
            alpha=locked_baseline_alpha,
        )
        challenger_bundle = _fit_lgbm_bundle(
            train_df=train_df,
            feature_cols=winner_feature_cols,
            cfg=cfg,
            ctx=ctx,
            param_overrides=tuned_best_params,
        )
        champion_bundle = challenger_bundle if promote_challenger else baseline_bundle

        joblib.dump(baseline_bundle, model_dir / "baseline_locked.joblib")
        joblib.dump(challenger_bundle, model_dir / "challenger_tuned.joblib")
        joblib.dump(champion_bundle, model_dir / "champion.joblib")

        final_comparison_df = pd.DataFrame(
            [
                {"model": f"baseline_{locked_baseline_name}", **baseline_final_ref},
                {"model": f"lightgbm_tuned_{winner_candidate}", **winner_final_ref},
            ]
        )
        final_comparison_df.to_csv(stage_dirs["final_decision"] / "final_comparison.csv", index=False)
        baseline_final_fold_df.to_csv(
            stage_dirs["final_decision"] / "baseline_final_backtest_folds.csv", index=False
        )
        winner_final_fold_df.to_csv(
            stage_dirs["final_decision"] / "winner_final_backtest_folds.csv", index=False
        )
        winner_recent_fold_df.to_csv(
            stage_dirs["final_decision"] / "winner_recent_backtest_folds.csv", index=False
        )
        pd.DataFrame([final_sig]).to_csv(
            stage_dirs["final_decision"] / "final_significance.csv", index=False
        )

        champion_metadata = {
            "champion_family": champion_family,
            "promote_challenger": promote_challenger,
            "locked_baseline_name": locked_baseline_name,
            "locked_baseline_alpha": locked_baseline_alpha,
            "winner_candidate": winner_candidate,
            "winner_feature_cols": winner_feature_cols,
            "tuned_best_params": tuned_best_params,
            "decision_logic": {
                "challenger_beats_baseline": challenger_beats_baseline,
                "challenger_gate_pass": challenger_gate,
                "recent_gate_pass": recent_gate,
                "significance_pass": significance_pass,
                "improvement_pass": improvement_pass,
                "improvement_pct_vs_baseline": improve_pct,
                "min_improvement_pct": min_improvement_pct,
                "ttest_pvalue": sig_pvalue,
                "significance_alpha": significance_alpha,
            },
        }
        _write_json(model_dir / "champion_metadata.json", champion_metadata)
        _write_json(
            stage_dirs["final_decision"] / "metrics.json",
            {
                "baseline_final_ref": baseline_final_ref,
                "winner_final_ref": winner_final_ref,
                "winner_recent_summary": winner_recent_summary,
                "final_significance": final_sig,
                "promote_challenger": promote_challenger,
                "champion_family": champion_family,
                "improvement_pct_vs_baseline": improve_pct,
            },
        )
        _mlflow_log_metrics(
            mlflow_enabled,
            {
                "baseline_pinball_mean_final": baseline_final_ref["pinball_mean"],
                "winner_pinball_mean_final": winner_final_ref["pinball_mean"],
                "promote_challenger": promote_challenger,
                "improvement_pct": improve_pct,
                "significance_pass": significance_pass,
            },
            prefix="stage5_final_decision",
        )
        _mlflow_log_df(
            mlflow_enabled,
            final_comparison_df,
            stage_dirs["final_decision"] / "final_comparison.csv",
        )

        # Stage 6: point baseline vs champion p50
        point_cfg = cfg.get("point_models", {}).get("linear_regression", {})
        point_feature_cols = list(
            feature_map.get(
                "quantile_lag24_lag168_hour_dow",
                feature_map[locked_baseline_name],
            )
        )
        lr = LinearRegression(fit_intercept=bool(point_cfg.get("fit_intercept", True)))
        X_train_point = train_df[point_feature_cols]
        y_train_point = train_df[target_col]
        X_test_point = test_df[point_feature_cols]
        y_test_point = test_df[target_col].to_numpy()
        lr.fit(X_train_point, y_train_point)
        point_pred = np.clip(
            lr.predict(X_test_point),
            float(ctx["target_lower_bound"]),
            float(ctx["target_upper_bound"]),
        )

        point_metrics = {
            "pinball_q50": pinball_loss(y_test_point, point_pred, q_med),
            "mae": float(mean_absolute_error(y_test_point, point_pred)),
            "rmse": rmse(y_test_point, point_pred),
            "r2": float(r2_score(y_test_point, point_pred)),
        }

        champion_p50_metrics_row = (
            winner_test_df[winner_test_df["quantile"] == q_med].iloc[0]
            if champion_family == "lightgbm_quantile"
            else baseline_test_df[baseline_test_df["quantile"] == q_med].iloc[0]
        )
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
        _mlflow_log_metrics(
            mlflow_enabled,
            {
                "point_mae": point_metrics["mae"],
                "champion_p50_mae": champion_p50_metrics["mae"],
                "point_rmse": point_metrics["rmse"],
                "champion_p50_rmse": champion_p50_metrics["rmse"],
            },
            prefix="stage6_point_vs_p50",
        )
        _mlflow_log_df(mlflow_enabled, point_vs_p50_df, stage_dirs["point_vs_p50"] / "point_vs_p50.csv")

        metrics_summary = {
            "locked_baseline": {
                "name": locked_baseline_name,
                "alpha": locked_baseline_alpha,
                "ref": baseline_ref,
            },
            "lgbm_winner_candidate": winner_candidate,
            "lgbm_tuned_params": tuned_best_params,
            "test_check": {
                "baseline_pinball_mean": baseline_test_pinball_mean,
                "winner_pinball_mean": winner_test_pinball_mean,
            },
            "final_decision": {
                "champion_family": champion_family,
                "promote_challenger": promote_challenger,
                "improvement_pct_vs_baseline": improve_pct,
                "significance_pvalue": sig_pvalue,
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
                "final_improvement_pct": improve_pct,
                "final_baseline_pinball_mean": baseline_final_ref["pinball_mean"],
                "final_winner_pinball_mean": winner_final_ref["pinball_mean"],
            },
            prefix="final",
        )

    LOGGER.info("Model training pipeline complete. Champion family: %s", champion_family)


if __name__ == "__main__":
    train_models()
