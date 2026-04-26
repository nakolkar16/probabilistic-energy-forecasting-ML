from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.utils.reproducibility import set_global_seed


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_params() -> dict[str, Any]:
    with open(Path("params.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dataset_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    ds_cfg = cfg.get("dataset", {})
    model_ready = Path(ds_cfg.get("model_ready_path", "data/modeling/model_ready.parquet"))
    train_path = Path(ds_cfg.get("train_path", "data/modeling/train.parquet"))
    test_path = Path(ds_cfg.get("test_path", "data/modeling/test.parquet"))
    metadata_path = Path(ds_cfg.get("feature_metadata_path", "data/modeling/feature_metadata.json"))
    stage_dir = Path("artifacts/stages/00_build_model_dataset")
    return {
        "model_ready_path": model_ready,
        "train_path": train_path,
        "test_path": test_path,
        "metadata_path": metadata_path,
        "stage_dir": stage_dir,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _build_feature_map(
    work_df: pd.DataFrame,
    generation_lagged_features: list[str],
) -> dict[str, list[str]]:
    gen_core_tokens = ("photovoltaics", "wind_onshore", "lignite", "fossil_gas", "hard_coal")
    gen_core = sorted([c for c in generation_lagged_features if any(tok in c for tok in gen_core_tokens)])

    feature_map: dict[str, list[str]] = {
        "quantile_lag24_lag168": ["lag_24", "lag_168"],
        "quantile_lag24_lag168_hour_dow": ["lag_24", "lag_168", "day_of_week", "hour"],
        "quantile_lag24_lag168_hour_dow_weekend": [
            "lag_24",
            "lag_168",
            "day_of_week",
            "hour",
            "is_weekend",
        ],
        "quantile_lag24_lag168_hour_dow_gen": ["lag_24", "lag_168", "day_of_week", "hour"] + generation_lagged_features,
        "quantile_lag24_lag168_hour_dow_weekend_gen": [
            "lag_24",
            "lag_168",
            "day_of_week",
            "hour",
            "is_weekend",
        ]
        + generation_lagged_features,
        "quantile_lag24_lag168_hour_dow_weekend_season": [
            "lag_24",
            "lag_168",
            "day_of_week",
            "hour",
            "is_weekend",
            "season_bucket",
        ],
        "quantile_lag24_lag168_hour_dow_weekend_season_gen": [
            "lag_24",
            "lag_168",
            "day_of_week",
            "hour",
            "is_weekend",
            "season_bucket",
        ]
        + generation_lagged_features,
        "lgbm_lag_only": ["lag_24", "lag_168"],
        "lgbm_lag_time": ["lag_24", "lag_168", "day_of_week", "hour"],
        "lgbm_lag_time_weekend": ["lag_24", "lag_168", "day_of_week", "hour", "is_weekend"],
        "lgbm_lag_time_season": ["lag_24", "lag_168", "day_of_week", "hour", "season_bucket"],
        "lgbm_lag_time_gen_core": ["lag_24", "lag_168", "day_of_week", "hour"] + gen_core,
        "lgbm_lag_time_gen": ["lag_24", "lag_168", "day_of_week", "hour"] + generation_lagged_features,
    }

    out: dict[str, list[str]] = {}
    for key, cols in feature_map.items():
        filtered = [c for c in cols if c in work_df.columns]
        out[key] = list(dict.fromkeys(filtered))
    return out


def _resolve_candidates(
    cfg: dict[str, Any],
    feature_map: dict[str, list[str]],
) -> tuple[list[str], list[str]]:
    benchmark_cfg = cfg.get("benchmark", {})
    baseline_cfg = benchmark_cfg.get("tune", {})
    baseline_candidates_cfg = list(baseline_cfg.get("candidates", []))

    if baseline_candidates_cfg:
        baseline_candidates = [c for c in baseline_candidates_cfg if c in feature_map]
    else:
        baseline_candidates = [
            c
            for c in [
                "quantile_lag24_lag168",
                "quantile_lag24_lag168_hour_dow",
                "quantile_lag24_lag168_hour_dow_weekend",
                "quantile_lag24_lag168_hour_dow_gen",
                "quantile_lag24_lag168_hour_dow_weekend_gen",
                "quantile_lag24_lag168_hour_dow_weekend_season",
                "quantile_lag24_lag168_hour_dow_weekend_season_gen",
            ]
            if c in feature_map
        ]

    locked_baseline = str(benchmark_cfg.get("locked_baseline", "quantile_lag24_lag168_hour_dow"))
    if locked_baseline in feature_map and locked_baseline not in baseline_candidates:
        baseline_candidates.append(locked_baseline)

    lgbm_cfg = cfg.get("lightgbm_tuning", {})
    lgbm_candidates_cfg = list(lgbm_cfg.get("candidates", []))
    if lgbm_candidates_cfg:
        lgbm_candidates = [c for c in lgbm_candidates_cfg if c in feature_map]
    else:
        lgbm_candidates = [
            c
            for c in [
                "lgbm_lag_only",
                "lgbm_lag_time",
                "lgbm_lag_time_weekend",
                "lgbm_lag_time_season",
                "lgbm_lag_time_gen_core",
                "lgbm_lag_time_gen",
            ]
            if c in feature_map
        ]

    return baseline_candidates, lgbm_candidates


def build_model_dataset() -> None:
    params = load_params()
    cfg = params["model_training"]
    paths = _dataset_paths(cfg)

    seed = int(params["global"]["seed"])
    set_global_seed(seed)

    input_path = Path(cfg["input_path"])
    time_col = str(cfg["time_col"])
    target_col = str(cfg["target_col"])
    lags_hours = [int(v) for v in cfg["lags_hours"]]
    generation_features = list(cfg.get("generation_features", []))
    generation_lag_hours = [int(v) for v in cfg.get("generation_lag_hours", lags_hours)]
    train_fraction = float(cfg["split"]["train_fraction"])

    logging.info("Reading validated data from %s", input_path)
    df_data = pd.read_parquet(input_path).sort_values(time_col).reset_index(drop=True)

    work_df = df_data.copy()
    for lag_h in lags_hours:
        work_df[f"lag_{lag_h}"] = work_df[target_col].shift(lag_h)

    work_df["day_of_week"] = work_df[time_col].dt.dayofweek
    work_df["hour"] = work_df[time_col].dt.hour
    work_df["is_weekend"] = (work_df["day_of_week"] >= 5).astype(int)
    work_df["month"] = work_df[time_col].dt.month.astype(int)
    season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    work_df["season_bucket"] = work_df["month"].map(season_map).astype(int)

    generation_base_features = [c for c in generation_features if c in work_df.columns and not work_df[c].isna().all()]
    generation_lagged_features: list[str] = []
    for base_col in generation_base_features:
        for lag_h in generation_lag_hours:
            lag_col = f"{base_col}_lag_{lag_h}"
            work_df[lag_col] = work_df[base_col].shift(lag_h)
            generation_lagged_features.append(lag_col)

    required_cols = [f"lag_{h}" for h in lags_hours] + generation_lagged_features
    work_df = work_df.dropna(subset=required_cols).copy()

    split_idx = int(len(work_df) * train_fraction)
    train_df = work_df.iloc[:split_idx].copy()
    test_df = work_df.iloc[split_idx:].copy()

    feature_map = _build_feature_map(work_df, generation_lagged_features)
    baseline_candidates, lgbm_candidates = _resolve_candidates(cfg, feature_map)

    quantiles = sorted(float(q) for q in cfg["quantiles"])
    metadata = {
        "seed": seed,
        "target_col": target_col,
        "time_col": time_col,
        "quantiles": quantiles,
        "nominal_coverage": float(quantiles[-1] - quantiles[0]),
        "target_lower_bound": float(train_df[target_col].min()),
        "target_upper_bound": float(train_df[target_col].max()),
        "rows_total": int(len(work_df)),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "generation_lagged_feature_count": int(len(generation_lagged_features)),
        "feature_map": feature_map,
        "baseline_candidates": baseline_candidates,
        "lgbm_candidates": lgbm_candidates,
    }

    for key in ["model_ready_path", "train_path", "test_path", "metadata_path"]:
        paths[key].parent.mkdir(parents=True, exist_ok=True)

    logging.info("Writing model-ready dataset to %s", paths["model_ready_path"])
    work_df.to_parquet(paths["model_ready_path"], index=False)
    train_df.to_parquet(paths["train_path"], index=False)
    test_df.to_parquet(paths["test_path"], index=False)
    _write_json(paths["metadata_path"], metadata)

    stage_dir = paths["stage_dir"]
    stage_dir.mkdir(parents=True, exist_ok=True)
    _write_json(stage_dir / "metrics.json", metadata)
    pd.DataFrame(
        [
            {"split": "train", "rows": len(train_df)},
            {"split": "test", "rows": len(test_df)},
        ]
    ).to_csv(stage_dir / "summary.csv", index=False)
    _write_json(stage_dir / "feature_spec.json", {"feature_map": feature_map})

    logging.info("Model dataset build complete")


if __name__ == "__main__":
    build_model_dataset()

