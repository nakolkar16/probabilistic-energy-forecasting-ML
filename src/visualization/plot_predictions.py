from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns
import yaml

from src.modeling.quantile_columns import low_median_high_quantiles, quantile_cols


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)


def load_params() -> dict[str, Any]:
    with open(Path("params.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _paths(cfg: dict[str, Any]) -> dict[str, Path]:
    infer_cfg = cfg.get("inference", {})
    return {
        "pred_dir": Path(infer_cfg.get("output_dir", "artifacts/predictions/models")),
        "manifest_path": Path(infer_cfg.get("manifest_path", "artifacts/predictions/prediction_manifest.csv")),
        "fig_dir": Path("artifacts/figures/predictions"),
        "stage_dir": Path("artifacts/stages/09_plot_predictions"),
    }


def _resolve_plot_dates(
    df: pd.DataFrame,
    local_time_col: str,
    plot_dates_cfg: list[Any] | None,
) -> list[pd.Timestamp]:
    available_dates = sorted(pd.to_datetime(df[local_time_col]).dt.date.unique().tolist())
    if not available_dates:
        raise ValueError("No dates available in prediction data.")

    if not plot_dates_cfg:
        return [pd.Timestamp(available_dates[-1])]

    selected_dates: list[pd.Timestamp] = []
    available_set = set(available_dates)
    for raw_date in plot_dates_cfg:
        try:
            date_obj = pd.Timestamp(str(raw_date)).date()
        except Exception as exc:
            raise ValueError(f"Invalid date in model_training.inference.plot_dates: {raw_date}") from exc
        if date_obj not in available_set:
            raise ValueError(
                f"Requested plot date {date_obj} not found in prediction data. "
                f"Available range: {available_dates[0]} to {available_dates[-1]}"
            )
        selected_dates.append(pd.Timestamp(date_obj))

    return sorted(list({d for d in selected_dates}))


def _save_overwrite(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    fig.savefig(path, dpi=160, bbox_inches="tight")


def _resolve_prediction_files(paths: dict[str, Path]) -> list[tuple[str, Path]]:
    if paths["manifest_path"].exists():
        manifest_df = pd.read_csv(paths["manifest_path"])
        rows: list[tuple[str, Path]] = []
        for _, rec in manifest_df.iterrows():
            rows.append((str(rec["model_name"]), Path(str(rec["prediction_path"]))))
        return rows

    rows = []
    for p in sorted(paths["pred_dir"].glob("*_predictions.parquet")):
        rows.append((p.stem.replace("_predictions", ""), p))
    return rows


def plot_predictions() -> None:
    params = load_params()
    cfg = params["model_training"]
    target_col = str(cfg["target_col"])
    time_col = str(cfg["time_col"])
    quantiles = sorted(float(q) for q in cfg["quantiles"])
    q_low, q_med, q_high = low_median_high_quantiles(quantiles)
    pred_cols = quantile_cols(quantiles)
    q_low_col = pred_cols[q_low]
    q_med_col = pred_cols[q_med]
    q_high_col = pred_cols[q_high]
    q_low_label = int(round(q_low * 100))
    q_med_label = int(round(q_med * 100))
    q_high_label = int(round(q_high * 100))
    inference_cfg = cfg.get("inference", {})
    plot_timezone = str(inference_cfg.get("plot_timezone", "Europe/Berlin"))
    plot_dates_cfg = inference_cfg.get("plot_dates", [])
    primary_model = str(inference_cfg.get("primary_model", "champion"))

    paths = _paths(cfg)
    pred_files = _resolve_prediction_files(paths)
    if not pred_files:
        raise FileNotFoundError("No prediction files found. Run predict stage first.")

    # Use first model to resolve plot dates; all model predictions come from same scoring batch.
    base_df = pd.read_parquet(pred_files[0][1]).copy()
    required_cols = [time_col, q_low_col, q_med_col, q_high_col]
    missing = [c for c in required_cols if c not in base_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in predictions file: {missing}")
    base_df[time_col] = pd.to_datetime(base_df[time_col], utc=True)
    base_local_time_col = f"{time_col}_local"
    base_df[base_local_time_col] = base_df[time_col].dt.tz_convert(plot_timezone)
    base_df = base_df.sort_values(time_col).reset_index(drop=True)

    selected_plot_days = _resolve_plot_dates(base_df, base_local_time_col, plot_dates_cfg)

    fig_dir = paths["fig_dir"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    primary_latest_png = fig_dir / "day_ahead_actual_vs_pred_latest.png"

    generated_pngs: list[str] = []
    rows_per_plot: dict[str, int] = {}
    has_actual_any = False
    sns.set_theme(style="whitegrid", context="talk")

    for model_name, pred_path in pred_files:
        if not pred_path.exists():
            continue
        df = pd.read_parquet(pred_path).copy()
        required_cols = [time_col, q_low_col, q_med_col, q_high_col]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in '{pred_path}': {missing}")

        df[time_col] = pd.to_datetime(df[time_col], utc=True)
        local_time_col = f"{time_col}_local"
        df[local_time_col] = df[time_col].dt.tz_convert(plot_timezone)
        df = df.sort_values(time_col).reset_index(drop=True)

        model_generated: list[Path] = []
        for plot_day in selected_plot_days:
            day_mask = df[local_time_col].dt.date == plot_day.date()
            day_df = df.loc[day_mask].copy()
            if day_df.empty:
                continue

            fig, ax = plt.subplots(figsize=(13, 5.8))
            has_actual = target_col in day_df.columns and day_df[target_col].notna().any()
            has_actual_any = has_actual_any or has_actual
            if has_actual:
                ax.plot(
                    day_df[local_time_col],
                    day_df[target_col],
                    label="Actual",
                    linewidth=2.4,
                    color="#2F4858",
                )
            ax.plot(
                day_df[local_time_col],
                day_df[q_med_col],
                label=f"Predicted p{q_med_label}",
                linewidth=2.4,
                color="#1F78B4",
            )
            ax.fill_between(
                day_df[local_time_col],
                day_df[q_low_col],
                day_df[q_high_col],
                alpha=0.18,
                label=f"Prediction band (p{q_low_label}-p{q_high_label})",
                color="#4EA3D8",
            )

            ax.set_title(
                f"Day-Ahead Forecast vs Actual | {model_name} ({plot_day.date()})",
                fontsize=16,
                pad=14,
            )
            ax.set_xlabel(f"Timestamp ({plot_timezone})")
            ax.set_ylabel("Residual Load (MWh)")
            ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f"{x:,.0f}"))
            ax.grid(axis="y", alpha=0.28, linewidth=0.8)
            ax.grid(axis="x", alpha=0.16, linewidth=0.6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(frameon=False, loc="upper left")
            fig.tight_layout()

            dated_png = fig_dir / f"day_ahead_actual_vs_pred_{model_name}_{plot_day.date()}.png"
            _save_overwrite(fig, dated_png)
            plt.close(fig)

            generated_pngs.append(str(dated_png))
            rows_per_plot[f"{model_name}:{plot_day.date()}"] = int(len(day_df))
            model_generated.append(dated_png)

        if model_generated:
            model_latest_png = fig_dir / f"day_ahead_actual_vs_pred_{model_name}_latest.png"
            if model_latest_png.exists():
                model_latest_png.unlink()
            model_latest_png.write_bytes(model_generated[-1].read_bytes())
            generated_pngs.append(str(model_latest_png))

            if model_name == primary_model:
                if primary_latest_png.exists():
                    primary_latest_png.unlink()
                primary_latest_png.write_bytes(model_generated[-1].read_bytes())
                generated_pngs.append(str(primary_latest_png))

    if not generated_pngs:
        raise ValueError("No plots generated for requested dates.")

    stage_dir = paths["stage_dir"]
    stage_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "prediction_dir": str(paths["pred_dir"]),
        "n_models_plotted": int(len(pred_files)),
        "model_names": [name for name, _ in pred_files],
        "plot_timezone": plot_timezone,
        "plot_days": [str(d.date()) for d in selected_plot_days],
        "rows_plotted_per_day": rows_per_plot,
        "has_actual": bool(has_actual_any),
        "generated_pngs": generated_pngs,
        "primary_model": primary_model,
        "primary_latest_png": str(primary_latest_png),
    }
    _write_json(stage_dir / "metrics.json", metrics)

    LOGGER.info("Saved %s plot file(s) across %s model(s)", len(generated_pngs), len(pred_files))


if __name__ == "__main__":
    plot_predictions()
