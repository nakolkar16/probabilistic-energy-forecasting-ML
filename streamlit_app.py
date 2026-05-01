from __future__ import annotations

import base64
from html import escape
import json
from pathlib import Path
from textwrap import dedent
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yaml

from src.modeling.model_artifacts import trusted_model_path
from src.modeling.quantile_columns import low_median_high_quantiles, quantile_cols


PROJECT_ROOT = Path(__file__).resolve().parent
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def load_params() -> dict[str, Any]:
    with open(PARAMS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def resolve_prediction_files(manifest_path: Path, pred_dir: Path) -> list[tuple[str, Path]]:
    """
    Returns [(model_name, prediction_path), ...] from manifest if available,
    otherwise falls back to scanning prediction directory.
    """
    files: list[tuple[str, Path]] = []
    if manifest_path.exists():
        manifest_df = pd.read_csv(manifest_path)
        for _, row in manifest_df.iterrows():
            model_name = str(row["model_name"])
            pred_path = Path(str(row["prediction_path"]))
            if not pred_path.is_absolute():
                pred_path = PROJECT_ROOT / pred_path
            files.append((model_name, pred_path))
        return files

    if pred_dir.exists():
        for p in sorted(pred_dir.glob("*_predictions.parquet")):
            files.append((p.stem.replace("_predictions", ""), p))
    return files


def ordered_model_names(model_names: list[str]) -> list[str]:
    order: list[str] = []
    if "baseline" in model_names:
        order.append("baseline")
    for name in ["champion", "challenger_1", "challenger_2"]:
        if name in model_names and name not in order:
            order.append(name)
    for name in sorted(model_names):
        if name not in order:
            order.append(name)
    return order


def build_model_labels(model_names: list[str]) -> dict[str, str]:
    labels: dict[str, str] = {}
    ordered = ordered_model_names(model_names)

    model_idx = 1
    for name in ordered:
        if name == "baseline":
            labels[name] = "Baseline"
        elif name == "champion":
            labels[name] = "Champion"
        elif name == "challenger_1":
            labels[name] = "Challenger 1"
        elif name == "challenger_2":
            labels[name] = "Challenger 2"
        else:
            labels[name] = f"Model {model_idx}"
            model_idx += 1
    return labels


def as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def gate_metrics_fresh(
    pred_files: list[tuple[str, Path]],
    manifest_path: Path,
    model_summary_path: Path,
    model_summary_df: pd.DataFrame | None,
) -> tuple[bool, str]:
    if model_summary_df is None or model_summary_df.empty:
        return False, "Gate metrics are missing. Regenerate artifacts with the offline DVC/MLflow pipeline."
    required_cols = {
        "model_name",
        "coverage_gate_pass",
        "crossing_gate_pass",
        "min_coverage_gate_pass",
        "overall_gate_pass",
    }
    missing_cols = sorted(required_cols - set(model_summary_df.columns))
    if missing_cols:
        return False, f"Gate metrics are incomplete. Missing columns: {missing_cols}."
    if not model_summary_path.exists():
        return False, f"Gate metrics file is missing: {model_summary_path}."

    expected_models = {name for name, _ in pred_files}
    evaluated_models = set(model_summary_df["model_name"].astype(str))
    missing_models = sorted(expected_models - evaluated_models)
    if missing_models:
        return False, f"Gate metrics do not cover all displayed prediction files. Missing: {missing_models}."

    metric_mtime = model_summary_path.stat().st_mtime
    freshness_sources = [path for _, path in pred_files if path.exists()]
    if manifest_path.exists():
        freshness_sources.append(manifest_path)
    stale_sources = [str(path) for path in freshness_sources if path.stat().st_mtime > metric_mtime]
    if stale_sources:
        return False, "Gate metrics are older than prediction artifacts. Regenerate artifacts with the offline pipeline."

    return True, ""


def approved_forecast_models(model_names: list[str], model_summary_df: pd.DataFrame | None) -> list[str]:
    if model_summary_df is None or model_summary_df.empty or "model_name" not in model_summary_df.columns:
        return []

    gate_col = "overall_gate_pass" if "overall_gate_pass" in model_summary_df.columns else "crossing_gate_pass"
    if gate_col not in model_summary_df.columns:
        return []

    gate_pass = model_summary_df[gate_col].map(as_bool)
    passed = set(model_summary_df.loc[gate_pass, "model_name"].astype(str))
    return [name for name in model_names if name in passed]


def blocked_forecast_models(model_names: list[str], model_summary_df: pd.DataFrame | None) -> list[str]:
    approved = set(approved_forecast_models(model_names, model_summary_df))
    return [name for name in model_names if name not in approved]


def resolve_q_model(model_map: dict[Any, Any], q: float = 0.5) -> Any:
    if q in model_map:
        return model_map[q]
    for key, model in model_map.items():
        try:
            if np.isclose(float(key), q):
                return model
        except Exception:
            continue
    raise KeyError(f"No model found for quantile={q}")


def feature_importance_df(bundle: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    feature_cols = list(bundle.get("feature_cols", []))
    family = str(bundle.get("family", "unknown"))

    if not feature_cols:
        return pd.DataFrame(columns=["feature", "importance"]), "No features found"

    q50_model = resolve_q_model(bundle["models"], q=0.5)
    if family == "quantile_regression":
        coef = np.asarray(q50_model.named_steps["model"].coef_, dtype=float)
        imp = np.abs(coef)
        label = "|coefficient| (q50)"
    elif family == "lightgbm_quantile":
        imp = np.asarray(q50_model.feature_importances_, dtype=float)
        label = "feature importance (q50)"
    else:
        return pd.DataFrame(columns=["feature", "importance"]), f"Unsupported family: {family}"

    df = pd.DataFrame({"feature": feature_cols, "importance": imp}).sort_values(
        "importance", ascending=False
    )
    return df, label


def feature_driver_group(feature: str) -> str:
    if feature.startswith("lag_"):
        return "Recent residual-load memory"
    if feature.startswith(("diff_", "rolling_")):
        return "Load momentum and volatility"
    if feature.startswith(("photovoltaics_", "wind_")):
        return "Renewable generation"
    if feature.startswith(("fossil_gas_", "lignite_", "hard_coal_", "gen_hydro_")):
        return "Dispatchable generation mix"
    if feature in {
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "season_bucket",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
    }:
        return "Calendar pattern"
    return "Other model signal"


def hour_phrase(hours: str) -> str:
    return "1 hour" if str(hours) == "1" else f"{hours} hours"


def hour_modifier(hours: str) -> str:
    return "1-hour" if str(hours) == "1" else f"{hours}-hour"


def friendly_feature_label(feature: str) -> str:
    special = {
        "lag_1": "Residual load 1 hour ago",
        "lag_24": "Residual load yesterday, same hour",
        "lag_168": "Residual load last week, same hour",
        "diff_1": "1-hour residual-load change",
        "diff_24": "Daily residual-load change",
        "diff_168": "Weekly residual-load change",
        "hour_sin": "Hour-of-day pattern",
        "hour_cos": "Hour-of-day pattern",
        "dow_sin": "Day-of-week pattern",
        "dow_cos": "Day-of-week pattern",
        "month_sin": "Seasonal pattern",
        "month_cos": "Seasonal pattern",
        "is_weekend": "Weekend flag",
        "season_bucket": "Season bucket",
        "hour": "Hour of day",
        "day_of_week": "Day of week",
        "month": "Month",
    }
    if feature in special:
        return special[feature]
    if feature.startswith("rolling_mean_"):
        hours = feature.removeprefix("rolling_mean_")
        return f"{hour_modifier(hours)} average residual load"
    if feature.startswith("rolling_std_"):
        hours = feature.removeprefix("rolling_std_")
        return f"{hour_modifier(hours)} residual-load volatility"
    if feature.startswith("rolling_min_"):
        hours = feature.removeprefix("rolling_min_")
        return f"{hour_modifier(hours)} minimum residual load"
    if feature.startswith("rolling_max_"):
        hours = feature.removeprefix("rolling_max_")
        return f"{hour_modifier(hours)} maximum residual load"
    if feature.startswith("lag_"):
        hours = feature.removeprefix("lag_")
        return f"Residual load {hour_phrase(hours)} ago"
    for prefix, label in {
        "fossil_gas_mwh_lag_": "Gas generation",
        "lignite_mwh_lag_": "Lignite generation",
        "hard_coal_mwh_lag_": "Hard-coal generation",
        "photovoltaics_mwh_lag_": "Solar generation",
        "wind_onshore_mwh_lag_": "Onshore wind generation",
        "gen_hydro_pumped_storage_mwh_lag_": "Pumped-storage generation",
    }.items():
        if feature.startswith(prefix):
            hours = feature.removeprefix(prefix)
            return f"{label} {hour_phrase(hours)} ago"
    return feature.replace("_", " ")


def driver_group_summary(importance_df: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    if importance_df.empty or "feature" not in importance_df.columns or "importance" not in importance_df.columns:
        return pd.DataFrame(columns=["Driver group", "Share", "Example features"])
    top_df = importance_df.head(top_n).copy()
    top_df["driver_group"] = top_df["feature"].map(feature_driver_group)
    total_importance = float(top_df["importance"].sum())
    rows: list[dict[str, Any]] = []
    for group, group_df in top_df.groupby("driver_group", sort=False):
        share = float(group_df["importance"].sum() / total_importance) if total_importance else 0.0
        readable_examples = []
        for feature in group_df["feature"]:
            label = friendly_feature_label(feature)
            if label not in readable_examples:
                readable_examples.append(label)
            if len(readable_examples) == 3:
                break
        examples = ", ".join(readable_examples)
        rows.append(
            {
                "Driver group": group,
                "_share_value": share,
                "Share": f"{share:.0%}",
                "Example features": examples,
            }
        )
    return pd.DataFrame(rows).sort_values("_share_value", ascending=False).drop(columns="_share_value").reset_index(drop=True)


def load_prediction_df(path: Path, time_col: str, plot_tz: str) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df["time_local"] = df[time_col].dt.tz_convert(plot_tz)
    df = df.sort_values(time_col).reset_index(drop=True)
    return df


def forecast_plot(
    forecast_df: pd.DataFrame,
    model_name: str,
    time_col_local: str,
    target_col: str,
    plot_tz: str,
    horizon_hours: int,
    q_low_col: str,
    q_med_col: str,
    q_high_col: str,
    q_low_label: int,
    q_med_label: int,
    q_high_label: int,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 4.8))

    has_actual = target_col in forecast_df.columns and forecast_df[target_col].notna().any()
    if has_actual:
        ax.plot(forecast_df[time_col_local], forecast_df[target_col], label="Actual", linewidth=2.0, color="#2F4858")

    ax.plot(
        forecast_df[time_col_local],
        forecast_df[q_med_col],
        label=f"Predicted p{q_med_label}",
        linewidth=2.0,
        color="#1F78B4",
    )
    ax.fill_between(
        forecast_df[time_col_local],
        forecast_df[q_low_col],
        forecast_df[q_high_col],
        alpha=0.18,
        label=f"p{q_low_label}-p{q_high_label} band",
        color="#4EA3D8",
    )

    start_label = forecast_df[time_col_local].iloc[0].strftime("%Y-%m-%d %H:%M")
    end_label = forecast_df[time_col_local].iloc[-1].strftime("%Y-%m-%d %H:%M")
    ax.set_title(f"{horizon_hours}-Hour Forecast | {model_name} | {start_label} to {end_label}", fontsize=14, pad=12)
    ax.set_xlabel(f"Time ({plot_tz})")
    ax.set_ylabel("Residual Load (MWh)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def image_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def compact_html(html: str) -> str:
    return "\n".join(line.strip() for line in dedent(html).splitlines() if line.strip())


def fmt_number(value: Any, decimals: int = 0, suffix: str = "") -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{num:,.{decimals}f}{suffix}"


def fmt_percent(value: Any, decimals: int = 1) -> str:
    try:
        num = float(value) * 100
    except (TypeError, ValueError):
        return "n/a"
    return f"{num:.{decimals}f}%"


def metric_from_row(row: pd.Series | None, col: str, default: str = "n/a") -> str:
    if row is None or col not in row or pd.isna(row[col]):
        return default
    return str(row[col])


def best_landing_row(model_summary_df: pd.DataFrame | None) -> pd.Series | None:
    if model_summary_df is None or model_summary_df.empty:
        return None
    df = model_summary_df.copy()
    if "model_name" in df.columns and (df["model_name"].astype(str) == "champion").any():
        return df[df["model_name"].astype(str) == "champion"].iloc[0]
    if "pinball_mean" in df.columns:
        return df.sort_values("pinball_mean").iloc[0]
    return df.iloc[0]


def gate_pill(passed: Any) -> str:
    if as_bool(passed):
        return '<span class="rl-pill rl-green">PASS</span>'
    return '<span class="rl-pill rl-red">FAIL</span>'


def model_role(model_name: str) -> str:
    if model_name == "champion":
        return '<span class="rl-pill rl-gold">Production</span>'
    if model_name == "baseline":
        return '<span class="rl-pill rl-blue">Reference</span>'
    return '<span class="rl-pill rl-blue">Shadow</span>'


def model_descriptor(model_name: str, champion_meta: dict[str, Any] | None) -> str:
    if model_name == "champion" and champion_meta:
        family = str(champion_meta.get("champion_family", "champion model")).replace("_", " ")
        selected = str(champion_meta.get("selected_candidate", "selected candidate"))
        return f"{family} · {selected}"
    if model_name.startswith("challenger"):
        return "LightGBM quantile · tuned challenger"
    if model_name == "baseline":
        return "Locked quantile baseline"
    return "Forecast model"


def landing_model_table(
    model_summary_df: pd.DataFrame | None,
    model_label_map: dict[str, str],
    champion_meta: dict[str, Any] | None,
) -> str:
    if model_summary_df is None or model_summary_df.empty:
        return '<div class="rl-empty">Model comparison metrics are missing. Produce them with the offline DVC/MLflow pipeline.</div>'

    df = model_summary_df.copy()
    ordered_names = ordered_model_names(df["model_name"].astype(str).tolist())
    rows: list[str] = []
    for name in ordered_names:
        row_df = df[df["model_name"].astype(str) == name]
        if row_df.empty:
            continue
        row = row_df.iloc[0]
        label = escape(model_label_map.get(name, name.replace("_", " ").title()))
        descriptor = escape(model_descriptor(name, champion_meta))
        rows.append(
            f"""
            <tr>
              <td><strong>{label}</strong><div class="rl-model-sub">{descriptor}</div></td>
              <td>{model_role(name)}</td>
              <td><strong>{fmt_number(row.get("pinball_mean"), 1)}</strong></td>
              <td>{fmt_number(row.get("mae_p50"), 0)}</td>
              <td>{fmt_percent(row.get("coverage"))}</td>
              <td>{fmt_percent(row.get("output_crossing_rate", row.get("crossing_rate")))}</td>
              <td>{gate_pill(row.get("overall_gate_pass", False))}</td>
            </tr>
            """
        )
    return compact_html("\n".join(rows))


def page_overview(
    model_summary_df: pd.DataFrame | None,
    pilot_metrics: dict[str, Any] | None,
    champion_meta: dict[str, Any] | None,
    model_label_map: dict[str, str],
) -> None:
    landing_row = best_landing_row(model_summary_df)
    rows_evaluated = metric_from_row(landing_row, "rows_evaluated")
    coverage = fmt_percent(metric_from_row(landing_row, "coverage"))
    pinball = fmt_number(metric_from_row(landing_row, "pinball_mean"), 1)
    mae = fmt_number(metric_from_row(landing_row, "mae_p50"), 0, " MWh")
    r2 = fmt_number(metric_from_row(landing_row, "r2_p50"), 3)
    interval_width = fmt_number(metric_from_row(landing_row, "interval_width"), 0, " MWh")
    crossings = fmt_percent(metric_from_row(landing_row, "output_crossing_rate", "0"))
    n_models = int(pilot_metrics.get("n_models_evaluated", 0)) if pilot_metrics else 0
    passed_models = 0
    if model_summary_df is not None and not model_summary_df.empty and "overall_gate_pass" in model_summary_df.columns:
        passed_models = int(model_summary_df["overall_gate_pass"].map(as_bool).sum())

    latest_plot = image_data_uri(PROJECT_ROOT / "artifacts/figures/predictions/day_ahead_actual_vs_pred_latest.png")
    forecast_visual = (
        f'<img class="rl-forecast-img" src="{latest_plot}" alt="Latest day-ahead forecast plot" />'
        if latest_plot
        else compact_html(
            """
        <svg class="rl-chart-svg" viewBox="0 0 600 280" preserveAspectRatio="none" aria-label="Day-ahead forecast preview">
          <defs>
            <linearGradient id="bandGrad" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stop-color="#4ea3ff" stop-opacity="0.55"/>
              <stop offset="100%" stop-color="#34d399" stop-opacity="0.18"/>
            </linearGradient>
          </defs>
          <line class="rl-grid-line" x1="40" y1="40" x2="590" y2="40" />
          <line class="rl-grid-line" x1="40" y1="100" x2="590" y2="100" />
          <line class="rl-grid-line" x1="40" y1="160" x2="590" y2="160" />
          <line class="rl-grid-line" x1="40" y1="220" x2="590" y2="220" />
          <path class="rl-band" d="M 40,180 L 70,170 L 100,150 L 130,120 L 160,90 L 190,70 L 220,80 L 250,95 L 280,105 L 310,100 L 340,85 L 370,75 L 400,80 L 430,95 L 460,115 L 490,140 L 520,160 L 550,175 L 580,185 L 580,235 L 550,225 L 520,210 L 490,190 L 460,165 L 430,145 L 400,130 L 370,125 L 340,135 L 310,150 L 280,155 L 250,145 L 220,130 L 190,120 L 160,140 L 130,170 L 100,200 L 70,220 L 40,230 Z" />
          <path class="rl-line-p50" d="M 40,205 L 70,195 L 100,175 L 130,145 L 160,115 L 190,95 L 220,105 L 250,120 L 280,130 L 310,125 L 340,110 L 370,100 L 400,105 L 430,120 L 460,140 L 490,165 L 520,185 L 550,200 L 580,210" />
          <path class="rl-line-actual" d="M 40,210 L 70,198 L 100,180 L 130,150 L 160,118 L 190,100 L 220,108 L 250,125 L 280,128 L 310,128 L 340,108 L 370,98 L 400,108 L 430,118 L 460,142 L 490,170 L 520,188 L 550,202 L 580,212" />
        </svg>
        """
        )
    )

    insight_cards: list[str] = []
    for title, desc, image_name in [
        (
            "Residual load trend",
            "The hourly series is noisy, while the weekly structure motivates lagged and smoothed temporal features.",
            "residual_load_time_series.png",
        ),
        (
            "Monthly residual share",
            "Seasonal renewable production changes how much dispatchable supply must cover.",
            "monthly_residual_share_pct.png",
        ),
        (
            "Grid load vs residual load",
            "Monthly totals show how renewable generation changes the gap between total demand and residual demand.",
            "monthly_grid_vs_residual_twh.png",
        ),
    ]:
        img_src = image_data_uri(PROJECT_ROOT / "artifacts/figures" / image_name)
        image_html = f'<img src="{img_src}" alt="{escape(title)}" />' if img_src else ""
        insight_cards.append(
            compact_html(
                f"""
            <div class="rl-insight">
              {image_html}
              <div class="rl-insight-body">
                <div class="rl-insight-title">{escape(title)}</div>
                <div class="rl-insight-desc">{escape(desc)}</div>
              </div>
            </div>
            """
            )
        )

    model_rows = landing_model_table(model_summary_df, model_label_map, champion_meta)

    st.markdown(
        compact_html(
            """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
          :root {
            --rl-bg: #0b1020;
            --rl-panel: #121935;
            --rl-panel-2: #161e3d;
            --rl-border: #1f2a52;
            --rl-text: #e7ecff;
            --rl-muted: #9aa6cf;
            --rl-accent: #4ea3ff;
            --rl-accent-2: #7dd3fc;
            --rl-good: #34d399;
            --rl-warn: #fbbf24;
            --rl-bad: #f87171;
            --rl-grad: linear-gradient(135deg, #4ea3ff 0%, #7dd3fc 50%, #34d399 100%);
          }
          .stApp {
            background: linear-gradient(180deg, #0b1020 0%, #101735 45%, #0b1020 100%);
            color: var(--rl-text);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
          }
          header[data-testid="stHeader"] { background: rgba(11,16,32,.78); }
          section[data-testid="stSidebar"] { background: #0f1530; border-right: 1px solid var(--rl-border); }
          .block-container { max-width: 1240px; padding-top: 1.2rem; }
          .rl-shell { color: var(--rl-text); font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
          .rl-nav {
            position: sticky; top: 0; z-index: 5;
            display: flex; align-items: center; justify-content: space-between;
            min-height: 62px; margin: -8px 0 28px; padding: 0 4px 16px;
            border-bottom: 1px solid var(--rl-border);
          }
          .rl-brand { display: flex; align-items: center; gap: 10px; font-weight: 800; letter-spacing: .2px; }
          .rl-brand-mark {
            width: 30px; height: 30px; border-radius: 8px; background: var(--rl-grad);
            display: grid; place-items: center; color: #04122b; font-weight: 800;
          }
          .rl-nav-links { display: flex; gap: 24px; align-items: center; }
          .rl-nav-links a { color: var(--rl-muted); text-decoration: none; font-size: 14px; font-weight: 600; }
          .rl-nav-links a:hover { color: var(--rl-text); }
          .rl-nav-cta {
            background: var(--rl-grad); color: #04122b !important; padding: 9px 16px;
            border-radius: 10px; text-decoration: none; font-weight: 700;
          }
          .rl-hero { display: grid; grid-template-columns: 1.12fr 1fr; gap: 52px; align-items: center; padding: 34px 0 54px; }
          .rl-eyebrow {
            display: inline-flex; align-items: center; gap: 8px; color: var(--rl-accent-2);
            font-weight: 700; font-size: 13px; background: rgba(125,211,252,.08);
            border: 1px solid rgba(125,211,252,.25); padding: 6px 12px; border-radius: 999px;
          }
          .rl-dot { width: 8px; height: 8px; border-radius: 999px; background: var(--rl-good); box-shadow: 0 0 12px var(--rl-good); }
          .rl-title { font-size: clamp(38px, 5vw, 64px); line-height: 1.05; margin: 18px 0; font-weight: 800; letter-spacing: 0; }
          .rl-grad-text { background: var(--rl-grad); -webkit-background-clip: text; background-clip: text; color: transparent; }
          .rl-lede { font-size: 18px; color: var(--rl-muted); max-width: 590px; line-height: 1.65; }
          .rl-hero-cta { display: flex; gap: 12px; margin-top: 28px; flex-wrap: wrap; }
          .rl-btn {
            padding: 12px 18px; border-radius: 12px; font-weight: 700; font-size: 14px;
            text-decoration: none; border: 1px solid var(--rl-border); color: var(--rl-text) !important;
            background: var(--rl-panel); display: inline-block;
          }
          .rl-btn-primary { background: var(--rl-grad); color: #04122b !important; border-color: transparent; }
          .rl-hero-stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-top: 36px; max-width: 590px; }
          .rl-hero-stat { border: 1px solid var(--rl-border); background: rgba(18,25,53,.7); border-radius: 12px; padding: 14px 16px; }
          .rl-hero-stat .v { font-weight: 800; font-size: 22px; }
          .rl-hero-stat .l { color: var(--rl-muted); font-size: 12px; margin-top: 2px; }
          .rl-card {
            background: linear-gradient(180deg, rgba(22,30,61,.96), rgba(18,25,53,.96));
            border: 1px solid var(--rl-border); border-radius: 18px; padding: 22px;
            box-shadow: 0 30px 80px -30px rgba(0,0,0,.7);
          }
          .rl-card-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; }
          .rl-card-title { font-weight: 700; font-size: 14px; color: var(--rl-muted); }
          .rl-badge-live { color: var(--rl-good); border: 1px solid rgba(52,211,153,.4); background: rgba(52,211,153,.08); font-size: 11px; padding: 4px 8px; border-radius: 999px; }
          .rl-legend { display: flex; gap: 14px; flex-wrap: wrap; margin: 8px 0 14px; font-size: 12px; color: var(--rl-muted); }
          .rl-swatch { width: 12px; height: 12px; border-radius: 3px; display: inline-block; margin-right: 6px; vertical-align: -2px; }
          .rl-forecast-img { width: 100%; border-radius: 12px; border: 1px solid var(--rl-border); background: white; display: block; }
          .rl-chart-svg { width: 100%; height: 280px; display: block; }
          .rl-grid-line { stroke: rgba(255,255,255,.08); stroke-width: 1; }
          .rl-band { fill: url(#bandGrad); opacity: .55; }
          .rl-line-p50 { fill: none; stroke: var(--rl-accent-2); stroke-width: 2.2; }
          .rl-line-actual { fill: none; stroke: #fff; stroke-width: 1.6; stroke-dasharray: 4 3; opacity: .9; }
          .rl-mini-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 14px; }
          .rl-mini { border-radius: 10px; padding: 10px; background: rgba(78,163,255,.08); border: 1px solid rgba(78,163,255,.3); }
          .rl-mini .label { font-size: 11px; color: var(--rl-muted); }
          .rl-mini .value { font-weight: 800; font-size: 18px; }
          .rl-section { padding: 56px 0; border-top: 1px solid var(--rl-border); }
          .rl-section-title { font-size: 12px; letter-spacing: .18em; text-transform: uppercase; color: var(--rl-accent-2); font-weight: 800; }
          .rl-h2 { font-size: clamp(28px, 3.4vw, 40px); margin: 8px 0 14px; font-weight: 800; letter-spacing: 0; }
          .rl-section-lede { color: var(--rl-muted); max-width: 760px; font-size: 16px; line-height: 1.65; }
          .rl-kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-top: 28px; }
          .rl-kpi, .rl-step, .rl-stack-item, .rl-insight { border: 1px solid var(--rl-border); background: var(--rl-panel); border-radius: 14px; }
          .rl-kpi { padding: 18px; }
          .rl-kpi .label { color: var(--rl-muted); font-size: 12px; letter-spacing: .04em; text-transform: uppercase; }
          .rl-kpi .value { font-size: 28px; font-weight: 800; margin-top: 6px; }
          .rl-kpi .sub { font-size: 12px; color: var(--rl-muted); margin-top: 6px; }
          .rl-pass { color: var(--rl-good); font-weight: 800; }
          .rl-why {
            display: grid; grid-template-columns: 1fr 1.2fr; gap: 22px; align-items: start;
            margin-top: 30px; padding: 22px; border: 1px solid rgba(125,211,252,.28);
            border-radius: 14px; background: linear-gradient(135deg, rgba(78,163,255,.11), rgba(52,211,153,.08));
          }
          .rl-why-title { font-size: 22px; font-weight: 800; }
          .rl-why-copy { color: var(--rl-muted); line-height: 1.65; }
          .rl-why-list { display: grid; gap: 10px; margin-top: 4px; }
          .rl-why-item { color: var(--rl-text); font-weight: 700; }
          .rl-why-item span { color: var(--rl-muted); font-weight: 500; }
          .rl-pipeline { display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; margin-top: 28px; }
          .rl-step { padding: 16px; background: linear-gradient(180deg, rgba(22,30,61,.7), rgba(18,25,53,.7)); }
          .rl-step .num { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--rl-accent-2); }
          .rl-step .name { font-weight: 800; margin-top: 6px; font-size: 14px; }
          .rl-step .desc { color: var(--rl-muted); font-size: 12px; margin-top: 6px; line-height: 1.45; }
          .rl-table-wrap { border: 1px solid var(--rl-border); border-radius: 14px; overflow: hidden; margin-top: 28px; background: var(--rl-panel); }
          .rl-table { width: 100%; border-collapse: collapse; }
          .rl-table th, .rl-table td { padding: 14px 16px; text-align: left; font-size: 14px; border-bottom: 1px solid var(--rl-border); }
          .rl-table th { color: var(--rl-muted); background: rgba(78,163,255,.06); font-size: 12px; letter-spacing: .05em; text-transform: uppercase; }
          .rl-model-sub { color: var(--rl-muted); font-size: 12px; margin-top: 2px; }
          .rl-pill { display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 11px; font-weight: 800; letter-spacing: .04em; }
          .rl-green { background: rgba(52,211,153,.12); color: var(--rl-good); border: 1px solid rgba(52,211,153,.3); }
          .rl-red { background: rgba(248,113,113,.12); color: var(--rl-bad); border: 1px solid rgba(248,113,113,.35); }
          .rl-gold { background: rgba(251,191,36,.12); color: var(--rl-warn); border: 1px solid rgba(251,191,36,.35); }
          .rl-blue { background: rgba(78,163,255,.12); color: var(--rl-accent); border: 1px solid rgba(78,163,255,.3); }
          .rl-stack-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-top: 28px; }
          .rl-stack-item { padding: 16px; }
          .rl-stack-item .tag { font-size: 11px; color: var(--rl-accent-2); font-weight: 800; letter-spacing: .08em; text-transform: uppercase; }
          .rl-stack-item .nm { font-weight: 800; margin-top: 6px; }
          .rl-stack-item .ds { color: var(--rl-muted); font-size: 13px; margin-top: 6px; }
          .rl-insights { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-top: 28px; }
          .rl-insight { overflow: hidden; }
          .rl-insight img { display: block; width: 100%; height: 190px; object-fit: cover; border-bottom: 1px solid var(--rl-border); background: white; }
          .rl-insight-body { padding: 16px; }
          .rl-insight-title { font-weight: 800; }
          .rl-insight-desc { color: var(--rl-muted); font-size: 13px; margin-top: 6px; line-height: 1.5; }
          .rl-empty { color: var(--rl-muted); padding: 18px; border: 1px solid var(--rl-border); border-radius: 12px; margin-top: 20px; }
          .rl-footer { padding: 34px 0 10px; color: var(--rl-muted); font-size: 13px; border-top: 1px solid var(--rl-border); display: flex; justify-content: space-between; gap: 18px; flex-wrap: wrap; }
          @media (max-width: 980px) {
            .rl-hero { grid-template-columns: 1fr; }
            .rl-kpis, .rl-stack-grid { grid-template-columns: repeat(2, 1fr); }
            .rl-why { grid-template-columns: 1fr; }
            .rl-pipeline { grid-template-columns: repeat(2, 1fr); }
            .rl-insights { grid-template-columns: 1fr; }
            .rl-nav-links { display: none; }
          }
          @media (max-width: 640px) {
            .rl-hero-stats, .rl-mini-grid, .rl-kpis, .rl-pipeline, .rl-stack-grid { grid-template-columns: 1fr; }
            .rl-title { font-size: 38px; }
          }
        </style>
        """
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        compact_html(
            f"""
        <div class="rl-shell">
          <nav class="rl-nav">
            <div class="rl-brand">
              <div class="rl-brand-mark">R</div>
              <div>ResidualLoad<span style="color:var(--rl-muted)">.ml</span></div>
            </div>
            <div class="rl-nav-links">
              <a href="#overview">Overview</a>
              <a href="#pipeline">Pipeline</a>
              <a href="#models">Models</a>
              <a href="#stack">Tech Stack</a>
              <a href="#insights">Insights</a>
              <a class="rl-nav-cta" href="?page=Forecast">Launch App</a>
            </div>
          </nav>

          <header class="rl-hero">
            <div>
              <div class="rl-eyebrow"><span class="rl-dot"></span> Artifact-backed · Day-ahead inference dashboard</div>
              <div class="rl-title">Probabilistic forecasting for the <span class="rl-grad-text">German residual load</span>.</div>
              <div class="rl-lede">
                End-to-end ML system that predicts residual electricity demand with calibrated
                <strong>P10 / P50 / P90</strong> uncertainty intervals, versioned artifacts,
                and quality gates before forecast display.
              </div>
              <div class="rl-hero-cta">
                <a class="rl-btn rl-btn-primary" href="?page=Forecast">Open forecast viewer</a>
                <a class="rl-btn" href="#pipeline">See the pipeline</a>
                <a class="rl-btn" href="?page=Model%20Compare">Compare models</a>
              </div>
              <div class="rl-hero-stats">
                <div class="rl-hero-stat"><div class="v">{escape(rows_evaluated)}</div><div class="l">Hours evaluated</div></div>
                <div class="rl-hero-stat"><div class="v">{coverage}</div><div class="l">P10-P90 coverage</div></div>
                <div class="rl-hero-stat"><div class="v">{passed_models} / {n_models}</div><div class="l">Models passing gates</div></div>
              </div>
            </div>

            <div class="rl-card">
              <div class="rl-card-header">
                <div class="rl-card-title">Day-ahead forecast · Champion model</div>
                <span class="rl-badge-live">● Live</span>
              </div>
              <div class="rl-legend">
                <span><span class="rl-swatch" style="background:rgba(78,163,255,.4); border:1px solid var(--rl-accent-2)"></span>P10-P90 band</span>
                <span><span class="rl-swatch" style="background:var(--rl-accent-2)"></span>P50 forecast</span>
                <span><span class="rl-swatch" style="background:#fff"></span>Actual replay</span>
              </div>
              {forecast_visual}
              <div class="rl-mini-grid">
                <div class="rl-mini"><div class="label">Pinball score</div><div class="value">{pinball}</div></div>
                <div class="rl-mini"><div class="label">Avg interval width</div><div class="value">{interval_width}</div></div>
                <div class="rl-mini"><div class="label">Quantile crossings</div><div class="value">{crossings}</div></div>
              </div>
            </div>
          </header>

          <section class="rl-section" id="overview">
            <div class="rl-section-title">What it does</div>
            <div class="rl-h2">Quantified uncertainty for grid operations.</div>
            <div class="rl-section-lede">
              Residual load is demand left after wind and solar generation are subtracted.
              This system turns that target into a calibrated forecast band, so planning can
              reason about uncertainty instead of a single brittle point estimate.
            </div>
            <div class="rl-kpis">
              <div class="rl-kpi"><div class="label">P10-P90 Coverage</div><div class="value">{coverage}</div><div class="sub">Target-aware interval quality · <span class="rl-pass">tracked</span></div></div>
              <div class="rl-kpi"><div class="label">Pinball Mean</div><div class="value">{pinball}</div><div class="sub">Champion-style probabilistic score</div></div>
              <div class="rl-kpi"><div class="label">MAE P50</div><div class="value">{mae}</div><div class="sub">R² = {r2} · median forecast</div></div>
              <div class="rl-kpi"><div class="label">Quantile Crossings</div><div class="value">{crossings}</div><div class="sub">Repair checked before serving · <span class="rl-pass">gated</span></div></div>
            </div>
            <div class="rl-why">
              <div>
                <div class="rl-section-title">Why this matters</div>
                <div class="rl-why-title">Residual-load uncertainty is operational risk.</div>
              </div>
              <div>
                <div class="rl-why-copy">
                  Residual load is the demand that remains after wind and solar generation.
                  A calibrated forecast band helps estimate flexibility needs before they
                  become balancing problems.
                </div>
                <div class="rl-why-list">
                  <div class="rl-why-item">Reserve planning <span>needs a range, not only a point forecast.</span></div>
                  <div class="rl-why-item">Storage and dispatch <span>depend on when uncertainty is widest.</span></div>
                  <div class="rl-why-item">Market risk <span>rises when renewable output shifts residual demand quickly.</span></div>
                </div>
              </div>
            </div>
          </section>

          <section class="rl-section" id="pipeline">
            <div class="rl-section-title">How it's built</div>
            <div class="rl-h2">A reproducible offline ML pipeline behind the storefront.</div>
            <div class="rl-section-lede">
              DVC stages chain processing, validation, feature engineering, model training,
              calibration, prediction, evaluation, and plotting. Parameters live in
              <code>params.yaml</code>. Streamlit reads the resulting artifacts; it does not
              train, tune, or back-test models from the UI.
            </div>
            <div class="rl-pipeline">
              <div class="rl-step"><div class="num">01</div><div class="name">Ingest</div><div class="desc">Load SMARD consumption and generation data.</div></div>
              <div class="rl-step"><div class="num">02</div><div class="name">Validate</div><div class="desc">Apply schema and data-quality contracts.</div></div>
              <div class="rl-step"><div class="num">03</div><div class="name">Engineer</div><div class="desc">Create lag, rolling, calendar, and generation features.</div></div>
              <div class="rl-step"><div class="num">04</div><div class="name">Train</div><div class="desc">Fit quantile baseline and LightGBM challengers.</div></div>
              <div class="rl-step"><div class="num">05</div><div class="name">Calibrate</div><div class="desc">Use split-conformal qhat and quantile-order repair.</div></div>
              <div class="rl-step"><div class="num">06</div><div class="name">Serve</div><div class="desc">Expose approved forecasts in Streamlit.</div></div>
            </div>
          </section>

          <section class="rl-section" id="models">
            <div class="rl-section-title">Champion / challenger</div>
            <div class="rl-h2">Models are shown only with their gate status.</div>
            <div class="rl-section-lede">
              Forecast display is backed by the latest pilot evaluation: pinball loss, median
              error, empirical coverage, crossing rate, and overall gate result.
            </div>
            <div class="rl-table-wrap">
              <table class="rl-table">
                <thead>
                  <tr><th>Model</th><th>Role</th><th>Pinball</th><th>MAE P50</th><th>Coverage</th><th>Crossings</th><th>Gate</th></tr>
                </thead>
                <tbody>{model_rows}</tbody>
              </table>
            </div>
          </section>

          <section class="rl-section" id="stack">
            <div class="rl-section-title">Engineering</div>
            <div class="rl-h2">Production patterns end-to-end.</div>
            <div class="rl-section-lede">The project combines reproducibility, tracking, probabilistic modeling, validation, and an inference surface.</div>
            <div class="rl-stack-grid">
              <div class="rl-stack-item"><div class="tag">Orchestration</div><div class="nm">DVC</div><div class="ds">Stage graph, deterministic re-runs, artifact lineage.</div></div>
              <div class="rl-stack-item"><div class="tag">Tracking</div><div class="nm">MLflow</div><div class="ds">Metrics, parameters, and experiment records.</div></div>
              <div class="rl-stack-item"><div class="tag">Modeling</div><div class="nm">scikit-learn · LightGBM</div><div class="ds">Quantile regression and boosted challengers.</div></div>
              <div class="rl-stack-item"><div class="tag">Tuning</div><div class="nm">Optuna</div><div class="ds">Search over LightGBM candidate families.</div></div>
              <div class="rl-stack-item"><div class="tag">Validation</div><div class="nm">Pandera · Great Expectations</div><div class="ds">Schema and quality checks before modeling.</div></div>
              <div class="rl-stack-item"><div class="tag">Calibration</div><div class="nm">Split-conformal</div><div class="ds">qhat widening for interval coverage.</div></div>
              <div class="rl-stack-item"><div class="tag">Explainability</div><div class="nm">SHAP</div><div class="ds">Feature attribution artifacts for model review.</div></div>
              <div class="rl-stack-item"><div class="tag">Serving</div><div class="nm">Streamlit</div><div class="ds">Inference views, gates, model comparison, and artifact visualisation.</div></div>
            </div>
          </section>

          <section class="rl-section" id="insights">
            <div class="rl-section-title">From the data</div>
            <div class="rl-h2">Artifacts that shaped the design.</div>
            <div class="rl-section-lede">These visuals are produced by the pipeline and rendered directly from <code>artifacts/figures/</code>.</div>
            <div class="rl-insights">{''.join(insight_cards)}</div>
          </section>

          <div class="rl-footer">
            <div><strong>ResidualLoad.ml</strong> · Day-ahead probabilistic forecasting on public energy data.</div>
            <div>DVC · MLflow · LightGBM · Streamlit · Conformal Prediction</div>
          </div>
        </div>
        """
        ),
        unsafe_allow_html=True,
    )


def page_forecast(
    pred_files: list[tuple[str, Path]],
    target_col: str,
    time_col: str,
    plot_tz: str,
    champion_meta: dict[str, Any] | None,
    model_label_map: dict[str, str],
    model_paths_cfg: dict[str, str],
    model_summary_df: pd.DataFrame | None,
    quantiles: list[float],
    gate_metrics_are_fresh: bool,
    gate_metrics_message: str,
) -> None:
    st.subheader("Generate Forecast (Historical Replay Demo)")

    if not gate_metrics_are_fresh:
        st.error(gate_metrics_message)
        return

    model_names = ordered_model_names([m for m, _ in pred_files])
    approved_models = approved_forecast_models(model_names, model_summary_df)
    blocked_models = blocked_forecast_models(model_names, model_summary_df)
    if blocked_models:
        blocked_labels = ", ".join(model_label_map.get(name, name) for name in blocked_models)
        st.warning(f"Blocked from forecast display because validation gates failed: {blocked_labels}")
    if not approved_models:
        st.error("No approved models are available. Showing failed/fallback models for inspection only.")
        selectable_models = model_names
    else:
        selectable_models = approved_models

    options = [model_label_map.get(name, name) for name in selectable_models]
    selected_label = st.selectbox("Model", options, index=0)
    selected_model = {v: k for k, v in model_label_map.items()}[selected_label]
    selected_path = dict(pred_files)[selected_model]
    if selected_model not in approved_models:
        st.warning(f"{selected_label} failed validation gates. Use this forecast for diagnosis, not approval.")

    if not selected_path.exists():
        st.error(f"Prediction file missing: {selected_path}")
        return

    pred_df = load_prediction_df(selected_path, time_col=time_col, plot_tz=plot_tz)
    q_low, q_med, q_high = low_median_high_quantiles(quantiles)
    pred_cols = quantile_cols(quantiles)
    q_low_col = pred_cols[q_low]
    q_med_col = pred_cols[q_med]
    q_high_col = pred_cols[q_high]
    q_low_label = int(round(q_low * 100))
    q_med_label = int(round(q_med * 100))
    q_high_label = int(round(q_high * 100))
    missing_pred_cols = [col for col in [q_low_col, q_med_col, q_high_col] if col not in pred_df.columns]
    if missing_pred_cols:
        st.error(f"Prediction file is missing configured quantile columns: {missing_pred_cols}")
        return

    available_dates = sorted(pred_df["time_local"].dt.date.unique().tolist())
    selected_date = st.selectbox("Forecast Start Date", available_dates, index=len(available_dates) - 1)
    horizon_hours = st.radio("Forecast Window", [24, 48], horizontal=True, index=0)

    window_start = pd.Timestamp(selected_date, tz=plot_tz)
    window_end = window_start + pd.Timedelta(hours=int(horizon_hours))
    forecast_df = pred_df[
        (pred_df["time_local"] >= window_start) & (pred_df["time_local"] < window_end)
    ].copy()
    if forecast_df.empty:
        st.warning("No data for selected forecast window.")
        return
    if len(forecast_df) < int(horizon_hours):
        st.info(f"Selected window has {len(forecast_df)} rows available out of {horizon_hours} expected hours.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows (Selected Window)", len(forecast_df))
    c2.metric(f"Average Predicted p{q_med_label}", f"{forecast_df[q_med_col].mean():,.0f}")
    c3.metric("Average Uncertainty Width", f"{(forecast_df[q_high_col] - forecast_df[q_low_col]).mean():,.0f}")

    # Traffic-light gate panel from latest pilot evaluation metrics for selected model.
    if model_summary_df is not None and not model_summary_df.empty and "model_name" in model_summary_df.columns:
        gate_row_df = model_summary_df[model_summary_df["model_name"] == selected_model]
        if not gate_row_df.empty:
            gate_row = gate_row_df.iloc[0]

            def gate_html(title: str, passed: bool) -> str:
                color = "#2E7D32" if passed else "#C62828"
                text = "PASS" if passed else "FAIL"
                return (
                    f"<div style='padding:8px 10px;border:1px solid #E0E0E0;border-radius:8px;'>"
                    f"<div style='font-size:0.85rem;color:#666;'>{title}</div>"
                    f"<div style='font-size:1rem;font-weight:600;color:{color};'>● {text}</div>"
                    f"</div>"
                )

            st.markdown("#### Gate Status")
            g1, g2, g3 = st.columns(3)
            g1.markdown(gate_html("Coverage Gate", as_bool(gate_row.get("coverage_gate_pass", False))), unsafe_allow_html=True)
            g2.markdown(gate_html("Crossing Gate", as_bool(gate_row.get("crossing_gate_pass", False))), unsafe_allow_html=True)
            g3.markdown(gate_html("Overall Gate", as_bool(gate_row.get("overall_gate_pass", False))), unsafe_allow_html=True)

    fig = forecast_plot(
        forecast_df=forecast_df,
        model_name=selected_label,
        time_col_local="time_local",
        target_col=target_col,
        plot_tz=plot_tz,
        horizon_hours=int(horizon_hours),
        q_low_col=q_low_col,
        q_med_col=q_med_col,
        q_high_col=q_high_col,
        q_low_label=q_low_label,
        q_med_label=q_med_label,
        q_high_label=q_high_label,
    )
    st.pyplot(fig, clear_figure=True)

    # Additional visual: uncertainty width by hour
    width_fig, width_ax = plt.subplots(figsize=(12, 3.4))
    width_df = forecast_df.copy()
    width_df["hour"] = width_df["time_local"].dt.strftime("%m-%d %H:%M")
    width_df["width"] = width_df[q_high_col] - width_df[q_low_col]
    width_ax.bar(width_df["hour"], width_df["width"], color="#7FB3D5")
    width_ax.set_title(f"Uncertainty Width by Hour (p{q_high_label} - p{q_low_label})", fontsize=12)
    width_ax.set_xlabel(f"Time ({plot_tz})")
    width_ax.set_ylabel("Width (MWh)")
    width_ax.tick_params(axis="x", rotation=45)
    width_fig.tight_layout()
    st.pyplot(width_fig, clear_figure=True)

    # Additional visual: residual error (actual - median quantile)
    has_actual = target_col in forecast_df.columns and forecast_df[target_col].notna().any()
    if has_actual:
        err_fig, err_ax = plt.subplots(figsize=(12, 3.6))
        err_df = forecast_df.copy()
        err_df["hour"] = err_df["time_local"].dt.strftime("%m-%d %H:%M")
        err_df["residual_error"] = err_df[target_col] - err_df[q_med_col]
        colors = np.where(err_df["residual_error"] >= 0, "#4CAF50", "#E57373")
        err_ax.bar(err_df["hour"], err_df["residual_error"], color=colors)
        err_ax.axhline(0, color="#333333", linewidth=1.2)
        err_ax.set_title(f"Residual Error by Hour (Actual - p{q_med_label})", fontsize=12)
        err_ax.set_xlabel(f"Time ({plot_tz})")
        err_ax.set_ylabel("Error (MWh)")
        err_ax.tick_params(axis="x", rotation=45)
        err_fig.tight_layout()
        st.pyplot(err_fig, clear_figure=True)

    # Top 3 risky hours (largest interval width)
    risky_df = forecast_df.copy()
    risky_df["interval_width"] = risky_df[q_high_col] - risky_df[q_low_col]
    risky_df["hour_local"] = risky_df["time_local"].dt.strftime("%Y-%m-%d %H:%M")
    top_risky = risky_df.sort_values("interval_width", ascending=False).head(3)
    risky_cols = ["hour_local", q_low_col, q_med_col, q_high_col, "interval_width"]
    if has_actual:
        risky_cols.insert(1, target_col)
    st.markdown("#### Top 3 Risky Hours (Largest Uncertainty)")
    st.dataframe(top_risky[risky_cols], use_container_width=True)

    # Feature importance visual (from selected model artifact)
    model_artifact = model_paths_cfg.get(selected_model)
    if model_artifact:
        try:
            bundle_path = trusted_model_path(model_artifact)
        except ValueError as exc:
            st.warning(f"Blocked untrusted model artifact for {selected_label}: {exc}")
            bundle_path = None
        if bundle_path and bundle_path.exists():
            try:
                bundle = joblib.load(bundle_path)
                fi_df, fi_label = feature_importance_df(bundle)
                if not fi_df.empty:
                    st.markdown(f"### Feature Importance ({selected_label})")
                    st.caption(
                        "This selected model is mostly explained by recent residual-load memory, "
                        "load momentum, generation mix, and calendar timing. The chart shows the "
                        "strongest q50 signals for the model currently selected above."
                    )
                    top_n = min(10, len(fi_df))
                    summary_df = driver_group_summary(fi_df, top_n=top_n)
                    if not summary_df.empty:
                        st.dataframe(summary_df, use_container_width=True, hide_index=True)
                    top_df = fi_df.head(top_n).copy()
                    top_df["readable_feature"] = top_df["feature"].map(friendly_feature_label)
                    top_df = top_df.sort_values("importance", ascending=True)
                    fi_fig, fi_ax = plt.subplots(figsize=(10, 5.5))
                    fi_ax.barh(top_df["readable_feature"], top_df["importance"], color="#4EA3D8")
                    fi_ax.set_xlabel(fi_label)
                    fi_ax.set_ylabel("")
                    fi_ax.set_title("Top 10 Features Driving This Model")
                    fi_fig.tight_layout()
                    st.pyplot(fi_fig, clear_figure=True)
                    with st.expander("Raw feature names"):
                        raw_df = top_df.sort_values("importance", ascending=False)[
                            ["feature", "readable_feature", "importance"]
                        ].rename(
                            columns={
                                "feature": "Raw feature",
                                "readable_feature": "Readable feature",
                                "importance": "Importance",
                            }
                        )
                        st.dataframe(raw_df, use_container_width=True, hide_index=True)
            except Exception as exc:
                st.warning(f"Could not load feature importance for {selected_label}: {exc}")

    cols = ["time_local", q_low_col, q_med_col, q_high_col]
    if target_col in forecast_df.columns:
        cols.insert(1, target_col)
    show_df = forecast_df[cols].copy().rename(columns={"time_local": f"time_{plot_tz}"})
    st.dataframe(show_df, use_container_width=True)

    csv_bytes = show_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV (selected window)",
        data=csv_bytes,
        file_name=f"{selected_model}_forecast_{selected_date}_{horizon_hours}h.csv",
        mime="text/csv",
    )

    with st.expander("Model Context"):
        if champion_meta:
            st.json(champion_meta)
        else:
            st.info("`artifacts/models/champion_metadata.json` not found.")


def page_model_compare(model_summary_df: pd.DataFrame | None, model_label_map: dict[str, str]) -> None:
    st.subheader("Model Compare")
    if model_summary_df is None or model_summary_df.empty:
        st.warning("No model comparison file found. Produce evaluation artifacts with the offline pipeline first.")
        return

    view_df = model_summary_df.copy()
    view_df["model_label"] = view_df["model_name"].map(model_label_map).fillna(view_df["model_name"])

    show_cols = [
        "model_label",
        "pinball_mean",
        "mae_p50",
        "rmse_p50",
        "coverage",
        "coverage_gap",
        "crossing_rate",
        "overall_gate_pass",
    ]
    for col in ["raw_crossing_rate", "calibrated_crossing_rate", "repair_rate", "output_crossing_rate"]:
        if col in view_df.columns and col not in show_cols:
            show_cols.insert(show_cols.index("overall_gate_pass"), col)
    st.dataframe(view_df[show_cols].sort_values("pinball_mean"), use_container_width=True)

    metric = st.selectbox("Metric for Bar Chart", ["pinball_mean", "mae_p50", "rmse_p50", "coverage"])
    bar_df = view_df[["model_label", metric]].set_index("model_label")
    st.bar_chart(bar_df)

    st.markdown("### What Drives the Forecast?")
    st.markdown(
        """
The champion model is expected to lean on recent residual-load memory, short-term changes,
and generation-mix signals. That is a useful trust check: the model is learning patterns that
match the energy-system definition of residual load, not just arbitrary correlations.
"""
    )
    shap_path = PROJECT_ROOT / "artifacts/stages/05_test_report/shap_importance_q50.csv"
    shap_df = safe_read_csv(shap_path)
    if shap_df is not None and not shap_df.empty:
        group_df = driver_group_summary(shap_df, top_n=12)
        st.dataframe(group_df, use_container_width=True, hide_index=True)
        top_features = shap_df.head(5).copy()
        top_features["Readable feature"] = top_features["feature"].map(friendly_feature_label)
        st.caption(
            "Top SHAP signals: "
            + ", ".join(top_features["Readable feature"].tolist())
            + "."
        )
    else:
        st.info("SHAP summary not found. It is produced by the offline training stage at `artifacts/stages/05_test_report/shap_importance_q50.csv`.")

    winner_importance_path = PROJECT_ROOT / "artifacts/stages/05_test_report/feature_importance_winners.png"
    st.markdown("### Winner Feature Importance")
    if winner_importance_path.exists():
        st.image(str(winner_importance_path), use_container_width=True)
        st.caption("Saved training artifact from `train_models`: baseline q50 importance vs selected LightGBM q50 importance.")
    else:
        st.info("Winner feature-importance artifact not found. Regenerate it with the offline training pipeline.")


def page_pilot_health(pilot_metrics: dict[str, Any] | None, model_label_map: dict[str, str]) -> None:
    st.subheader("Pilot Health")
    if not pilot_metrics:
        st.warning("No pilot metrics found. Produce evaluation artifacts with the offline pipeline first.")
        return

    st.metric("Models Evaluated", pilot_metrics.get("n_models_evaluated", 0))
    best_by_pinball = pilot_metrics.get("best_by_pinball") or {}
    best_pinball_raw = str(best_by_pinball.get("model_name", pilot_metrics.get("best_model_by_pinball_mean", "n/a")))
    st.metric("Best by Pinball", model_label_map.get(best_pinball_raw, best_pinball_raw))
    st.metric("Best Pinball Mean", f"{float(best_by_pinball.get('pinball_mean', pilot_metrics.get('best_pinball_mean', 0))):,.3f}")
    best_approved = pilot_metrics.get("best_approved_model")
    if best_approved:
        approved_raw = str(best_approved.get("model_name", "n/a"))
        st.metric("Best Approved Model", model_label_map.get(approved_raw, approved_raw))
    else:
        st.metric("Best Approved Model", "None")

    models = pilot_metrics.get("models", [])
    if models:
        health_df = pd.DataFrame(models)
        health_df["model_label"] = health_df["model_name"].map(model_label_map).fillna(health_df["model_name"])
        cols = [
            "model_label",
            "pinball_mean",
            "coverage",
            "coverage_gap",
            "crossing_rate",
            "overall_gate_pass",
        ]
        for col in ["raw_crossing_rate", "calibrated_crossing_rate", "repair_rate", "output_crossing_rate"]:
            if col in health_df.columns and col not in cols:
                cols.insert(cols.index("overall_gate_pass"), col)
        st.dataframe(health_df[cols].sort_values("pinball_mean"), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Probabilistic Residual Load Forecast", layout="wide")

    params = load_params()
    cfg = params["model_training"]
    infer_cfg = cfg.get("inference", {})

    target_col = str(cfg["target_col"])
    time_col = str(cfg["time_col"])
    quantiles = sorted(float(q) for q in cfg["quantiles"])
    plot_tz = str(infer_cfg.get("plot_timezone", "Europe/Berlin"))

    manifest_path = PROJECT_ROOT / infer_cfg.get("manifest_path", "artifacts/predictions/prediction_manifest.csv")
    pred_dir = PROJECT_ROOT / infer_cfg.get("output_dir", "artifacts/predictions/models")
    pilot_metrics_path = PROJECT_ROOT / infer_cfg.get(
        "pilot_metrics_path", "artifacts/metrics/pilot_evaluation_metrics.json"
    )
    model_summary_path = PROJECT_ROOT / "artifacts/stages/08_evaluate_pilot/model_summary_metrics.csv"
    champion_meta_path = PROJECT_ROOT / "artifacts/models/champion_metadata.json"
    model_paths_cfg = {str(k): str(v) for k, v in infer_cfg.get("model_paths", {}).items()}
    page_options = ["Overview", "Forecast", "Model Compare", "Pilot Health"]
    try:
        query_page = st.query_params.get("page", "Overview")
    except Exception:
        query_values = st.experimental_get_query_params()
        query_page = query_values.get("page", ["Overview"])[0]
    page_index = page_options.index(query_page) if query_page in page_options else 0

    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Page", page_options, index=page_index)
        st.caption("This app reads trained model, prediction, and evaluation artifacts produced by the offline DVC + MLflow pipeline.")

    pred_files = resolve_prediction_files(manifest_path=manifest_path, pred_dir=pred_dir)
    model_summary_df = safe_read_csv(model_summary_path)
    pilot_metrics = safe_read_json(pilot_metrics_path)
    champion_meta = safe_read_json(champion_meta_path)
    model_label_map = build_model_labels([name for name, _ in pred_files]) if pred_files else {}
    gate_metrics_are_fresh, gate_metrics_message = gate_metrics_fresh(
        pred_files=pred_files,
        manifest_path=manifest_path,
        model_summary_path=model_summary_path,
        model_summary_df=model_summary_df,
    )

    if not pred_files:
        st.warning(
            "No prediction files found yet. Produce artifacts with the offline pipeline first:\n"
            "`dvc repro predict evaluate_pilot plot_predictions`"
        )

    if page == "Overview":
        page_overview(
            model_summary_df=model_summary_df,
            pilot_metrics=pilot_metrics,
            champion_meta=champion_meta,
            model_label_map=model_label_map,
        )
    elif page == "Forecast":
        if pred_files:
            page_forecast(
                pred_files=pred_files,
                target_col=target_col,
                time_col=time_col,
                plot_tz=plot_tz,
                champion_meta=champion_meta,
                model_label_map=model_label_map,
                model_paths_cfg=model_paths_cfg,
                model_summary_df=model_summary_df,
                quantiles=quantiles,
                gate_metrics_are_fresh=gate_metrics_are_fresh,
                gate_metrics_message=gate_metrics_message,
            )
    elif page == "Model Compare":
        page_model_compare(model_summary_df, model_label_map=model_label_map)
    elif page == "Pilot Health":
        page_pilot_health(pilot_metrics, model_label_map=model_label_map)


if __name__ == "__main__":
    main()
