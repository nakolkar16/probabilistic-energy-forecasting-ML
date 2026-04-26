from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yaml


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


def run_module(module_name: str) -> tuple[bool, str]:
    """
    Runs one of the project modules with current python env.
    """
    cmd = [sys.executable, "-m", module_name]
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    ok = proc.returncode == 0
    logs = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return ok, logs.strip()


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
    for name in ["champion", "challenger"]:
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
        else:
            labels[name] = f"Model {model_idx}"
            model_idx += 1
    return labels


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


def load_prediction_df(path: Path, time_col: str, plot_tz: str) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df["time_local"] = df[time_col].dt.tz_convert(plot_tz)
    df = df.sort_values(time_col).reset_index(drop=True)
    return df


def day_plot(
    day_df: pd.DataFrame,
    model_name: str,
    time_col_local: str,
    target_col: str,
    plot_tz: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 4.8))

    has_actual = target_col in day_df.columns and day_df[target_col].notna().any()
    if has_actual:
        ax.plot(day_df[time_col_local], day_df[target_col], label="Actual", linewidth=2.0, color="#2F4858")

    ax.plot(day_df[time_col_local], day_df["pred_q50"], label="Predicted p50", linewidth=2.0, color="#1F78B4")
    ax.fill_between(
        day_df[time_col_local],
        day_df["pred_q10"],
        day_df["pred_q90"],
        alpha=0.18,
        label="p10-p90 band",
        color="#4EA3D8",
    )

    day_label = str(pd.to_datetime(day_df[time_col_local]).dt.date.iloc[0])
    ax.set_title(f"Day-ahead Forecast | {model_name} | {day_label}", fontsize=14, pad=12)
    ax.set_xlabel(f"Time ({plot_tz})")
    ax.set_ylabel("Residual Load (MWh)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def page_overview() -> None:
    st.subheader("Overview")
    st.markdown(
        """
### What This Project Is About
This project forecasts **German residual load** for the next 24 hours with uncertainty:
- **P10**: optimistic/low-load scenario
- **P50**: central forecast
- **P90**: conservative/high-load scenario

Residual load means: demand that still needs to be covered after variable renewables.
"""
    )
    st.info(
        "This app is inference-focused. Training, tuning, and backtesting stay in the offline pipeline (DVC + MLflow)."
    )
    st.markdown(
        """
### App Workflow
1. Load approved model artifacts
2. Generate forecast artifacts (`predict -> evaluate_pilot -> plot_predictions`)
3. Visualize forecasts, uncertainty, and model health
"""
    )

    latest_plot = PROJECT_ROOT / "artifacts/figures/predictions/day_ahead_actual_vs_pred_latest.png"
    if latest_plot.exists():
        st.markdown("### Latest Forecast Snapshot")
        st.image(str(latest_plot), use_container_width=True)


def page_forecast(
    pred_files: list[tuple[str, Path]],
    target_col: str,
    time_col: str,
    plot_tz: str,
    champion_meta: dict[str, Any] | None,
    model_label_map: dict[str, str],
    model_paths_cfg: dict[str, str],
    model_summary_df: pd.DataFrame | None,
) -> None:
    st.subheader("Generate Forecast (Historical Replay Demo)")

    model_names = ordered_model_names([m for m, _ in pred_files])
    options = [model_label_map.get(name, name) for name in model_names]
    selected_label = st.selectbox("Model", options, index=0)
    selected_model = {v: k for k, v in model_label_map.items()}[selected_label]
    selected_path = dict(pred_files)[selected_model]

    if not selected_path.exists():
        st.error(f"Prediction file missing: {selected_path}")
        return

    pred_df = load_prediction_df(selected_path, time_col=time_col, plot_tz=plot_tz)
    available_dates = sorted(pred_df["time_local"].dt.date.unique().tolist())
    selected_date = st.selectbox("Forecast Date", available_dates, index=len(available_dates) - 1)

    day_df = pred_df[pred_df["time_local"].dt.date == selected_date].copy()
    if day_df.empty:
        st.warning("No data for selected date.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows (Selected Day)", len(day_df))
    c2.metric("Average Predicted p50", f"{day_df['pred_q50'].mean():,.0f}")
    c3.metric("Average Uncertainty Width", f"{(day_df['pred_q90'] - day_df['pred_q10']).mean():,.0f}")

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
            g1.markdown(gate_html("Coverage Gate", bool(gate_row.get("coverage_gate_pass", False))), unsafe_allow_html=True)
            g2.markdown(gate_html("Crossing Gate", bool(gate_row.get("crossing_gate_pass", False))), unsafe_allow_html=True)
            g3.markdown(gate_html("Overall Gate", bool(gate_row.get("overall_gate_pass", False))), unsafe_allow_html=True)

    fig = day_plot(
        day_df=day_df,
        model_name=selected_label,
        time_col_local="time_local",
        target_col=target_col,
        plot_tz=plot_tz,
    )
    st.pyplot(fig, clear_figure=True)

    # Additional visual: uncertainty width by hour
    width_fig, width_ax = plt.subplots(figsize=(12, 3.4))
    width_df = day_df.copy()
    width_df["hour"] = width_df["time_local"].dt.strftime("%H:%M")
    width_df["width"] = width_df["pred_q90"] - width_df["pred_q10"]
    width_ax.bar(width_df["hour"], width_df["width"], color="#7FB3D5")
    width_ax.set_title("Uncertainty Width by Hour (p90 - p10)", fontsize=12)
    width_ax.set_xlabel(f"Hour ({plot_tz})")
    width_ax.set_ylabel("Width (MWh)")
    width_ax.tick_params(axis="x", rotation=45)
    width_fig.tight_layout()
    st.pyplot(width_fig, clear_figure=True)

    # Additional visual: residual error (actual - p50)
    has_actual = target_col in day_df.columns and day_df[target_col].notna().any()
    if has_actual:
        err_fig, err_ax = plt.subplots(figsize=(12, 3.6))
        err_df = day_df.copy()
        err_df["hour"] = err_df["time_local"].dt.strftime("%H:%M")
        err_df["residual_error"] = err_df[target_col] - err_df["pred_q50"]
        colors = np.where(err_df["residual_error"] >= 0, "#4CAF50", "#E57373")
        err_ax.bar(err_df["hour"], err_df["residual_error"], color=colors)
        err_ax.axhline(0, color="#333333", linewidth=1.2)
        err_ax.set_title("Residual Error by Hour (Actual - p50)", fontsize=12)
        err_ax.set_xlabel(f"Hour ({plot_tz})")
        err_ax.set_ylabel("Error (MWh)")
        err_ax.tick_params(axis="x", rotation=45)
        err_fig.tight_layout()
        st.pyplot(err_fig, clear_figure=True)

    # Top 3 risky hours (largest interval width)
    risky_df = day_df.copy()
    risky_df["interval_width"] = risky_df["pred_q90"] - risky_df["pred_q10"]
    risky_df["hour_local"] = risky_df["time_local"].dt.strftime("%Y-%m-%d %H:%M")
    top_risky = risky_df.sort_values("interval_width", ascending=False).head(3)
    risky_cols = ["hour_local", "pred_q10", "pred_q50", "pred_q90", "interval_width"]
    if has_actual:
        risky_cols.insert(1, target_col)
    st.markdown("#### Top 3 Risky Hours (Largest Uncertainty)")
    st.dataframe(top_risky[risky_cols], use_container_width=True)

    # Feature importance visual (from selected model artifact)
    model_artifact = model_paths_cfg.get(selected_model)
    if model_artifact:
        bundle_path = PROJECT_ROOT / model_artifact
        if bundle_path.exists():
            try:
                bundle = joblib.load(bundle_path)
                fi_df, fi_label = feature_importance_df(bundle)
                if not fi_df.empty:
                    st.markdown(f"### Feature Importance ({selected_label})")
                    top_n = min(15, len(fi_df))
                    top_df = fi_df.head(top_n).sort_values("importance", ascending=True)
                    fi_fig, fi_ax = plt.subplots(figsize=(10, 5.5))
                    fi_ax.barh(top_df["feature"], top_df["importance"], color="#4EA3D8")
                    fi_ax.set_xlabel(fi_label)
                    fi_ax.set_ylabel("")
                    fi_ax.set_title("Top Features Driving This Model")
                    fi_fig.tight_layout()
                    st.pyplot(fi_fig, clear_figure=True)
            except Exception as exc:
                st.warning(f"Could not load feature importance for {selected_label}: {exc}")

    cols = ["time_local", "pred_q10", "pred_q50", "pred_q90"]
    if target_col in day_df.columns:
        cols.insert(1, target_col)
    show_df = day_df[cols].copy().rename(columns={"time_local": f"time_{plot_tz}"})
    st.dataframe(show_df, use_container_width=True)

    csv_bytes = show_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV (selected day)",
        data=csv_bytes,
        file_name=f"{selected_model}_forecast_{selected_date}.csv",
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
        st.warning("No model comparison file found. Run `evaluate_pilot` first.")
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
    st.dataframe(view_df[show_cols].sort_values("pinball_mean"), use_container_width=True)

    metric = st.selectbox("Metric for Bar Chart", ["pinball_mean", "mae_p50", "rmse_p50", "coverage"])
    bar_df = view_df[["model_label", metric]].set_index("model_label")
    st.bar_chart(bar_df)


def page_pilot_health(pilot_metrics: dict[str, Any] | None, model_label_map: dict[str, str]) -> None:
    st.subheader("Pilot Health")
    if not pilot_metrics:
        st.warning("No pilot metrics found. Run `evaluate_pilot` first.")
        return

    st.metric("Models Evaluated", pilot_metrics.get("n_models_evaluated", 0))
    best_raw = str(pilot_metrics.get("best_model_by_pinball_mean", "n/a"))
    st.metric("Best Model (pinball)", model_label_map.get(best_raw, best_raw))
    st.metric("Best Pinball Mean", f"{pilot_metrics.get('best_pinball_mean', 0):,.3f}")

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
        st.dataframe(health_df[cols].sort_values("pinball_mean"), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Probabilistic Residual Load Forecast", layout="wide")
    st.title("Probabilistic Residual Load Forecast (Day-Ahead)")

    params = load_params()
    cfg = params["model_training"]
    infer_cfg = cfg.get("inference", {})

    target_col = str(cfg["target_col"])
    time_col = str(cfg["time_col"])
    plot_tz = str(infer_cfg.get("plot_timezone", "Europe/Berlin"))

    manifest_path = PROJECT_ROOT / infer_cfg.get("manifest_path", "artifacts/predictions/prediction_manifest.csv")
    pred_dir = PROJECT_ROOT / infer_cfg.get("output_dir", "artifacts/predictions/models")
    pilot_metrics_path = PROJECT_ROOT / infer_cfg.get(
        "pilot_metrics_path", "artifacts/metrics/pilot_evaluation_metrics.json"
    )
    model_summary_path = PROJECT_ROOT / "artifacts/stages/08_evaluate_pilot/model_summary_metrics.csv"
    champion_meta_path = PROJECT_ROOT / "artifacts/models/champion_metadata.json"
    model_paths_cfg = {str(k): str(v) for k, v in infer_cfg.get("model_paths", {}).items()}

    with st.sidebar:
        st.header("Controls")
        if st.button("Generate Latest Forecast Artifacts"):
            with st.spinner("Running predict -> evaluate_pilot -> plot_predictions ..."):
                ok1, log1 = run_module("src.modeling.predict")
                ok2, log2 = run_module("src.modeling.evaluate_pilot")
                ok3, log3 = run_module("src.visualization.plot_predictions")
            if ok1 and ok2 and ok3:
                st.success("Artifacts refreshed.")
            else:
                st.error("One or more steps failed. See logs below.")
            with st.expander("Execution Logs"):
                st.code(f"[predict]\n{log1}\n\n[evaluate_pilot]\n{log2}\n\n[plot_predictions]\n{log3}")

        page = st.radio("Page", ["Overview", "Forecast", "Model Compare", "Pilot Health"])

    pred_files = resolve_prediction_files(manifest_path=manifest_path, pred_dir=pred_dir)
    model_summary_df = safe_read_csv(model_summary_path)
    pilot_metrics = safe_read_json(pilot_metrics_path)
    champion_meta = safe_read_json(champion_meta_path)
    model_label_map = build_model_labels([name for name, _ in pred_files]) if pred_files else {}

    if not pred_files:
        st.warning(
            "No prediction files found yet. Run pipeline first:\n"
            "`dvc repro predict evaluate_pilot plot_predictions`"
        )

    if page == "Overview":
        page_overview()
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
            )
    elif page == "Model Compare":
        page_model_compare(model_summary_df, model_label_map=model_label_map)
    elif page == "Pilot Health":
        page_pilot_health(pilot_metrics, model_label_map=model_label_map)


if __name__ == "__main__":
    main()
