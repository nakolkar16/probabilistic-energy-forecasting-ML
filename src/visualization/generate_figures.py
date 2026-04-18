from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from src.data.column_schema import CANONICAL_TO_LABEL


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}


def load_params() -> dict:
    with open(Path("params.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_required_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved figure: %s", path)


def annotate_seasons(
    ax: plt.Axes,
    df: pd.DataFrame,
    season_col: str = "season",
    y_position: float | None = None,
    y_mult: float = 1.01,
) -> None:
    season_change_idx = [
        i
        for i in range(1, len(df))
        if df[season_col].iloc[i] != df[season_col].iloc[i - 1]
    ]
    for i in season_change_idx:
        ax.axvline(i - 0.5, color="#BDBDBD", linestyle=":", linewidth=1.2)

    if y_position is None:
        y_position = ax.get_ylim()[1] * y_mult

    start = 0
    for i in range(1, len(df) + 1):
        boundary = i == len(df) or df[season_col].iloc[i] != df[season_col].iloc[i - 1]
        if boundary:
            end = i - 1
            center = (start + end) / 2
            ax.text(
                center,
                y_position,
                df[season_col].iloc[start],
                ha="center",
                va="bottom",
                fontsize=10,
                color="#666666",
            )
            start = i


def prepare_base_frames(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    _ensure_required_columns(df, ["timestamp", "residual_load_mwh", "grid_load_mwh"])

    df_plot = df.copy()
    df_plot["timestamp"] = pd.to_datetime(df_plot["timestamp"], errors="coerce")
    df_plot = (
        df_plot.dropna(subset=["timestamp", "residual_load_mwh"]).sort_values("timestamp")
    )

    df_plot["residual_load_ma_24h"] = (
        df_plot["residual_load_mwh"].rolling(window=24, min_periods=1).mean()
    )
    df_plot["residual_load_ma_168h"] = (
        df_plot["residual_load_mwh"].rolling(window=168, min_periods=1).mean()
    )

    df_subset = df_plot[["timestamp", "residual_load_mwh", "grid_load_mwh"]].copy()
    df_monthly = (
        df_subset.set_index("timestamp")
        .resample("ME")
        .agg(
            monthly_grid_load=("grid_load_mwh", "sum"),
            monthly_residual_load=("residual_load_mwh", "sum"),
        )
        .reset_index()
    )

    # Drop first/last partial months, matching notebook logic.
    df_monthly = df_monthly.iloc[1:-1].copy()
    df_monthly["residual_share_pct"] = (
        df_monthly["monthly_residual_load"] / df_monthly["monthly_grid_load"] * 100
    )
    df_monthly["month_year"] = df_monthly["timestamp"].dt.strftime("%Y-%m")
    df_monthly["monthly_grid_load_twh"] = df_monthly["monthly_grid_load"] / 1_000_000
    df_monthly["monthly_residual_load_twh"] = (
        df_monthly["monthly_residual_load"] / 1_000_000
    )
    df_monthly["season"] = df_monthly["timestamp"].dt.month.map(SEASON_MAP)
    df_monthly["below_50"] = df_monthly["residual_share_pct"] < 50

    return df_plot, df_monthly


def plot_residual_time_series(df_plot: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))

    sns.lineplot(
        data=df_plot,
        x="timestamp",
        y="residual_load_mwh",
        ax=ax,
        label=CANONICAL_TO_LABEL["residual_load_mwh"],
        color="#C9CDD3",
        alpha=0.5,
        linewidth=0.7,
    )
    sns.lineplot(
        data=df_plot,
        x="timestamp",
        y="residual_load_ma_24h",
        ax=ax,
        label="24h rolling average",
        color="#4C78A8",
        linewidth=2.2,
    )
    sns.lineplot(
        data=df_plot,
        x="timestamp",
        y="residual_load_ma_168h",
        ax=ax,
        label="7-day rolling average",
        color="#2F5D50",
        linewidth=2.8,
    )

    ax.set_title("Residual load over time", fontsize=18, pad=18)
    ax.text(
        0,
        1.03,
        "Hourly values with short-term and weekly smoothing",
        transform=ax.transAxes,
        fontsize=11,
        color="#555555",
    )
    ax.set_xlabel("Time", fontsize=13)
    ax.set_ylabel(f"{CANONICAL_TO_LABEL['residual_load_mwh']} (thousand)", fontsize=13)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f"{x / 1000:.0f}"))

    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=11, loc="upper right")

    _save_figure(fig, output_dir / "residual_load_time_series.png", dpi=dpi)


def plot_monthly_load_lines(df_monthly: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(14, 5.5))

    sns.lineplot(
        data=df_monthly,
        x="month_year",
        y="monthly_grid_load_twh",
        marker="o",
        linewidth=2.5,
        label=CANONICAL_TO_LABEL["grid_load_mwh"],
        ax=ax,
    )
    sns.lineplot(
        data=df_monthly,
        x="month_year",
        y="monthly_residual_load_twh",
        marker="o",
        linewidth=2.5,
        label=CANONICAL_TO_LABEL["residual_load_mwh"],
        ax=ax,
    )

    xticks = ax.get_xticks()
    xlabels = [label.get_text() for label in ax.get_xticklabels()]
    ax.set_xticks(xticks[::2])
    ax.set_xticklabels(xlabels[::2], rotation=45, ha="right", fontsize=11)

    ax.set_ylabel("Monthly energy (TWh)", fontsize=13)
    ax.set_xlabel("Month", fontsize=13)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    annotate_seasons(ax, df_monthly, y_mult=1.01)

    ax.set_title("Monthly grid load and residual load", fontsize=16, pad=24)
    fig.text(
        0.125,
        0.92,
        "Monthly totals shown in TWh for easier comparison",
        fontsize=10.5,
        color="#555555",
    )
    ax.legend(frameon=False, fontsize=11, loc="upper left")

    _save_figure(fig, output_dir / "monthly_grid_vs_residual_twh.png", dpi=dpi)


def plot_monthly_residual_share(df_monthly: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(14, 5.5))

    sns.barplot(
        data=df_monthly,
        x="month_year",
        y="residual_share_pct",
        hue="below_50",
        palette={True: "#4C9F70", False: "#C9CDD3"},
        dodge=False,
        legend=False,
        ax=ax,
    )

    ax.axhline(50, color="black", linestyle="--", linewidth=1.5)
    ax.text(
        len(df_monthly) - 0.2,
        51.5,
        "50% threshold",
        ha="right",
        va="bottom",
        fontsize=10,
    )

    ax.set_ylim(0, 105)
    ax.set_ylabel(CANONICAL_TO_LABEL["residual_share_pct"], fontsize=14)
    ax.set_xlabel("Month", fontsize=14)
    ax.set_yticks(range(0, 101, 10))
    ax.set_yticklabels([f"{i}%" for i in range(0, 101, 10)], fontsize=11)

    xticks = ax.get_xticks()
    xlabels = [label.get_text() for label in ax.get_xticklabels()]
    ax.set_xticks(xticks[::2])
    ax.set_xticklabels(xlabels[::2], rotation=45, ha="right", fontsize=11)

    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    annotate_seasons(ax, df_monthly, y_position=102.5)

    ax.set_title(
        "Monthly residual load share of total grid load",
        fontsize=16,
        pad=28,
    )
    fig.text(
        0.125,
        0.89,
        "Green bars indicate months where residual load share was below 50%",
        fontsize=11,
        color="#555555",
    )

    _save_figure(fig, output_dir / "monthly_residual_share_pct.png", dpi=dpi)


def plot_top_correlations(
    df: pd.DataFrame,
    output_dir: Path,
    target: str,
    top_n: int,
    dpi: int,
) -> tuple[pd.DataFrame, pd.Series]:
    _ensure_required_columns(df, [target])

    corr_df = df.select_dtypes(include="number").copy()
    _ensure_required_columns(corr_df, [target])

    target_corr = (
        corr_df.corr(numeric_only=True)[target]
        .drop(target)
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )

    plot_df = target_corr.reset_index()
    plot_df.columns = ["feature", "correlation"]
    plot_df["feature_label"] = plot_df["feature"].map(CANONICAL_TO_LABEL).fillna(
        plot_df["feature"]
    )

    plot_df["abs_corr"] = plot_df["correlation"].abs()
    plot_df = (
        plot_df.sort_values("abs_corr", ascending=False)
        .head(top_n)
        .sort_values("correlation")
    )
    plot_df["direction"] = plot_df["correlation"].ge(0).map(
        {True: "Positive", False: "Negative"}
    )

    fig, ax = plt.subplots(figsize=(10, 6.5))

    sns.barplot(
        data=plot_df,
        x="correlation",
        y="feature_label",
        hue="direction",
        dodge=False,
        palette={"Positive": "#6B8FB3", "Negative": "#D79A9A"},
        ax=ax,
    )

    if ax.legend_:
        ax.legend_.remove()

    ax.axvline(0, color="#333333", linewidth=1.2)
    target_label = CANONICAL_TO_LABEL.get(target, target)
    ax.set_title(f"Top feature correlations with {target_label}", fontsize=17, pad=14, loc="left")
    ax.text(
        0,
        1.00,
        "Numeric candidate features ranked by absolute Pearson correlation",
        transform=ax.transAxes,
        fontsize=10.5,
        color="#666666",
    )

    ax.set_xlabel("Correlation coefficient", fontsize=12)
    ax.set_ylabel("")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", labelsize=11)
    ax.tick_params(axis="x", labelsize=11)

    _save_figure(fig, output_dir / "top_feature_correlations.png", dpi=dpi)
    return corr_df, target_corr


def plot_correlation_heatmap(
    corr_df: pd.DataFrame,
    target_corr: pd.Series,
    output_dir: Path,
    target: str,
    top_n: int,
    dpi: int,
) -> None:
    selected_features = target_corr.head(top_n).index.tolist()
    heatmap_cols = [target] + selected_features

    corr_small = corr_df[heatmap_cols].corr()
    corr_small = corr_small.rename(index=CANONICAL_TO_LABEL, columns=CANONICAL_TO_LABEL)
    corr_small = corr_small.rename(
        index=lambda s: s.replace(" (MWh)", ""),
        columns=lambda s: s.replace(" (MWh)", ""),
    )

    mask = np.triu(np.ones_like(corr_small, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(10, 7))

    sns.heatmap(
        corr_small,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=1,
        linecolor="white",
        square=False,
        cbar_kws={"shrink": 0.85, "label": "Correlation"},
        annot_kws={"size": 11},
        ax=ax,
    )

    ax.set_title("Shortlisted feature correlation heatmap", fontsize=17, pad=16, loc="left")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

    fig.text(
        0.125,
        0.92,
        "Target plus top numeric candidate features",
        fontsize=10.5,
        color="#666666",
    )

    fig.subplots_adjust(left=0.30, bottom=0.22, top=0.85)
    _save_figure(fig, output_dir / "shortlisted_correlation_heatmap.png", dpi=dpi)


def generate_figures(
    input_path: Path,
    output_dir: Path,
    target: str = "residual_load_mwh",
    top_corr_n: int = 12,
    heatmap_n: int = 6,
    dpi: int = 150,
) -> None:
    logging.info("Loading validated data from %s", input_path)
    df = pd.read_parquet(input_path)
    logging.info("Loaded %s rows and %s columns", f"{len(df):,}", len(df.columns))

    sns.set_theme(style="whitegrid", context="talk")

    df_plot, df_monthly = prepare_base_frames(df)

    plot_residual_time_series(df_plot, output_dir=output_dir, dpi=dpi)
    plot_monthly_load_lines(df_monthly, output_dir=output_dir, dpi=dpi)
    plot_monthly_residual_share(df_monthly, output_dir=output_dir, dpi=dpi)

    corr_df, target_corr = plot_top_correlations(
        df,
        output_dir=output_dir,
        target=target,
        top_n=top_corr_n,
        dpi=dpi,
    )
    plot_correlation_heatmap(
        corr_df,
        target_corr,
        output_dir=output_dir,
        target=target,
        top_n=heatmap_n,
        dpi=dpi,
    )

    logging.info("Figure generation complete. Output directory: %s", output_dir)


if __name__ == "__main__":
    params = load_params()
    viz = params["visualization"]

    generate_figures(
        input_path=Path(viz["input_path"]),
        output_dir=Path(viz["output_dir"]),
        target=viz["target"],
        top_corr_n=int(viz["top_corr_n"]),
        heatmap_n=int(viz["heatmap_n"]),
        dpi=int(viz["dpi"]),
    )
