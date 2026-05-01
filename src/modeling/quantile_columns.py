from __future__ import annotations


def quantile_col(q: float) -> str:
    return f"pred_q{int(round(float(q) * 100))}"


def quantile_cols(quantiles: list[float]) -> dict[float, str]:
    return {float(q): quantile_col(float(q)) for q in quantiles}


def low_median_high_quantiles(quantiles: list[float]) -> tuple[float, float, float]:
    ordered = sorted(float(q) for q in quantiles)
    if len(ordered) < 3:
        raise ValueError("At least three quantiles are required for low/median/high forecast display.")
    return ordered[0], ordered[len(ordered) // 2], ordered[-1]
