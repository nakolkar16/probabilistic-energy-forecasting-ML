from __future__ import annotations

import numpy as np


def quantile_crossing_rate(pred_map: dict[float, np.ndarray], quantiles: list[float]) -> float:
    ordered_quantiles = sorted(float(q) for q in quantiles)
    q_matrix = np.column_stack([np.asarray(pred_map[q], dtype=float) for q in ordered_quantiles])
    return float((np.diff(q_matrix, axis=1) < 0).any(axis=1).mean())


def repair_quantile_order(
    pred_map: dict[float, np.ndarray],
    quantiles: list[float],
) -> tuple[dict[float, np.ndarray], float]:
    ordered_quantiles = sorted(float(q) for q in quantiles)
    q_matrix = np.column_stack([np.asarray(pred_map[q], dtype=float) for q in ordered_quantiles])
    crossed_rows = (np.diff(q_matrix, axis=1) < 0).any(axis=1)
    repaired_matrix = np.sort(q_matrix, axis=1)
    repaired = {
        q: repaired_matrix[:, idx].copy()
        for idx, q in enumerate(ordered_quantiles)
    }
    return repaired, float(crossed_rows.mean())
