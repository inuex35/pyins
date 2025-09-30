"""Single-difference helpers using cssrlib-based residuals."""

from __future__ import annotations

from typing import Dict

import numpy as np

from pyins.gnss.residuals import compute_zero_diff_residuals, ResidualEntry


def form_single_differences(
    rover_obs,
    base_obs,
    nav,
    rover_position: np.ndarray,
    base_position: np.ndarray,
    min_elevation_deg: float = 10.0,
) -> Dict[str, Dict[str, object]]:
    """Compute single-difference measurements keyed by satellite ID."""

    rover_res = compute_zero_diff_residuals(
        rover_obs, nav, rover_position, min_elevation_deg)
    base_res = compute_zero_diff_residuals(
        base_obs, nav, base_position, min_elevation_deg)

    common_ids = sorted(set(rover_res.keys()) & set(base_res.keys()))
    results: Dict[str, Dict[str, object]] = {}

    for sat_id in common_ids:
        rover_entry: ResidualEntry = rover_res[sat_id]
        base_entry: ResidualEntry = base_res[sat_id]

        sd_observed = rover_entry.pseudorange - base_entry.pseudorange
        sd_predicted = rover_entry.predicted - base_entry.predicted
        sd_residual = rover_entry.residual - base_entry.residual

        results[sat_id] = {
            'system': rover_entry.system,
            'sat': sat_id,
            'sd_obs': sd_observed,
            'sd_predicted': sd_predicted,
            'sd_residual': sd_residual,
            'rover_entry': rover_entry,
            'base_entry': base_entry,
        }

    return results


__all__ = ['form_single_differences']
