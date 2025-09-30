"""Double-difference helpers using cssrlib-based residuals."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from cssrlib.gnss import sat2id, rCST

from pyins.gnss.geometry import compute_satellite_positions
from pyins.gnss.residuals import compute_zero_diff_residuals


def corrected_range(sat_pos: np.ndarray, sat_clk: float, receiver_pos: np.ndarray) -> float:
    """Geometric range corrected by satellite clock (meters)."""
    return float(np.linalg.norm(sat_pos - receiver_pos) - rCST.CLIGHT * sat_clk)


def form_double_differences(
    rover_obs,
    base_obs,
    nav,
    rover_position: np.ndarray,
    base_position: np.ndarray,
    min_elevation_deg: float = 10.0,
) -> List[Dict[str, object]]:
    """Compute double-difference measurements from cssrlib observations."""

    rover_res = compute_zero_diff_residuals(rover_obs, nav, rover_position, min_elevation_deg)
    base_res = compute_zero_diff_residuals(base_obs, nav, base_position, min_elevation_deg)

    common = sorted(set(rover_res.keys()) & set(base_res.keys()))
    if len(common) < 2:
        return []

    # Group satellites by constellation
    system_groups: Dict[str, List[str]] = {}
    for sat_id in common:
        sys = sat_id[0]
        system_groups.setdefault(sys, []).append(sat_id)

    dd_entries: List[Dict[str, object]] = []

    for sys_char, sat_ids in system_groups.items():
        if len(sat_ids) < 2:
            continue
        sat_ids.sort()
        ref_sat_id = sat_ids[0]

        ref_rover = rover_res[ref_sat_id]
        ref_base = base_res[ref_sat_id]

        for sat_id in sat_ids[1:]:
            rover_data = rover_res[sat_id]
            base_data = base_res[sat_id]

            dd_obs = (
                rover_data.pseudorange - ref_rover.pseudorange
                - (base_data.pseudorange - ref_base.pseudorange)
            )

            dd_pred = (
                rover_data.predicted - ref_rover.predicted
                - (base_data.predicted - ref_base.predicted)
            )

            dd_res = dd_obs - dd_pred

            entry = {
                'sys': sys_char,
                'sat': sat_id,
                'ref_sat': ref_sat_id,
                'dd_obs': dd_obs,
                'dd_range_m': dd_pred,
                'dd_residual_m': dd_res,
                'sat_pos': rover_data.sat_pos,
                'sat_clk': rover_data.sat_clk,
                'ref_sat_pos': ref_rover.sat_pos,
                'ref_sat_clk': ref_rover.sat_clk,
                'base_sat_pos': base_data.sat_pos,
                'base_sat_clk': base_data.sat_clk,
                'base_ref_sat_pos': ref_base.sat_pos,
                'base_ref_sat_clk': ref_base.sat_clk,
            }

            dd_entries.append(entry)

    return dd_entries


__all__ = ['form_double_differences', 'corrected_range']
