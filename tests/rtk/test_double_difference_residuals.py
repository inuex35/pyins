"""Validate cssrlib-based double-difference residual computation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from cssrlib.gnss import time2gpst

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "test_data"
CSSRLIB_SRC = REPO_ROOT / "cssrlib" / "src"

if str(CSSRLIB_SRC) not in sys.path:
    sys.path.insert(0, str(CSSRLIB_SRC))

from pyins.io import parse_rinex_approx_position, read_nav, read_obs, reference_position_from_pos
from pyins.rtk.double_difference import corrected_range, form_double_differences


def _gps_seconds(obs) -> float:
    week, tow = time2gpst(obs.t)
    return week * 604800.0 + tow


def _nearest_epoch(epochs, gps_time):
    return min(epochs, key=lambda epoch: abs(_gps_seconds(epoch) - gps_time))


def test_dd_residual_matches_cssrlib_geometry():
    base_obs_path = DATA_DIR / "base.obs"
    rover_obs_path = DATA_DIR / "main.obs"
    rover_nav_path = DATA_DIR / "rover.nav"
    pos_path = DATA_DIR / "main.pos"

    base_ecef = parse_rinex_approx_position(base_obs_path)
    gps_time_ref, rover_ref_ecef = reference_position_from_pos(pos_path, base_ecef)

    nav = read_nav(str(rover_nav_path))
    rover_epochs = read_obs(str(rover_obs_path))
    base_epochs = read_obs(str(base_obs_path))

    rover_epoch = _nearest_epoch(rover_epochs, gps_time_ref)
    base_epoch = _nearest_epoch(base_epochs, gps_time_ref)

    dd_entries = form_double_differences(
        rover_epoch,
        base_epoch,
        nav,
        rover_ref_ecef,
        base_ecef,
        min_elevation_deg=10.0,
    )

    assert dd_entries, "Expected double-difference measurements"

    residuals_m = []

    for dd in dd_entries:
        dd_range = (
            corrected_range(dd["sat_pos"], dd["sat_clk"], rover_ref_ecef)
            - corrected_range(dd["ref_sat_pos"], dd["ref_sat_clk"], rover_ref_ecef)
            - (
                corrected_range(dd["base_sat_pos"], dd["base_sat_clk"], base_ecef)
                - corrected_range(dd["base_ref_sat_pos"], dd["base_ref_sat_clk"], base_ecef)
            )
        )

        assert np.isclose(dd["dd_range_m"], dd_range, atol=0.1)

        expected_residual = dd["dd_obs"] - dd_range
        assert np.isclose(dd["dd_residual_m"], expected_residual, atol=0.2)
        residuals_m.append(dd["dd_residual_m"])

    assert residuals_m, "Expected at least one residual computed in meters"
    assert np.all(np.isfinite(residuals_m))
