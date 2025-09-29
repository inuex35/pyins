"""Validate double-difference residual computation against geometry."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "test_data"
CSSRLIB_SRC = REPO_ROOT / "cssrlib" / "src"

if str(CSSRLIB_SRC) not in sys.path:
    sys.path.insert(0, str(CSSRLIB_SRC))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyins.coordinate.transforms import ecef2llh
from pyins.io import (
    RinexNavReader,
    RinexObsReader,
    parse_rinex_approx_position,
    reference_position_from_pos,
)
from pyins.rtk.double_difference import corrected_range, form_double_differences


def _nearest_epoch(epochs, gps_time):
    return min(epochs, key=lambda epoch: abs(epoch["gps_time"] - gps_time))


def test_dd_residual_matches_cssrlib_geometry():
    base_obs_path = DATA_DIR / "base.obs"
    rover_obs_path = DATA_DIR / "main.obs"
    rover_nav_path = DATA_DIR / "rover.nav"
    pos_path = DATA_DIR / "main.pos"

    base_ecef = parse_rinex_approx_position(base_obs_path)
    gps_time_ref, rover_ref_ecef = reference_position_from_pos(pos_path, base_ecef)

    nav = RinexNavReader(str(rover_nav_path)).read()
    rover_epochs = RinexObsReader(str(rover_obs_path)).read()
    base_epochs = RinexObsReader(str(base_obs_path)).read()

    rover_epoch = _nearest_epoch(rover_epochs, gps_time_ref)
    base_epoch = _nearest_epoch(base_epochs, gps_time_ref)

    base_llh_deg = np.degrees(ecef2llh(base_ecef))

    dd_entries = form_double_differences(
        rover_epoch["observations"],
        base_epoch["observations"],
        nav,
        rover_epoch["gps_time"],
        base_ecef,
        base_llh_deg,
        rover_position=rover_ref_ecef,
        use_systems=["G", "E", "C", "R", "J", "S", "I"],
        cutoff_angle=10.0,
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

        if dd["dd_range_m"] is not None:
            assert np.isclose(dd["dd_range_m"], dd_range, atol=1e-6)

        if dd["dd_obs"] is not None and dd["dd_residual_m"] is not None:
            expected_residual = dd["dd_obs"] - dd_range
            assert np.isclose(dd["dd_residual_m"], expected_residual, atol=1e-6)
            residuals_m.append(dd["dd_residual_m"])

        if (
            dd["dd_range_cycles"] is not None
            and dd["dd_carrier"] is not None
        ):
            expected_cycles = dd["dd_carrier"] - dd["dd_range_cycles"]
            assert np.isclose(dd["dd_residual"], expected_cycles, atol=1e-6)

    assert residuals_m, "Expected at least one residual computed in meters"
    assert np.all(np.isfinite(residuals_m))
