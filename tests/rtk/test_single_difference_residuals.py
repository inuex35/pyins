"""Validate cssrlib-based single-difference residual computation."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "test_data"
CSSRLIB_SRC = REPO_ROOT / "cssrlib" / "src"

if str(CSSRLIB_SRC) not in sys.path:
    sys.path.insert(0, str(CSSRLIB_SRC))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from copy import deepcopy

import numpy as np

from cssrlib.gnss import time2gpst
from cssrlib.gnss import sat2id
from cssrlib.pntpos import stdpos

from pyins.gnss.geometry import compute_satellite_positions
from pyins.io import (parse_rinex_approx_position, read_nav, read_obs,
                      reference_position_from_pos)
from pyins.rtk.single_difference import form_single_differences


def _gps_seconds(obs) -> float:
    week, tow = time2gpst(obs.t)
    return week * 604800.0 + tow


def _nearest_epoch(epochs, gps_time):
    return min(epochs, key=lambda epoch: abs(_gps_seconds(epoch) - gps_time))


def test_single_difference_residual_matches_cssrlib_geometry():
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

    sd_entries = form_single_differences(
        rover_epoch,
        base_epoch,
        nav,
        rover_ref_ecef,
        base_ecef,
        min_elevation_deg=10.0,
    )

    assert sd_entries, "Expected single-difference measurements"

    positions_rover, clocks_rover, _ = compute_satellite_positions(rover_epoch, nav)
    positions_base, clocks_base, _ = compute_satellite_positions(base_epoch, nav)

    cssr_rover = _cssrlib_residuals(nav, rover_epoch, rover_ref_ecef, positions_rover, clocks_rover)
    cssr_base = _cssrlib_residuals(nav, base_epoch, base_ecef, positions_base, clocks_base)

    matched = 0
    for sat_id, entry in sd_entries.items():
        if sat_id not in cssr_rover or sat_id not in cssr_base:
            continue
        cssr_sd = cssr_rover[sat_id] - cssr_base[sat_id]
        assert np.isclose(entry["sd_residual"], cssr_sd, atol=0.01)

        expected_predicted = entry["sd_obs"] - entry["sd_residual"]
        assert np.isclose(entry["sd_predicted"], expected_predicted, atol=0.01)
        matched += 1

    assert matched, "Expected at least one satellite matched for CSSRLIB comparison"


def _cssrlib_residuals(nav, obs_epoch, receiver_pos, positions, clocks):
    nav_copy = deepcopy(nav)
    solver = stdpos(nav_copy, pos0=receiver_pos)
    solver.nav.t = obs_epoch.t
    vs = np.zeros_like(positions)

    y, *_ = solver.zdres(obs_epoch, None, None, positions, vs, clocks, solver.nav.x)
    residuals = {}
    for idx, sat in enumerate(obs_epoch.sat):
        sat_id = sat2id(sat)
        if not sat_id:
            continue
        if idx >= obs_epoch.P.shape[0] or obs_epoch.P[idx][0] <= 0:
            continue
        residuals[sat_id] = float(y[idx, 0])
    return residuals
