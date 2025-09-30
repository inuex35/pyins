"""Satellite geometry helpers backed by cssrlib."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from cssrlib.ephemeris import satpos as cssr_satpos
from cssrlib.gnss import gpst2bdt, gpst2utc, sat2id, timeadd, rCST


def _first_valid_pseudorange(p_values: np.ndarray) -> float | None:
    for value in p_values:
        if value > 0:
            return float(value)
    return None


def _transmit_time(reception_time, pseudorange: float, sys_char: str):
    from cssrlib.gnss import timeadd

    return timeadd(reception_time, -pseudorange / rCST.CLIGHT)


def compute_satellite_positions(obs_epoch, nav) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return satellite positions, clocks, and health flags using cssrlib data."""

    n = len(obs_epoch.sat)
    positions = np.zeros((n, 3))
    clocks = np.zeros(n)
    health = np.full(n, -1, dtype=int)

    for idx, sat in enumerate(obs_epoch.sat):
        pr = _first_valid_pseudorange(np.asarray(obs_epoch.P[idx], dtype=float))
        if pr is None:
            continue

        sat_id = sat2id(sat)
        sys_char = sat_id[0] if sat_id else 'G'
        t_tx = _transmit_time(obs_epoch.t, pr, sys_char)

        rs, _, dts, svh = cssr_satpos(sat, t_tx, nav)
        if rs is None or np.isnan(rs).any() or dts is None or np.isnan(dts).any():
            continue

        positions[idx] = rs[0]
        clocks[idx] = dts[0]
        health[idx] = svh[0]

    return positions, clocks, health
