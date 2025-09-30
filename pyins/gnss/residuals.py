"""Zero-difference residual helpers built on cssrlib."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from cssrlib.gnss import ecef2pos, geodist, rCST, satazel, sat2id, tropmapf, tropmodel
from cssrlib.pntpos import ionmodel

from .geometry import compute_satellite_positions


@dataclass
class ResidualEntry:
    sat: int
    sat_id: str
    system: str
    pseudorange: float
    predicted: float
    residual: float
    geom: float
    trop: float
    iono: float
    sat_pos: np.ndarray
    sat_clk: float
    elevation: float


def _first_valid_pseudorange(p_values: np.ndarray) -> float | None:
    for value in p_values:
        if value > 0:
            return float(value)
    return None


def _transmit_time(reception_time, pseudorange: float, sys_char: str):
    from cssrlib.gnss import gpst2bdt, gpst2utc, timeadd

    t_obs = reception_time
    if sys_char == 'C':  # BeiDou uses BDT
        t_obs = gpst2bdt(t_obs)
    elif sys_char == 'R':  # GLONASS uses UTC+3h
        t_obs = gpst2utc(t_obs)
        t_obs = timeadd(t_obs, 10800.0)
    return timeadd(t_obs, -pseudorange / rCST.CLIGHT)


def compute_zero_diff_residuals(obs_epoch, nav, receiver_pos: np.ndarray,
                                min_elevation_deg: float = 10.0) -> Dict[str, ResidualEntry]:
    """Return zero-difference residual information keyed by satellite ID."""

    positions, clocks, health = compute_satellite_positions(obs_epoch, nav)
    pos_llh = ecef2pos(receiver_pos)

    results: Dict[str, ResidualEntry] = {}

    for idx, sat in enumerate(obs_epoch.sat):
        sat_id = sat2id(sat)
        if not sat_id:
            continue
        sys_char = sat_id[0]

        if health[idx] != 0:
            continue

        pr = _first_valid_pseudorange(np.asarray(obs_epoch.P[idx], dtype=float))
        if pr is None:
            continue

        sat_pos = positions[idx]
        if not np.isfinite(sat_pos).all() or np.linalg.norm(sat_pos) < 1.0:
            continue

        geom_dist, los = geodist(sat_pos, receiver_pos)
        az, el = satazel(pos_llh, los)
        if np.degrees(el) < min_elevation_deg:
            continue

        trop_hs, trop_wet, _ = tropmodel(obs_epoch.t, pos_llh)
        mapfh, mapfw = tropmapf(obs_epoch.t, pos_llh, el)
        trop = mapfh * trop_hs + mapfw * trop_wet
        iono = ionmodel(obs_epoch.t, pos_llh, az, el, nav=nav)

        geom = geom_dist - rCST.CLIGHT * clocks[idx]
        predicted = geom + trop + iono
        residual = pr - predicted

        results[sat_id] = ResidualEntry(
            sat=sat,
            sat_id=sat_id,
            system=sys_char,
            pseudorange=pr,
            predicted=predicted,
            residual=residual,
            geom=geom,
            trop=trop,
            iono=iono,
            sat_pos=sat_pos.copy(),
            sat_clk=clocks[idx],
            elevation=el,
        )

    return results
