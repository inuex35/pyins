# Copyright 2024 inuex35
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Satellite position computation from ephemeris"""


import math

import numpy as np

import cssrlib.ephemeris
from cssrlib.gnss import gpst2time

from ..core.constants import *
from ..core.data_structures import Ephemeris, GloEphemeris
from ..core.time import timediff

_WEEK_SECONDS = 604800.0


def _to_gtime(seconds: float, *, week_hint: int | None = None, reference_seconds: float | None = None):
    if week_hint is not None and 0.0 <= seconds < _WEEK_SECONDS:
        week = week_hint
        tow = seconds
    elif reference_seconds is not None and 0.0 <= seconds < _WEEK_SECONDS:
        ref_week = math.floor(reference_seconds / _WEEK_SECONDS)
        ref_tow = reference_seconds - ref_week * _WEEK_SECONDS
        week = int(ref_week)
        tow = seconds
        delta = tow - ref_tow
        half_week = _WEEK_SECONDS / 2.0
        if delta > half_week:
            week -= 1
        elif delta < -half_week:
            week += 1
    else:
        week = int(math.floor(seconds / _WEEK_SECONDS))
        tow = seconds - week * _WEEK_SECONDS
    return gpst2time(week, tow)


def compute_satellite_position(eph: Ephemeris, time: float) -> tuple[np.ndarray, float, np.ndarray]:
    """Compute satellite position and clock bias using cssrlib."""

    if sat2sys(eph.sat) == SYS_GLO:
        raise ValueError("Use compute_glonass_position() for GLONASS ephemerides")

    if eph.A == 0.0:
        nan = np.array([np.nan, np.nan, np.nan], dtype=float)
        return nan, float('nan'), nan.copy()

    reference = eph.week * _WEEK_SECONDS + eph.toe
    t_cssr = _to_gtime(time, week_hint=getattr(eph, 'week', None), reference_seconds=reference)

    rs, dts = cssrlib.ephemeris.eph2pos(t_cssr, eph)
    ura = ura_value(eph.sva)
    var = np.full(3, ura * ura, dtype=float)

    return np.asarray(rs, dtype=float), float(dts), var


def compute_satellite_velocity(eph: Ephemeris, time: float) -> tuple[np.ndarray, float]:
    """Compute satellite velocity and clock drift using cssrlib."""

    if sat2sys(eph.sat) == SYS_GLO:
        raise ValueError("Use compute_glonass_position() for GLONASS ephemerides")

    if eph.A == 0.0:
        nan = np.array([np.nan, np.nan, np.nan], dtype=float)
        return nan, float('nan')

    reference = eph.week * _WEEK_SECONDS + eph.toe
    week_hint = getattr(eph, 'week', None)
    t_cssr = _to_gtime(time, week_hint=week_hint, reference_seconds=reference)

    _, vs, dts = cssrlib.ephemeris.eph2pos(t_cssr, eph, flg_v=True)

    delta = 1.0
    t_forward = _to_gtime(time + delta, week_hint=week_hint, reference_seconds=reference)
    t_backward = _to_gtime(time - delta, week_hint=week_hint, reference_seconds=reference)
    clk_forward = cssrlib.ephemeris.eph2clk(t_forward, eph)
    clk_backward = cssrlib.ephemeris.eph2clk(t_backward, eph)
    ddts = (clk_forward - clk_backward) / (2.0 * delta)

    return np.asarray(vs, dtype=float), float(ddts)


def ura_value(sva: int) -> float:
    """
    Convert User Range Accuracy (URA) index to actual accuracy value in meters.

    The URA provides an indication of the expected ranging accuracy for
    a satellite. This function converts the 4-bit URA index broadcast
    in the navigation message to the corresponding accuracy value.

    Parameters
    ----------
    sva : int
        URA index (0-15) from satellite ephemeris

    Returns
    -------
    float
        URA accuracy value in meters

    Notes
    -----
    URA index mapping (GPS Interface Specification):
    - 0: 2.4 m     - 8: 96.0 m
    - 1: 3.4 m     - 9: 192.0 m
    - 2: 4.85 m    - 10: 384.0 m
    - 3: 6.85 m    - 11: 768.0 m
    - 4: 9.65 m    - 12: 1536.0 m
    - 5: 13.65 m   - 13: 3072.0 m
    - 6: 24.0 m    - 14: 6144.0 m
    - 7: 48.0 m    - 15: No accuracy prediction

    Values outside the 0-15 range return 0.0 meters.

    Examples
    --------
    >>> accuracy = ura_value(2)  # Returns 4.85 meters
    >>> print(f"Expected ranging accuracy: {accuracy:.2f} m")
    """
    ura_eph = [
        2.4, 3.4, 4.85, 6.85, 9.65, 13.65, 24.0, 48.0,
        96.0, 192.0, 384.0, 768.0, 1536.0, 3072.0, 6144.0, 0.0
    ]
    return ura_eph[sva] if 0 <= sva <= 15 else 0.0


def geph2clk(geph: GloEphemeris, time: float) -> float:
    """Compute GLONASS satellite clock bias using cssrlib."""

    t_cssr = _to_gtime(time, reference_seconds=geph.toe)
    return float(cssrlib.ephemeris.geph2clk(t_cssr, geph))


def compute_glonass_position(geph: GloEphemeris, time: float) -> tuple[np.ndarray, float, np.ndarray]:
    """Compute GLONASS satellite position using cssrlib."""

    t_cssr = _to_gtime(time, reference_seconds=geph.toe)
    result = cssrlib.ephemeris.geph2pos(t_cssr, geph)

    if len(result) == 2:
        rs, dts = result
    else:
        rs, _, dts = result

    var = np.full(3, ERREPH_GLO**2, dtype=float)
    return np.asarray(rs, dtype=float), float(dts), var




def compute_glonass_velocity(geph: GloEphemeris, time: float) -> tuple[np.ndarray, float]:
    """Compute GLONASS satellite velocity using cssrlib."""

    t_cssr = _to_gtime(time, reference_seconds=geph.toe)
    _, vs, _ = cssrlib.ephemeris.geph2pos(t_cssr, geph, flg_v=True)
    return np.asarray(vs, dtype=float), float(geph.gamn)


def satpos_sbas(eph: Ephemeris, time: float) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Compute SBAS (Satellite-Based Augmentation System) satellite position.

    SBAS satellites (e.g., WAAS, EGNOS, MSAS) use geostationary orbits
    with simplified ephemeris parameters. This function computes their
    positions using a linear propagation model rather than Keplerian
    orbital mechanics.

    Parameters
    ----------
    eph : Ephemeris
        SBAS ephemeris containing simplified orbital parameters:
        - A: X-coordinate or semi-major axis (m)
        - e: Y-coordinate or eccentricity
        - i0: Z-coordinate or inclination (rad)
        - deln: X-velocity or mean motion correction (m/s)
        - OMGd: Y-velocity or rate of right ascension (m/s)
        - idot: Z-velocity or inclination rate (m/s)
        - f0, f1, f2: Clock parameters
        - toe: Reference time

    time : float
        Time of interest in GPS time (seconds)

    Returns
    -------
    tuple[np.ndarray, float, np.ndarray]
        - rs : np.ndarray, shape (3,)
            Satellite position in ECEF coordinates (m)
        - dts : float
            Satellite clock bias (seconds)
        - var : np.ndarray, shape (3,)
            Position variance (m², simplified to 100 m²)

    Notes
    -----
    SBAS ephemeris format differs significantly from GPS/Galileo:
    - Uses Cartesian coordinates instead of Keplerian elements
    - Employs linear propagation: pos(t) = pos0 + vel * (t - t0)
    - Clock model is similar to GPS (polynomial)
    - Simplified variance model due to geostationary nature

    The linear model is appropriate for SBAS satellites because:
    1. They maintain geostationary orbits
    2. Orbital corrections are small over short periods
    3. High precision is less critical than for navigation satellites

    Examples
    --------
    >>> pos, clk, var = satpos_sbas(sbas_eph, gps_time)
    >>> print(f"SBAS position: {pos/1000:.0f} km")
    >>> print(f"Clock bias: {clk*1e9:.1f} ns")
    """
    # Time from ephemeris
    t = timediff(time, eph.toe)

    # Position
    rs = np.array([
        eph.A + eph.deln * t,
        eph.e + eph.OMGd * t,
        eph.i0 + eph.idot * t
    ])

    # Clock
    dts = eph.f0 + eph.f1 * t + eph.f2 * t**2

    # Variance
    var = np.ones(3) * 100.0  # Simplified

    return rs, dts, var
