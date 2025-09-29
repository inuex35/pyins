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

"""Ephemeris selection and satellite position computation.

This module provides comprehensive functionality for handling both broadcast
ephemeris (from RINEX navigation files) and precise SP3 ephemeris data. It
implements satellite position calculation, clock bias computation, and
ephemeris selection algorithms following GNSS standards and RTKLIB conventions.

Key Features:
- Multi-constellation support (GPS, GLONASS, Galileo, BeiDou, QZSS)
- Broadcast ephemeris processing with Kepler orbital elements
- GLONASS-specific ephemeris handling with numerical integration
- SP3 precise ephemeris interpolation
- Relativistic corrections and Earth rotation effects
- Robust ephemeris selection with time rollover handling
- Iterative satellite clock and position calculation

Time Systems:
- GPS Time: Standard for GPS, Galileo, QZSS satellites
- GLONASS Time: For GLONASS satellites (handled internally)
- BeiDou Time (BDT): For BeiDou satellites (14s offset from GPS)
- All outputs normalized to GPS Time system

Ephemeris Types Supported:
- GPS/QZSS: Standard Keplerian ephemeris
- Galileo: Compatible with GPS format
- BeiDou: Keplerian with GEO satellite special handling
- GLONASS: Numerical integration of equations of motion

Functions:
    timediff: Handle time differences with week rollover
    dtadjust: Adjust time differences for week boundaries
    seleph: Select best ephemeris for satellite at given time
    eph2clk: Calculate satellite clock bias from ephemeris
    eph2pos: Compute satellite position from broadcast ephemeris
    satpos: Compute positions for all satellites in observation
    satpos_sp3: Compute positions using SP3 precise ephemeris
    compute_satellite_position: Unified position computation interface

Notes:
    All position outputs are in ECEF coordinates (meters).
    Clock biases are in seconds, normalized to GPS time system.
    Variance outputs represent position uncertainty in m².
"""

from typing import Optional, Union

import math

import numpy as np

import cssrlib.ephemeris
from cssrlib.gnss import gpst2time, gtime_t, time2gpst

from ..core.constants import *
from ..core.unified_time import TimeCore
from ..satellite.ephemeris import geph2clk as sat_geph2clk
from ..satellite.ephemeris import geph2pos as sat_geph2pos
from ..satellite.satellite_position import ura_value

_WEEK_SECONDS = 604800.0


def _to_seconds(value):
    if isinstance(value, gtime_t):
        week, tow = time2gpst(value)
        return week * _WEEK_SECONDS + tow
    return float(value)


def timediff(t1, t2):
    """Calculate time difference in seconds with proper week rollover handling.

    Computes the difference between two GPS times, properly handling the weekly
    rollover that occurs every 604,800 seconds (1 week). This is essential for
    GNSS processing where times are often represented as Time of Week (TOW).

    Parameters
    ----------
    t1 : float
        First time in seconds. Can be either GPS seconds or Time of Week (TOW).
        Values > 604800 are treated as full GPS time and converted to TOW.
    t2 : float
        Second time in seconds. Can be either GPS seconds or Time of Week (TOW).
        Values > 604800 are treated as full GPS time and converted to TOW.

    Returns
    -------
    float
        Time difference (t1 - t2) in seconds, adjusted for week boundaries.
        The result is normalized to the range [-302400, 302400] seconds
        (±half week) to ensure the shortest time difference is returned.

    Notes
    -----
    GPS weeks roll over every 604,800 seconds (7 days). This function ensures
    that time differences across week boundaries are calculated correctly by:
    1. Converting full GPS times to TOW if necessary
    2. Computing the raw difference
    3. Adjusting for week rollover if the difference exceeds ±half week

    Examples
    --------
    >>> # Times within same week
    >>> dt = timediff(100000, 50000)
    >>> print(dt)  # 50000.0

    >>> # Times across week boundary
    >>> dt = timediff(10000, 590000)  # Near week rollover
    >>> print(dt)  # Should be negative, adjusted for rollover

    >>> # Full GPS times (automatically converted to TOW)
    >>> dt = timediff(1234567890, 1234567800)
    >>> print(dt)  # 90.0
    """
    # Convert both times to TOW if one is GPS time
    t1 = _to_seconds(t1)
    t2 = _to_seconds(t2)

    if t1 > 604800:
        t1 = t1 % 604800
    if t2 > 604800:
        t2 = t2 % 604800

    # Calculate difference
    dt = t1 - t2

    # Handle week rollover
    if dt > 302400:  # More than half week
        dt -= 604800
    elif dt < -302400:
        dt += 604800

    return dt


def dtadjust(t1, t2, tw=604800):
    """Calculate time difference with week rollover adjustment.

    Computes the time difference between two GPS times with proper handling of
    week boundaries. This function is commonly used in GNSS ephemeris processing
    where accurate time differences are critical for position calculations.

    Parameters
    ----------
    t1 : float
        First time in seconds. Can be GPS seconds or Time of Week (TOW).
    t2 : float
        Second time in seconds. Can be GPS seconds or Time of Week (TOW).
    tw : float, optional
        Week duration in seconds. Default is 604800 (GPS week length).
        Other values can be used for different time systems if needed.

    Returns
    -------
    float
        Adjusted time difference (t1 - t2) in seconds, normalized to the range
        [-tw/2, tw/2] to ensure the shortest possible time difference.

    Notes
    -----
    This function builds upon timediff() by adding an additional adjustment
    step to ensure the time difference falls within ±half the specified period.
    This is particularly important for ephemeris validity checks and time
    synchronization in GNSS processing.

    The adjustment algorithm:
    1. Calculate initial time difference using timediff()
    2. If difference > tw/2, subtract tw
    3. If difference < -tw/2, add tw

    Examples
    --------
    >>> # Normal time difference within same week
    >>> dt = dtadjust(100000, 50000)
    >>> print(dt)  # 50000.0

    >>> # Time difference across week boundary
    >>> dt = dtadjust(50000, 550000)  # Should be adjusted
    >>> print(dt)  # Will be adjusted to avoid large positive difference

    >>> # Custom period (e.g., for testing)
    >>> dt = dtadjust(350000, 50000, tw=300000)
    >>> print(dt)  # Adjusted for 300000-second period
    """
    dt = timediff(t1, t2)
    if dt > tw / 2:
        dt -= tw
    elif dt < -tw / 2:
        dt += tw
    return dt


def seleph(nav, t, sat):
    """Select the best ephemeris for a satellite at a given time.

    Searches through available broadcast ephemeris data to find the most
    appropriate ephemeris for a specific satellite at a given time. The
    selection algorithm minimizes the time difference between the requested
    time and the ephemeris time of ephemeris (toe) or time of clock (toc).

    Parameters
    ----------
    nav : NavigationData
        Navigation data object containing ephemeris collections:
        - nav.eph: GPS/Galileo/BeiDou/QZSS ephemerides
        - nav.geph: GLONASS ephemerides
    t : float or TimeCore
        Time of interest. Can be:
        - TimeCore object with GPS time information
        - Float representing GPS seconds (full GPS time)
        - Float representing Time of Week (TOW < 604800)
    sat : int
        Satellite number in the internal numbering system:
        - GPS: 1-32, GLONASS: 65-96, Galileo: 301-336, etc.

    Returns
    -------
    Ephemeris or GloEphemeris or None
        Best matching ephemeris for the satellite at the given time.
        Returns None if no suitable ephemeris is found.
        - For non-GLONASS: returns Ephemeris object
        - For GLONASS: returns GloEphemeris object

    Notes
    -----
    Selection Strategy:
    - Non-GLONASS satellites: Uses Time of Week (TOW) for comparison
    - GLONASS satellites: Uses full GPS time with preference for past ephemerides
    - Ephemerides are assumed to be sorted by time in the navigation data
    - Selection terminates early when time differences start increasing

    Time Handling:
    - GPS, Galileo, BeiDou, QZSS: Compared using TOW modulo 604800
    - GLONASS: Compared using full GPS time with 1-hour penalty for future ephemerides
    - BeiDou GEO satellites: Standard selection but may need special handling

    Validity Thresholds:
    - Standard satellites: Selects closest ephemeris by toe
    - GLONASS: Prefers past ephemeris, adds penalty for future ones
    - Maximum search continues until time differences increase consistently

    Examples
    --------
    >>> import numpy as np
    >>> from pyins.core.unified_time import TimeCore
    >>>
    >>> # Select ephemeris using TimeCore
    >>> time = TimeCore(gps_seconds=1234567890)
    >>> eph = seleph(nav, time, sat=1)  # GPS satellite 1
    >>>
    >>> # Select ephemeris using TOW
    >>> eph = seleph(nav, 345600.0, sat=65)  # GLONASS satellite
    >>>
    >>> # Check if ephemeris was found
    >>> if eph is not None:
    >>>     print(f"Found ephemeris for satellite {sat}")
    >>>     print(f"Ephemeris toe: {eph.toe}")
    """
    dt_best = 1e10
    eph_best = None

    # Handle TimeCore input
    if hasattr(t, 'get_tow'):
        # It's a TimeCore object
        t_tow = t.get_tow()
        t_gps = t.get_gps_seconds()
    else:
        # It's a float - could be TOW or GPS seconds
        if t > 604800:
            t_tow = t % 604800
            t_gps = t
        else:
            t_tow = t
            t_gps = t  # Assume it's TOW for now

    # Get satellite system
    sys = sat2sys(sat)

    if sys != SYS_GLO:
        # GPS, Galileo, BeiDou, QZSS, etc. - use TOW
        for eph in nav.eph:
            if eph.sat != sat:
                continue

            # For Galileo, check signal availability
            if sys == SYS_GAL:
                # TODO: Check Galileo signal codes when needed
                pass

            # Calculate time difference from time of ephemeris using TOW
            dt = abs(dtadjust(t_tow, eph.toe))

            # Select ephemeris with minimum time difference
            if dt <= dt_best:
                dt_best = dt
                eph_best = eph
            else:
                # Ephemerides are sorted by time, so we can break
                break

        return eph_best

    else:
        # GLONASS uses geph and needs full GPS time
        # For GLONASS, prefer past ephemeris over future ones
        # This matches RTKLIB behavior
        for geph in nav.geph:
            if geph.sat != sat:
                continue

            # GLONASS ephemeris toe is in full GPS time, not TOW
            # So we need to use full GPS time for comparison
            dt_raw = t_gps - _to_seconds(geph.toe)

            # For GLONASS, prefer past ephemeris (dt_raw >= 0)
            # Add small penalty for future ephemeris
            if dt_raw < 0:
                # Future ephemeris - add penalty
                dt = abs(dt_raw) + 3600.0  # Add 1 hour penalty
            else:
                # Past ephemeris - use as is
                dt = abs(dt_raw)

            if dt <= dt_best:
                dt_best = dt
                eph_best = geph
            else:
                # Since ephemerides are sorted by time, we can break
                # only if we're past the best match
                if dt_best < 7200.0:  # If we have a good match (< 2 hours)
                    break
        # Fallback: some CSSR readers may store GLONASS ephemerides in nav.eph
        # using generic Eph structures. If nothing found in nav.geph, attempt
        # to locate a matching entry in nav.eph before giving up.
        if eph_best is None:
            for eph in nav.eph:
                if eph.sat != sat:
                    continue

                dt = abs(dtadjust(t_tow, eph.toe))
                if dt <= dt_best:
                    dt_best = dt
                    eph_best = eph
                else:
                    break

        return eph_best


def _extract_time_seconds(t: Union[float, TimeCore]) -> tuple[float, float]:
    """Return (gps_seconds, tow) from input time."""

    if hasattr(t, "get_gps_seconds"):
        gps_seconds = float(t.get_gps_seconds())
        tow = float(t.get_tow())
    else:
        gps_seconds = float(t)
        if gps_seconds >= _WEEK_SECONDS:
            tow = gps_seconds % _WEEK_SECONDS
        else:
            tow = gps_seconds
    return gps_seconds, tow


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


def eph2clk(t, eph):
    """Calculate satellite clock bias using cssrlib."""

    gps_seconds, tow = _extract_time_seconds(t)

    if hasattr(eph, "taun"):
        return sat_geph2clk(eph, gps_seconds)

    reference = eph.week * _WEEK_SECONDS + _to_seconds(eph.toc)

    if gps_seconds >= _WEEK_SECONDS:
        t_cssr = _to_gtime(gps_seconds, reference_seconds=reference)
    else:
        t_cssr = _to_gtime(tow, week_hint=getattr(eph, 'week', None), reference_seconds=reference)

    return float(cssrlib.ephemeris.eph2clk(t_cssr, eph))


def eph2pos(t, eph):
    """Compute satellite position using cssrlib."""

    gps_seconds, tow = _extract_time_seconds(t)

    if hasattr(eph, "taun"):
        return sat_geph2pos(eph, gps_seconds)

    reference = eph.week * _WEEK_SECONDS + _to_seconds(eph.toe)

    if gps_seconds >= _WEEK_SECONDS:
        t_cssr = _to_gtime(gps_seconds, reference_seconds=reference)
    else:
        t_cssr = _to_gtime(tow, week_hint=getattr(eph, 'week', None), reference_seconds=reference)

    rs, dts = cssrlib.ephemeris.eph2pos(t_cssr, eph)
    var = ura_value(eph.sva) ** 2
    return np.asarray(rs, dtype=float), var, float(dts)


def satpos(obs, nav):
    """
    Compute satellite positions and clocks for all satellites in observation
    (Based on SPP implementation)

    Parameters
    ----------
    obs : list of Observation
        Observation data
    nav : NavigationData or SP3Ephemeris
        Navigation data (broadcast) or SP3 ephemeris handler

    Returns
    -------
    rs : np.ndarray
        Satellite positions (n x 3)
    dts : np.ndarray
        Satellite clock biases (n)
    var : np.ndarray
        Position variances (n)
    svh : np.ndarray
        Satellite health (n)
    """
    # Check if SP3 or broadcast ephemeris
    from .sp3_ephemeris import SP3Ephemeris
    if isinstance(nav, SP3Ephemeris):
        return satpos_sp3(obs, nav)
    n = len(obs)
    rs = np.zeros((n, 3))
    dts = np.zeros(n)
    var = np.zeros(n)
    svh = np.zeros(n, dtype=int)

    for i, ob in enumerate(obs):
        # Get pseudorange (prefer L1)
        pr = ob.P[0] if ob.P[0] > 0 else ob.P[1]
        if pr == 0:
            svh[i] = -1
            continue

        # Get time as float (GPS seconds)
        if hasattr(ob, 'time'):
            obs_time = ob.time if isinstance(ob.time, (int, float)) else ob.time
        else:
            svh[i] = -1
            continue

        # Convert UNIX time to GPS time if needed
        if obs_time > 1.5e9:  # Likely UNIX time
            GPS_UNIX_OFFSET = 315964800
            LEAP_SECONDS = 18
            gps_time = obs_time - GPS_UNIX_OFFSET + LEAP_SECONDS
        else:
            gps_time = obs_time

        # Calculate transmission time
        t_tx = gps_time - pr / CLIGHT

        # Convert to week and TOW for ephemeris selection
        from ..core.time import gps_seconds_to_week_tow
        week, tow = gps_seconds_to_week_tow(t_tx)

        # Select ephemeris
        # GLONASS needs full GPS time, others can use TOW
        sys = sat2sys(ob.sat)
        if sys == SYS_GLO:
            eph = seleph(nav, t_tx, ob.sat)  # Use full GPS time for GLONASS
        else:
            eph = seleph(nav, tow, ob.sat)  # Use TOW for others
        if eph is None:
            svh[i] = -1
            continue

        try:
            # For GLONASS, use full GPS time; for others, use full GPS time too
            # Using full GPS time for all systems ensures consistent clock computation
            if sys == SYS_GLO:  # GLONASS (use integer constant)
                # Use full GPS time for GLONASS (t_tx is already full GPS time)
                time_for_eph = t_tx  # Full GPS time
            else:
                # Use full GPS time for GPS/Galileo/BeiDou for better precision
                # This ensures clock differences are properly computed for close times
                time_for_eph = t_tx  # Use full GPS time instead of TOW

            # Satellite clock correction
            dt_sat = eph2clk(time_for_eph, eph)
            dts[i] = dt_sat

            # Satellite position at corrected transmission time
            result = eph2pos(time_for_eph - dt_sat, eph)

            if result is not None:
                # Handle different return types
                if isinstance(result, tuple) and len(result) >= 2:
                    rs[i] = result[0]
                    if len(result) >= 2 and not np.isnan(result[1]).any():
                        var[i] = result[1] if np.isscalar(result[1]) else 0.0
                else:
                    svh[i] = -1
                    continue

                # Sanity check on position
                radius = np.linalg.norm(rs[i])
                if radius < 20000e3 or radius > 50000e3:
                    svh[i] = -1
                    rs[i] = np.zeros(3)
                    dts[i] = 0.0
                    continue

            else:
                svh[i] = -1

        except Exception:
            svh[i] = -1
            continue
        # Check satellite health
        svh[i] = getattr(eph, 'svh', 0)

    return rs, dts, var, svh


def satpos_sp3(obs, sp3_eph, poly_degree: int = 10, method: str = 'neville'):
    """
    Compute satellite positions using SP3 precise ephemeris

    Parameters
    ----------
    obs : list of Observation
        Observation data
    sp3_eph : SP3Ephemeris
        SP3 ephemeris handler with loaded data
    poly_degree : int
        Polynomial degree for interpolation (default 10 for RTKLIB)
    method : str
        Interpolation method ('neville', 'polyfit', 'lagrange')

    Returns
    -------
    rs : np.ndarray
        Satellite positions (n x 3)
    dts : np.ndarray
        Satellite clock biases (n)
    var : np.ndarray
        Position variances (n)
    svh : np.ndarray
        Satellite health (n)
    """
    n = len(obs)
    rs = np.zeros((n, 3))
    dts = np.zeros(n)
    var = np.zeros(n)
    svh = np.zeros(n, dtype=int)

    for i, ob in enumerate(obs):
        # Get pseudorange (prefer L1)
        pr = ob.P[0] if ob.P[0] > 0 else ob.P[1]
        if pr == 0:
            svh[i] = -1
            continue

        # Get time
        if hasattr(ob, 'time'):
            if isinstance(ob.time, TimeCore):
                time = ob.time
            else:
                # Assume GPS seconds
                time = TimeCore(ob.time)
        else:
            svh[i] = -1
            continue

        # Calculate transmission time
        tau = pr / CLIGHT
        transmit_time = time - tau

        # Iterate for better accuracy
        for _ in range(2):
            # Get satellite position from SP3
            pos, clk, position_var = sp3_eph.interpolate_position(
                ob.sat, transmit_time,
                poly_degree=poly_degree,
                method=method
            )

            if pos is None:
                svh[i] = -1
                break

            # Update transmission time
            tau = np.linalg.norm(pos) / CLIGHT
            transmit_time = time - tau

        if pos is not None:
            # Earth rotation correction
            omega_e = 7.2921151467e-5  # Earth rotation rate (rad/s)
            theta = omega_e * tau

            # Rotation matrix
            R = np.array([
                [np.cos(theta), np.sin(theta), 0],
                [-np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])

            # Apply Earth rotation correction
            rs[i] = R @ pos
            dts[i] = clk
            var[i] = position_var
            svh[i] = 0  # SP3 doesn't have health flags, assume healthy

            # Sanity check on position
            radius = np.linalg.norm(rs[i])
            if radius < 20000e3 or radius > 50000e3:
                svh[i] = -1
                rs[i] = np.zeros(3)
                dts[i] = 0.0
        else:
            svh[i] = -1

    return rs, dts, var, svh


def compute_satellite_position(sat_num: int, time: Union[TimeCore, float],
                              nav_or_sp3, pseudorange: Optional[float] = None,
                              prefer_mgex: bool = True,
                              **kwargs) -> tuple[np.ndarray, float, float]:
    """
    Unified function to compute satellite position from either broadcast or SP3 ephemeris

    Parameters
    ----------
    sat_num : int
        Satellite number
    time : TimeCore or float
        Time (GPS seconds or TimeCore object)
    nav_or_sp3 : dict or SP3Ephemeris
        Navigation data or SP3 ephemeris object
    pseudorange : float, optional
        Pseudorange for signal transmission time calculation
    prefer_mgex : bool
        If True and using SP3, prefer MGEX products for multi-GNSS
    **kwargs
        Additional arguments for SP3 interpolation

    Parameters
    ----------
    sat_num : int
        Satellite number
    time : TimeCore or float
        Time (TimeCore object or GPS seconds)
    nav_or_sp3 : NavigationData or SP3Ephemeris
        Navigation data (broadcast) or SP3 ephemeris handler
    pseudorange : float, optional
        Pseudorange for transmission time calculation
    **kwargs : dict
        Additional arguments for SP3 interpolation (poly_degree, method)

    Returns
    -------
    position : np.ndarray
        Satellite position [x, y, z] in ECEF meters
    clock : float
        Satellite clock correction in seconds
    variance : float
        Position variance
    """
    from .sp3_ephemeris import SP3Ephemeris

    # Convert time to TimeCore if needed
    if not isinstance(time, TimeCore):
        time = TimeCore(time)

    # Handle SP3 ephemeris
    if isinstance(nav_or_sp3, SP3Ephemeris):
        # Calculate transmission time if pseudorange provided
        if pseudorange and pseudorange > 0:
            tau = pseudorange / CLIGHT
            transmit_time = time - tau

            # Iterate for better accuracy
            for _ in range(2):
                pos, clk, var = nav_or_sp3.interpolate_position(
                    sat_num, transmit_time,
                    poly_degree=kwargs.get('poly_degree', 10),
                    method=kwargs.get('method', 'neville')
                )
                if pos is None:
                    return None, None, None
                tau = np.linalg.norm(pos) / CLIGHT
                transmit_time = time - tau
        else:
            transmit_time = time

        # Get position
        pos, clk, var = nav_or_sp3.interpolate_position(
            sat_num, transmit_time,
            poly_degree=kwargs.get('poly_degree', 10),
            method=kwargs.get('method', 'neville')
        )

        # Earth rotation correction if pseudorange provided
        if pos is not None and pseudorange and pseudorange > 0:
            omega_e = 7.2921151467e-5
            theta = omega_e * tau
            R = np.array([
                [np.cos(theta), np.sin(theta), 0],
                [-np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
            pos = R @ pos

        return pos, clk, var

    # Handle broadcast ephemeris
    else:
        gps_seconds = time.get_gps_seconds()

        # Select ephemeris
        eph = seleph(nav_or_sp3, gps_seconds, sat_num)
        if eph is None:
            return None, None, None

        # Calculate transmission time if pseudorange provided
        if pseudorange and pseudorange > 0:
            tau = pseudorange / CLIGHT
            transmit_time = gps_seconds - tau

            # Iterate for better accuracy
            for _ in range(2):
                pos, vel, clk = eph2pos(transmit_time, eph)
                if pos is None:
                    return None, None, None
                tau = np.linalg.norm(pos) / CLIGHT
                transmit_time = gps_seconds - tau
        else:
            transmit_time = gps_seconds

        # Compute position
        pos, vel, clk = eph2pos(transmit_time, eph)

        if pos is None:
            return None, None, None

        # Earth rotation correction if pseudorange provided
        if pseudorange and pseudorange > 0:
            omega_e = 7.2921151467e-5
            theta = omega_e * tau
            R = np.array([
                [np.cos(theta), np.sin(theta), 0],
                [-np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
            pos = R @ pos

        # Estimate variance from SV accuracy
        if hasattr(eph, 'sva'):
            if eph.sva <= 6:
                var = (2**(1 + eph.sva/2))**2
            else:
                var = (2**eph.sva)**2
        else:
            var = 4.0**2  # Default 4m

        return pos, clk, var
