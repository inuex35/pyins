# Copyright 2024 The PyIns Authors. All Rights Reserved.
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

"""Single Point Positioning (SPP) core implementation"""

import numpy as np
from numpy.linalg import norm

from pyins.coordinate import ecef2llh
from pyins.core.constants import (
    CLIGHT, OMGE, RE_WGS84,
    SYS_GPS, SYS_GLO, SYS_GAL, SYS_BDS, SYS_QZS,
    sat2sys, sys2char
)
from pyins.core.data_structures import Solution
from pyins.core.unified_time import TimeCore
from pyins.gnss.ephemeris import eph2pos, seleph
from pyins.gnss.raim import raim_fde


# Constants
MAXITR = 10          # max iterations
MIN_EL = 5.0         # min elevation in degrees
ERR_SAAS = 0.3       # Saastamoinen model error
ERR_ION = 5.0        # Ionosphere error
ERR_CBIAS = 0.3      # Code bias error


def tropmodel_simple(pos, el):
    """Simple tropospheric model (Saastamoinen-like)"""
    if el < np.deg2rad(MIN_EL):
        return 0.0

    # Standard atmosphere at sea level
    P0 = 1013.25  # hPa
    T0 = 288.15   # K
    e0 = 11.75    # hPa (water vapor pressure)

    # Height correction
    h = pos[2] if len(pos) > 2 else 0.0
    if h < 0:
        h = 0.0
    # Limit height to valid range (max ~44km for this formula)
    if h > 44330:
        h = 44330

    # Pressure and temperature at height
    # Ensure the base is positive before exponentiation
    base = 1 - 2.26e-5 * h
    if base <= 0:
        P = 0.0  # Essentially no atmosphere
    else:
        P = P0 * base ** 5.225
    T = T0 - 6.5e-3 * h
    e = e0 * (T / T0) ** 4.0

    # Zenith delays
    zhd = 0.0022768 * P / (1 - 0.00266 * np.cos(2 * pos[0]) - 0.00028e-3 * h)
    zwd = 0.0022768 * (1255 / T + 0.05) * e

    # Mapping function (simple)
    mapf = 1.0 / np.sin(el)

    return (zhd + zwd) * mapf


def sagnac_correction(sat_pos, rec_pos):
    """Sagnac effect correction"""
    return (OMGE / CLIGHT) * (sat_pos[0] * rec_pos[1] - sat_pos[1] * rec_pos[0])


def geodist(sat_pos, rec_pos):
    """Geometric distance and unit vector"""
    diff = sat_pos - rec_pos
    r = norm(diff)
    if r > 0:
        e = diff / r
    else:
        e = np.zeros(3)
    return r, e


def satazel(pos, e):
    """Satellite azimuth/elevation from receiver position and line-of-sight vector"""
    lat, lon, _h = pos[0], pos[1], pos[2]

    # ENU transformation matrix
    R = np.array([
        [-np.sin(lon), np.cos(lon), 0],
        [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)],
        [np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)]
    ])

    # Transform to ENU
    enu = R @ e

    # Azimuth and elevation
    az = np.arctan2(enu[0], enu[1])
    if az < 0:
        az += 2 * np.pi
    el = np.arcsin(enu[2])

    return az, el


def varerr(sys, el):
    """Variance of pseudorange error"""
    a = 0.003  # Base error (m)
    b = 0.003  # Elevation-dependent error (m)

    s_el = np.sin(el)
    if s_el <= 0:
        return 100.0  # Large error for negative elevation

    # Basic elevation-dependent model
    var = (a ** 2) + (b / s_el) ** 2

    # Add system-specific errors
    if sys == SYS_GLO:
        var *= 1.5  # GLONASS typically has larger errors
    elif sys == SYS_BDS:
        var *= 1.2  # BeiDou slightly larger errors

    return var


def single_point_positioning(observations, nav_data, initial_pos=None,
                           systems_to_use=None, use_raim=True, raim_threshold=30.0):
    """
    Perform single point positioning using iterative least squares with RAIM

    Parameters:
    -----------
    observations : list
        List of GNSS observations
    nav_data : NavigationData
        Navigation data with ephemerides
    initial_pos : np.ndarray, optional
        Initial position estimate (ECEF)
    systems_to_use : list, optional
        List of satellite system characters to use (e.g., ['G', 'R'])
    use_raim : bool, optional
        Enable RAIM for fault detection (default: True)
    raim_threshold : float, optional
        RAIM residual threshold in meters (default: 30.0)

    Returns:
    --------
    solution : Solution
        Position solution
    used_sats : list
        List of satellites used in the solution
    """
    if not observations:
        return None, []

    # Default systems to use
    if systems_to_use is None:
        systems_to_use = ['G', 'E', 'C', 'J', 'R']

    # Filter observations by system
    filtered_obs = []
    for obs in observations:
        sys = sat2sys(obs.sat)
        if sys == 0:
            continue
        sys_char = sys2char(sys)
        if sys_char in systems_to_use:
            filtered_obs.append(obs)

    if len(filtered_obs) < 4:
        return None, []

    # Initial state [x, y, z, dtr_gps, dtr_glo, dtr_gal, dtr_bds]
    if initial_pos is None:
        x = np.zeros(7)
    else:
        x = np.zeros(7)
        x[:3] = initial_pos.copy()

    # Iterative least squares
    for iteration in range(MAXITR):
        H = []
        v = []
        var = []
        used_sats = []

        pos = x[:3]
        dtr = x[3:]  # Clock biases in seconds

        # Convert to LLH for elevation calculation
        if norm(pos) > RE_WGS84:
            llh = ecef2llh(pos)
        else:
            llh = np.array([0, 0, 0])

        for obs in filtered_obs:
            # Get pseudorange
            pr = obs.P[0] if obs.P[0] > 0 else obs.P[1]
            if pr <= 0:
                continue

            # Calculate transmission time
            if isinstance(obs.time, (int, float)):
                gps_week = int(obs.time // 604800)
                gps_tow = obs.time % 604800
                tc_rx = TimeCore.from_gps(gps_week, gps_tow)
            else:
                tc_rx = obs.time

            tc_tx = tc_rx - (pr / CLIGHT)

            # Get ephemeris
            eph = seleph(nav_data, tc_tx, obs.sat)
            if eph is None:
                continue

            # Get satellite position and clock
            sat_pos, sat_var, dts = eph2pos(tc_tx, eph)
            if np.any(np.isnan(sat_pos)) or norm(sat_pos) < RE_WGS84:
                continue

            # Geometric range and unit vector
            r, e = geodist(sat_pos, pos)
            if r <= 0:
                continue

            # Elevation check
            if norm(pos) > RE_WGS84:
                az, el = satazel(llh, e)
                if el < np.deg2rad(MIN_EL):
                    continue
            else:
                el = np.pi/4

            # System-specific clock bias
            sys = sat2sys(obs.sat)
            if sys == SYS_GPS or sys == SYS_QZS:
                clk_bias = dtr[0]
                clk_idx = 0
            elif sys == SYS_GLO:
                clk_bias = dtr[0] + dtr[1]
                clk_idx = 1
            elif sys == SYS_GAL:
                clk_bias = dtr[0] + dtr[2]
                clk_idx = 2
            elif sys == SYS_BDS:
                clk_bias = dtr[0] + dtr[3]
                clk_idx = 3
            else:
                continue

            # Corrections
            dtrp = tropmodel_simple(llh, el) if iteration > 0 else 0.0
            dion = 0.0  # No ionosphere correction for single frequency
            sagnac = sagnac_correction(sat_pos, pos)

            # Pseudorange residual
            res = pr - (r + sagnac + clk_bias * CLIGHT - dts * CLIGHT + dion + dtrp)

            # Design matrix row
            H_row = np.zeros(7)
            H_row[:3] = -e
            H_row[3] = CLIGHT  # GPS clock
            if clk_idx > 0 and clk_idx < 4:
                H_row[3 + clk_idx] = CLIGHT  # ISB term

            H.append(H_row)
            v.append(res)
            var.append(varerr(sys, el))
            used_sats.append(obs.sat)

        if len(v) < 4:
            return None, []

        # Convert to arrays
        H = np.array(H)
        v = np.array(v)
        var = np.array(var)

        # Apply RAIM if enabled
        if use_raim and len(v) > 5:
            rms_residual = np.sqrt(np.mean(v**2))
            if rms_residual > raim_threshold * 1000 or iteration > 0:
                H, v, var, used_sats, excluded = raim_fde(H, v, var, used_sats, raim_threshold)
                if len(v) < 4:
                    return None, []

        # Remove unused clock parameters
        active_params = [True, True, True, True]  # x, y, z, dtr_gps always active
        for i in range(1, 4):
            if 3+i < H.shape[1]:
                col_sum = np.sum(np.abs(H[:, 3+i]))
                active_params.append(col_sum > 0)

        active_idx = np.where(active_params[:H.shape[1]])[0]
        H_reduced = H[:, active_idx]

        # Weighted least squares
        W = np.diag(1.0 / var)

        try:
            N = H_reduced.T @ W @ H_reduced
            b = H_reduced.T @ W @ v
            dx_reduced = np.linalg.solve(N, b)
            dx = np.zeros(7)
            dx[active_idx] = dx_reduced
        except np.linalg.LinAlgError:
            return None, []

        # Update state
        x += dx

        # Check convergence
        if norm(dx[:3]) < 1e-4:
            break

    # Create solution
    pos = x[:3]
    dtr = x[3:]

    try:
        Q = np.linalg.inv(H.T @ W @ H)
    except:
        Q = np.eye(7) * 100.0

    solution = Solution(
        time=observations[0].time,
        type=5,  # Single point
        rr=pos,
        dtr=dtr,
        qr=Q[:3, :3],
        ns=len(used_sats)
    )

    return solution, used_sats