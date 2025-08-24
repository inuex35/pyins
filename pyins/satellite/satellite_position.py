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


import numpy as np

from ..core.constants import *
from ..core.data_structures import Ephemeris, GloEphemeris
from ..core.time import timediff


def compute_satellite_position(eph: Ephemeris, time: float) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Compute satellite position and clock from broadcast ephemeris

    Parameters:
    -----------
    eph : Ephemeris
        Broadcast ephemeris
    time : float
        Signal transmission time (GPST)

    Returns:
    --------
    rs : np.ndarray
        Satellite position in ECEF (m)
    dts : float
        Satellite clock bias (s)
    var : np.ndarray
        Satellite position and clock variance
    """

    # Check if this is GLONASS (different ephemeris format)
    if 33 <= eph.sat <= 59:  # GLONASS satellite range
        # GLONASS uses different ephemeris format, should use compute_glonass_position
        raise ValueError("Use compute_glonass_position() for GLONASS satellites")

    # Check for invalid parameters
    if eph.A == 0.0:
        return np.array([np.nan, np.nan, np.nan]), np.nan, np.array([np.nan, np.nan, np.nan])

    # Time from ephemeris reference epoch
    tk = timediff(time, eph.toe)

    # Mean anomaly
    n0 = np.sqrt(GME / eph.A**3)  # Computed mean motion
    n = n0 + eph.deln  # Corrected mean motion
    M = eph.M0 + n * tk  # Mean anomaly

    # Eccentric anomaly (Kepler's equation by Newton-Raphson)
    E = M
    for _ in range(10):
        E_old = E
        E = M + eph.e * np.sin(E)
        if abs(E - E_old) < 1e-12:
            break

    # True anomaly
    sin_E = np.sin(E)
    cos_E = np.cos(E)
    v = np.arctan2(np.sqrt(1.0 - eph.e**2) * sin_E, cos_E - eph.e)

    # Argument of latitude
    u = v + eph.omg

    # Corrections
    sin_2u = np.sin(2.0 * u)
    cos_2u = np.cos(2.0 * u)
    du = eph.cuc * cos_2u + eph.cus * sin_2u  # Argument of latitude correction
    dr = eph.crc * cos_2u + eph.crs * sin_2u  # Radius correction
    di = eph.cic * cos_2u + eph.cis * sin_2u  # Inclination correction

    # Corrected values
    u = u + du
    r = eph.A * (1.0 - eph.e * cos_E) + dr
    i = eph.i0 + eph.idot * tk + di

    # Positions in orbital plane
    x = r * np.cos(u)
    y = r * np.sin(u)

    # Corrected longitude of ascending node
    OMG = eph.OMG0 + (eph.OMGd - OMGE) * tk - OMGE * eph.toes

    # Earth-fixed coordinates
    sin_OMG = np.sin(OMG)
    cos_OMG = np.cos(OMG)
    sin_i = np.sin(i)
    cos_i = np.cos(i)

    # Check if BeiDou GEO/IGSO satellite (PRN 1-5, 59-63)
    sys = sat2sys(eph.sat)
    prn = sat2prn(eph.sat)

    if sys == SYS_BDS and (prn <= 5 or prn >= 59):
        # BeiDou GEO/IGSO satellite coordinate transformation
        # Reference: BeiDou ICD
        xg = x * cos_OMG - y * cos_i * sin_OMG
        yg = x * sin_OMG + y * cos_i * cos_OMG
        zg = y * sin_i

        # Additional rotation for BeiDou GEO satellites
        sin_5 = np.sin(np.deg2rad(-5))
        cos_5 = np.cos(np.deg2rad(-5))
        sin_omge_tk = np.sin(OMGE * tk)
        cos_omge_tk = np.cos(OMGE * tk)

        rs = np.array([
            xg * cos_omge_tk + yg * sin_omge_tk * cos_5 + zg * sin_omge_tk * sin_5,
            -xg * sin_omge_tk + yg * cos_omge_tk * cos_5 + zg * cos_omge_tk * sin_5,
            -yg * sin_5 + zg * cos_5
        ])
    else:
        # Standard transformation for other satellites
        rs = np.array([
            x * cos_OMG - y * cos_i * sin_OMG,
            x * sin_OMG + y * cos_i * cos_OMG,
            y * sin_i
        ])

    # Satellite clock correction (including relativistic correction)
    F = -2.0 * np.sqrt(GME) / CLIGHT**2
    dts = eph.f0 + eph.f1 * tk + eph.f2 * tk**2 + F * eph.e * np.sqrt(eph.A) * sin_E

    # Variance (simplified - based on URA)
    var = np.array([ura_value(eph.sva)**2, ura_value(eph.sva)**2, ura_value(eph.sva)**2])

    return rs, dts, var


def compute_satellite_velocity(eph: Ephemeris, time: float) -> tuple[np.ndarray, float]:
    """
    Compute satellite velocity and clock drift

    Parameters:
    -----------
    eph : Ephemeris
        Broadcast ephemeris
    time : float
        Signal transmission time (GPST)

    Returns:
    --------
    vs : np.ndarray
        Satellite velocity in ECEF (m/s)
    ddts : float
        Satellite clock drift (s/s)
    """

    # Time from ephemeris reference epoch
    tk = timediff(time, eph.toe)

    # Mean motion
    n0 = np.sqrt(GME / eph.A**3)
    n = n0 + eph.deln
    M = eph.M0 + n * tk

    # Eccentric anomaly
    E = M
    for _ in range(10):
        E_old = E
        E = M + eph.e * np.sin(E)
        if abs(E - E_old) < 1e-12:
            break

    # Derivatives
    sin_E = np.sin(E)
    cos_E = np.cos(E)
    dE_dt = n / (1.0 - eph.e * cos_E)

    # True anomaly rate
    sqrt_1_e2 = np.sqrt(1.0 - eph.e**2)
    dv_dt = sqrt_1_e2 * dE_dt / (1.0 - eph.e * cos_E)

    # Argument of latitude
    v = np.arctan2(sqrt_1_e2 * sin_E, cos_E - eph.e)
    u = v + eph.omg

    # Corrections
    sin_2u = np.sin(2.0 * u)
    cos_2u = np.cos(2.0 * u)
    du = eph.cuc * cos_2u + eph.cus * sin_2u
    dr = eph.crc * cos_2u + eph.crs * sin_2u
    di = eph.cic * cos_2u + eph.cis * sin_2u

    # Corrected values and rates
    u = u + du
    r = eph.A * (1.0 - eph.e * cos_E) + dr
    i = eph.i0 + eph.idot * tk + di

    du_dt = dv_dt + 2.0 * dv_dt * (eph.cuc * (-sin_2u) + eph.cus * cos_2u)
    dr_dt = eph.A * eph.e * sin_E * dE_dt + 2.0 * dv_dt * (eph.crc * (-sin_2u) + eph.crs * cos_2u)
    di_dt = eph.idot + 2.0 * dv_dt * (eph.cic * (-sin_2u) + eph.cis * cos_2u)

    # Velocity in orbital plane
    dx_dt = dr_dt * np.cos(u) - r * np.sin(u) * du_dt
    dy_dt = dr_dt * np.sin(u) + r * np.cos(u) * du_dt

    # ECEF velocity
    OMG = eph.OMG0 + (eph.OMGd - OMGE) * tk - OMGE * eph.toes
    dOMG_dt = eph.OMGd - OMGE

    sin_OMG = np.sin(OMG)
    cos_OMG = np.cos(OMG)
    sin_i = np.sin(i)
    cos_i = np.cos(i)

    x = r * np.cos(u)
    y = r * np.sin(u)

    vs = np.array([
        dx_dt * cos_OMG - dy_dt * cos_i * sin_OMG + y * sin_i * sin_OMG * di_dt - (x * sin_OMG + y * cos_i * cos_OMG) * dOMG_dt,
        dx_dt * sin_OMG + dy_dt * cos_i * cos_OMG - y * sin_i * cos_OMG * di_dt + (x * cos_OMG - y * cos_i * sin_OMG) * dOMG_dt,
        dy_dt * sin_i + y * cos_i * di_dt
    ])

    # Clock drift
    F = -2.0 * np.sqrt(GME) / CLIGHT**2
    ddts = eph.f1 + 2.0 * eph.f2 * tk + F * eph.e * np.sqrt(eph.A) * cos_E * dE_dt

    return vs, ddts


def ura_value(sva: int) -> float:
    """Convert URA index to value in meters"""
    ura_eph = [
        2.4, 3.4, 4.85, 6.85, 9.65, 13.65, 24.0, 48.0,
        96.0, 192.0, 384.0, 768.0, 1536.0, 3072.0, 6144.0, 0.0
    ]
    return ura_eph[sva] if 0 <= sva <= 15 else 0.0


def geph2clk(geph: GloEphemeris, time: float) -> float:
    """
    GLONASS ephemeris to satellite clock bias

    Based on rtklib-py's geph2clk() function.

    Parameters:
    -----------
    geph : GloEphemeris
        GLONASS ephemeris
    time : float
        Time of interest (GPST)

    Returns:
    --------
    dts : float
        Satellite clock bias (s)
    """
    t = ts = timediff(time, geph.toe)

    # Iterative solution for clock bias (same as rtklib-py)
    for _ in range(2):
        t = ts - (-geph.taun + geph.gamn * t)

    dts = -geph.taun + geph.gamn * t

    return dts


def compute_glonass_position(geph: GloEphemeris, time: float) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Compute GLONASS satellite position and clock from broadcast ephemeris

    Based on rtklib-py's geph2pos() function.

    Parameters:
    -----------
    geph : GloEphemeris
        GLONASS broadcast ephemeris
    time : float
        Signal transmission time (GPST)

    Returns:
    --------
    rs : np.ndarray
        Satellite position in ECEF (m)
    dts : float
        Satellite clock bias (s)
    var : np.ndarray
        Satellite position and clock variance
    """

    # Time difference from ephemeris reference time
    t = timediff(time, geph.toe)

    # Clock bias (same as rtklib-py geph2pos)
    dts = -geph.taun + geph.gamn * t

    # Initial state vector [pos, vel]
    x = np.array([*geph.pos, *geph.vel])

    # Integration step (same as rtklib-py)
    TSTEP = 120.0  # 120 second steps

    # Runge-Kutta integration (same as rtklib-py geph2pos)
    tt = -TSTEP if t < 0 else TSTEP
    while abs(t) > 1E-5:  # rtklib-py uses 1E-5
        if abs(t) < TSTEP:
            tt = t
        x = glorbit(tt, x, geph.acc)
        t -= tt

    # Extract position
    rs = x[0:3].copy()

    # Variance (ERREPH_GLO^2 = 5.0^2)
    var = np.array([25.0, 25.0, 25.0])

    return rs, dts, var


def glorbit(t: float, x: np.ndarray, acc: np.ndarray) -> np.ndarray:
    """
    GLONASS orbit integration using Runge-Kutta 4th order

    Based on rtklib-py's glorbit() function.

    Parameters:
    -----------
    t : float
        Time step
    x : np.ndarray
        State vector [pos, vel]
    acc : np.ndarray
        Acceleration

    Returns:
    --------
    x_new : np.ndarray
        Updated state vector
    """
    k1 = deq(x, acc)
    w = x + k1 * t / 2
    k2 = deq(w, acc)
    w = x + k2 * t / 2
    k3 = deq(w, acc)
    w = x + k3 * t
    k4 = deq(w, acc)
    x += (k1 + 2 * k2 + 2 * k3 + k4) * t / 6
    return x


def deq(x: np.ndarray, acc: np.ndarray) -> np.ndarray:
    """
    GLONASS orbit differential equations

    Based on rtklib-py's deq() function.

    Parameters:
    -----------
    x : np.ndarray
        State vector [pos, vel]
    acc : np.ndarray
        Acceleration

    Returns:
    --------
    xdot : np.ndarray
        State derivatives
    """
    xdot = np.zeros(6)
    r2 = np.dot(x[0:3], x[0:3])
    if r2 <= 0.0:
        return xdot

    r3 = r2 * np.sqrt(r2)
    omg2 = OMGE_GLO**2

    # Constants from rtklib-py
    J2_GLO = 1.0826257E-3     # 2nd zonal harmonic
    MU_GLO = 3.9860044E14     # gravitational constant
    RE_GLO = 6378136.0        # radius of earth (m)

    a = 1.5 * J2_GLO * MU_GLO * RE_GLO**2 / r2 / r3
    b = 5.0 * x[2]**2 / r2
    c = -MU_GLO / r3 - a * (1.0 - b)

    xdot[0:3] = x[3:6]
    xdot[3] = (c + omg2) * x[0] + 2.0 * OMGE_GLO * x[4] + acc[0]
    xdot[4] = (c + omg2) * x[1] - 2.0 * OMGE_GLO * x[3] + acc[1]
    xdot[5] = (c - 2.0 * a) * x[2] + acc[2]

    return xdot


def compute_glonass_velocity(geph: GloEphemeris, time: float) -> tuple[np.ndarray, float]:
    """
    Compute GLONASS satellite velocity and clock drift

    Parameters:
    -----------
    geph : GloEphemeris
        GLONASS broadcast ephemeris
    time : float
        Signal transmission time (GPST)

    Returns:
    --------
    vs : np.ndarray
        Satellite velocity in ECEF (m/s)
    ddts : float
        Satellite clock drift (s/s)
    """

    # Time difference from ephemeris time
    t = timediff(time, geph.toe)

    # Check time validity
    if abs(t) > 7200.0:
        return np.array([np.nan, np.nan, np.nan]), np.nan

    # Initial state vector
    x = np.zeros(6)
    x[0:3] = geph.pos
    x[3:6] = geph.vel

    # Integration
    TSTEP = 120.0  # Same as rtklib-py
    n = int(abs(t) / TSTEP)
    h = TSTEP if t >= 0 else -TSTEP

    for _ in range(n):
        x = glorbit(h, x, geph.acc)

    h = t - n * h
    if h != 0:
        x = glorbit(h, x, geph.acc)

    # Extract velocity
    vs = x[3:6].copy()

    # Clock drift
    ddts = geph.gamn

    return vs, ddts


def satpos_sbas(eph: Ephemeris, time: float) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Compute SBAS satellite position

    Parameters:
    -----------
    eph : Ephemeris
        SBAS ephemeris
    time : float
        Time (GPST)

    Returns:
    --------
    rs : np.ndarray
        Satellite position (m)
    dts : float
        Satellite clock bias (s)
    var : np.ndarray
        Variance
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
