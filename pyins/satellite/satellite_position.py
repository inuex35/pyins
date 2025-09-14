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
    Compute satellite position and clock bias from broadcast ephemeris.

    This is the primary function for computing satellite positions from
    broadcast ephemeris data. It implements the standard Keplerian orbital
    model with perturbation corrections as specified in satellite system
    Interface Control Documents (GPS, Galileo, BeiDou, QZSS).

    Parameters
    ----------
    eph : Ephemeris
        Broadcast ephemeris containing all necessary orbital parameters:
        - Keplerian elements (A, e, i0, OMG0, omg, M0)
        - Perturbation corrections (cuc, cus, crc, crs, cic, cis)
        - Time parameters (toe, toc)
        - Clock coefficients (f0, f1, f2)
        - Accuracy indicators (sva)
    time : float
        Signal transmission time in GPS time (seconds)

    Returns
    -------
    tuple[np.ndarray, float, np.ndarray]
        - rs : np.ndarray, shape (3,)
            Satellite position in ECEF coordinates (meters)
        - dts : float
            Satellite clock bias (seconds, positive = fast clock)
        - var : np.ndarray, shape (3,)
            Position variance based on URA (m²)

    Raises
    ------
    ValueError
        If ephemeris is for GLONASS satellite (use compute_glonass_position)

    Notes
    -----
    The computation follows these steps:
    1. Validate ephemeris parameters (A > 0)
    2. Compute mean motion and mean anomaly
    3. Solve Kepler's equation for eccentric anomaly
    4. Calculate true anomaly and argument of latitude
    5. Apply harmonic corrections (cuc, cus, crc, crs, cic, cis)
    6. Transform from orbital plane to ECEF coordinates
    7. Apply system-specific transformations (BeiDou GEO/IGSO)
    8. Compute clock bias with relativistic correction

    Special handling for BeiDou GEO/IGSO satellites (PRN 1-5, 59-63):
    - Additional coordinate transformation
    - 5-degree rotation about Z-axis
    - Earth rotation correction

    Examples
    --------
    >>> pos, clk_bias, variance = compute_satellite_position(gps_eph, gps_time)
    >>> print(f"Position: {pos/1000:.1f} km")
    >>> print(f"Clock bias: {clk_bias*1e9:.1f} ns")
    >>> print(f"Position accuracy: {np.sqrt(variance[0]):.1f} m")
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
    Compute satellite velocity and clock drift from broadcast ephemeris.

    This function calculates the satellite velocity in Earth-Centered Earth-Fixed
    (ECEF) coordinates and clock drift rate using Keplerian orbital mechanics
    with perturbation corrections. The computation includes proper derivatives
    of all orbital elements and corrections.

    Parameters
    ----------
    eph : Ephemeris
        Broadcast ephemeris containing orbital parameters and clock coefficients
    time : float
        Signal transmission time in GPS time (seconds)

    Returns
    -------
    tuple[np.ndarray, float]
        - vs : np.ndarray, shape (3,)
            Satellite velocity in ECEF coordinates (m/s)
        - ddts : float
            Satellite clock drift rate (s/s)

    Notes
    -----
    The velocity computation includes:
    - Keplerian motion derivatives
    - Harmonic correction derivatives (cuc, cus, crc, crs, cic, cis)
    - Orbital plane to ECEF coordinate transformation
    - Earth rotation effects
    - Relativistic clock drift correction

    The algorithm follows standard GPS/GNSS Interface Control Documents
    and is compatible with all supported satellite systems (GPS, Galileo,
    BeiDou, QZSS).

    Examples
    --------
    >>> vel, drift = compute_satellite_velocity(eph, gps_time)
    >>> speed = np.linalg.norm(vel)
    >>> print(f"Satellite speed: {speed/1000:.1f} km/s")
    >>> print(f"Clock drift: {drift*1e12:.1f} ps/s")
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
    """
    Compute GLONASS satellite clock bias from ephemeris parameters.

    This function calculates the GLONASS satellite clock correction using
    the linear clock model with iterative solution for improved accuracy.
    The implementation follows the RTKLIB approach with sign convention
    adjusted to match GPS standards.

    Parameters
    ----------
    geph : GloEphemeris
        GLONASS ephemeris containing clock parameters:
        - taun: Clock bias at reference time (s)
        - gamn: Relative frequency bias (dimensionless)
        - toe: Reference time for ephemeris (GPST)

    time : float
        Time of interest in GPS time (seconds)

    Returns
    -------
    float
        Satellite clock bias in seconds (positive = satellite clock fast)

    Notes
    -----
    The GLONASS clock model differs from GPS:
    - Uses linear correction: -τn + γn * t
    - Includes iterative solution for improved accuracy
    - Sign convention adjusted to match GPS (positive = fast clock)

    The iterative solution accounts for the coupling between time
    correction and clock bias computation.

    References
    ----------
    - GLONASS Interface Control Document
    - RTKLIB source code (geph2clk function)

    Examples
    --------
    >>> clock_bias = geph2clk(glonass_eph, gps_time)
    >>> correction_ns = clock_bias * 1e9
    >>> print(f"GLONASS clock bias: {correction_ns:.1f} ns")
    """
    t = ts = timediff(time, geph.toe)

    # Iterative solution for clock bias (same as rtklib-py)
    for _ in range(2):
        t = ts - (-geph.taun + geph.gamn * t)

    # GLONASS clock correction with sign convention matching GPS
    # Original: dts = -geph.taun + geph.gamn * t
    # Negated to match GPS convention (positive means satellite clock is fast)
    dts = -(-geph.taun + geph.gamn * t)

    return dts


def compute_glonass_position(geph: GloEphemeris, time: float) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Compute GLONASS satellite position and clock bias from broadcast ephemeris.

    GLONASS uses a different orbital model compared to GPS-type satellites.
    Instead of Keplerian elements, GLONASS broadcasts position, velocity,
    and acceleration in Earth-fixed coordinates, requiring numerical
    integration to propagate the orbit.

    Parameters
    ----------
    geph : GloEphemeris
        GLONASS broadcast ephemeris containing:
        - pos: Position at reference time toe (m, ECEF)
        - vel: Velocity at reference time toe (m/s, ECEF)
        - acc: Acceleration terms (m/s², ECEF)
        - toe: Reference time (GPST)
        - taun: Clock bias at toe (s)
        - gamn: Relative frequency bias (dimensionless)
    time : float
        Signal transmission time in GPS time (seconds)

    Returns
    -------
    tuple[np.ndarray, float, np.ndarray]
        - rs : np.ndarray, shape (3,)
            Satellite position in ECEF coordinates (meters)
        - dts : float
            Satellite clock bias (seconds)
        - var : np.ndarray, shape (3,)
            Position variance (25 m² for each component)

    Notes
    -----
    The GLONASS position computation:
    1. Calculates time difference from ephemeris reference
    2. Applies GLONASS clock correction model
    3. Sets up initial state vector [position, velocity]
    4. Numerically integrates differential equations using Runge-Kutta
    5. Uses 120-second integration steps (TSTEP)
    6. Stops when |time_remaining| < 1×10⁻⁵ seconds

    The differential equations include:
    - Central gravitational acceleration
    - J2 zonal harmonic perturbation
    - Earth rotation effects (Coriolis, centrifugal)
    - Solar radiation pressure and other accelerations

    GLONASS uses the PZ-90.02 coordinate system, which is very close
    to WGS-84 but has slight parameter differences.

    Accuracy: GLONASS ephemeris typically provides ~5m positioning
    accuracy (ERREPH_GLO = 5.0m), hence variance = 25 m².

    References
    ----------
    - GLONASS Interface Control Document
    - RTKLIB source code implementation
    - "GLONASS Principles, Methods and Applications" by various authors

    Examples
    --------
    >>> pos, clk, var = compute_glonass_position(glo_eph, gps_time)
    >>> print(f"GLONASS position: {pos/1000:.2f} km")
    >>> print(f"Clock bias: {clk*1e9:.1f} ns")
    >>> print(f"Expected accuracy: {np.sqrt(var[0]):.1f} m")
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
    Integrate GLONASS satellite orbit using 4th-order Runge-Kutta method.

    This function performs one step of numerical integration for GLONASS
    satellite motion using the classical 4th-order Runge-Kutta algorithm.
    It integrates the coupled differential equations of motion including
    gravitational and non-gravitational accelerations.

    Parameters
    ----------
    t : float
        Integration time step in seconds (typically 120 seconds)
    x : np.ndarray, shape (6,)
        Current state vector [px, py, pz, vx, vy, vz] where:
        - px, py, pz: position components in PZ-90 coordinates (m)
        - vx, vy, vz: velocity components in PZ-90 coordinates (m/s)
    acc : np.ndarray, shape (3,)
        Acceleration vector [ax, ay, az] in PZ-90 coordinates (m/s²)
        Includes solar radiation pressure and other perturbations

    Returns
    -------
    np.ndarray, shape (6,)
        Updated state vector after one integration step

    Notes
    -----
    The Runge-Kutta 4th-order method provides high accuracy for orbit
    propagation while maintaining stability. The algorithm evaluates
    the derivative function (deq) at four points within the step interval:

    1. k1 = f(x, t)
    2. k2 = f(x + k1*t/2, t + t/2)
    3. k3 = f(x + k2*t/2, t + t/2)
    4. k4 = f(x + k3*t, t + t)

    Final update: x_new = x + (k1 + 2*k2 + 2*k3 + k4) * t/6

    Examples
    --------
    >>> # Propagate GLONASS satellite for 120 seconds
    >>> state_new = glorbit(120.0, initial_state, accelerations)
    >>> pos_new = state_new[:3]  # Extract position
    >>> vel_new = state_new[3:]  # Extract velocity
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
    Compute derivatives for GLONASS orbital differential equations.

    This function calculates the time derivatives of position and velocity
    for GLONASS satellite motion in the PZ-90.02 coordinate system.
    It includes Earth's gravitational field effects, J2 perturbations,
    and Earth rotation.

    Parameters
    ----------
    x : np.ndarray, shape (6,)
        State vector [px, py, pz, vx, vy, vz] where:
        - px, py, pz: position components (m)
        - vx, vy, vz: velocity components (m/s)
    acc : np.ndarray, shape (3,)
        External acceleration vector [ax, ay, az] (m/s²)
        Includes solar radiation pressure, atmospheric drag, etc.

    Returns
    -------
    np.ndarray, shape (6,)
        State derivatives [dpx/dt, dpy/dt, dpz/dt, dvx/dt, dvy/dt, dvz/dt]

    Notes
    -----
    The differential equations include:

    1. Position derivatives: dx/dt = v
    2. Velocity derivatives: dv/dt = F_total

    Where F_total includes:
    - Central gravitational acceleration: -μ/r³ * r
    - J2 zonal harmonic perturbation (Earth's oblateness)
    - Coriolis acceleration: 2 * Ω × v
    - Centrifugal acceleration: Ω × (Ω × r)
    - External accelerations (solar pressure, drag, etc.)

    GLONASS-specific constants:
    - μ = 3.9860044×14 m³/s² (PZ-90 gravitational parameter)
    - J2 = 1.0826257×10⁻³ (2nd zonal harmonic)
    - Re = 6378136.0 m (PZ-90 Earth radius)
    - Ωe = 7.292115×10⁻⁵ rad/s (Earth rotation rate)

    Examples
    --------
    >>> derivatives = deq(state_vector, external_acc)
    >>> pos_derivatives = derivatives[:3]  # Velocity
    >>> vel_derivatives = derivatives[3:]  # Acceleration
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
    Compute GLONASS satellite velocity and clock drift from ephemeris.

    This function calculates GLONASS satellite velocity by numerically
    integrating the orbital equations from the ephemeris reference time
    to the requested time. The integration uses the same Runge-Kutta
    method as position computation for consistency.

    Parameters
    ----------
    geph : GloEphemeris
        GLONASS ephemeris containing:
        - pos: Initial position at toe (m)
        - vel: Initial velocity at toe (m/s)
        - acc: Acceleration terms (m/s²)
        - toe: Reference time (GPST)
        - gamn: Clock frequency bias

    time : float
        Signal transmission time in GPS time (seconds)

    Returns
    -------
    tuple[np.ndarray, float]
        - vs : np.ndarray, shape (3,)
            Satellite velocity in ECEF coordinates (m/s)
            Returns NaN if time difference exceeds 2 hours
        - ddts : float
            Satellite clock drift rate (s/s)

    Notes
    -----
    The velocity computation:
    1. Checks time validity (must be within 2 hours of toe)
    2. Initializes state vector with ephemeris pos/vel
    3. Integrates using Runge-Kutta with 120-second steps
    4. Handles fractional time steps at the end
    5. Extracts final velocity from integrated state

    Clock drift for GLONASS is simply the frequency bias parameter
    gamn, which represents the relative frequency offset.

    Time validation prevents unreliable extrapolation beyond the
    validity period of GLONASS ephemerides.

    Examples
    --------
    >>> vel, drift = compute_glonass_velocity(glo_eph, gps_time)
    >>> if not np.isnan(vel).any():
    ...     speed = np.linalg.norm(vel)
    ...     print(f"GLONASS velocity: {speed/1000:.2f} km/s")
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
