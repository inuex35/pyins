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

"""Azimuth-Elevation-Range (AER) coordinate transformations"""


import numpy as np

from .eci_transforms import eci2enu, eci2lla
from .transforms import ecef2enu, ecef2llh, lla2enu


def enu2aer(enu_t: np.ndarray, enu_r: np.ndarray) -> np.ndarray:
    """Convert ENU to Azimuth-Elevation-Range coordinates

    Computes the relative azimuth, elevation, and range from a reference
    point to a target point in the local ENU coordinate system.

    Parameters
    ----------
    enu_t : np.ndarray
        Target ENU coordinates [e, n, u] in meters
    enu_r : np.ndarray
        Reference ENU coordinates [e, n, u] in meters

    Returns
    -------
    np.ndarray
        Relative AER coordinates [azimuth, elevation, range] where:
        - azimuth: angle from north towards east (0-2π rad)
        - elevation: angle above horizontal plane (-π/2 to π/2 rad)
        - range: distance between points (m)

    Notes
    -----
    Azimuth is measured clockwise from north (0°) towards east (90°).
    Elevation is positive above the horizontal plane.

    Examples
    --------
    >>> import numpy as np
    >>> enu_ref = np.array([0, 0, 0])
    >>> enu_target = np.array([100, 100, 50])
    >>> aer = enu2aer(enu_target, enu_ref)
    >>> print(f"Azimuth: {np.degrees(aer[0]):.1f}°")
    >>> print(f"Elevation: {np.degrees(aer[1]):.1f}°")
    """
    de, dn, du = enu_t - enu_r

    r = np.hypot(de, dn)
    az = np.mod(np.arctan2(de, dn), 2 * np.pi)
    el = np.arctan2(du, r)
    rng = np.hypot(r, du)

    return np.array([az, el, rng])


def ned2aer(ned_t: np.ndarray, ned_r: np.ndarray) -> np.ndarray:
    """Convert NED to Azimuth-Elevation-Range coordinates

    Computes the relative azimuth, elevation, and range from a reference
    point to a target point in the local NED coordinate system.

    Parameters
    ----------
    ned_t : np.ndarray
        Target NED coordinates [n, e, d] in meters
    ned_r : np.ndarray
        Reference NED coordinates [n, e, d] in meters

    Returns
    -------
    np.ndarray
        Relative AER coordinates [azimuth, elevation, range] where:
        - azimuth: angle from north towards east (0-2π rad)
        - elevation: angle above horizontal plane (-π/2 to π/2 rad)
        - range: distance between points (m)

    Notes
    -----
    In NED coordinates, down is positive, so elevation is computed
    from the negative of the down component.

    Examples
    --------
    >>> import numpy as np
    >>> ned_ref = np.array([0, 0, 0])
    >>> ned_target = np.array([100, 100, -50])  # Note: negative for altitude
    >>> aer = ned2aer(ned_target, ned_ref)
    """
    dn, de, dd = ned_t - ned_r

    r = np.hypot(de, dn)
    az = np.mod(np.arctan2(de, dn), 2 * np.pi)
    el = np.arctan2(-dd, r)
    rng = np.hypot(r, -dd)

    return np.array([az, el, rng])


def aer2enu(aer: np.ndarray, enu_r: np.ndarray) -> np.ndarray:
    """Convert Azimuth-Elevation-Range to ENU coordinates

    Converts relative AER measurements from a reference point to
    absolute ENU coordinates of the target point.

    Parameters
    ----------
    aer : np.ndarray
        AER coordinates [azimuth, elevation, range] where:
        - azimuth: angle from north towards east (rad)
        - elevation: angle above horizontal plane (rad)
        - range: distance from reference point (m)
    enu_r : np.ndarray
        Reference ENU coordinates [e, n, u] in meters

    Returns
    -------
    np.ndarray
        Target ENU coordinates [e, n, u] in meters

    Notes
    -----
    This is the inverse operation of enu2aer().

    Examples
    --------
    >>> import numpy as np
    >>> aer = np.array([np.pi/4, np.pi/6, 200])  # 45° azimuth, 30° elevation, 200m range
    >>> enu_ref = np.array([1000, 2000, 100])
    >>> enu_target = aer2enu(aer, enu_ref)
    """
    az, el, rng = aer

    # Horizontal range
    r = rng * np.cos(el)

    # ENU offsets
    de = r * np.sin(az)
    dn = r * np.cos(az)
    du = rng * np.sin(el)

    return enu_r + np.array([de, dn, du])


def aer2ned(aer: np.ndarray, ned_r: np.ndarray) -> np.ndarray:
    """Convert Azimuth-Elevation-Range to NED coordinates

    Converts relative AER measurements from a reference point to
    absolute NED coordinates of the target point.

    Parameters
    ----------
    aer : np.ndarray
        AER coordinates [azimuth, elevation, range] where:
        - azimuth: angle from north towards east (rad)
        - elevation: angle above horizontal plane (rad)
        - range: distance from reference point (m)
    ned_r : np.ndarray
        Reference NED coordinates [n, e, d] in meters

    Returns
    -------
    np.ndarray
        Target NED coordinates [n, e, d] in meters

    Notes
    -----
    This is the inverse operation of ned2aer(). The down component
    is computed as negative of the vertical component.

    Examples
    --------
    >>> import numpy as np
    >>> aer = np.array([0, np.pi/4, 100])  # North direction, 45° up, 100m
    >>> ned_ref = np.array([0, 0, 0])
    >>> ned_target = aer2ned(aer, ned_ref)
    """
    az, el, rng = aer

    # Horizontal range
    r = rng * np.cos(el)

    # NED offsets
    dn = r * np.cos(az)
    de = r * np.sin(az)
    dd = -rng * np.sin(el)

    return ned_r + np.array([dn, de, dd])


def lla2aer(lla_t: np.ndarray, lla_r: np.ndarray) -> np.ndarray:
    """Convert geodetic coordinates to Azimuth-Elevation-Range

    Computes the relative azimuth, elevation, and range between two
    points specified in geodetic (latitude, longitude, altitude) coordinates.

    Parameters
    ----------
    lla_t : np.ndarray
        Target geodetic coordinates [lat, lon, height] where:
        - lat, lon: in radians
        - height: in meters above ellipsoid
    lla_r : np.ndarray
        Reference geodetic coordinates [lat, lon, height] where:
        - lat, lon: in radians
        - height: in meters above ellipsoid

    Returns
    -------
    np.ndarray
        Relative AER coordinates [azimuth, elevation, range] where:
        - azimuth: angle from north towards east (0-2π rad)
        - elevation: angle above horizontal plane (-π/2 to π/2 rad)
        - range: distance between points (m)

    Notes
    -----
    This function converts both points to the local ENU frame centered
    at the reference point, then computes AER coordinates.

    Examples
    --------
    >>> import numpy as np
    >>> lla_ref = np.array([np.radians(35.0), np.radians(139.0), 100])  # Tokyo
    >>> lla_target = np.array([np.radians(35.1), np.radians(139.1), 200])
    >>> aer = lla2aer(lla_target, lla_ref)
    """
    enu_t = lla2enu(lla_t, lla_r)
    enu_r = np.zeros(3)  # Reference is at origin in its own ENU frame
    return enu2aer(enu_t, enu_r)


def ecef2aer(ecef_t: np.ndarray, ecef_r: np.ndarray) -> np.ndarray:
    """Convert ECEF to Azimuth-Elevation-Range coordinates

    Computes the relative azimuth, elevation, and range between two
    points specified in Earth-Centered Earth-Fixed (ECEF) coordinates.

    Parameters
    ----------
    ecef_t : np.ndarray
        Target ECEF coordinates [x, y, z] in meters
    ecef_r : np.ndarray
        Reference ECEF coordinates [x, y, z] in meters

    Returns
    -------
    np.ndarray
        Relative AER coordinates [azimuth, elevation, range] where:
        - azimuth: angle from north towards east (0-2π rad)
        - elevation: angle above horizontal plane (-π/2 to π/2 rad)
        - range: distance between points (m)

    Notes
    -----
    This function first converts the reference point to geodetic coordinates,
    then transforms the target point to the local ENU frame centered at the
    reference point, and finally computes AER coordinates.

    Examples
    --------
    >>> import numpy as np
    >>> ecef_ref = np.array([-3962108.7, 3312993.9, 3565877.8])  # Example ECEF
    >>> ecef_target = np.array([-3961000.0, 3313000.0, 3566000.0])
    >>> aer = ecef2aer(ecef_target, ecef_ref)
    """
    lla_r = ecef2llh(ecef_r)
    enu_t = ecef2enu(ecef_t, lla_r)
    enu_r = np.zeros(3)  # Reference is at origin in its own ENU frame
    return enu2aer(enu_t, enu_r)


def eci2aer(eci_t: np.ndarray, eci_r: np.ndarray, t: float) -> np.ndarray:
    """Convert ECI to Azimuth-Elevation-Range coordinates

    Computes the relative azimuth, elevation, and range between two
    points specified in Earth-Centered Inertial (ECI) coordinates.

    Parameters
    ----------
    eci_t : np.ndarray
        Target ECI coordinates [x, y, z] in meters
    eci_r : np.ndarray
        Reference ECI coordinates [x, y, z] in meters
    t : float
        Time since reference epoch in seconds (for Earth rotation)

    Returns
    -------
    np.ndarray
        Relative AER coordinates [azimuth, elevation, range] where:
        - azimuth: angle from north towards east (0-2π rad)
        - elevation: angle above horizontal plane (-π/2 to π/2 rad)
        - range: distance between points (m)

    Notes
    -----
    This function accounts for Earth's rotation by using the time parameter
    to properly transform from the inertial frame to the Earth-fixed frame.
    Both points are converted to the local ENU frame at the reference
    location before computing AER coordinates.

    Examples
    --------
    >>> import numpy as np
    >>> eci_ref = np.array([6378137.0, 0, 0])  # Example ECI position
    >>> eci_target = np.array([6378137.0, 1000, 1000])
    >>> t = 3600.0  # 1 hour since epoch
    >>> aer = eci2aer(eci_target, eci_ref, t)
    """
    lla_r = eci2lla(eci_r, t)
    enu_t = eci2enu(eci_t, lla_r, t)
    enu_r = eci2enu(eci_r, lla_r, t)
    return enu2aer(enu_t, enu_r)
