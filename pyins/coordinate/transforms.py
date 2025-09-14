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

"""Coordinate transformation utilities"""


import numpy as np

from ..core.constants import FE_WGS84, RE_WGS84


def ecef2llh(xyz: np.ndarray) -> np.ndarray:
    """Convert ECEF coordinates to geodetic coordinates

    Converts Earth-Centered Earth-Fixed (ECEF) Cartesian coordinates
    to geodetic coordinates using an iterative algorithm.

    Parameters
    ----------
    xyz : np.ndarray
        ECEF coordinates [x, y, z] in meters

    Returns
    -------
    np.ndarray
        Geodetic coordinates [lat, lon, height] where:
        - lat: latitude in radians (-π/2 to π/2)
        - lon: longitude in radians (-π to π)
        - height: height above WGS84 ellipsoid in meters

    Notes
    -----
    Uses the WGS84 ellipsoid parameters. The algorithm typically
    converges in 3-4 iterations with high precision.

    Examples
    --------
    >>> import numpy as np
    >>> ecef = np.array([4193790.895, 454436.195, 4768166.813])  # Mount Fuji approx.
    >>> llh = ecef2llh(ecef)
    >>> lat_deg, lon_deg = np.degrees(llh[0]), np.degrees(llh[1])
    >>> print(f"Lat: {lat_deg:.6f}°, Lon: {lon_deg:.6f}°, Height: {llh[2]:.1f}m")
    """
    x, y, z = xyz[0], xyz[1], xyz[2]

    # Longitude
    lon = np.arctan2(y, x)

    # Iterative computation of latitude and height
    p = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, p * (1.0 - FE_WGS84))

    for _ in range(5):  # Usually converges in 3-4 iterations
        N = RE_WGS84 / np.sqrt(1.0 - FE_WGS84 * (2.0 - FE_WGS84) * np.sin(lat)**2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1.0 - FE_WGS84 * (2.0 - FE_WGS84) * N / (N + h)))

    return np.array([lat, lon, h])


def llh2ecef(llh: np.ndarray) -> np.ndarray:
    """Convert geodetic coordinates to ECEF coordinates

    Converts geodetic coordinates (latitude, longitude, height) to
    Earth-Centered Earth-Fixed (ECEF) Cartesian coordinates.

    Parameters
    ----------
    llh : np.ndarray
        Geodetic coordinates [lat, lon, height] where:
        - lat: latitude in radians
        - lon: longitude in radians
        - height: height above WGS84 ellipsoid in meters

    Returns
    -------
    np.ndarray
        ECEF coordinates [x, y, z] in meters

    Notes
    -----
    Uses the WGS84 ellipsoid parameters. This transformation is exact
    (no iterations required).

    Examples
    --------
    >>> import numpy as np
    >>> llh = np.array([np.radians(35.3606), np.radians(138.7274), 3776])  # Mount Fuji
    >>> ecef = llh2ecef(llh)
    >>> print(f"ECEF: [{ecef[0]:.1f}, {ecef[1]:.1f}, {ecef[2]:.1f}] m")
    """
    lat, lon, h = llh[0], llh[1], llh[2]

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    N = RE_WGS84 / np.sqrt(1.0 - FE_WGS84 * (2.0 - FE_WGS84) * sin_lat**2)

    x = (N + h) * cos_lat * cos_lon
    y = (N + h) * cos_lat * sin_lon
    z = (N * (1.0 - FE_WGS84 * (2.0 - FE_WGS84)) + h) * sin_lat

    return np.array([x, y, z])


def ecef2enu(xyz: np.ndarray, org_llh: np.ndarray) -> np.ndarray:
    """Convert ECEF to local ENU coordinates

    Transforms ECEF coordinates to local East-North-Up (ENU) coordinates
    relative to a specified origin point.

    Parameters
    ----------
    xyz : np.ndarray
        ECEF coordinates [x, y, z] in meters
    org_llh : np.ndarray
        Origin geodetic coordinates [lat, lon, height] where:
        - lat, lon: in radians
        - height: in meters above ellipsoid

    Returns
    -------
    np.ndarray
        Local ENU coordinates [e, n, u] in meters where:
        - e: eastward displacement from origin
        - n: northward displacement from origin
        - u: upward displacement from origin

    Notes
    -----
    The ENU coordinate system is a local Cartesian system with:
    - East axis pointing towards local east
    - North axis pointing towards local north
    - Up axis pointing away from Earth's center (local vertical)

    Examples
    --------
    >>> import numpy as np
    >>> origin_llh = np.array([np.radians(35.0), np.radians(139.0), 0])
    >>> point_ecef = np.array([4193790.0, 454436.0, 4768166.0])
    >>> enu = ecef2enu(point_ecef, origin_llh)
    >>> print(f"ENU: E={enu[0]:.1f}m, N={enu[1]:.1f}m, U={enu[2]:.1f}m")
    """
    # Origin in ECEF
    org_xyz = llh2ecef(org_llh)

    # Vector from origin
    dx = xyz - org_xyz

    # Rotation matrix
    lat, lon = org_llh[0], org_llh[1]
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    R = np.array([
        [-sin_lon, cos_lon, 0],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]
    ])

    return R @ dx


def enu2ecef(enu: np.ndarray, org_llh: np.ndarray) -> np.ndarray:
    """Convert local ENU to ECEF coordinates

    Transforms local East-North-Up (ENU) coordinates to ECEF coordinates
    using a specified origin point.

    Parameters
    ----------
    enu : np.ndarray
        Local ENU coordinates [e, n, u] in meters where:
        - e: eastward displacement from origin
        - n: northward displacement from origin
        - u: upward displacement from origin
    org_llh : np.ndarray
        Origin geodetic coordinates [lat, lon, height] where:
        - lat, lon: in radians
        - height: in meters above ellipsoid

    Returns
    -------
    np.ndarray
        ECEF coordinates [x, y, z] in meters

    Notes
    -----
    This is the inverse transformation of ecef2enu().

    Examples
    --------
    >>> import numpy as np
    >>> origin_llh = np.array([np.radians(35.0), np.radians(139.0), 0])
    >>> enu_local = np.array([100.0, 200.0, 50.0])  # 100m E, 200m N, 50m up
    >>> ecef = enu2ecef(enu_local, origin_llh)
    """
    # Origin in ECEF
    org_xyz = llh2ecef(org_llh)

    # Rotation matrix (transpose of ecef2enu)
    lat, lon = org_llh[0], org_llh[1]
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    R = np.array([
        [-sin_lon, -sin_lat * cos_lon, cos_lat * cos_lon],
        [cos_lon, -sin_lat * sin_lon, cos_lat * sin_lon],
        [0, cos_lat, sin_lat]
    ])

    return org_xyz + R @ enu


def ecef2ned(xyz: np.ndarray, org_llh: np.ndarray) -> np.ndarray:
    """
    Convert ECEF to local NED coordinates

    Parameters:
    -----------
    xyz : np.ndarray
        ECEF coordinates [x, y, z] (m)
    org_llh : np.ndarray
        Origin geodetic coordinates [lat, lon, height] (rad, rad, m)

    Returns:
    --------
    ned : np.ndarray
        Local NED coordinates [n, e, d] (m)
    """
    enu = ecef2enu(xyz, org_llh)
    return np.array([enu[1], enu[0], -enu[2]])


def ned2ecef(ned: np.ndarray, org_llh: np.ndarray) -> np.ndarray:
    """
    Convert local NED to ECEF coordinates

    Parameters:
    -----------
    ned : np.ndarray
        Local NED coordinates [n, e, d] (m)
    org_llh : np.ndarray
        Origin geodetic coordinates [lat, lon, height] (rad, rad, m)

    Returns:
    --------
    xyz : np.ndarray
        ECEF coordinates [x, y, z] (m)
    """
    enu = np.array([ned[1], ned[0], -ned[2]])
    return enu2ecef(enu, org_llh)


def covecef2enu(llh: np.ndarray, P_ecef: np.ndarray) -> np.ndarray:
    """Transform covariance matrix from ECEF to ENU coordinate system

    Rotates a covariance matrix from ECEF coordinates to local ENU coordinates
    using the rotation matrix at the specified location.

    Parameters
    ----------
    llh : np.ndarray
        Geodetic coordinates [lat, lon, height] where rotation is computed:
        - lat, lon: in radians
        - height: in meters (not used for rotation)
    P_ecef : np.ndarray
        Covariance matrix in ECEF coordinates (3x3)

    Returns
    -------
    np.ndarray
        Covariance matrix in ENU coordinates (3x3)

    Notes
    -----
    Uses the transformation: P_enu = R * P_ecef * R^T where R is the
    rotation matrix from ECEF to ENU at the given location.

    Examples
    --------
    >>> import numpy as np
    >>> llh = np.array([np.radians(35.0), np.radians(139.0), 100])
    >>> P_ecef = np.diag([1.0, 1.0, 1.0])  # 1m standard deviation in each axis
    >>> P_enu = covecef2enu(llh, P_ecef)
    """
    lat, lon = llh[0], llh[1]
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    R = np.array([
        [-sin_lon, cos_lon, 0],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]
    ])

    return R @ P_ecef @ R.T


def covenu2ecef(llh: np.ndarray, P_enu: np.ndarray) -> np.ndarray:
    """
    Transform covariance matrix from ENU to ECEF

    Parameters:
    -----------
    llh : np.ndarray
        Geodetic coordinates [lat, lon, height] (rad, rad, m)
    P_enu : np.ndarray
        Covariance matrix in ENU (3x3)

    Returns:
    --------
    P_ecef : np.ndarray
        Covariance matrix in ECEF (3x3)
    """
    lat, lon = llh[0], llh[1]
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    R = np.array([
        [-sin_lon, -sin_lat * cos_lon, cos_lat * cos_lon],
        [cos_lon, -sin_lat * sin_lon, cos_lat * sin_lon],
        [0, cos_lat, sin_lat]
    ])

    return R @ P_enu @ R.T


def xyz2enu_jacobian(org_llh: np.ndarray) -> np.ndarray:
    """Get Jacobian matrix for ECEF to ENU transformation

    Computes the Jacobian (rotation) matrix that transforms vectors
    from ECEF to ENU coordinates at the specified location.

    Parameters
    ----------
    org_llh : np.ndarray
        Origin geodetic coordinates [lat, lon, height] where:
        - lat, lon: in radians
        - height: in meters (not used for rotation)

    Returns
    -------
    np.ndarray
        Jacobian/rotation matrix (3x3) for ECEF to ENU transformation

    Notes
    -----
    This matrix is the same as the rotation matrix used in ecef2enu()
    and can be used for transforming vectors, covariances, or computing
    partial derivatives.

    Examples
    --------
    >>> import numpy as np
    >>> llh = np.array([np.radians(35.0), np.radians(139.0), 0])
    >>> J = xyz2enu_jacobian(llh)
    >>> # Transform a vector from ECEF to ENU
    >>> vec_ecef = np.array([1, 0, 0])  # Unit vector along ECEF X-axis
    >>> vec_enu = J @ vec_ecef
    """
    lat, lon = org_llh[0], org_llh[1]
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    return np.array([
        [-sin_lon, cos_lon, 0],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]
    ])


def lla2ned(lla: np.ndarray, lla0: np.ndarray) -> np.ndarray:
    """
    Convert geodetic coordinates to local NED coordinates

    Parameters:
    -----------
    lla : np.ndarray
        Geodetic coordinates [lat, lon, height] (rad, rad, m)
    lla0 : np.ndarray
        Origin geodetic coordinates [lat, lon, height] (rad, rad, m)

    Returns:
    --------
    ned : np.ndarray
        Local NED coordinates [n, e, d] (m)
    """
    xyz = llh2ecef(lla)
    return ecef2ned(xyz, lla0)


def ned2lla(ned: np.ndarray, lla0: np.ndarray) -> np.ndarray:
    """
    Convert local NED to geodetic coordinates

    Parameters:
    -----------
    ned : np.ndarray
        Local NED coordinates [n, e, d] (m)
    lla0 : np.ndarray
        Origin geodetic coordinates [lat, lon, height] (rad, rad, m)

    Returns:
    --------
    lla : np.ndarray
        Geodetic coordinates [lat, lon, height] (rad, rad, m)
    """
    xyz = ned2ecef(ned, lla0)
    return ecef2llh(xyz)


def lla2enu(lla: np.ndarray, lla0: np.ndarray) -> np.ndarray:
    """
    Convert geodetic coordinates to local ENU coordinates

    Parameters:
    -----------
    lla : np.ndarray
        Geodetic coordinates [lat, lon, height] (rad, rad, m)
    lla0 : np.ndarray
        Origin geodetic coordinates [lat, lon, height] (rad, rad, m)

    Returns:
    --------
    enu : np.ndarray
        Local ENU coordinates [e, n, u] (m)
    """
    xyz = llh2ecef(lla)
    return ecef2enu(xyz, lla0)


def enu2lla(enu: np.ndarray, lla0: np.ndarray) -> np.ndarray:
    """
    Convert local ENU to geodetic coordinates

    Parameters:
    -----------
    enu : np.ndarray
        Local ENU coordinates [e, n, u] (m)
    lla0 : np.ndarray
        Origin geodetic coordinates [lat, lon, height] (rad, rad, m)

    Returns:
    --------
    lla : np.ndarray
        Geodetic coordinates [lat, lon, height] (rad, rad, m)
    """
    xyz = enu2ecef(enu, lla0)
    return ecef2llh(xyz)


def enu2ned(enu: np.ndarray) -> np.ndarray:
    """Convert ENU to NED coordinates

    Simple coordinate system conversion between East-North-Up and
    North-East-Down coordinate systems.

    Parameters
    ----------
    enu : np.ndarray
        ENU coordinates [e, n, u] in meters where:
        - e: eastward displacement
        - n: northward displacement
        - u: upward displacement

    Returns
    -------
    np.ndarray
        NED coordinates [n, e, d] in meters where:
        - n: northward displacement (same as input)
        - e: eastward displacement (same as input)
        - d: downward displacement (negative of upward)

    Notes
    -----
    This is a simple axis reordering and sign change:
    [n, e, d] = [n, e, -u]

    Examples
    --------
    >>> import numpy as np
    >>> enu = np.array([100, 200, 50])  # 100m E, 200m N, 50m up
    >>> ned = enu2ned(enu)
    >>> print(ned)  # [200, 100, -50] -> 200m N, 100m E, 50m down
    """
    return np.array([enu[1], enu[0], -enu[2]])


def ned2enu(ned: np.ndarray) -> np.ndarray:
    """Convert NED to ENU coordinates

    Simple coordinate system conversion between North-East-Down and
    East-North-Up coordinate systems.

    Parameters
    ----------
    ned : np.ndarray
        NED coordinates [n, e, d] in meters where:
        - n: northward displacement
        - e: eastward displacement
        - d: downward displacement

    Returns
    -------
    np.ndarray
        ENU coordinates [e, n, u] in meters where:
        - e: eastward displacement (same as input)
        - n: northward displacement (same as input)
        - u: upward displacement (negative of downward)

    Notes
    -----
    This is a simple axis reordering and sign change:
    [e, n, u] = [e, n, -d]
    This is the inverse of enu2ned().

    Examples
    --------
    >>> import numpy as np
    >>> ned = np.array([200, 100, -50])  # 200m N, 100m E, 50m down
    >>> enu = ned2enu(ned)
    >>> print(enu)  # [100, 200, 50] -> 100m E, 200m N, 50m up
    """
    return np.array([ned[1], ned[0], -ned[2]])


def compute_rotation_matrix_enu(llh: np.ndarray) -> np.ndarray:
    """Compute rotation matrix from ECEF to ENU coordinates

    Calculates the 3x3 rotation matrix that transforms vectors from
    Earth-Centered Earth-Fixed (ECEF) to local East-North-Up (ENU)
    coordinate system at the specified geodetic location.

    Parameters
    ----------
    llh : np.ndarray
        Geodetic coordinates [lat, lon, height] where:
        - lat: latitude in radians
        - lon: longitude in radians
        - height: in meters (not used in rotation calculation)

    Returns
    -------
    np.ndarray
        Rotation matrix (3x3) for ECEF to ENU transformation

    Notes
    -----
    The rotation matrix R transforms ECEF vectors to ENU: v_enu = R @ v_ecef
    The matrix elements depend only on latitude and longitude.

    Examples
    --------
    >>> import numpy as np
    >>> llh = np.array([np.radians(45.0), np.radians(0.0), 0])
    >>> R = compute_rotation_matrix_enu(llh)
    >>> # Transform ECEF unit vector to ENU
    >>> ecef_north = np.array([0, 0, 1])  # ECEF Z-axis (towards north pole)
    >>> enu_north = R @ ecef_north  # Should be [0, 1, 0] at equator
    """
    lat, lon = llh[0], llh[1]
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    return np.array([
        [-sin_lon, cos_lon, 0],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]
    ])


def compute_rotation_matrix_ned(llh: np.ndarray) -> np.ndarray:
    """Compute rotation matrix from ECEF to NED coordinates

    Calculates the 3x3 rotation matrix that transforms vectors from
    Earth-Centered Earth-Fixed (ECEF) to local North-East-Down (NED)
    coordinate system at the specified geodetic location.

    Parameters
    ----------
    llh : np.ndarray
        Geodetic coordinates [lat, lon, height] where:
        - lat: latitude in radians
        - lon: longitude in radians
        - height: in meters (not used in rotation calculation)

    Returns
    -------
    np.ndarray
        Rotation matrix (3x3) for ECEF to NED transformation

    Notes
    -----
    The rotation matrix R transforms ECEF vectors to NED: v_ned = R @ v_ecef
    NED coordinate system is commonly used in navigation applications.

    Examples
    --------
    >>> import numpy as np
    >>> llh = np.array([np.radians(45.0), np.radians(0.0), 0])
    >>> R = compute_rotation_matrix_ned(llh)
    >>> # Transform ECEF unit vector to NED
    >>> ecef_up = np.array([0, 0, 1])  # ECEF Z-axis
    >>> ned_up = R @ ecef_up  # Should have negative down component
    """
    lat, lon = llh[0], llh[1]
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    return np.array([
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [-sin_lon, cos_lon, 0],
        [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat]
    ])
