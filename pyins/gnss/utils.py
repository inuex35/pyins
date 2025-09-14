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

"""GNSS utility functions based on rtklib-py.

This module provides essential utility functions for GNSS processing, including coordinate
transformations, geometric calculations, satellite visibility analysis, and satellite
health checks. These functions form the foundation for more complex GNSS algorithms.

The utilities are designed to be compatible with the rtklib library conventions while
providing modern Python implementations with comprehensive error handling and documentation.

Functions:
    ecef2pos: Transform ECEF coordinates to geodetic position (rtklib compatibility alias)
    geodist: Compute geometric distance and line-of-sight unit vector
    satazel: Calculate satellite azimuth and elevation angles
    satexclude: Determine if a satellite should be excluded from processing

Notes:
    All functions use standard GNSS conventions:
    - ECEF coordinates in meters
    - Geodetic coordinates: latitude/longitude in radians, height in meters
    - Angles in radians unless otherwise specified
    - Distance measurements in meters
"""

import numpy as np

from ..coordinate.transforms import ecef2llh


def ecef2pos(r):
    """Transform ECEF position to geodetic position.

    This is an alias function for ecef2llh to maintain compatibility with rtklib
    conventions. It converts Earth-Centered, Earth-Fixed (ECEF) Cartesian coordinates
    to geodetic coordinates (latitude, longitude, height).

    Parameters
    ----------
    r : array_like of shape (3,)
        ECEF position vector [x, y, z] in meters.

    Returns
    -------
    pos : numpy.ndarray of shape (3,)
        Geodetic position [latitude, longitude, height] where:
        - latitude in radians (-π/2 to π/2)
        - longitude in radians (-π to π)
        - height in meters above the reference ellipsoid

    See Also
    --------
    pyins.coordinate.transforms.ecef2llh : The underlying transformation function

    Examples
    --------
    >>> import numpy as np
    >>> ecef_pos = np.array([4194304.0, 486594.0, 4688237.0])
    >>> llh_pos = ecef2pos(ecef_pos)
    >>> print(f"Lat: {np.degrees(llh_pos[0]):.6f}°")
    >>> print(f"Lon: {np.degrees(llh_pos[1]):.6f}°")
    >>> print(f"Height: {llh_pos[2]:.3f} m")
    """
    return ecef2llh(r)


def geodist(rs, rr):
    """Calculate geometric distance and receiver-to-satellite unit vector.

    Computes the Euclidean distance between receiver and satellite positions in ECEF
    coordinates and returns the corresponding unit vector pointing from receiver to
    satellite. This is fundamental for GNSS positioning calculations.

    Parameters
    ----------
    rs : array_like of shape (3,)
        Satellite position in ECEF coordinates [x, y, z] in meters.
    rr : array_like of shape (3,)
        Receiver position in ECEF coordinates [x, y, z] in meters.

    Returns
    -------
    r : float
        Geometric distance from receiver to satellite in meters.
        Returns -1 if distance is invalid (≤ 0).
    e : numpy.ndarray of shape (3,)
        Unit vector from receiver to satellite in ECEF coordinates.
        Returns zero vector if distance is invalid.

    Notes
    -----
    The function performs basic validation to ensure the computed distance is positive.
    If the distance is non-positive, it returns -1 and a zero vector, indicating an
    error condition that calling functions should handle appropriately.

    Examples
    --------
    >>> import numpy as np
    >>> satellite_pos = np.array([20000000.0, 10000000.0, 15000000.0])
    >>> receiver_pos = np.array([4194304.0, 486594.0, 4688237.0])
    >>> distance, unit_vector = geodist(satellite_pos, receiver_pos)
    >>> print(f"Distance: {distance:.3f} km")
    >>> print(f"Unit vector: {unit_vector}")
    """
    e = rs - rr
    r = np.linalg.norm(e)

    if r <= 0:
        return -1, np.zeros(3)

    e = e / r

    return r, e


def satazel(pos, e):
    """Calculate satellite azimuth and elevation angles.

    Computes the azimuth and elevation angles of a satellite as seen from a receiver
    position. The calculation transforms the satellite line-of-sight vector from ECEF
    coordinates to local East-North-Up (ENU) coordinates at the receiver position.

    Parameters
    ----------
    pos : array_like of shape (3,)
        Geodetic position of receiver [latitude, longitude, height] where:
        - latitude in radians (-π/2 to π/2)
        - longitude in radians (-π to π)
        - height in meters (not used in calculation)
    e : array_like of shape (3,)
        Unit vector from receiver to satellite in ECEF coordinates.

    Returns
    -------
    az : float
        Azimuth angle in radians, measured clockwise from North (0 to 2π).
        - 0 = North, π/2 = East, π = South, 3π/2 = West
    el : float
        Elevation angle in radians above the horizon (-π/2 to π/2).
        - π/2 = zenith (directly overhead)
        - 0 = horizon
        - -π/2 = nadir (directly below, theoretical)

    Notes
    -----
    The transformation uses the standard ECEF to ENU rotation matrix:
    - E (East) = -sin(lon)*X + cos(lon)*Y
    - N (North) = -sin(lat)*cos(lon)*X - sin(lat)*sin(lon)*Y + cos(lat)*Z
    - U (Up) = cos(lat)*cos(lon)*X + cos(lat)*sin(lon)*Y + sin(lat)*Z

    The azimuth is normalized to the range [0, 2π] and the elevation is computed
    using the arcsin of the Up component.

    Examples
    --------
    >>> import numpy as np
    >>> receiver_pos = np.array([np.radians(40.0), np.radians(-74.0), 100.0])
    >>> los_vector = np.array([0.5, 0.5, 0.707])  # Normalized unit vector
    >>> azimuth, elevation = satazel(receiver_pos, los_vector)
    >>> print(f"Azimuth: {np.degrees(azimuth):.1f}°")
    >>> print(f"Elevation: {np.degrees(elevation):.1f}°")
    """
    lat = pos[0]
    lon = pos[1]

    # Rotation matrix from ECEF to ENU
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    # Transform to ENU
    enu = np.array([
        -sin_lon * e[0] + cos_lon * e[1],
        -sin_lat * cos_lon * e[0] - sin_lat * sin_lon * e[1] + cos_lat * e[2],
        cos_lat * cos_lon * e[0] + cos_lat * sin_lon * e[1] + sin_lat * e[2]
    ])

    # Azimuth and elevation
    az = np.arctan2(enu[0], enu[1])
    if az < 0:
        az += 2 * np.pi

    el = np.arcsin(enu[2])

    return az, el


def satexclude(sat, var, svh, nav=None):
    """Test whether a satellite should be excluded from processing.

    Evaluates various criteria to determine if a satellite should be excluded from
    GNSS positioning calculations. The function performs health checks, quality
    assessments, and other exclusion tests to ensure reliable satellite data.

    Parameters
    ----------
    sat : int
        Satellite number (PRN) in the range specific to each constellation:
        - GPS: 1-32, GLONASS: 1-24, Galileo: 1-36, BeiDou: 1-63, etc.
    var : float
        Satellite position variance in m². This represents the uncertainty
        in the satellite position computation.
    svh : int
        Satellite health flag from navigation message:
        - Negative values indicate unhealthy satellites
        - 0 typically indicates healthy status
        - Positive values may indicate various health conditions
    nav : object, optional
        Navigation data object containing additional satellite information.
        Currently not used but reserved for future enhancements.

    Returns
    -------
    excluded : bool
        True if the satellite should be excluded from processing, False otherwise.

    Notes
    -----
    Current exclusion criteria:
    1. Health check: satellites with svh < 0 are excluded
    2. Position quality: satellites with variance > 10,000 m² (σ > 100m) are excluded

    Future enhancements may include:
    - Navigation message age checks
    - Constellation-specific exclusion rules
    - User-defined satellite exclusion lists
    - Signal quality metrics

    Examples
    --------
    >>> # Healthy GPS satellite with good position accuracy
    >>> exclude = satexclude(sat=1, var=25.0, svh=0)
    >>> print(f"Exclude satellite: {exclude}")  # False

    >>> # Unhealthy satellite
    >>> exclude = satexclude(sat=5, var=100.0, svh=-1)
    >>> print(f"Exclude satellite: {exclude}")  # True

    >>> # Satellite with poor position quality
    >>> exclude = satexclude(sat=10, var=15000.0, svh=0)
    >>> print(f"Exclude satellite: {exclude}")  # True
    """
    # Basic health check
    if svh < 0:
        return True

    # Variance check
    if var > 100.0**2:  # 100m std threshold
        return True

    # TODO: Add more exclusion criteria if needed
    # - Navigation message age
    # - Satellite specific exclusions

    return False
