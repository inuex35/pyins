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

"""Geodetic computations and utilities"""


import numpy as np

from ..core.constants import FE_WGS84, RE_WGS84


def geodetic_distance(llh1: np.ndarray, llh2: np.ndarray) -> float:
    """
    Compute geodetic distance between two points using Vincenty's formula

    Parameters:
    -----------
    llh1 : np.ndarray
        First point [lat, lon, height] (rad, rad, m)
    llh2 : np.ndarray
        Second point [lat, lon, height] (rad, rad, m)

    Returns:
    --------
    distance : float
        Geodetic distance (m)
    """
    lat1, lon1 = llh1[0], llh1[1]
    lat2, lon2 = llh2[0], llh2[1]

    # Vincenty's inverse formula
    a = RE_WGS84
    f = FE_WGS84
    b = a * (1 - f)

    L = lon2 - lon1
    U1 = np.arctan((1 - f) * np.tan(lat1))
    U2 = np.arctan((1 - f) * np.tan(lat2))

    sin_U1 = np.sin(U1)
    cos_U1 = np.cos(U1)
    sin_U2 = np.sin(U2)
    cos_U2 = np.cos(U2)

    lam = L
    for _ in range(100):  # Max iterations
        sin_lam = np.sin(lam)
        cos_lam = np.cos(lam)

        sin_sigma = np.sqrt((cos_U2 * sin_lam)**2 +
                           (cos_U1 * sin_U2 - sin_U1 * cos_U2 * cos_lam)**2)

        if sin_sigma == 0:
            return 0.0  # Coincident points

        cos_sigma = sin_U1 * sin_U2 + cos_U1 * cos_U2 * cos_lam
        sigma = np.arctan2(sin_sigma, cos_sigma)

        sin_alpha = cos_U1 * cos_U2 * sin_lam / sin_sigma
        cos2_alpha = 1 - sin_alpha**2

        if cos2_alpha == 0:
            cos_2sigma_m = 0
        else:
            cos_2sigma_m = cos_sigma - 2 * sin_U1 * sin_U2 / cos2_alpha

        C = f / 16 * cos2_alpha * (4 + f * (4 - 3 * cos2_alpha))

        lam_prev = lam
        lam = L + (1 - C) * f * sin_alpha * (
            sigma + C * sin_sigma * (
                cos_2sigma_m + C * cos_sigma * (-1 + 2 * cos_2sigma_m**2)
            )
        )

        if abs(lam - lam_prev) < 1e-12:
            break

    u2 = cos2_alpha * (a**2 - b**2) / b**2
    A = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))

    delta_sigma = B * sin_sigma * (
        cos_2sigma_m + B / 4 * (
            cos_sigma * (-1 + 2 * cos_2sigma_m**2) -
            B / 6 * cos_2sigma_m * (-3 + 4 * sin_sigma**2) * (-3 + 4 * cos_2sigma_m**2)
        )
    )

    s = b * A * (sigma - delta_sigma)

    # Add height difference
    dh = llh2[2] - llh1[2]
    distance = np.sqrt(s**2 + dh**2)

    return distance


def geodetic_azimuth(llh1: np.ndarray, llh2: np.ndarray) -> tuple[float, float]:
    """
    Compute forward and reverse azimuth between two points

    Parameters:
    -----------
    llh1 : np.ndarray
        First point [lat, lon, height] (rad, rad, m)
    llh2 : np.ndarray
        Second point [lat, lon, height] (rad, rad, m)

    Returns:
    --------
    fwd_azimuth : float
        Forward azimuth from point 1 to point 2 (rad)
    rev_azimuth : float
        Reverse azimuth from point 2 to point 1 (rad)
    """
    lat1, lon1 = llh1[0], llh1[1]
    lat2, lon2 = llh2[0], llh2[1]

    dlon = lon2 - lon1

    # Forward azimuth
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    fwd_azimuth = np.arctan2(y, x)

    # Reverse azimuth
    y = -np.sin(dlon) * np.cos(lat1)
    x = np.cos(lat2) * np.sin(lat1) - np.sin(lat2) * np.cos(lat1) * np.cos(dlon)
    rev_azimuth = np.arctan2(y, x)

    return fwd_azimuth, rev_azimuth


def radius_of_curvature(lat: float) -> tuple[float, float]:
    """
    Compute radii of curvature at given latitude

    Parameters:
    -----------
    lat : float
        Latitude (rad)

    Returns:
    --------
    M : float
        Meridional radius of curvature (m)
    N : float
        Prime vertical radius of curvature (m)
    """
    sin_lat = np.sin(lat)
    e2 = FE_WGS84 * (2.0 - FE_WGS84)

    # Prime vertical radius
    N = RE_WGS84 / np.sqrt(1.0 - e2 * sin_lat**2)

    # Meridional radius
    M = RE_WGS84 * (1.0 - e2) / (1.0 - e2 * sin_lat**2)**1.5

    return M, N


def gravity_model(lat: float, h: float) -> float:
    """
    Compute local gravity using WGS84 gravity model

    Parameters:
    -----------
    lat : float
        Latitude (rad)
    h : float
        Height above ellipsoid (m)

    Returns:
    --------
    g : float
        Local gravity (m/s^2)
    """
    # WGS84 gravity model parameters
    ge = 9.7803253359  # Gravity at equator (m/s^2)

    sin_lat = np.sin(lat)
    sin2_lat = sin_lat**2

    # Normal gravity at ellipsoid surface (Somigliana formula)
    g0 = ge * (1.0 + 0.00193185265241 * sin2_lat) / \
         np.sqrt(1.0 - 0.00669437999014 * sin2_lat)

    # Height correction (free-air reduction)
    g = g0 * (1.0 - 2.0 * h / RE_WGS84)

    return g


def local_gravity_vector(llh: np.ndarray) -> np.ndarray:
    """
    Compute local gravity vector in ECEF coordinates

    Parameters:
    -----------
    llh : np.ndarray
        Geodetic coordinates [lat, lon, height] (rad, rad, m)

    Returns:
    --------
    g_ecef : np.ndarray
        Gravity vector in ECEF (m/s^2)
    """
    lat, lon, h = llh[0], llh[1], llh[2]

    # Magnitude of gravity
    g = gravity_model(lat, h)

    # Direction (towards Earth center, adjusted for ellipsoid)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    # Normal to ellipsoid surface
    n = np.array([
        cos_lat * cos_lon,
        cos_lat * sin_lon,
        sin_lat
    ])

    # Gravity vector (negative of normal)
    g_ecef = -g * n

    return g_ecef


class GeodeticGrid:
    """Grid-based geodetic computations"""

    def __init__(self, lat_min: float, lat_max: float,
                 lon_min: float, lon_max: float,
                 lat_step: float, lon_step: float):
        """
        Initialize geodetic grid

        Parameters:
        -----------
        lat_min, lat_max : float
            Latitude bounds (rad)
        lon_min, lon_max : float
            Longitude bounds (rad)
        lat_step, lon_step : float
            Grid spacing (rad)
        """
        self.lat_grid = np.arange(lat_min, lat_max + lat_step, lat_step)
        self.lon_grid = np.arange(lon_min, lon_max + lon_step, lon_step)

        # Precompute gravity and radii of curvature
        self.gravity_grid = np.zeros((len(self.lat_grid), len(self.lon_grid)))
        self.M_grid = np.zeros(len(self.lat_grid))
        self.N_grid = np.zeros(len(self.lat_grid))

        for i, lat in enumerate(self.lat_grid):
            self.M_grid[i], self.N_grid[i] = radius_of_curvature(lat)
            for j, _lon in enumerate(self.lon_grid):
                self.gravity_grid[i, j] = gravity_model(lat, 0.0)

    def interpolate_gravity(self, lat: float, lon: float) -> float:
        """Interpolate gravity from grid"""
        # Find grid indices
        i = np.searchsorted(self.lat_grid, lat) - 1
        j = np.searchsorted(self.lon_grid, lon) - 1

        # Boundary check
        i = np.clip(i, 0, len(self.lat_grid) - 2)
        j = np.clip(j, 0, len(self.lon_grid) - 2)

        # Bilinear interpolation
        lat1, lat2 = self.lat_grid[i], self.lat_grid[i + 1]
        lon1, lon2 = self.lon_grid[j], self.lon_grid[j + 1]

        w_lat = (lat - lat1) / (lat2 - lat1)
        w_lon = (lon - lon1) / (lon2 - lon1)

        g00 = self.gravity_grid[i, j]
        g01 = self.gravity_grid[i, j + 1]
        g10 = self.gravity_grid[i + 1, j]
        g11 = self.gravity_grid[i + 1, j + 1]

        g0 = g00 * (1 - w_lon) + g01 * w_lon
        g1 = g10 * (1 - w_lon) + g11 * w_lon

        return g0 * (1 - w_lat) + g1 * w_lat
