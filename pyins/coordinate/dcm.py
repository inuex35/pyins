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

"""Direction Cosine Matrix (DCM) transformations for coordinate systems"""

import numpy as np

from ..core.constants import OMGE


def eci2ecef_dcm(t: float) -> np.ndarray:
    """
    Earth-Centered-Inertial to Earth-Centered-Earth-Fixed direction cosine matrix

    Parameters:
    -----------
    t : float
        Time since reference epoch (s)

    Returns:
    --------
    C_i_e : np.ndarray
        ECI->ECEF direction cosine matrix (3x3)
    """
    sin_wie = np.sin(OMGE * t)
    cos_wie = np.cos(OMGE * t)

    C_i_e = np.array([
        [cos_wie, sin_wie, 0.0],
        [-sin_wie, cos_wie, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    return C_i_e


def ecef2eci_dcm(t: float) -> np.ndarray:
    """
    Earth-Centered-Earth-Fixed to Earth-Centered-Inertial direction cosine matrix

    Parameters:
    -----------
    t : float
        Time since reference epoch (s)

    Returns:
    --------
    C_e_i : np.ndarray
        ECEF->ECI direction cosine matrix (3x3)
    """
    sin_wie = np.sin(OMGE * t)
    cos_wie = np.cos(OMGE * t)

    C_e_i = np.array([
        [cos_wie, -sin_wie, 0.0],
        [sin_wie, cos_wie, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    return C_e_i


def ecef2ned_dcm(lla: np.ndarray) -> np.ndarray:
    """
    Earth-Centered-Earth-Fixed to North-East-Down direction cosine matrix

    Parameters:
    -----------
    lla : np.ndarray
        Geodetic coordinates [lat, lon, height] (rad, rad, m)

    Returns:
    --------
    C_e_n : np.ndarray
        ECEF->NED direction cosine matrix (3x3)
    """
    sin_lat = np.sin(lla[0])
    cos_lat = np.cos(lla[0])
    sin_lon = np.sin(lla[1])
    cos_lon = np.cos(lla[1])

    C_e_n = np.array([
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [-sin_lon, cos_lon, 0.0],
        [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat]
    ], dtype=np.float64)

    return C_e_n


def ned2ecef_dcm(lla: np.ndarray) -> np.ndarray:
    """
    North-East-Down to Earth-Centered-Earth-Fixed direction cosine matrix

    Parameters:
    -----------
    lla : np.ndarray
        Geodetic coordinates [lat, lon, height] (rad, rad, m)

    Returns:
    --------
    C_n_e : np.ndarray
        NED->ECEF direction cosine matrix (3x3)
    """
    sin_lat = np.sin(lla[0])
    cos_lat = np.cos(lla[0])
    sin_lon = np.sin(lla[1])
    cos_lon = np.cos(lla[1])

    C_n_e = np.array([
        [-sin_lat * cos_lon, -sin_lon, -cos_lat * cos_lon],
        [-sin_lat * sin_lon, cos_lon, -cos_lat * sin_lon],
        [cos_lat, 0.0, -sin_lat]
    ], dtype=np.float64)

    return C_n_e


def ecef2enu_dcm(lla: np.ndarray) -> np.ndarray:
    """
    Earth-Centered-Earth-Fixed to East-North-Up direction cosine matrix

    Parameters:
    -----------
    lla : np.ndarray
        Geodetic coordinates [lat, lon, height] (rad, rad, m)

    Returns:
    --------
    C_e_n : np.ndarray
        ECEF->ENU direction cosine matrix (3x3)
    """
    sin_lat = np.sin(lla[0])
    cos_lat = np.cos(lla[0])
    sin_lon = np.sin(lla[1])
    cos_lon = np.cos(lla[1])

    C_e_n = np.array([
        [-sin_lon, cos_lon, 0.0],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]
    ], dtype=np.float64)

    return C_e_n


def enu2ecef_dcm(lla: np.ndarray) -> np.ndarray:
    """
    East-North-Up to Earth-Centered-Earth-Fixed direction cosine matrix

    Parameters:
    -----------
    lla : np.ndarray
        Geodetic coordinates [lat, lon, height] (rad, rad, m)

    Returns:
    --------
    C_n_e : np.ndarray
        ENU->ECEF direction cosine matrix (3x3)
    """
    sin_lat = np.sin(lla[0])
    cos_lat = np.cos(lla[0])
    sin_lon = np.sin(lla[1])
    cos_lon = np.cos(lla[1])

    C_n_e = np.array([
        [-sin_lon, -cos_lon * sin_lat, cos_lon * cos_lat],
        [cos_lon, -sin_lon * sin_lat, sin_lon * cos_lat],
        [0.0, cos_lat, sin_lat]
    ], dtype=np.float64)

    return C_n_e


def eci2ned_dcm(lla: np.ndarray, t: float) -> np.ndarray:
    """
    Earth-Centered-Inertial to North-East-Down direction cosine matrix

    Parameters:
    -----------
    lla : np.ndarray
        Geodetic coordinates [lat, lon, height] (rad, rad, m)
    t : float
        Time since reference epoch (s)

    Returns:
    --------
    C_i_n : np.ndarray
        ECI->NED direction cosine matrix (3x3)
    """
    sin_lat = np.sin(lla[0])
    cos_lat = np.cos(lla[0])
    sin_lam_wie = np.sin(lla[1] + OMGE * t)
    cos_lam_wie = np.cos(lla[1] + OMGE * t)

    C_i_n = np.array([
        [-sin_lat * cos_lam_wie, -sin_lat * sin_lam_wie, cos_lat],
        [-sin_lam_wie, cos_lam_wie, 0.0],
        [-cos_lat * cos_lam_wie, -cos_lat * sin_lam_wie, -sin_lat]
    ], dtype=np.float64)

    return C_i_n


def ned2eci_dcm(lla: np.ndarray, t: float) -> np.ndarray:
    """
    North-East-Down to Earth-Centered-Inertial direction cosine matrix

    Parameters:
    -----------
    lla : np.ndarray
        Geodetic coordinates [lat, lon, height] (rad, rad, m)
    t : float
        Time since reference epoch (s)

    Returns:
    --------
    C_n_i : np.ndarray
        NED->ECI direction cosine matrix (3x3)
    """
    sin_lat = np.sin(lla[0])
    cos_lat = np.cos(lla[0])
    sin_lam_wie = np.sin(lla[1] + OMGE * t)
    cos_lam_wie = np.cos(lla[1] + OMGE * t)

    C_n_i = np.array([
        [-sin_lat * cos_lam_wie, -sin_lam_wie, -cos_lat * cos_lam_wie],
        [-sin_lat * sin_lam_wie, cos_lam_wie, -cos_lat * sin_lam_wie],
        [cos_lat, 0.0, -sin_lat]
    ], dtype=np.float64)

    return C_n_i


def eci2enu_dcm(lla: np.ndarray, t: float) -> np.ndarray:
    """
    Earth-Centered-Inertial to East-North-Up direction cosine matrix

    Parameters:
    -----------
    lla : np.ndarray
        Geodetic coordinates [lat, lon, height] (rad, rad, m)
    t : float
        Time since reference epoch (s)

    Returns:
    --------
    C_i_n : np.ndarray
        ECI->ENU direction cosine matrix (3x3)
    """
    sin_lat = np.sin(lla[0])
    cos_lat = np.cos(lla[0])
    sin_lam_wie = np.sin(lla[1] + OMGE * t)
    cos_lam_wie = np.cos(lla[1] + OMGE * t)

    C_i_n = np.array([
        [-sin_lam_wie, cos_lam_wie, 0.0],
        [-sin_lat * cos_lam_wie, -sin_lat * sin_lam_wie, cos_lat],
        [cos_lat * cos_lam_wie, cos_lat * sin_lam_wie, sin_lat]
    ], dtype=np.float64)

    return C_i_n


def enu2eci_dcm(lla: np.ndarray, t: float) -> np.ndarray:
    """
    East-North-Up to Earth-Centered-Inertial direction cosine matrix

    Parameters:
    -----------
    lla : np.ndarray
        Geodetic coordinates [lat, lon, height] (rad, rad, m)
    t : float
        Time since reference epoch (s)

    Returns:
    --------
    C_n_i : np.ndarray
        ENU->ECI direction cosine matrix (3x3)
    """
    sin_lat = np.sin(lla[0])
    cos_lat = np.cos(lla[0])
    sin_lam_wie = np.sin(lla[1] + OMGE * t)
    cos_lam_wie = np.cos(lla[1] + OMGE * t)

    C_n_i = np.array([
        [-sin_lam_wie, -sin_lat * cos_lam_wie, cos_lat * cos_lam_wie],
        [cos_lam_wie, -sin_lat * sin_lam_wie, cos_lat * sin_lam_wie],
        [0.0, cos_lat, sin_lat]
    ], dtype=np.float64)

    return C_n_i
