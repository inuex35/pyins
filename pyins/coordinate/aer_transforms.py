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
from typing import Tuple
from .transforms import llh2ecef, ecef2llh, ecef2enu, lla2enu
from .eci_transforms import eci2enu, eci2lla


def enu2aer(enu_t: np.ndarray, enu_r: np.ndarray) -> np.ndarray:
    """
    Convert ENU to Azimuth-Elevation-Range coordinates
    
    Parameters:
    -----------
    enu_t : np.ndarray
        Target ENU coordinates [e, n, u] (m)
    enu_r : np.ndarray
        Reference ENU coordinates [e, n, u] (m)
        
    Returns:
    --------
    aer : np.ndarray
        Relative AER [azimuth, elevation, range] (rad, rad, m)
    """
    de, dn, du = enu_t - enu_r
    
    r = np.hypot(de, dn)
    az = np.mod(np.arctan2(de, dn), 2 * np.pi)
    el = np.arctan2(du, r)
    rng = np.hypot(r, du)
    
    return np.array([az, el, rng])


def ned2aer(ned_t: np.ndarray, ned_r: np.ndarray) -> np.ndarray:
    """
    Convert NED to Azimuth-Elevation-Range coordinates
    
    Parameters:
    -----------
    ned_t : np.ndarray
        Target NED coordinates [n, e, d] (m)
    ned_r : np.ndarray
        Reference NED coordinates [n, e, d] (m)
        
    Returns:
    --------
    aer : np.ndarray
        Relative AER [azimuth, elevation, range] (rad, rad, m)
    """
    dn, de, dd = ned_t - ned_r
    
    r = np.hypot(de, dn)
    az = np.mod(np.arctan2(de, dn), 2 * np.pi)
    el = np.arctan2(-dd, r)
    rng = np.hypot(r, -dd)
    
    return np.array([az, el, rng])


def aer2enu(aer: np.ndarray, enu_r: np.ndarray) -> np.ndarray:
    """
    Convert Azimuth-Elevation-Range to ENU coordinates
    
    Parameters:
    -----------
    aer : np.ndarray
        AER coordinates [azimuth, elevation, range] (rad, rad, m)
    enu_r : np.ndarray
        Reference ENU coordinates [e, n, u] (m)
        
    Returns:
    --------
    enu : np.ndarray
        Target ENU coordinates [e, n, u] (m)
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
    """
    Convert Azimuth-Elevation-Range to NED coordinates
    
    Parameters:
    -----------
    aer : np.ndarray
        AER coordinates [azimuth, elevation, range] (rad, rad, m)
    ned_r : np.ndarray
        Reference NED coordinates [n, e, d] (m)
        
    Returns:
    --------
    ned : np.ndarray
        Target NED coordinates [n, e, d] (m)
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
    """
    Convert geodetic coordinates to Azimuth-Elevation-Range
    
    Parameters:
    -----------
    lla_t : np.ndarray
        Target geodetic coordinates [lat, lon, height] (rad, rad, m)
    lla_r : np.ndarray
        Reference geodetic coordinates [lat, lon, height] (rad, rad, m)
        
    Returns:
    --------
    aer : np.ndarray
        Relative AER [azimuth, elevation, range] (rad, rad, m)
    """
    enu_t = lla2enu(lla_t, lla_r)
    enu_r = np.zeros(3)  # Reference is at origin in its own ENU frame
    return enu2aer(enu_t, enu_r)


def ecef2aer(ecef_t: np.ndarray, ecef_r: np.ndarray) -> np.ndarray:
    """
    Convert ECEF to Azimuth-Elevation-Range coordinates
    
    Parameters:
    -----------
    ecef_t : np.ndarray
        Target ECEF coordinates [x, y, z] (m)
    ecef_r : np.ndarray
        Reference ECEF coordinates [x, y, z] (m)
        
    Returns:
    --------
    aer : np.ndarray
        Relative AER [azimuth, elevation, range] (rad, rad, m)
    """
    lla_r = ecef2llh(ecef_r)
    enu_t = ecef2enu(ecef_t, lla_r)
    enu_r = np.zeros(3)  # Reference is at origin in its own ENU frame
    return enu2aer(enu_t, enu_r)


def eci2aer(eci_t: np.ndarray, eci_r: np.ndarray, t: float) -> np.ndarray:
    """
    Convert ECI to Azimuth-Elevation-Range coordinates
    
    Parameters:
    -----------
    eci_t : np.ndarray
        Target ECI coordinates [x, y, z] (m)
    eci_r : np.ndarray
        Reference ECI coordinates [x, y, z] (m)
    t : float
        Time since reference epoch (s)
        
    Returns:
    --------
    aer : np.ndarray
        Relative AER [azimuth, elevation, range] (rad, rad, m)
    """
    lla_r = eci2lla(eci_r, t)
    enu_t = eci2enu(eci_t, lla_r, t)
    enu_r = eci2enu(eci_r, lla_r, t)
    return enu2aer(enu_t, enu_r)