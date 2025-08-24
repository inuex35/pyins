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
from typing import Tuple
from ..core.constants import RE_WGS84, FE_WGS84

def ecef2llh(xyz: np.ndarray) -> np.ndarray:
    """
    Convert ECEF coordinates to geodetic coordinates
    
    Parameters:
    -----------
    xyz : np.ndarray
        ECEF coordinates [x, y, z] (m)
        
    Returns:
    --------
    llh : np.ndarray
        Geodetic coordinates [lat, lon, height] (rad, rad, m)
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
    """
    Convert geodetic coordinates to ECEF coordinates
    
    Parameters:
    -----------
    llh : np.ndarray
        Geodetic coordinates [lat, lon, height] (rad, rad, m)
        
    Returns:
    --------
    xyz : np.ndarray
        ECEF coordinates [x, y, z] (m)
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
    """
    Convert ECEF to local ENU coordinates
    
    Parameters:
    -----------
    xyz : np.ndarray
        ECEF coordinates [x, y, z] (m)
    org_llh : np.ndarray
        Origin geodetic coordinates [lat, lon, height] (rad, rad, m)
        
    Returns:
    --------
    enu : np.ndarray
        Local ENU coordinates [e, n, u] (m)
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
    """
    Convert local ENU to ECEF coordinates
    
    Parameters:
    -----------
    enu : np.ndarray
        Local ENU coordinates [e, n, u] (m)
    org_llh : np.ndarray
        Origin geodetic coordinates [lat, lon, height] (rad, rad, m)
        
    Returns:
    --------
    xyz : np.ndarray
        ECEF coordinates [x, y, z] (m)
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
    """
    Transform covariance matrix from ECEF to ENU
    
    Parameters:
    -----------
    llh : np.ndarray
        Geodetic coordinates [lat, lon, height] (rad, rad, m)
    P_ecef : np.ndarray
        Covariance matrix in ECEF (3x3)
        
    Returns:
    --------
    P_enu : np.ndarray
        Covariance matrix in ENU (3x3)
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
    """
    Get Jacobian matrix for ECEF to ENU transformation
    
    Parameters:
    -----------
    org_llh : np.ndarray
        Origin geodetic coordinates [lat, lon, height] (rad, rad, m)
        
    Returns:
    --------
    J : np.ndarray
        Jacobian matrix (3x3)
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
    """
    Convert ENU to NED coordinates
    
    Parameters:
    -----------
    enu : np.ndarray
        ENU coordinates [e, n, u] (m)
        
    Returns:
    --------
    ned : np.ndarray
        NED coordinates [n, e, d] (m)
    """
    return np.array([enu[1], enu[0], -enu[2]])


def ned2enu(ned: np.ndarray) -> np.ndarray:
    """
    Convert NED to ENU coordinates
    
    Parameters:
    -----------
    ned : np.ndarray
        NED coordinates [n, e, d] (m)
        
    Returns:
    --------
    enu : np.ndarray
        ENU coordinates [e, n, u] (m)
    """
    return np.array([ned[1], ned[0], -ned[2]])


def compute_rotation_matrix_enu(llh: np.ndarray) -> np.ndarray:
    """
    Compute rotation matrix from ECEF to ENU
    
    Parameters:
    -----------
    llh : np.ndarray
        Geodetic coordinates [lat, lon, height] (rad, rad, m)
        
    Returns:
    --------
    R : np.ndarray
        Rotation matrix (3x3)
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
    """
    Compute rotation matrix from ECEF to NED
    
    Parameters:
    -----------
    llh : np.ndarray
        Geodetic coordinates [lat, lon, height] (rad, rad, m)
        
    Returns:
    --------
    R : np.ndarray
        Rotation matrix (3x3)
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