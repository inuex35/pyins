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

"""GNSS utility functions based on rtklib-py"""

import numpy as np
from ..core.constants import RE_WGS84
from ..coordinate.transforms import ecef2llh


def ecef2pos(r):
    """Transform ECEF position to geodetic position
    
    Alias for ecef2llh for rtklib compatibility
    """
    return ecef2llh(r)


def geodist(rs, rr):
    """Geometric distance and receiver-to-satellite unit vector
    
    Parameters
    ----------
    rs : np.ndarray
        Satellite position (m) ECEF
    rr : np.ndarray  
        Receiver position (m) ECEF
        
    Returns
    -------
    r : float
        Geometric distance (m)
    e : np.ndarray
        Unit vector from receiver to satellite
    """
    e = rs - rr
    r = np.linalg.norm(e)
    
    if r <= 0:
        return -1, np.zeros(3)
        
    e = e / r
    
    return r, e


def satazel(pos, e):
    """Satellite azimuth/elevation angle
    
    Parameters
    ----------
    pos : np.ndarray
        Geodetic position of receiver (lat, lon, h) in rad, rad, m
    e : np.ndarray
        Unit vector from receiver to satellite in ECEF
        
    Returns
    -------
    az : float
        Azimuth angle (rad) (0 to 2*pi)
    el : float
        Elevation angle (rad) (-pi/2 to pi/2)
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
    """Test satellite exclusion
    
    Parameters
    ----------
    sat : int
        Satellite number
    var : float
        Satellite position variance
    svh : int
        Satellite health flag
    nav : Navigation data (optional)
        
    Returns
    -------
    excluded : bool
        True if satellite should be excluded
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