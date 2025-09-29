#!/usr/bin/env python3
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

"""
Elevation angle computation for GNSS satellites.

This module provides functionality to compute satellite elevation angles for Global
Navigation Satellite System (GNSS) applications. The elevation angle is a critical
parameter in GNSS processing as it affects signal quality, multipath effects, and
positioning accuracy.

The primary function compute_elevation_angle() calculates the elevation angle of a
satellite as observed from a receiver position, taking into account the Earth's
curvature through coordinate transformations from ECEF to local ENU frame.

Functions
---------
compute_elevation_angle : function
    Compute the elevation angle of a GNSS satellite as observed from a receiver.

Notes
-----
Elevation angles are commonly used in GNSS processing for:
- Satellite visibility determination
- Signal quality assessment
- Multipath mitigation
- Tropospheric delay modeling
- Satellite geometry analysis (DOP computation)

Typical elevation mask angles range from 5° to 15° depending on the application
and environment.
"""

import numpy as np

from ..coordinate.transforms import ecef2enu


def compute_elevation_angle(sat_pos: np.ndarray, rcv_pos: np.ndarray, reference_llh: np.ndarray) -> float:
    """
    Compute the elevation angle of a GNSS satellite as observed from a receiver.

    The elevation angle is the angle between the horizontal plane and the line-of-sight
    vector from the receiver to the satellite. This function transforms the satellite
    and receiver positions from ECEF coordinates to a local East-North-Up (ENU)
    coordinate frame to compute the elevation angle.

    Parameters
    ----------
    sat_pos : np.ndarray
        Satellite position in Earth-Centered Earth-Fixed (ECEF) coordinates [m].
        Expected shape: (3,) for [x, y, z] coordinates.
    rcv_pos : np.ndarray
        Receiver position in Earth-Centered Earth-Fixed (ECEF) coordinates [m].
        Expected shape: (3,) for [x, y, z] coordinates.
    reference_llh : np.ndarray
        Reference position for the local coordinate frame in [latitude, longitude, height].
        Expected shape: (3,) where:
        - latitude in degrees
        - longitude in degrees
        - height in meters above ellipsoid

    Returns
    -------
    float
        Elevation angle in degrees. Range: [-90, 90] where:
        - 90° indicates the satellite is directly overhead (zenith)
        - 0° indicates the satellite is on the horizon
        - Negative values indicate the satellite is below the horizon

    Notes
    -----
    The elevation angle is computed using the following steps:
    1. Calculate the line-of-sight vector from receiver to satellite in ECEF
    2. Transform this vector to local East-North-Up (ENU) coordinates
    3. Compute the elevation as arctan2(up_component, horizontal_distance)

    The reference position is used to define the local coordinate frame origin
    and should typically be the receiver position in geodetic coordinates.

    Examples
    --------
    >>> import numpy as np
    >>> sat_pos = np.array([20000000, 10000000, 15000000])  # ECEF [m]
    >>> rcv_pos = np.array([4000000, 3000000, 5000000])     # ECEF [m]
    >>> ref_llh = np.array([35.0, 139.0, 100.0])           # [deg, deg, m]
    >>> elevation = compute_elevation_angle(sat_pos, rcv_pos, ref_llh)
    >>> print(f"Elevation angle: {elevation:.2f}°")
    """
    # reference_llh is expected in radians (lat, lon) and meters (height)
    reference_llh_rad = np.array([reference_llh[0], reference_llh[1], reference_llh[2]])
    
    # ecef2enu expects the vector from receiver to satellite
    # and the reference LLH in radians
    los_vector = sat_pos - rcv_pos
    enu = ecef2enu(los_vector, reference_llh_rad)
    
    # Compute elevation angle
    horizontal = np.sqrt(enu[0]**2 + enu[1]**2)
    elevation = np.arctan2(enu[2], horizontal)
    
    return np.degrees(elevation)
