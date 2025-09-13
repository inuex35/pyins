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
Elevation angle computation for GNSS satellites
"""

import numpy as np

from ..coordinate.transforms import ecef2enu


def compute_elevation_angle(sat_pos: np.ndarray, rcv_pos: np.ndarray, reference_llh: np.ndarray) -> float:
    """
    Compute satellite elevation angle

    Parameters:
    -----------
    sat_pos : np.ndarray
        Satellite position in ECEF
    rcv_pos : np.ndarray
        Receiver position in ECEF
    reference_llh : np.ndarray
        Reference position in [lat(deg), lon(deg), height(m)] for local frame

    Returns:
    --------
    float
        Elevation angle in degrees
    """
    # Convert LLH from degrees to radians for ecef2enu
    reference_llh_rad = np.array([
        np.radians(reference_llh[0]),  # lat in radians
        np.radians(reference_llh[1]),  # lon in radians
        reference_llh[2]                # height in meters
    ])
    
    # ecef2enu expects the vector from receiver to satellite
    # and the reference LLH in radians
    los_vector = sat_pos - rcv_pos
    enu = ecef2enu(los_vector, reference_llh_rad)
    
    # Compute elevation angle
    horizontal = np.sqrt(enu[0]**2 + enu[1]**2)
    elevation = np.arctan2(enu[2], horizontal)
    
    return np.degrees(elevation)
