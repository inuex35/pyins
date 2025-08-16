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

"""BeiDou GEO/IGSO satellite coordinate transformation"""

import numpy as np
from ..core.constants import OMGE

def is_beidou_geo_igso(prn):
    """
    Check if BeiDou satellite is GEO or IGSO
    Based on rtklib-py approach
    
    GEO: C01-C05, C59-C63
    IGSO: C06-C18, C31-C40, C56-C58
    """
    return (prn <= 5) or (prn >= 59)  # Simplified check for GEO satellites

def beidou_geo_transform(rs, tk):
    """
    Apply BeiDou GEO satellite coordinate transformation
    Based on rtklib-py implementation
    
    Parameters
    ----------
    rs : array_like
        Satellite position in ECEF [x, y, z] (m)
    tk : float
        Time since ephemeris reference epoch (s)
        
    Returns
    -------
    rs_transformed : ndarray
        Transformed satellite position (m)
    """
    # Constants
    sin5 = np.sin(np.deg2rad(-5))
    cos5 = np.cos(np.deg2rad(-5))
    
    # Extract coordinates
    xg, yg, zg = rs[0], rs[1], rs[2]
    
    # Rotation angle
    sino = np.sin(OMGE * tk)
    coso = np.cos(OMGE * tk)
    
    # Apply transformation (from CGCS2000 to WGS84)
    x = xg * coso + yg * sino * cos5 + zg * sino * sin5
    y = -xg * sino + yg * coso * cos5 + zg * coso * sin5
    z = -yg * sin5 + zg * cos5
    
    return np.array([x, y, z])