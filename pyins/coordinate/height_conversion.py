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
Height conversion utilities for converting between ellipsoidal and orthometric heights.

Ellipsoidal height (h): Height above the reference ellipsoid (WGS84)
Orthometric height (H): Height above the geoid (mean sea level)
Geoid height (N): Height of the geoid above the ellipsoid

Relationship: h = H + N
"""

import numpy as np
from typing import Union, Tuple
from enum import Enum

class HeightSystem(Enum):
    """Height system enumeration."""
    ELLIPSOIDAL = "ellipsoidal"  # Height above WGS84 ellipsoid
    ORTHOMETRIC = "orthometric"  # Height above geoid (MSL)


class EGM96:
    """
    Simple EGM96 geoid model implementation.
    This is a simplified version - for production use, consider using pygeodesy or similar.
    """
    
    def __init__(self):
        # These are example values for demonstration
        # In practice, you would load the full EGM96 grid data
        self._grid_resolution = 0.25  # degrees
        self._lat_range = (-90, 90)
        self._lon_range = (-180, 180)
        
    def get_geoid_height(self, lat: float, lon: float) -> float:
        """
        Get geoid height (N) at given location.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            
        Returns:
            Geoid height in meters (geoid above ellipsoid)
        """
        # Simplified geoid model - returns approximate values
        # For Japan/California area, geoid heights are typically -30 to -40m
        # This is a very rough approximation!
        
        # Normalize longitude to [-180, 180]
        lon = ((lon + 180) % 360) - 180
        
        # Very rough approximation based on location
        if 30 <= lat <= 45 and 130 <= lon <= 145:  # Japan region
            return -35.0 + 2.0 * np.sin(np.radians(lat - 35))
        elif 30 <= lat <= 42 and -125 <= lon <= -115:  # California region  
            return -30.0 + 3.0 * np.sin(np.radians(lat - 36))
        else:
            # Global rough approximation
            return -30.0 + 20.0 * np.sin(np.radians(lat))


# Global instance
_egm96 = EGM96()


def ellipsoidal_to_orthometric(
    h_ellipsoidal: Union[float, np.ndarray],
    lat: Union[float, np.ndarray], 
    lon: Union[float, np.ndarray],
    geoid_model: str = "EGM96"
) -> Union[float, np.ndarray]:
    """
    Convert ellipsoidal height to orthometric height.
    
    Args:
        h_ellipsoidal: Ellipsoidal height(s) in meters
        lat: Latitude(s) in degrees
        lon: Longitude(s) in degrees
        geoid_model: Geoid model to use (currently only "EGM96" supported)
        
    Returns:
        Orthometric height(s) in meters
    """
    if geoid_model != "EGM96":
        raise ValueError(f"Unsupported geoid model: {geoid_model}")
        
    # Get geoid height
    if isinstance(lat, np.ndarray):
        N = np.array([_egm96.get_geoid_height(la, lo) for la, lo in zip(lat, lon)])
    else:
        N = _egm96.get_geoid_height(lat, lon)
    
    # H = h - N
    return h_ellipsoidal - N


def orthometric_to_ellipsoidal(
    h_orthometric: Union[float, np.ndarray],
    lat: Union[float, np.ndarray],
    lon: Union[float, np.ndarray],
    geoid_model: str = "EGM96"
) -> Union[float, np.ndarray]:
    """
    Convert orthometric height to ellipsoidal height.
    
    Args:
        h_orthometric: Orthometric height(s) in meters
        lat: Latitude(s) in degrees
        lon: Longitude(s) in degrees
        geoid_model: Geoid model to use (currently only "EGM96" supported)
        
    Returns:
        Ellipsoidal height(s) in meters
    """
    if geoid_model != "EGM96":
        raise ValueError(f"Unsupported geoid model: {geoid_model}")
        
    # Get geoid height
    if isinstance(lat, np.ndarray):
        N = np.array([_egm96.get_geoid_height(la, lo) for la, lo in zip(lat, lon)])
    else:
        N = _egm96.get_geoid_height(lat, lon)
    
    # h = H + N
    return h_orthometric + N


def get_geoid_height(
    lat: Union[float, np.ndarray],
    lon: Union[float, np.ndarray],
    geoid_model: str = "EGM96"
) -> Union[float, np.ndarray]:
    """
    Get geoid height at given location(s).
    
    Args:
        lat: Latitude(s) in degrees
        lon: Longitude(s) in degrees  
        geoid_model: Geoid model to use (currently only "EGM96" supported)
        
    Returns:
        Geoid height(s) in meters (geoid above ellipsoid)
    """
    if geoid_model != "EGM96":
        raise ValueError(f"Unsupported geoid model: {geoid_model}")
        
    if isinstance(lat, np.ndarray):
        return np.array([_egm96.get_geoid_height(la, lo) for la, lo in zip(lat, lon)])
    else:
        return _egm96.get_geoid_height(lat, lon)


def convert_height(
    height: Union[float, np.ndarray],
    lat: Union[float, np.ndarray],
    lon: Union[float, np.ndarray],
    from_system: HeightSystem,
    to_system: HeightSystem,
    geoid_model: str = "EGM96"
) -> Union[float, np.ndarray]:
    """
    Convert height between different height systems.
    
    Args:
        height: Height value(s) to convert
        lat: Latitude(s) in degrees
        lon: Longitude(s) in degrees
        from_system: Source height system
        to_system: Target height system
        geoid_model: Geoid model to use
        
    Returns:
        Converted height value(s)
    """
    if from_system == to_system:
        return height
        
    if from_system == HeightSystem.ELLIPSOIDAL and to_system == HeightSystem.ORTHOMETRIC:
        return ellipsoidal_to_orthometric(height, lat, lon, geoid_model)
    elif from_system == HeightSystem.ORTHOMETRIC and to_system == HeightSystem.ELLIPSOIDAL:
        return orthometric_to_ellipsoidal(height, lat, lon, geoid_model)
    else:
        raise ValueError(f"Unsupported conversion: {from_system} to {to_system}")