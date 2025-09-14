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

from enum import Enum
from typing import Union

import numpy as np


class HeightSystem(Enum):
    """Enumeration of supported height systems

    Defines the different height reference systems used in geodetic
    and navigation applications.

    Attributes
    ----------
    ELLIPSOIDAL : str
        Height above the reference ellipsoid (WGS84)
        Also known as geodetic height or GPS height
    ORTHOMETRIC : str
        Height above the geoid (mean sea level)
        Also known as sea level height or MSL height

    Examples
    --------
    >>> print(HeightSystem.ELLIPSOIDAL.value)  # 'ellipsoidal'
    >>> print(HeightSystem.ORTHOMETRIC.value)  # 'orthometric'
    """
    ELLIPSOIDAL = "ellipsoidal"  # Height above WGS84 ellipsoid
    ORTHOMETRIC = "orthometric"  # Height above geoid (MSL)


class EGM96:
    """Simple EGM96 geoid model implementation

    Provides geoid height calculations using a simplified approximation
    of the EGM96 (Earth Gravitational Model 1996) geoid model.

    Notes
    -----
    This is a simplified implementation for demonstration purposes.
    For production applications requiring high accuracy, consider using
    dedicated geoid libraries like pygeodesy or PROJ.

    The implementation provides rough approximations based on geographic
    regions and should not be used for high-precision applications.

    Attributes
    ----------
    _grid_resolution : float
        Grid resolution in degrees (not used in current implementation)
    _lat_range : tuple
        Latitude range in degrees (-90, 90)
    _lon_range : tuple
        Longitude range in degrees (-180, 180)
    """

    def __init__(self):
        # These are example values for demonstration
        # In practice, you would load the full EGM96 grid data
        self._grid_resolution = 0.25  # degrees
        self._lat_range = (-90, 90)
        self._lon_range = (-180, 180)

    def get_geoid_height(self, lat: float, lon: float) -> float:
        """Get geoid height (N) at specified location

        Computes the height of the geoid above the WGS84 ellipsoid
        at the given geographic coordinates using regional approximations.

        Parameters
        ----------
        lat : float
            Latitude in degrees (-90 to 90)
        lon : float
            Longitude in degrees (-180 to 180)

        Returns
        -------
        float
            Geoid height in meters (positive when geoid is above ellipsoid)

        Notes
        -----
        This implementation uses very rough approximations:
        - Japan region (30-45°N, 130-145°E): ~-35m with small variations
        - California region (30-42°N, 125-115°W): ~-30m with small variations
        - Global: ~-30m + latitude-dependent variation

        These values are order-of-magnitude estimates only and should not
        be used for precise applications.

        Examples
        --------
        >>> egm96 = EGM96()
        >>> N_tokyo = egm96.get_geoid_height(35.6762, 139.6503)  # Tokyo
        >>> print(f"Geoid height in Tokyo: {N_tokyo:.1f} m")
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
    """Convert ellipsoidal height to orthometric height

    Transforms height above the WGS84 ellipsoid to height above mean sea level
    (geoid) using the specified geoid model.

    Parameters
    ----------
    h_ellipsoidal : float or np.ndarray
        Ellipsoidal height(s) in meters (height above WGS84 ellipsoid)
    lat : float or np.ndarray
        Latitude(s) in degrees (-90 to 90)
    lon : float or np.ndarray
        Longitude(s) in degrees (-180 to 180)
    geoid_model : str, optional
        Geoid model to use, by default "EGM96"
        Currently only "EGM96" is supported

    Returns
    -------
    float or np.ndarray
        Orthometric height(s) in meters (height above geoid/MSL)

    Raises
    ------
    ValueError
        If an unsupported geoid model is specified

    Notes
    -----
    The relationship is: H = h - N
    where H is orthometric height, h is ellipsoidal height, and N is geoid height.

    Examples
    --------
    >>> # Convert GPS height to sea level height
    >>> gps_height = 150.0  # meters above WGS84 ellipsoid
    >>> lat, lon = 35.0, 139.0  # degrees
    >>> msl_height = ellipsoidal_to_orthometric(gps_height, lat, lon)
    >>> print(f"Height above sea level: {msl_height:.1f} m")
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
    """Convert height between different height systems

    General-purpose function to convert heights between ellipsoidal and
    orthometric height systems using the specified geoid model.

    Parameters
    ----------
    height : float or np.ndarray
        Height value(s) to convert in meters
    lat : float or np.ndarray
        Latitude(s) in degrees (-90 to 90)
    lon : float or np.ndarray
        Longitude(s) in degrees (-180 to 180)
    from_system : HeightSystem
        Source height system (HeightSystem.ELLIPSOIDAL or HeightSystem.ORTHOMETRIC)
    to_system : HeightSystem
        Target height system (HeightSystem.ELLIPSOIDAL or HeightSystem.ORTHOMETRIC)
    geoid_model : str, optional
        Geoid model to use for conversion, by default "EGM96"

    Returns
    -------
    float or np.ndarray
        Converted height value(s) in meters

    Raises
    ------
    ValueError
        If an unsupported conversion or geoid model is specified

    Notes
    -----
    If from_system equals to_system, the input height is returned unchanged.
    This function serves as a unified interface for all height conversions.

    Examples
    --------
    >>> from pyins.coordinate import HeightSystem
    >>> # Convert GPS height to orthometric height
    >>> gps_height = 200.0
    >>> lat, lon = 35.0, 139.0
    >>> ortho_height = convert_height(
    ...     gps_height, lat, lon,
    ...     HeightSystem.ELLIPSOIDAL, HeightSystem.ORTHOMETRIC
    ... )
    >>> print(f"Orthometric height: {ortho_height:.1f} m")
    """
    if from_system == to_system:
        return height

    if from_system == HeightSystem.ELLIPSOIDAL and to_system == HeightSystem.ORTHOMETRIC:
        return ellipsoidal_to_orthometric(height, lat, lon, geoid_model)
    elif from_system == HeightSystem.ORTHOMETRIC and to_system == HeightSystem.ELLIPSOIDAL:
        return orthometric_to_ellipsoidal(height, lat, lon, geoid_model)
    else:
        raise ValueError(f"Unsupported conversion: {from_system} to {to_system}")
