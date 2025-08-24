#!/usr/bin/env python3
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
        Reference position in [lat, lon, height] for local frame

    Returns:
    --------
    float
        Elevation angle in degrees
    """
    # Vector from receiver to satellite
    los = sat_pos - rcv_pos

    # Convert to ENU
    enu = ecef2enu(los, reference_llh)

    # Compute elevation angle
    horizontal = np.sqrt(enu[0]**2 + enu[1]**2)
    elevation = np.arctan2(enu[2], horizontal)

    return np.degrees(elevation)
