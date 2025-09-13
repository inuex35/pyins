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
    
    # ecef2enu expects the satellite position, not the LOS vector
    # and the reference LLH in radians
    enu = ecef2enu(sat_pos, reference_llh_rad)
    
    # Compute elevation angle
    horizontal = np.sqrt(enu[0]**2 + enu[1]**2)
    elevation = np.arctan2(enu[2], horizontal)
    
    return np.degrees(elevation)
