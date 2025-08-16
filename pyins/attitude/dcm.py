"""
Attitude conversion from direction cosine matrices.

This module provides functions for converting from Direction Cosine Matrices (DCM) to
other attitude representations. All rotations assume right-hand coordinate frames with 
euler angles in the order 'roll-pitch-yaw' and DCMs with the order of 'ZYX'.

References:
    Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems
    - (2013) Paul D. Groves

Author: Migrated from navlib by Daniel Sturdivant <sturdivant20@gmail.com>
Date: December 2023
"""

import numpy as np
from numba import njit
from .euler import euler2quat


@njit(cache=True, fastmath=True)
def dcm2euler(C):
    """
    Convert 'ZYX' DCM matrix into corresponding euler angles (roll-pitch-yaw).
    
    The conversion is based on the ZYX rotation sequence where:
    - Roll (φ): rotation about x-axis
    - Pitch (θ): rotation about y-axis  
    - Yaw (ψ): rotation about z-axis
    
    Parameters
    ----------
    C : array_like, shape (3, 3)
        'XYZ' direction cosine matrix
        
    Returns
    -------
    e : ndarray, shape (3,)
        Euler angles [roll, pitch, yaw] in radians
    """
    e = np.array([np.arctan2(C[1,2], C[2,2]), 
                  np.arcsin(-C[0,2]), 
                  np.arctan2(C[0,1], C[0,0])], 
                 dtype=np.double)
    return e


@njit(cache=True, fastmath=True)
def dcm2quat(C):
    """
    Convert DCM matrix into corresponding quaternion.
    
    This function converts a direction cosine matrix to a quaternion by first
    converting to euler angles and then to quaternion representation.
    
    Parameters
    ----------
    C : array_like, shape (3, 3)
        'XYZ' direction cosine matrix
        
    Returns
    -------
    q : ndarray, shape (4,)
        Quaternion [w, x, y, z]
    """
    q = euler2quat(dcm2euler(C))
    return q