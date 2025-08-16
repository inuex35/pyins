"""
Attitude angle wrapping utilities.

This module provides functions for wrapping angles to specific ranges. All rotations 
assume right-hand coordinate frames with euler angles in the order 'roll-pitch-yaw'
and DCMs with the order of 'ZYX'.

References:
    Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems
    - (2013) Paul D. Groves

Author: Migrated from navlib by Daniel Sturdivant <sturdivant20@gmail.com>
Date: December 2023
"""

import numpy as np
from numba import njit

# Constants for angle wrapping
TWO_PI = 2 * np.pi
HALF_PI = 0.5 * np.pi


@njit(cache=True, fastmath=True)
def wrapTo2Pi(v1):
    """
    Wrap angles to [0, 2π] range.
    
    Parameters
    ----------
    v1 : array_like
        Vector of angles in radians
        
    Returns
    -------
    v2 : ndarray
        Vector of normalized angles in radians [0, 2π]
    """
    i = v1 > 0
    v1 = np.mod(v1, TWO_PI)
    v2 = v1.copy()
    v2[(v1 == 0) & i] = TWO_PI
    return v2


@njit(cache=True, fastmath=True)
def wrapToPi(v1):
    """
    Wrap angles to [-π, π] range.
    
    Parameters
    ----------
    v1 : array_like
        Vector of angles in radians
        
    Returns
    -------
    v2 : ndarray
        Vector of normalized angles in radians [-π, π]
    """
    v2 = v1.copy()
    i = (v1 < -np.pi) | (np.pi < v1)
    if np.any(i):
        v2[i] = wrapTo2Pi(v1[i] + np.pi) - np.pi
    return v2


@njit(cache=True, fastmath=True)
def wrapEulerAngles(e):
    """
    Wrap euler angles to proper ranges.
    
    This function ensures:
    - Roll and yaw are wrapped to [-π, π]
    - Pitch is kept within [-π/2, π/2], with appropriate adjustments
      to roll and yaw when pitch exceeds these bounds
    
    Parameters
    ----------
    e : array_like, shape (3,)
        Vector of euler angles (roll-pitch-yaw) in radians
        
    Returns
    -------
    e2 : ndarray, shape (3,)
        Vector of normalized euler angles (roll-pitch-yaw) in radians
    """
    e2 = e.copy()
    
    if e2[1] > HALF_PI:
        e2[1] = np.pi - e2[1]
        e2[0] = wrapToPi(np.array([e2[0] + np.pi]))[0]
        e2[2] = wrapToPi(np.array([e2[2] + np.pi]))[0]
    elif e2[1] < -HALF_PI:
        e2[1] = -np.pi - e2[1]
        e2[0] = wrapToPi(np.array([e2[0] + np.pi]))[0]
        e2[2] = wrapToPi(np.array([e2[2] + np.pi]))[0]
    else:
        e2[0] = wrapToPi(np.array([e2[0]]))[0]
        e2[2] = wrapToPi(np.array([e2[2]]))[0]
    
    return e2