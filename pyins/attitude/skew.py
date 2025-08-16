"""
Attitude skew symmetric form utilities.

This module provides functions for converting between vectors and their skew symmetric forms.
All rotations assume right-hand coordinate frames with euler angles in the order 
'roll-pitch-yaw' and DCMs with the order of 'ZYX'.

References:
    Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems
    - (2013) Paul D. Groves

Author: Migrated from navlib by Daniel Sturdivant <sturdivant20@gmail.com>
Date: December 2023
"""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def skew(v):
    """
    Convert vector into its skew symmetric form.
    
    The skew symmetric matrix of a vector v = [v1, v2, v3] is:
    [  0  -v3   v2 ]
    [ v3    0  -v1 ]
    [-v2   v1    0 ]
    
    Parameters
    ----------
    v : array_like, shape (3,)
        Input vector
        
    Returns
    -------
    M : ndarray, shape (3, 3)
        Skew symmetric form of input vector
    """
    M = np.array([[  0.0, -v[2],  v[1]], 
                  [ v[2],   0.0, -v[0]], 
                  [-v[1],  v[0],   0.0]], 
                 dtype=np.double)
    return M


@njit(cache=True, fastmath=True)
def deskew(M):
    """
    Convert skew symmetric form into its respective vector.
    
    Extracts the vector v from a skew symmetric matrix M where:
    v1 = M[2,1], v2 = M[0,2], v3 = M[1,0]
    
    Parameters
    ----------
    M : array_like, shape (3, 3)
        Skew symmetric form of vector
        
    Returns
    -------
    v : ndarray, shape (3,)
        Output vector
    """
    v = np.array([M[2,1], M[0,2], M[1,0]], dtype=np.double)
    return v