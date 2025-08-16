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
Attitude conversion from euler angles.

This module provides functions for converting from euler angles to other attitude 
representations. All rotations assume right-hand coordinate frames with euler angles 
in the order 'roll-pitch-yaw' and DCMs with the order of 'ZYX'.

References:
    Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems
    - (2013) Paul D. Groves

Author: Migrated from navlib by Daniel Sturdivant <sturdivant20@gmail.com>
Date: December 2023
"""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def euler2dcm(e):
    """
    Convert euler angles (roll-pitch-yaw) to corresponding 'ZYX' DCM.
    
    The rotation sequence is:
    1. Yaw (ψ) about z-axis
    2. Pitch (θ) about y-axis
    3. Roll (φ) about x-axis
    
    Parameters
    ----------
    e : array_like, shape (3,)
        RPY euler angles [roll, pitch, yaw] in radians
        
    Returns
    -------
    C : ndarray, shape (3, 3)
        'XYZ' direction cosine matrix
    """
    sinP, sinT, sinS = np.sin(e)
    cosP, cosT, cosS = np.cos(e)
    C = np.array([[cosT*cosS, cosT*sinS, -sinT], 
                  [sinP*sinT*cosS - cosP*sinS, sinP*sinT*sinS + cosP*cosS, cosT*sinP], 
                  [sinT*cosP*cosS + sinS*sinP, sinT*cosP*sinS - cosS*sinP, cosT*cosP]],
                 dtype=np.double)
    return C


@njit(cache=True, fastmath=True)
def euler2quat(e):
    """
    Convert euler angles (roll-pitch-yaw) to corresponding quaternion.
    
    The quaternion is computed using the half-angle formulas for the
    ZYX rotation sequence.
    
    Parameters
    ----------
    e : array_like, shape (3,)
        RPY euler angles [roll, pitch, yaw] in radians
        
    Returns
    -------
    q : ndarray, shape (4,)
        Quaternion [w, x, y, z]
    """
    sinX, sinY, sinZ = np.sin(e * 0.5)
    cosX, cosY, cosZ = np.cos(e * 0.5)
    q = np.array([cosZ*cosY*cosX + sinZ*sinY*sinX,
                  cosZ*cosY*sinX - sinZ*sinY*cosX,
                  cosZ*sinY*cosX + sinZ*cosY*sinX,
                  sinZ*cosY*cosX - cosZ*sinY*sinX], 
                 dtype=np.double)
    return q


@njit(cache=True, fastmath=True)
def rot_x(phi):
    """
    Convert single euler angle to corresponding 'X' DCM.
    
    Creates a rotation matrix for rotation about the x-axis.
    
    Parameters
    ----------
    phi : float
        Euler angle in radians
        
    Returns
    -------
    R : ndarray, shape (3, 3)
        Direction cosine matrix for x-axis rotation
    """
    sinP = np.sin(phi)
    cosP = np.cos(phi)
    R = np.array([[1.0,  0.0,   0.0], 
                  [0.0, cosP, -sinP], 
                  [0.0, sinP,  cosP]], 
                 dtype=np.double)
    return R


@njit(cache=True, fastmath=True)
def rot_y(theta):
    """
    Convert single euler angle to corresponding 'Y' DCM.
    
    Creates a rotation matrix for rotation about the y-axis.
    
    Parameters
    ----------
    theta : float
        Euler angle in radians
        
    Returns
    -------
    R : ndarray, shape (3, 3)
        Direction cosine matrix for y-axis rotation
    """
    sinT = np.sin(theta)
    cosT = np.cos(theta)
    R = np.array([[ cosT, 0.0, sinT], 
                  [  0.0, 1.0,  0.0], 
                  [-sinT, 0.0, cosT]], 
                 dtype=np.double)
    return R


@njit(cache=True, fastmath=True)
def rot_z(psi):
    """
    Convert single euler angle to corresponding 'Z' DCM.
    
    Creates a rotation matrix for rotation about the z-axis.
    
    Parameters
    ----------
    psi : float
        Euler angle in radians
        
    Returns
    -------
    R : ndarray, shape (3, 3)
        Direction cosine matrix for z-axis rotation
    """
    sinS = np.sin(psi)
    cosS = np.cos(psi)
    R = np.array([[cosS, -sinS, 0.0], 
                  [sinS,  cosS, 0.0], 
                  [ 0.0,   0.0, 1.0]], 
                 dtype=np.double)
    return R