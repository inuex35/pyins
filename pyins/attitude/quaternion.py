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
Attitude conversion from quaternions.

This module provides functions for converting from quaternions to other attitude
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
def quat2euler(q):
    """
    Convert quaternion to corresponding euler angles (roll-pitch-yaw).
    
    The conversion uses the standard formulas for extracting euler angles
    from a quaternion in the ZYX rotation sequence.
    
    Parameters
    ----------
    q : array_like, shape (4,)
        Quaternion [w, x, y, z]
        
    Returns
    -------
    e : ndarray, shape (3,)
        RPY euler angles [roll, pitch, yaw] in radians
    """
    w, x, y, z = q
    e = np.array([np.arctan2(2*(w*x + y*z), (w*w - x*x - y*y + z*z)), 
                  np.arcsin(-2*(-w*y + x*z)), 
                  np.arctan2(2*(w*z + x*y), (w*w + x*x - y*y - z*z))], 
                 dtype=np.double)
    return e


@njit(cache=True, fastmath=True)
def quat2dcm(q):
    """
    Convert quaternion to corresponding 'XYZ' DCM.
    
    The direction cosine matrix is computed directly from the quaternion
    components using the standard conversion formulas.
    
    Parameters
    ----------
    q : array_like, shape (4,)
        Quaternion [w, x, y, z]
        
    Returns
    -------
    C : ndarray, shape (3, 3)
        'XYZ' direction cosine matrix
    """
    w, x, y, z = q
    C = np.array([[w*w + x*x - y*y - z*z,         2*(x*y - w*z),          2*(w*y + x*z)], 
                  [        2*(w*z + x*y), w*w - x*x + y*y - z*z,          2*(y*z - w*x)], 
                  [        2*(x*z - w*y),         2*(y*z + w*x),  w*w - x*x - y*y + z*z]], 
                 dtype=np.double)
    return C