"""
Attitude module for coordinate transformations and rotations.

This module provides functions for converting between different attitude representations:
- Euler angles (roll-pitch-yaw)
- Direction Cosine Matrices (DCM)
- Quaternions
- Skew symmetric matrices

All rotations assume right-hand coordinate frames with euler angles in the order 
'roll-pitch-yaw' and DCMs with the order of 'ZYX'.

References:
    Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems
    - (2013) Paul D. Groves
"""

from .skew import skew, deskew
from .dcm import dcm2euler, dcm2quat
from .euler import euler2dcm, euler2quat, rot_x, rot_y, rot_z
from .quaternion import quat2euler, quat2dcm
from .wrap import wrapTo2Pi, wrapToPi, wrapEulerAngles

__all__ = [
    'skew', 'deskew',
    'dcm2euler', 'dcm2quat',
    'euler2dcm', 'euler2quat', 'rot_x', 'rot_y', 'rot_z',
    'quat2euler', 'quat2dcm',
    'wrapTo2Pi', 'wrapToPi', 'wrapEulerAngles'
]