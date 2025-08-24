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

from .dcm import dcm2euler, dcm2quat
from .euler import euler2dcm, euler2quat, rot_x, rot_y, rot_z
from .quaternion import quat2dcm, quat2euler
from .skew import deskew, skew
from .wrap import wrapEulerAngles, wrapTo2Pi, wrapToPi

__all__ = [
    'skew', 'deskew',
    'dcm2euler', 'dcm2quat',
    'euler2dcm', 'euler2quat', 'rot_x', 'rot_y', 'rot_z',
    'quat2euler', 'quat2dcm',
    'wrapTo2Pi', 'wrapToPi', 'wrapEulerAngles'
]