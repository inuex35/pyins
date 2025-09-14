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
Geometry utilities for GNSS processing.

This package provides geometric computation utilities essential for Global Navigation
Satellite System (GNSS) processing and inertial navigation applications. The geometry
module includes functions for satellite positioning, elevation angle calculations,
and coordinate frame transformations.

Key geometric computations include:
- Satellite elevation angle determination
- Line-of-sight vector calculations
- Coordinate frame transformations (ECEF to local frames)
- Geometric dilution of precision (GDOP) related computations

Modules
-------
elevation : module
    Satellite elevation angle computation utilities

Functions
---------
compute_elevation_angle : function
    Compute the elevation angle of a GNSS satellite as observed from a receiver

Notes
-----
Accurate geometric computations are fundamental to GNSS positioning accuracy.
The functions in this package handle the coordinate transformations and geometric
relationships necessary for precise positioning and navigation solutions.

Examples
--------
>>> from pyins.geometry import compute_elevation_angle
>>> import numpy as np
>>> sat_pos = np.array([20000000, 10000000, 15000000])  # ECEF [m]
>>> rcv_pos = np.array([4000000, 3000000, 5000000])     # ECEF [m]
>>> ref_llh = np.array([35.0, 139.0, 100.0])           # [deg, deg, m]
>>> elevation = compute_elevation_angle(sat_pos, rcv_pos, ref_llh)
"""

from .elevation import compute_elevation_angle

__all__ = ['compute_elevation_angle']
