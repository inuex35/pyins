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

"""Coordinate transformation and rotation utilities

This module provides comprehensive coordinate transformations including:
- Basic transforms (ECEF, LLH, ENU, NED)
- ECI (Earth-Centered Inertial) transforms
- AER (Azimuth-Elevation-Range) transforms
- DCM (Direction Cosine Matrix) for coordinate frames
- General rotation utilities

For attitude-specific transformations with Numba optimization,
use pyins.attitude module.
"""

# Basic coordinate transforms
# AER transforms
from .aer_transforms import aer2enu, aer2ned, ecef2aer, eci2aer, enu2aer, lla2aer, ned2aer

# Object-oriented coordinate transformer
from .coordinate_transformer import CoordinateTransformer

# DCM for coordinate frames
from .dcm import (
    ecef2eci_dcm,
    ecef2enu_dcm,
    ecef2ned_dcm,
    eci2ecef_dcm,
    eci2enu_dcm,
    eci2ned_dcm,
    enu2ecef_dcm,
    enu2eci_dcm,
    ned2ecef_dcm,
    ned2eci_dcm,
)

# ECI transforms
from .eci_transforms import ecef2eci, eci2ecef, eci2enu, eci2lla, eci2ned, enu2eci, lla2eci, ned2eci

# Geodetic utilities
from .geodetic import *

# Height conversion utilities
from .height_conversion import (
    HeightSystem,
    convert_height,
    ellipsoidal_to_orthometric,
    get_geoid_height,
    orthometric_to_ellipsoidal,
)

# Rotation utilities (general purpose)
from .rotation import (
    RotationIntegrator,
    axis_angle2dcm,
    dcm2axis_angle,
    dcm2euler,
    dcm2quaternion,
    euler2dcm,
    quaternion2dcm,
    quaternion_conjugate,
    quaternion_inverse,
    quaternion_multiply,
    rotate_vector,
)
from .transforms import (
    covecef2enu,
    covenu2ecef,
    ecef2enu,
    ecef2llh,
    ecef2ned,
    enu2ecef,
    enu2lla,
    lla2enu,
    lla2ned,
    llh2ecef,
    ned2ecef,
    ned2lla,
    xyz2enu_jacobian,
)
