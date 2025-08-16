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
from .transforms import (
    ecef2llh, llh2ecef,
    ecef2enu, enu2ecef,
    ecef2ned, ned2ecef,
    lla2ned, ned2lla,
    lla2enu, enu2lla,
    covecef2enu, covenu2ecef,
    xyz2enu_jacobian
)

# Geodetic utilities
from .geodetic import *

# Rotation utilities (general purpose)
from .rotation import (
    euler2dcm, dcm2euler,
    quaternion2dcm, dcm2quaternion,
    axis_angle2dcm, dcm2axis_angle,
    rotate_vector,
    quaternion_multiply, quaternion_conjugate, quaternion_inverse,
    RotationIntegrator
)

# ECI transforms
from .eci_transforms import (
    eci2ecef, ecef2eci,
    lla2eci, eci2lla,
    eci2ned, ned2eci,
    eci2enu, enu2eci
)

# AER transforms
from .aer_transforms import (
    enu2aer, ned2aer,
    aer2enu, aer2ned,
    lla2aer, ecef2aer, eci2aer
)

# DCM for coordinate frames
from .dcm import (
    eci2ecef_dcm, ecef2eci_dcm,
    ecef2ned_dcm, ned2ecef_dcm,
    ecef2enu_dcm, enu2ecef_dcm,
    eci2ned_dcm, ned2eci_dcm,
    eci2enu_dcm, enu2eci_dcm
)

# Height conversion utilities
from .height_conversion import (
    HeightSystem,
    ellipsoidal_to_orthometric,
    orthometric_to_ellipsoidal,
    get_geoid_height,
    convert_height
)

# Object-oriented coordinate transformer
from .coordinate_transformer import CoordinateTransformer