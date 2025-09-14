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
PyINS Utils Module

This module provides utility functions and classes for GNSS (Global Navigation Satellite System)
processing and analysis, including:

- Satellite system identification and frequency information
- LAMBDA method implementation for ambiguity resolution
- Processing options and satellite exclusion logic
- Utility functions for satellite constellation management

The module supports multiple satellite systems including GPS, GLONASS, Galileo, BeiDou,
QZSS, and SBAS.

Examples
--------
>>> from pyins.utils import get_satellite_system, SatelliteSystem
>>> system = get_satellite_system(15)  # GPS satellite
>>> print(system)
SatelliteSystem.GPS

>>> from pyins.utils import lambda_reduction
>>> import numpy as np
>>> Q = np.array([[4.0, 2.0], [2.0, 3.0]])
>>> Z, L, D = lambda_reduction(Q)
"""

from .ambiguity import *
from .satellite_utils import *
