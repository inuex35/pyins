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

"""Core GNSS Processing Module.

This module provides fundamental components for GNSS data processing including:

- **Constants and Parameters**: GNSS system constants, frequencies, physical parameters,
  satellite system definitions, and error model parameters
- **Data Structures**: Core data types for observations, ephemeris data, navigation
  messages, and positioning solutions
- **Time Systems**: Comprehensive time handling for GPS, Galileo, BeiDou, GLONASS,
  and other time systems with proper leap second handling
- **Statistical Parameters**: RTK/PPP processing parameters including error models,
  process noise, ambiguity resolution thresholds, and innovation validation
- **Satellite Numbering**: Unified satellite numbering system for consistent
  satellite identification across all GNSS constellations

The module serves as the foundation for all GNSS processing algorithms in pyins,
providing consistent interfaces and data structures throughout the codebase.

Key Features:
- Multi-constellation support (GPS, GLONASS, Galileo, BeiDou, QZSS, SBAS, IRNSS)
- Comprehensive time system conversions
- RTKLIB-compatible statistical parameters
- Type-safe data structures with validation
- Unified satellite identification system

Example Usage:
    >>> from pyins.core import *
    >>>
    >>> # Create an observation
    >>> obs = Observation(time=123456.0, sat=1, system=SYS_GPS)
    >>> obs.L[0] = 12345678.9  # L1 carrier phase in cycles
    >>> obs.P[0] = 23456789.1  # L1 pseudorange in meters
    >>>
    >>> # Create GNSS time
    >>> gps_time = GNSSTime(2200, 432000.0, 'GPS')
    >>>
    >>> # Compute observation variance
    >>> var = compute_obs_variance(np.pi/4, is_phase=True)
"""

from .constants import *
from .data_structures import *
from .time import *
