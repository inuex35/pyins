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
Satellite computation module for GNSS positioning systems.

This module provides comprehensive satellite position and clock computation
capabilities for multiple Global Navigation Satellite Systems (GNSS) including
GPS, GLONASS, Galileo, BeiDou, and QZSS. It implements standard algorithms
from satellite system Interface Control Documents and follows established
practices from the GNSS community.

Modules
-------
clock : module
    Satellite clock bias and drift computation with TGD corrections
ephemeris : module
    Ephemeris selection, validation, and management utilities
satellite_position : module
    Satellite position and velocity computation from broadcast ephemeris

Key Features
------------
- Multi-GNSS support (GPS, GLONASS, Galileo, BeiDou, QZSS, SBAS)
- High-precision position computation using Keplerian orbital mechanics
- GLONASS-specific numerical integration with PZ-90 coordinate system
- Comprehensive clock correction models including relativistic effects
- Time Group Delay (TGD) corrections for multi-frequency observations
- Ephemeris validation and selection algorithms
- Efficient caching mechanisms for performance optimization
- System-specific accuracy models and validity periods

Usage Examples
--------------
Basic satellite position computation:

    >>> from pyins.satellite import compute_satellite_position
    >>> pos, clk, var = compute_satellite_position(gps_eph, gps_time)
    >>> print(f"Satellite position: {pos} m")

GLONASS satellite handling:

    >>> from pyins.satellite import compute_glonass_position
    >>> pos, clk, var = compute_glonass_position(glo_eph, gps_time)

Ephemeris management:

    >>> from pyins.satellite import EphemerisManager, select_ephemeris
    >>> manager = EphemerisManager()
    >>> manager.add_ephemeris(new_eph)
    >>> best_eph = manager.get_ephemeris(sat=1, time=current_time)

Clock corrections with caching:

    >>> from pyins.satellite import SatelliteClockManager
    >>> clock_mgr = SatelliteClockManager()
    >>> dts, ddts = clock_mgr.get_clock_correction(eph, time)

Standards Compliance
-------------------
The implementation follows these standards and references:
- GPS Interface Specification IS-GPS-200
- GLONASS Interface Control Document
- Galileo Open Service Signal In Space ICD
- BeiDou Navigation Satellite System Signal In Space ICD
- RTCM Standards for differential GNSS
- RTKLIB algorithms for compatibility

Notes
-----
Time systems: All functions expect GPS time (GPST) as input. Proper time
system conversions should be applied when working with other time references.

Coordinate systems: Positions are computed in Earth-Centered Earth-Fixed
(ECEF) coordinates. For GLONASS, the PZ-90.02 system is used, which is
very close to WGS-84.

Performance: The module includes caching mechanisms and efficient algorithms
to support real-time applications and batch processing scenarios.
"""

from .clock import *
from .ephemeris import *
from .satellite_position import *
