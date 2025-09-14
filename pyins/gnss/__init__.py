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

"""GNSS processing module.

This module provides functionality for Global Navigation Satellite System (GNSS) processing,
including satellite position computation, single point positioning (SPP), and various GNSS-related
utilities and algorithms.

The module supports multiple GNSS constellations:
- GPS (Global Positioning System)
- GLONASS (Globalnaya Navigazionnaya Sputnikovaya Sistema)
- Galileo
- BeiDou
- QZSS (Quasi-Zenith Satellite System)
- IRNSS (Indian Regional Navigation Satellite System)
- SBAS (Satellite-Based Augmentation Systems)

Key Components:
- Single Point Positioning (SPP)
- Satellite position and velocity computation
- Ephemeris handling (broadcast and precise SP3)
- Atmospheric corrections
- GNSS signal processing utilities
- Frequency and wavelength management
- Receiver Autonomous Integrity Monitoring (RAIM)
- Double difference processing for RTK applications

Examples:
    Basic usage for single point positioning:

    >>> from pyins.gnss import single_point_positioning, compute_satellite_info
    >>> # Process GNSS observations for positioning
    >>> pos, vel = single_point_positioning(obs_data, nav_data)
    >>> # Compute satellite positions and clock corrections
    >>> sat_info = compute_satellite_info(obs_data, nav_data, time)
"""

from .satellite_positions import compute_satellite_info
from .spp import single_point_positioning

__all__ = ['single_point_positioning', 'compute_satellite_info']
