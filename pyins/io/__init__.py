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

"""I/O module for pyins - comprehensive file format support for GNSS and IMU data.

This module provides comprehensive input/output functionality for various
file formats commonly used in GNSS-INS processing. It includes readers for
RINEX files, IMU data, and velocity measurements with proper time system
handling and data validation.

Classes
-------
RinexObsReader
    RINEX observation file reader with RTKLIB-compatible signal processing
RinexNavReader
    RINEX navigation file reader for satellite ephemeris data
IMUReader
    IMU data reader supporting CSV and text formats with time system conversion
IMUPreprocessor
    IMU data preprocessing utilities including bias removal and synchronization
VelocityReader
    Velocity data reader with coordinate system conversion

Functions
---------
load_imu_data
    Convenience function to load IMU data from files
preprocess_imu_data
    Convenience function to preprocess IMU data
sync_gnss_imu
    Convenience function to synchronize GNSS and IMU data

Supported File Formats
----------------------
RINEX:
    - RINEX 3 observation files (.obs, .rnx)
    - RINEX 3 navigation files (.nav, .rnx)
    - Multiple GNSS constellations (GPS, Galileo, BeiDou, GLONASS, QZSS, IRNSS)

IMU Data:
    - CSV format with flexible column naming
    - Text format with space-separated values
    - Time system conversion (GPS, UNIX time)
    - Automatic bias estimation and removal

Velocity Data:
    - CSV format with coordinate system detection
    - NED, ENU, and body frame support
    - Temporal interpolation capabilities

Features
--------
- Automatic time system conversions
- RTKLIB-compatible signal priority handling
- Robust error handling and data validation
- Efficient data caching and processing
- Comprehensive logging for debugging"""

from .imu_reader import (
    IMUPreprocessor,
    IMUReader,
    load_imu_data,
    preprocess_imu_data,
    sync_gnss_imu,
)
from .rinex import RinexNavReader, RinexObsReader
from .velocity_reader import VelocityReader

__all__ = [
    'RinexObsReader', 'RinexNavReader',
    'IMUReader', 'IMUPreprocessor', 'VelocityReader',
    'load_imu_data', 'preprocess_imu_data', 'sync_gnss_imu'
]
