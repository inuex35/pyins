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
Sensor components for the pyins navigation system.

This module provides sensor interfaces, data structures, and processing utilities
for inertial navigation systems. It includes support for:

- IMU sensors with preintegration capabilities
- Lever arm compensation for multi-sensor systems
- Robot-specific lever arm management
- Base sensor interfaces and data structures

Classes:
    SensorInterface: Abstract base class for all sensors
    SensorData: Base data container for sensor measurements
    SensorConfig: Configuration parameters for sensors
    IMUSensor: IMU sensor implementation with bias correction
    LeverArm: Lever arm compensation for position/velocity/acceleration
    RobotLeverArm: Management system for robot component configurations

Examples:
    Basic IMU sensor setup:

    >>> from pyins.sensors import IMUConfig, IMUSensor
    >>> config = IMUConfig('imu1', SensorType.IMU, 100.0)
    >>> imu = IMUSensor(config)
    >>> imu.initialize()

    Lever arm compensation:

    >>> from pyins.sensors import LeverArm
    >>> lever = LeverArm(np.array([0.1, 0.0, 0.05]))  # 10cm forward, 5cm up
    >>> compensated_pos = lever.compensate_position(body_pos, rotation_matrix)
"""

from .imu import *
from .lever_arm import *
from .sensor_base import *
