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

"""Base sensor classes and interfaces"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ..core.data_structures import SensorType


@dataclass
class SensorConfig:
    """Sensor configuration parameters"""
    sensor_id: str
    sensor_type: SensorType
    sampling_rate: float  # Hz
    noise_params: dict[str, float] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)
    lever_arm: np.ndarray = field(default_factory=lambda: np.zeros(3))  # lever arm in body frame (m)
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))     # rotation from sensor to body frame

    def get_noise_std(self, param_name: str, default: float = 0.0) -> float:
        """Get noise standard deviation for a parameter.

        Parameters
        ----------
        param_name : str
            Name of the noise parameter to retrieve
        default : float, optional
            Default value to return if parameter is not found (default: 0.0)

        Returns
        -------
        float
            Noise standard deviation value for the specified parameter

        Examples
        --------
        >>> config = SensorConfig('imu1', SensorType.IMU, 100.0)
        >>> config.noise_params = {'gyro_noise': 0.001}
        >>> std = config.get_noise_std('gyro_noise')
        >>> print(std)
        0.001
        """
        return self.noise_params.get(param_name, default)


@dataclass
class SensorData:
    """Base sensor data container"""
    timestamp: float  # Unix timestamp or GNSS time
    sensor_id: str
    sensor_type: SensorType
    data: np.ndarray
    covariance: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """Check if sensor data is valid.

        Returns
        -------
        bool
            True if data exists and contains only finite values, False otherwise

        Notes
        -----
        This method checks that:
        - data attribute is not None
        - All data values are finite (not NaN or infinity)
        """
        return self.data is not None and np.all(np.isfinite(self.data))


class SensorInterface(ABC):
    """Abstract base class for sensor interfaces"""

    def __init__(self, config: SensorConfig):
        self.config = config
        self._is_initialized = False

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the sensor.

        Returns
        -------
        bool
            True if initialization was successful, False otherwise

        Notes
        -----
        This method must be implemented by all sensor subclasses.
        It should perform any necessary setup operations for the sensor.
        """
        pass

    @abstractmethod
    def read(self) -> Optional[SensorData]:
        """Read data from the sensor.

        Returns
        -------
        Optional[SensorData]
            Latest sensor measurement, or None if no data is available

        Notes
        -----
        This method must be implemented by all sensor subclasses.
        It should return the most recent sensor measurement.
        """
        pass

    @abstractmethod
    def calibrate(self, calibration_data: dict[str, Any]) -> bool:
        """Apply calibration to the sensor.

        Parameters
        ----------
        calibration_data : dict[str, Any]
            Dictionary containing calibration parameters specific to the sensor type

        Returns
        -------
        bool
            True if calibration was applied successfully, False otherwise

        Notes
        -----
        This method must be implemented by all sensor subclasses.
        The expected format of calibration_data depends on the sensor type.
        """
        pass

    def get_lever_arm(self) -> np.ndarray:
        """Get lever arm vector from sensor to body frame.

        Returns
        -------
        np.ndarray
            3D lever arm vector in body frame coordinates (meters)

        Notes
        -----
        The lever arm represents the translation from the sensor location
        to the body frame origin.
        """
        return self.config.lever_arm

    def get_rotation(self) -> np.ndarray:
        """Get rotation matrix from sensor to body frame.

        Returns
        -------
        np.ndarray
            3x3 rotation matrix for transforming sensor measurements to body frame

        Notes
        -----
        This matrix transforms vectors from sensor frame to body frame:
        vector_body = R @ vector_sensor
        """
        return self.config.rotation

    def transform_to_body(self, sensor_data: np.ndarray) -> np.ndarray:
        """Transform sensor data from sensor frame to body frame.

        Parameters
        ----------
        sensor_data : np.ndarray
            Sensor measurement in sensor frame coordinates

        Returns
        -------
        np.ndarray
            Sensor measurement transformed to body frame coordinates

        Notes
        -----
        This method applies the rotation matrix stored in the sensor configuration
        to transform measurements from sensor frame to body frame.
        """
        return self.config.rotation @ sensor_data

    @property
    def is_initialized(self) -> bool:
        """
        Check if sensor has been successfully initialized.

        Returns:
        --------
        bool
            True if sensor initialization was successful, False otherwise

        Notes:
            This property should be checked before attempting to read data
            from the sensor. Uninitialized sensors may not provide valid data.
        """
        return self._is_initialized
