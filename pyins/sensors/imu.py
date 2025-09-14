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

"""IMU sensor data structures and processing"""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .sensor_base import SensorConfig, SensorData, SensorInterface, SensorType


@dataclass
class IMUData(SensorData):
    """
    Specialized data container for IMU measurements.

    This class extends SensorData to provide IMU-specific functionality
    including separate access to acceleration and angular velocity components,
    and gravity compensation methods.

    Attributes:
        sensor_type (SensorType): Always set to SensorType.IMU
        data (np.ndarray): 6D array [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]

    Notes:
        The data array must be exactly 6 elements long, containing 3 acceleration
        values (m/s²) followed by 3 angular velocity values (rad/s).

    Examples:
        Creating IMU data:

        >>> imu_data = IMUData(
        ...     timestamp=time.time(),
        ...     sensor_id='imu1',
        ...     data=np.array([0.1, 0.2, 9.8, 0.01, 0.02, 0.03])
        ... )
        >>> print(imu_data.acceleration)
        [0.1 0.2 9.8]
    """
    sensor_type: SensorType = field(default=SensorType.IMU, init=False)

    def __post_init__(self):
        """
        Validate IMU data format after initialization.

        Raises:
            ValueError: If data array is not exactly 6 elements
        """
        self.sensor_type = SensorType.IMU
        if self.data.shape != (6,):
            raise ValueError("IMU data must be 6D [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]")

    @property
    def acceleration(self) -> np.ndarray:
        """
        Get acceleration measurements from IMU data.

        Returns:
        --------
        np.ndarray
            3D acceleration vector [ax, ay, az] in m/s²

        Notes:
            Returns the first 3 elements of the data array, representing
            acceleration measurements along the x, y, and z axes.
        """
        return self.data[:3]

    @property
    def angular_velocity(self) -> np.ndarray:
        """
        Get angular velocity measurements from IMU data.

        Returns:
        --------
        np.ndarray
            3D angular velocity vector [wx, wy, wz] in rad/s

        Notes:
            Returns the last 3 elements of the data array, representing
            angular velocity measurements about the x, y, and z axes.
        """
        return self.data[3:6]

    def remove_gravity(self, gravity_vector: np.ndarray) -> np.ndarray:
        """
        Remove gravity from acceleration measurements.

        Parameters:
        -----------
        gravity_vector : np.ndarray
            3D gravity vector in the same frame as the acceleration (typically [0, 0, 9.81])

        Returns:
        --------
        np.ndarray
            3D acceleration vector with gravity removed (m/s²)

        Notes:
            This method subtracts the provided gravity vector from the measured
            acceleration to obtain the specific force (acceleration due to motion).

        Examples:
            >>> gravity = np.array([0, 0, 9.81])  # Standard gravity
            >>> specific_force = imu_data.remove_gravity(gravity)
        """
        return self.acceleration - gravity_vector


@dataclass
class IMUConfig(SensorConfig):
    """
    Configuration parameters specific to IMU sensors.

    This class extends SensorConfig with IMU-specific default noise parameters.
    It automatically sets appropriate default values for accelerometer and
    gyroscope noise characteristics.

    Attributes:
        sensor_type (SensorType): Automatically set to SensorType.IMU
        noise_params (dict): Default IMU noise parameters including:
            - acc_noise: Accelerometer noise (m/s²)
            - gyro_noise: Gyroscope noise (rad/s)
            - acc_bias_walk: Accelerometer bias random walk (m/s³)
            - gyro_bias_walk: Gyroscope bias random walk (rad/s²)

    Examples:
        Creating IMU configuration:

        >>> config = IMUConfig(
        ...     sensor_id='imu_main',
        ...     sampling_rate=100.0
        ... )
        >>> print(config.noise_params['acc_noise'])
        0.01
    """
    def __post_init__(self):
        """
        Initialize IMU configuration with default noise parameters.

        Sets the sensor type to IMU and provides default noise parameters
        if they are not already specified in the configuration.
        """
        self.sensor_type = SensorType.IMU

        # Default IMU noise parameters if not provided
        default_noise = {
            'acc_noise': 0.01,      # m/s^2
            'gyro_noise': 0.001,    # rad/s
            'acc_bias_walk': 0.0001,  # m/s^3
            'gyro_bias_walk': 0.00001 # rad/s^2
        }

        for key, value in default_noise.items():
            if key not in self.noise_params:
                self.noise_params[key] = value


@dataclass
class IMUBias:
    """
    IMU bias state for sensor calibration and error modeling.

    This class stores and manages accelerometer and gyroscope bias estimates,
    providing methods for bias correction of IMU measurements.

    Attributes:
        acc_bias (np.ndarray): 3D accelerometer bias vector (m/s²)
        gyro_bias (np.ndarray): 3D gyroscope bias vector (rad/s)

    Notes:
        Biases represent systematic errors in IMU measurements that should be
        subtracted from raw readings. These can be estimated through calibration
        procedures or online estimation algorithms.

    Examples:
        Creating and applying bias correction:

        >>> bias = IMUBias(
        ...     acc_bias=np.array([0.01, -0.02, 0.05]),
        ...     gyro_bias=np.array([0.001, 0.002, -0.001])
        ... )
        >>> corrected_data = bias.correct_measurement(raw_imu_data)
    """
    acc_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))   # accelerometer bias (m/s^2)
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))  # gyroscope bias (rad/s)

    def correct_measurement(self, imu_data: IMUData) -> IMUData:
        """
        Apply bias correction to IMU measurements.

        Parameters:
        -----------
        imu_data : IMUData
            Raw IMU measurement data to be corrected

        Returns:
        --------
        IMUData
            New IMUData object with bias-corrected measurements

        Notes:
            This method subtracts the stored bias values from the raw measurements
            to produce corrected data. The original data object is not modified.

        Examples:
            >>> corrected = bias.correct_measurement(raw_data)
            >>> print(np.allclose(corrected.acceleration,
            ...                   raw_data.acceleration - bias.acc_bias))
            True
        """
        corrected_data = imu_data.data.copy()
        corrected_data[:3] -= self.acc_bias
        corrected_data[3:6] -= self.gyro_bias

        return IMUData(
            timestamp=imu_data.timestamp,
            sensor_id=imu_data.sensor_id,
            data=corrected_data,
            covariance=imu_data.covariance,
            metadata=imu_data.metadata
        )


@dataclass
class IMUPreintegration:
    """
    IMU preintegration for factor graph-based optimization.

    This class implements IMU preintegration theory for use in factor graph
    SLAM/VIO systems. It integrates IMU measurements between keyframes and
    maintains Jacobians for bias correction.

    Attributes:
        dt (float): Total integration time interval (seconds)
        dP (np.ndarray): Preintegrated position displacement (meters)
        dV (np.ndarray): Preintegrated velocity change (m/s)
        dR (np.ndarray): Preintegrated rotation matrix
        dP_dba (np.ndarray): Jacobian of position w.r.t. accelerometer bias
        dP_dbg (np.ndarray): Jacobian of position w.r.t. gyroscope bias
        dV_dba (np.ndarray): Jacobian of velocity w.r.t. accelerometer bias
        dV_dbg (np.ndarray): Jacobian of velocity w.r.t. gyroscope bias
        dR_dbg (np.ndarray): Jacobian of rotation w.r.t. gyroscope bias
        cov (np.ndarray): Preintegrated measurement covariance matrix
        acc_noise (float): Accelerometer noise standard deviation
        gyro_noise (float): Gyroscope noise standard deviation

    Notes:
        This implementation provides the core preintegration functionality
        needed for IMU factors in graph-based SLAM systems. The Jacobians
        enable efficient bias correction without re-integration.

    References:
        Forster et al. "IMU Preintegration on Manifold for Efficient
        Visual-Inertial Maximum-a-Posteriori Estimation" RSS 2015
    """
    dt: float = 0.0                                           # integration time interval
    dP: np.ndarray = field(default_factory=lambda: np.zeros(3))     # preintegrated position
    dV: np.ndarray = field(default_factory=lambda: np.zeros(3))     # preintegrated velocity
    dR: np.ndarray = field(default_factory=lambda: np.eye(3))       # preintegrated rotation

    # Jacobians with respect to bias
    dP_dba: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    dP_dbg: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    dV_dba: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    dV_dbg: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    dR_dbg: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))

    # Covariance
    cov: np.ndarray = field(default_factory=lambda: np.zeros((9, 9)))

    # Noise parameters
    acc_noise: float = 0.01
    gyro_noise: float = 0.001

    def integrate(self, acc: np.ndarray, gyro: np.ndarray, dt: float):
        """
        Integrate a single IMU measurement into the preintegration.

        Parameters:
        -----------
        acc : np.ndarray
            3D acceleration measurement (m/s²) in body frame
        gyro : np.ndarray
            3D angular velocity measurement (rad/s) in body frame
        dt : float
            Time step for this measurement (seconds)

        Notes:
            This method updates the preintegrated values (dP, dV, dR) and their
            associated Jacobians. It uses Rodrigues' formula for rotation integration
            and maintains numerical stability for small rotation angles.

        Examples:
            >>> preint = IMUPreintegration()
            >>> preint.integrate(acc_measurement, gyro_measurement, 0.01)
        """
        # Store previous values
        dP_prev = self.dP.copy()
        dV_prev = self.dV.copy()
        dR_prev = self.dR.copy()

        # Compute rotation increment
        theta = gyro * dt
        theta_norm = np.linalg.norm(theta)

        if theta_norm > 1e-8:
            axis = theta / theta_norm
            K = np.array([[0, -axis[2], axis[1]],
                         [axis[2], 0, -axis[0]],
                         [-axis[1], axis[0], 0]])
            dR_inc = np.eye(3) + np.sin(theta_norm) * K + (1 - np.cos(theta_norm)) * K @ K
        else:
            # Small angle approximation
            K = np.array([[0, -theta[2], theta[1]],
                         [theta[2], 0, -theta[0]],
                         [-theta[1], theta[0], 0]])
            dR_inc = np.eye(3) + K

        # Update preintegrated values
        self.dR = dR_prev @ dR_inc
        self.dV = dV_prev + dR_prev @ acc * dt
        self.dP = dP_prev + dV_prev * dt + 0.5 * dR_prev @ acc * dt * dt

        # Update Jacobians (simplified - full implementation would be more complex)
        np.eye(3)
        self.dP_dba += self.dV_dba * dt - 0.5 * dR_prev * dt * dt
        self.dP_dbg += self.dV_dbg * dt
        self.dV_dba -= dR_prev * dt

        # Update time
        self.dt += dt

        # Update covariance (simplified)
        np.diag([self.acc_noise**2] * 3 + [self.gyro_noise**2] * 3) * dt
        # Full covariance propagation would go here

    def predict(self, pos_i: np.ndarray, vel_i: np.ndarray, rot_i: np.ndarray,
                bias: IMUBias, gravity: np.ndarray) -> tuple:
        """
        Predict state at time j given state at time i using preintegration.

        Parameters:
        -----------
        pos_i : np.ndarray
            Position at time i in navigation frame (3x1)
        vel_i : np.ndarray
            Velocity at time i in navigation frame (3x1)
        rot_i : np.ndarray
            Rotation matrix from body to navigation frame at time i (3x3)
        bias : IMUBias
            Current bias estimates for correction
        gravity : np.ndarray
            Gravity vector in navigation frame (3x1)

        Returns:
        --------
        tuple
            Predicted (position_j, velocity_j, rotation_j) at time j

        Notes:
            This method applies bias correction to the preintegrated values
            before using them to predict the state at the next keyframe.
            The prediction incorporates gravity effects.

        Examples:
            >>> pos_j, vel_j, rot_j = preint.predict(pos_i, vel_i, rot_i, bias, gravity)
        """
        # Apply bias correction to preintegrated values
        dP_corrected = self.dP + self.dP_dba @ bias.acc_bias + self.dP_dbg @ bias.gyro_bias
        dV_corrected = self.dV + self.dV_dba @ bias.acc_bias + self.dV_dbg @ bias.gyro_bias

        # TODO: Implement dR correction for gyro bias
        dR_corrected = self.dR

        # Predict next state
        pos_j = pos_i + vel_i * self.dt + 0.5 * gravity * self.dt * self.dt + rot_i @ dP_corrected
        vel_j = vel_i + gravity * self.dt + rot_i @ dV_corrected
        rot_j = rot_i @ dR_corrected

        return pos_j, vel_j, rot_j


def skew(v: np.ndarray) -> np.ndarray:
    """
    Create skew-symmetric matrix from a 3D vector.

    The skew-symmetric matrix enables representation of the cross product
    operation as matrix multiplication: skew(v) @ u = v × u

    Parameters:
    -----------
    v : np.ndarray
        3D input vector [v1, v2, v3]

    Returns:
    --------
    np.ndarray
        3x3 skew-symmetric matrix

    Notes:
        For vector v = [v1, v2, v3], the skew-symmetric matrix is:
        [[  0, -v3,  v2],
         [ v3,   0, -v1],
         [-v2,  v1,   0]]

    Examples:
        >>> omega = np.array([0.1, 0.2, 0.3])
        >>> omega_skew = skew(omega)
        >>> # Cross product: omega × v = omega_skew @ v
    """
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def exp_so3(v: np.ndarray) -> np.ndarray:
    """
    Exponential map from so(3) to SO(3) using Rodrigues' formula.

    This function converts a rotation vector (axis-angle representation)
    to a rotation matrix using the matrix exponential.

    Parameters:
    -----------
    v : np.ndarray
        3D rotation vector where ||v|| is the rotation angle (radians)
        and v/||v|| is the rotation axis

    Returns:
    --------
    np.ndarray
        3x3 rotation matrix in SO(3)

    Notes:
        Uses Rodrigues' formula: R = I + sin(θ)K + (1-cos(θ))K²
        where K is the skew-symmetric matrix of the unit axis and θ is the angle.
        For small angles, uses first-order approximation for numerical stability.

    Examples:
        >>> angle_axis = np.array([0, 0, np.pi/4])  # 45° rotation about z-axis
        >>> R = exp_so3(angle_axis)
        >>> print(R[0:2, 0:2])  # Should be 2D rotation matrix
    """
    theta = np.linalg.norm(v)
    if theta < 1e-8:
        return np.eye(3) + skew(v)

    axis = v / theta
    K = skew(axis)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


class IMUSensor(SensorInterface):
    """
    IMU sensor interface implementation.

    This class provides a complete implementation of the SensorInterface
    for IMU devices, including bias management, preintegration capabilities,
    and measurement processing.

    Attributes:
        config (IMUConfig): IMU-specific configuration parameters
        bias (IMUBias): Current bias estimates for the IMU
        preintegration (Optional[IMUPreintegration]): Current preintegration object

    Examples:
        Setting up an IMU sensor:

        >>> config = IMUConfig('imu1', SensorType.IMU, 100.0)
        >>> imu = IMUSensor(config)
        >>> imu.initialize()
        >>> imu.start_preintegration()
    """

    def __init__(self, config: IMUConfig):
        """
        Initialize IMU sensor with configuration.

        Parameters:
        -----------
        config : IMUConfig
            IMU-specific configuration parameters
        """
        super().__init__(config)
        self.bias = IMUBias()
        self.preintegration = None

    def initialize(self) -> bool:
        """
        Initialize the IMU sensor.

        Returns:
        --------
        bool
            True if initialization was successful, False otherwise

        Notes:
            This method should be implemented to perform any necessary
            hardware initialization, driver setup, or communication checks.
            Currently returns True as a placeholder.
        """
        # Initialization logic here
        self._is_initialized = True
        return True

    def read(self) -> Optional[IMUData]:
        """
        Read current IMU measurement data.

        Returns:
        --------
        Optional[IMUData]
            Latest IMU measurement, or None if no data is available

        Notes:
            This method should interface with the actual IMU hardware/driver
            to retrieve the latest acceleration and angular velocity measurements.
            Currently returns None as a placeholder implementation.
        """
        # Implementation would interface with actual IMU hardware/driver
        # This is a placeholder
        return None

    def calibrate(self, calibration_data: dict[str, Any]) -> bool:
        """
        Apply calibration parameters to the IMU sensor.

        Parameters:
        -----------
        calibration_data : dict[str, Any]
            Dictionary containing calibration parameters such as:
            - 'acc_bias': 3D accelerometer bias vector
            - 'gyro_bias': 3D gyroscope bias vector
            - Additional sensor-specific calibration parameters

        Returns:
        --------
        bool
            True if calibration was applied successfully, False otherwise

        Examples:
            >>> calibration = {
            ...     'acc_bias': [0.01, -0.02, 0.05],
            ...     'gyro_bias': [0.001, 0.002, -0.001]
            ... }
            >>> success = imu.calibrate(calibration)
        """
        if 'acc_bias' in calibration_data:
            self.bias.acc_bias = np.array(calibration_data['acc_bias'])
        if 'gyro_bias' in calibration_data:
            self.bias.gyro_bias = np.array(calibration_data['gyro_bias'])

        # Update configuration with calibration data
        self.config.calibration.update(calibration_data)
        return True

    def start_preintegration(self):
        """
        Start a new preintegration period.

        This method creates a new IMUPreintegration object with noise parameters
        from the sensor configuration. It should be called at the beginning of
        each keyframe interval in a SLAM/VIO system.

        Notes:
            Any existing preintegration data is discarded when this method is called.
            The new preintegration will accumulate measurements until the next keyframe.
        """
        self.preintegration = IMUPreintegration(
            acc_noise=self.config.get_noise_std('acc_noise'),
            gyro_noise=self.config.get_noise_std('gyro_noise')
        )

    def add_measurement(self, imu_data: IMUData, dt: float):
        """
        Add an IMU measurement to the current preintegration.

        Parameters:
        -----------
        imu_data : IMUData
            IMU measurement to add to preintegration
        dt : float
            Time step for this measurement (seconds)

        Notes:
            The measurement is first corrected for bias, then integrated into
            the preintegration. If no preintegration is active, a new one is
            automatically started.

        Examples:
            >>> imu.add_measurement(imu_data, 0.01)  # 100Hz measurement
        """
        if self.preintegration is None:
            self.start_preintegration()

        # Apply bias correction
        corrected = self.bias.correct_measurement(imu_data)

        # Integrate
        self.preintegration.integrate(
            corrected.acceleration,
            corrected.angular_velocity,
            dt
        )
