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

"""Lever arm compensation for multi-sensor systems"""

from typing import Optional

import numpy as np


class LeverArm:
    """
    Lever arm compensation for multi-sensor navigation systems.

    This class provides methods to compensate position, velocity, and acceleration
    measurements for the lever arm effect - the physical offset between sensors
    and the body/IMU center.

    Attributes:
        lever_arm (np.ndarray): 3D lever arm vector from IMU/body center to sensor (meters)

    Notes:
        The lever arm compensation accounts for the rotational motion of the platform,
        which causes different sensors to experience different linear motions due to
        their spatial separation.

    Examples:
        Basic lever arm setup:

        >>> lever = LeverArm(np.array([0.5, 0.0, 0.1]))  # 50cm forward, 10cm up
        >>> compensated_pos = lever.compensate_position(body_pos, rotation_matrix)
    """

    def __init__(self, lever_arm: np.ndarray = np.zeros(3)):
        """
        Initialize lever arm

        Parameters:
        -----------
        lever_arm : np.ndarray
            3D lever arm vector from IMU/body center to sensor (m)
        """
        self.lever_arm = lever_arm

    def compensate_position(self,
                          pos_body: np.ndarray,
                          R_body: np.ndarray,
                          lever_arm: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compensate position for lever arm effect

        Parameters:
        -----------
        pos_body : np.ndarray
            Position of body/IMU center (3x1)
        R_body : np.ndarray
            Rotation matrix from body to navigation frame (3x3)
        lever_arm : np.ndarray, optional
            Override default lever arm

        Returns:
        --------
        pos_sensor : np.ndarray
            Position of sensor after lever arm compensation
        """
        la = lever_arm if lever_arm is not None else self.lever_arm
        return pos_body + R_body @ la

    def compensate_velocity(self,
                          vel_body: np.ndarray,
                          omega_body: np.ndarray,
                          R_body: np.ndarray,
                          lever_arm: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compensate velocity for lever arm effect

        Parameters:
        -----------
        vel_body : np.ndarray
            Velocity of body/IMU center in navigation frame (3x1)
        omega_body : np.ndarray
            Angular velocity in body frame (3x1)
        R_body : np.ndarray
            Rotation matrix from body to navigation frame (3x3)
        lever_arm : np.ndarray, optional
            Override default lever arm

        Returns:
        --------
        vel_sensor : np.ndarray
            Velocity of sensor after lever arm compensation
        """
        la = lever_arm if lever_arm is not None else self.lever_arm
        # v_sensor = v_body + R_body * (omega_body x lever_arm)
        return vel_body + R_body @ np.cross(omega_body, la)

    def compensate_acceleration(self,
                              acc_body: np.ndarray,
                              omega_body: np.ndarray,
                              alpha_body: np.ndarray,
                              R_body: np.ndarray,
                              lever_arm: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compensate acceleration for lever arm effect

        Parameters:
        -----------
        acc_body : np.ndarray
            Acceleration of body/IMU center in navigation frame (3x1)
        omega_body : np.ndarray
            Angular velocity in body frame (3x1)
        alpha_body : np.ndarray
            Angular acceleration in body frame (3x1)
        R_body : np.ndarray
            Rotation matrix from body to navigation frame (3x3)
        lever_arm : np.ndarray, optional
            Override default lever arm

        Returns:
        --------
        acc_sensor : np.ndarray
            Acceleration of sensor after lever arm compensation
        """
        la = lever_arm if lever_arm is not None else self.lever_arm

        # Centripetal acceleration: omega x (omega x lever_arm)
        centripetal = np.cross(omega_body, np.cross(omega_body, la))

        # Tangential acceleration: alpha x lever_arm
        tangential = np.cross(alpha_body, la)

        # Total acceleration
        return acc_body + R_body @ (centripetal + tangential)

    def compute_relative_motion(self,
                              omega_body: np.ndarray,
                              alpha_body: np.ndarray,
                              lever_arm_1: np.ndarray,
                              lever_arm_2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute relative velocity and acceleration between two sensors

        Parameters:
        -----------
        omega_body : np.ndarray
            Angular velocity in body frame (3x1)
        alpha_body : np.ndarray
            Angular acceleration in body frame (3x1)
        lever_arm_1 : np.ndarray
            Lever arm of sensor 1 (3x1)
        lever_arm_2 : np.ndarray
            Lever arm of sensor 2 (3x1)

        Returns:
        --------
        rel_vel : np.ndarray
            Relative velocity of sensor 2 w.r.t sensor 1 in body frame
        rel_acc : np.ndarray
            Relative acceleration of sensor 2 w.r.t sensor 1 in body frame
        """
        # Relative lever arm
        rel_lever = lever_arm_2 - lever_arm_1

        # Relative velocity: omega x relative_lever_arm
        rel_vel = np.cross(omega_body, rel_lever)

        # Relative acceleration
        centripetal = np.cross(omega_body, np.cross(omega_body, rel_lever))
        tangential = np.cross(alpha_body, rel_lever)
        rel_acc = centripetal + tangential

        return rel_vel, rel_acc

    @staticmethod
    def estimate_lever_arm(pos_sensor_nav: np.ndarray,
                          pos_body_nav: np.ndarray,
                          R_body: np.ndarray) -> np.ndarray:
        """
        Estimate lever arm given sensor and body positions

        Parameters:
        -----------
        pos_sensor_nav : np.ndarray
            Sensor position in navigation frame (3x1)
        pos_body_nav : np.ndarray
            Body/IMU position in navigation frame (3x1)
        R_body : np.ndarray
            Rotation matrix from body to navigation frame (3x3)

        Returns:
        --------
        lever_arm : np.ndarray
            Estimated lever arm in body frame (3x1)
        """
        # lever_arm = R_body^T @ (pos_sensor - pos_body)
        return R_body.T @ (pos_sensor_nav - pos_body_nav)

    def get_jacobian_position(self, R_body: np.ndarray) -> np.ndarray:
        """
        Get Jacobian of sensor position w.r.t body position and orientation

        Parameters:
        -----------
        R_body : np.ndarray
            Rotation matrix from body to navigation frame (3x3)

        Returns:
        --------
        J : np.ndarray
            Jacobian matrix (3x6) [d_pos/d_pos_body, d_pos/d_euler]
        """
        J = np.zeros((3, 6))
        J[:, :3] = np.eye(3)  # Derivative w.r.t position

        # Derivative w.r.t orientation (simplified for small angles)
        J[:, 3:6] = -R_body @ skew_symmetric(self.lever_arm)

        return J

    def get_jacobian_velocity(self,
                            omega_body: np.ndarray,
                            R_body: np.ndarray) -> np.ndarray:
        """
        Get Jacobian of sensor velocity w.r.t body velocity and angular velocity

        Parameters:
        -----------
        omega_body : np.ndarray
            Angular velocity in body frame (3x1)
        R_body : np.ndarray
            Rotation matrix from body to navigation frame (3x3)

        Returns:
        --------
        J : np.ndarray
            Jacobian matrix (3x6) [d_vel/d_vel_body, d_vel/d_omega]
        """
        J = np.zeros((3, 6))
        J[:, :3] = np.eye(3)  # Derivative w.r.t velocity
        J[:, 3:6] = -R_body @ skew_symmetric(self.lever_arm)  # Derivative w.r.t omega

        return J


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """
    Create skew-symmetric matrix from a 3D vector.

    The skew-symmetric matrix is used to represent the cross product operation
    as a matrix multiplication: skew(v) @ u = v × u

    Parameters:
    -----------
    v : np.ndarray
        3D vector to convert to skew-symmetric matrix

    Returns:
    --------
    np.ndarray
        3x3 skew-symmetric matrix representation of the input vector

    Notes:
        For vector v = [v1, v2, v3], the skew-symmetric matrix is:
        [[  0, -v3,  v2],
         [ v3,   0, -v1],
         [-v2,  v1,   0]]

    Examples:
        >>> v = np.array([1, 2, 3])
        >>> skew_v = skew_symmetric(v)
        >>> print(skew_v)
        [[ 0 -3  2]
         [ 3  0 -1]
         [-2  1  0]]
    """
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


class MultiSensorLeverArm:
    """
    Manage lever arms for multiple sensors in a navigation system.

    This class provides a convenient interface for handling multiple sensors,
    each with their own lever arm configurations. It allows bulk operations
    on all registered sensors.

    Attributes:
        lever_arms (dict): Dictionary mapping sensor IDs to LeverArm objects

    Examples:
        Multi-sensor setup:

        >>> multi_lever = MultiSensorLeverArm()
        >>> multi_lever.add_sensor('gnss1', np.array([0.2, 0, 0.3]))
        >>> multi_lever.add_sensor('gnss2', np.array([-0.2, 0, 0.3]))
        >>> positions = multi_lever.compensate_all_positions(body_pos, rotation)
    """

    def __init__(self):
        self.lever_arms = {}  # sensor_id -> lever_arm

    def add_sensor(self, sensor_id: str, lever_arm: np.ndarray):
        """
        Add a sensor with its lever arm configuration.

        Parameters:
        -----------
        sensor_id : str
            Unique identifier for the sensor
        lever_arm : np.ndarray
            3D lever arm vector from body center to sensor (meters)

        Notes:
            If a sensor with the same ID already exists, it will be replaced.
        """
        self.lever_arms[sensor_id] = LeverArm(lever_arm)

    def get_lever_arm(self, sensor_id: str) -> Optional[LeverArm]:
        """
        Get lever arm configuration for a specific sensor.

        Parameters:
        -----------
        sensor_id : str
            Sensor identifier to look up

        Returns:
        --------
        Optional[LeverArm]
            LeverArm object for the sensor, or None if sensor not found
        """
        return self.lever_arms.get(sensor_id)

    def compensate_all_positions(self,
                                pos_body: np.ndarray,
                                R_body: np.ndarray) -> dict:
        """
        Compensate positions for all registered sensors.

        Parameters:
        -----------
        pos_body : np.ndarray
            Position of body/IMU center in navigation frame (3x1)
        R_body : np.ndarray
            Rotation matrix from body to navigation frame (3x3)

        Returns:
        --------
        dict
            Dictionary mapping sensor IDs to their compensated positions
        """
        compensated = {}
        for sensor_id, lever_arm in self.lever_arms.items():
            compensated[sensor_id] = lever_arm.compensate_position(pos_body, R_body)
        return compensated

    def compensate_all_velocities(self,
                                 vel_body: np.ndarray,
                                 omega_body: np.ndarray,
                                 R_body: np.ndarray) -> dict:
        """
        Compensate velocities for all registered sensors.

        Parameters:
        -----------
        vel_body : np.ndarray
            Velocity of body/IMU center in navigation frame (3x1)
        omega_body : np.ndarray
            Angular velocity in body frame (3x1)
        R_body : np.ndarray
            Rotation matrix from body to navigation frame (3x3)

        Returns:
        --------
        dict
            Dictionary mapping sensor IDs to their compensated velocities
        """
        compensated = {}
        for sensor_id, lever_arm in self.lever_arms.items():
            compensated[sensor_id] = lever_arm.compensate_velocity(
                vel_body, omega_body, R_body)
        return compensated
