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

"""Rotation representations and conversions"""


import numpy as np


def euler2dcm(euler: np.ndarray, sequence: str = 'ZYX') -> np.ndarray:
    """
    Convert Euler angles to Direction Cosine Matrix (DCM)

    Parameters:
    -----------
    euler : np.ndarray
        Euler angles [roll, pitch, yaw] or as specified by sequence (rad)
    sequence : str
        Rotation sequence (e.g., 'ZYX', 'XYZ', 'ZYZ')

    Returns:
    --------
    dcm : np.ndarray
        Direction Cosine Matrix (3x3)
    """
    angles = euler.copy()

    # Create rotation matrices
    def Rx(theta):
        return np.array([
            [1, 0, 0],
            [0, np.cos(theta), np.sin(theta)],
            [0, -np.sin(theta), np.cos(theta)]
        ])

    def Ry(theta):
        return np.array([
            [np.cos(theta), 0, -np.sin(theta)],
            [0, 1, 0],
            [np.sin(theta), 0, np.cos(theta)]
        ])

    def Rz(theta):
        return np.array([
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])

    # Map axes to rotation functions
    rot_map = {'X': Rx, 'Y': Ry, 'Z': Rz}

    # Apply rotations in sequence (intrinsic rotations)
    dcm = np.eye(3)
    for i, axis in enumerate(sequence):
        dcm = rot_map[axis](angles[i]) @ dcm

    return dcm


def dcm2euler(dcm: np.ndarray, sequence: str = 'ZYX') -> np.ndarray:
    """Convert Direction Cosine Matrix to Euler angles

    Extracts Euler angles from a direction cosine matrix using the
    specified rotation sequence.

    Parameters
    ----------
    dcm : np.ndarray
        Direction Cosine Matrix (3x3), must be orthogonal
    sequence : str, optional
        Rotation sequence string, by default 'ZYX'
        Supported sequences: 'ZYX' (roll-pitch-yaw), 'XYZ'

    Returns
    -------
    np.ndarray
        Euler angles [angle1, angle2, angle3] in radians
        For 'ZYX': [roll, pitch, yaw]
        For 'XYZ': [roll, pitch, yaw] in XYZ order

    Raises
    ------
    NotImplementedError
        If the specified rotation sequence is not implemented

    Notes
    -----
    - 'ZYX' sequence is the aerospace/navigation convention (roll-pitch-yaw)
    - Gimbal lock can occur when the middle angle is ±90 degrees
    - In gimbal lock situations, one degree of freedom is lost

    Examples
    --------
    >>> import numpy as np
    >>> # Create DCM from known Euler angles
    >>> euler_in = np.array([0.1, 0.2, 0.3])  # roll, pitch, yaw
    >>> dcm = euler2dcm(euler_in)
    >>> euler_out = dcm2euler(dcm, 'ZYX')
    >>> np.allclose(euler_in, euler_out)  # Should be True
    """
    if sequence == 'ZYX':
        # Roll-Pitch-Yaw convention
        pitch = np.arcsin(-dcm[2, 0])

        if np.cos(pitch) > 1e-6:
            roll = np.arctan2(dcm[2, 1], dcm[2, 2])
            yaw = np.arctan2(dcm[1, 0], dcm[0, 0])
        else:
            # Gimbal lock
            roll = 0
            yaw = np.arctan2(-dcm[0, 1], dcm[1, 1])

        return np.array([roll, pitch, yaw])

    elif sequence == 'XYZ':
        pitch = np.arcsin(dcm[0, 2])

        if np.cos(pitch) > 1e-6:
            roll = np.arctan2(-dcm[1, 2], dcm[2, 2])
            yaw = np.arctan2(-dcm[0, 1], dcm[0, 0])
        else:
            roll = np.arctan2(dcm[2, 1], dcm[1, 1])
            yaw = 0

        return np.array([roll, pitch, yaw])

    else:
        raise NotImplementedError(f"Sequence {sequence} not implemented")


def quaternion2dcm(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to Direction Cosine Matrix

    Parameters:
    -----------
    q : np.ndarray
        Quaternion [w, x, y, z] (scalar-first convention)

    Returns:
    --------
    dcm : np.ndarray
        Direction Cosine Matrix (3x3)
    """
    q = q / np.linalg.norm(q)  # Normalize
    w, x, y, z = q[0], q[1], q[2], q[3]

    dcm = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])

    return dcm


def dcm2quaternion(dcm: np.ndarray) -> np.ndarray:
    """
    Convert Direction Cosine Matrix to quaternion

    Parameters:
    -----------
    dcm : np.ndarray
        Direction Cosine Matrix (3x3)

    Returns:
    --------
    q : np.ndarray
        Quaternion [w, x, y, z]
    """
    # Shepperd's method for numerical stability
    trace = np.trace(dcm)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (dcm[2, 1] - dcm[1, 2]) * s
        y = (dcm[0, 2] - dcm[2, 0]) * s
        z = (dcm[1, 0] - dcm[0, 1]) * s
    elif dcm[0, 0] > dcm[1, 1] and dcm[0, 0] > dcm[2, 2]:
        s = 2.0 * np.sqrt(1.0 + dcm[0, 0] - dcm[1, 1] - dcm[2, 2])
        w = (dcm[2, 1] - dcm[1, 2]) / s
        x = 0.25 * s
        y = (dcm[0, 1] + dcm[1, 0]) / s
        z = (dcm[0, 2] + dcm[2, 0]) / s
    elif dcm[1, 1] > dcm[2, 2]:
        s = 2.0 * np.sqrt(1.0 + dcm[1, 1] - dcm[0, 0] - dcm[2, 2])
        w = (dcm[0, 2] - dcm[2, 0]) / s
        x = (dcm[0, 1] + dcm[1, 0]) / s
        y = 0.25 * s
        z = (dcm[1, 2] + dcm[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + dcm[2, 2] - dcm[0, 0] - dcm[1, 1])
        w = (dcm[1, 0] - dcm[0, 1]) / s
        x = (dcm[0, 2] + dcm[2, 0]) / s
        y = (dcm[1, 2] + dcm[2, 1]) / s
        z = 0.25 * s

    return np.array([w, x, y, z])


def axis_angle2dcm(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Convert axis-angle representation to DCM

    Parameters:
    -----------
    axis : np.ndarray
        Rotation axis (3x1), will be normalized
    angle : float
        Rotation angle (rad)

    Returns:
    --------
    dcm : np.ndarray
        Direction Cosine Matrix (3x3)
    """
    axis = axis / np.linalg.norm(axis)
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c

    x, y, z = axis[0], axis[1], axis[2]

    dcm = np.array([
        [t*x*x + c, t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c, t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c]
    ])

    return dcm


def dcm2axis_angle(dcm: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Convert DCM to axis-angle representation

    Parameters:
    -----------
    dcm : np.ndarray
        Direction Cosine Matrix (3x3)

    Returns:
    --------
    axis : np.ndarray
        Rotation axis (3x1)
    angle : float
        Rotation angle (rad)
    """
    # Rotation angle
    angle = np.arccos((np.trace(dcm) - 1) / 2)

    if angle < 1e-6:
        # No rotation
        return np.array([0, 0, 1]), 0.0
    elif angle > np.pi - 1e-6:
        # 180 degree rotation
        # Find the largest diagonal element
        i = np.argmax(np.diag(dcm))
        axis = np.zeros(3)
        axis[i] = np.sqrt((dcm[i, i] + 1) / 2)
        for j in range(3):
            if j != i:
                axis[j] = dcm[i, j] / (2 * axis[i])
        return axis, np.pi
    else:
        # General case
        axis = np.array([
            dcm[2, 1] - dcm[1, 2],
            dcm[0, 2] - dcm[2, 0],
            dcm[1, 0] - dcm[0, 1]
        ]) / (2 * np.sin(angle))
        return axis, angle


def rotate_vector(v: np.ndarray, dcm: np.ndarray) -> np.ndarray:
    """
    Rotate vector using DCM

    Parameters:
    -----------
    v : np.ndarray
        Vector to rotate (3x1)
    dcm : np.ndarray
        Direction Cosine Matrix (3x3)

    Returns:
    --------
    v_rot : np.ndarray
        Rotated vector
    """
    return dcm @ v


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions

    Parameters:
    -----------
    q1, q2 : np.ndarray
        Quaternions [w, x, y, z]

    Returns:
    --------
    q : np.ndarray
        Product quaternion
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return np.array([w, x, y, z])


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """Get quaternion conjugate

    Computes the conjugate of a quaternion, which reverses the sign
    of the vector components while keeping the scalar component unchanged.

    Parameters
    ----------
    q : np.ndarray
        Quaternion [w, x, y, z] in scalar-first convention

    Returns
    -------
    np.ndarray
        Conjugate quaternion [w, -x, -y, -z]

    Notes
    -----
    For a unit quaternion representing rotation, the conjugate represents
    the inverse rotation.

    Examples
    --------
    >>> import numpy as np
    >>> q = np.array([0.7071, 0.7071, 0, 0])  # 90° rotation around X-axis
    >>> q_conj = quaternion_conjugate(q)
    >>> print(q_conj)  # [0.7071, -0.7071, 0, 0]
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_inverse(q: np.ndarray) -> np.ndarray:
    """Get quaternion inverse

    Computes the multiplicative inverse of a quaternion.

    Parameters
    ----------
    q : np.ndarray
        Quaternion [w, x, y, z] in scalar-first convention

    Returns
    -------
    np.ndarray
        Inverse quaternion q^(-1)

    Notes
    -----
    For a unit quaternion, the inverse equals the conjugate.
    For non-unit quaternions: q^(-1) = q* / |q|^2
    where q* is the conjugate and |q| is the norm.

    Examples
    --------
    >>> import numpy as np
    >>> q = np.array([1.0, 1.0, 0.0, 0.0])  # Non-unit quaternion
    >>> q_inv = quaternion_inverse(q)
    >>> # Verify: q * q_inv should give identity [1, 0, 0, 0]
    """
    return quaternion_conjugate(q) / np.linalg.norm(q)**2


class RotationIntegrator:
    """Integrate angular velocity to update rotation state

    This class provides methods to integrate angular velocity measurements
    over time to update rotation representations (quaternions or DCM).
    It supports different integration methods for various accuracy and
    computational requirements.

    Parameters
    ----------
    method : str, optional
        Integration method to use, by default 'quaternion'.
        Options: 'quaternion', 'dcm', 'euler'

    Attributes
    ----------
    method : str
        Current integration method

    Notes
    -----
    Quaternion integration is generally preferred as it avoids singularities
    and maintains numerical stability better than Euler angles.

    Examples
    --------
    >>> integrator = RotationIntegrator(method='quaternion')
    >>> q0 = np.array([1, 0, 0, 0])  # Identity quaternion
    >>> omega = np.array([0.1, 0.0, 0.0])  # Angular velocity rad/s
    >>> dt = 0.01  # Time step
    >>> q1 = integrator.integrate(q0, omega, dt)
    """

    def __init__(self, method: str = 'quaternion'):
        """
        Initialize rotation integrator

        Parameters:
        -----------
        method : str
            Integration method ('quaternion', 'dcm', 'euler')
        """
        self.method = method

    def integrate(self, R: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
        """Integrate rotation state using angular velocity

        Updates the rotation state by integrating angular velocity over
        a specified time step using the selected integration method.

        Parameters
        ----------
        R : np.ndarray
            Current rotation state in the format specified by self.method:
            - quaternion: [w, x, y, z] (4-element array)
            - dcm: direction cosine matrix (3x3 array)
            - euler: Euler angles (3-element array)
        omega : np.ndarray
            Angular velocity vector [wx, wy, wz] in rad/s
        dt : float
            Integration time step in seconds

        Returns
        -------
        np.ndarray
            Updated rotation state in the same format as input

        Raises
        ------
        ValueError
            If an unknown integration method is specified

        Notes
        -----
        - Quaternion method: Uses first-order integration with normalization
        - DCM method: Uses skew-symmetric matrix integration with SVD orthogonalization
        - Euler method: Not yet implemented

        The quaternion method is recommended for most applications due to
        its numerical stability and absence of singularities.

        Examples
        --------
        >>> integrator = RotationIntegrator('quaternion')
        >>> q = np.array([1, 0, 0, 0])  # Identity rotation
        >>> omega = np.array([0, 0, 0.1])  # Yaw at 0.1 rad/s
        >>> dt = 0.01
        >>> q_new = integrator.integrate(q, omega, dt)
        """
        if self.method == 'quaternion':
            # Quaternion integration
            q = R if len(R) == 4 else dcm2quaternion(R)

            # Quaternion derivative
            omega_q = np.array([0, omega[0], omega[1], omega[2]])
            q_dot = 0.5 * quaternion_multiply(q, omega_q)

            # Update quaternion
            q_new = q + q_dot * dt
            q_new = q_new / np.linalg.norm(q_new)  # Normalize

            return q_new

        elif self.method == 'dcm':
            # DCM integration
            dcm = R if R.shape == (3, 3) else quaternion2dcm(R)

            # Skew-symmetric matrix
            Omega = np.array([
                [0, -omega[2], omega[1]],
                [omega[2], 0, -omega[0]],
                [-omega[1], omega[0], 0]
            ])

            # Update DCM
            dcm_new = dcm @ (np.eye(3) + Omega * dt)

            # Orthogonalize (simple method)
            U, _, Vt = np.linalg.svd(dcm_new)
            dcm_new = U @ Vt

            return dcm_new

        else:
            raise ValueError(f"Unknown integration method: {self.method}")
