"""Rotation representations and conversions"""

import numpy as np
from typing import Tuple

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
    """
    Convert Direction Cosine Matrix to Euler angles
    
    Parameters:
    -----------
    dcm : np.ndarray
        Direction Cosine Matrix (3x3)
    sequence : str
        Rotation sequence
        
    Returns:
    --------
    euler : np.ndarray
        Euler angles (rad)
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


def dcm2axis_angle(dcm: np.ndarray) -> Tuple[np.ndarray, float]:
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
    """Get quaternion conjugate"""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_inverse(q: np.ndarray) -> np.ndarray:
    """Get quaternion inverse"""
    return quaternion_conjugate(q) / np.linalg.norm(q)**2


class RotationIntegrator:
    """Integrate angular velocity to update rotation"""
    
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
        """
        Integrate rotation with angular velocity
        
        Parameters:
        -----------
        R : np.ndarray
            Current rotation (quaternion, DCM, or Euler angles)
        omega : np.ndarray
            Angular velocity (rad/s)
        dt : float
            Time step (s)
            
        Returns:
        --------
        R_new : np.ndarray
            Updated rotation
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