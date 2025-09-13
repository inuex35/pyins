"""
Velocity factor for GTSAM factor graph.
"""

import gtsam
import numpy as np
from typing import List
from pyins.sensors.lever_arm import LeverArm


class VelocityFactor(gtsam.CustomFactor):
    """
    Factor for velocity measurements.
    Constrains the velocity state based on external velocity measurements.
    Handles lever arm correction from odometer to IMU/GNSS position.
    """
    
    def __init__(self, pose_key: int, velocity_key: int, measured_velocity: np.ndarray, 
                 lever_arm: np.ndarray, gyro_measurement: np.ndarray, 
                 noise_model: gtsam.noiseModel.Base):
        """
        Initialize velocity factor with lever arm correction using pyins.
        
        Parameters:
        -----------
        pose_key : int
            Key for the pose variable (needed for lever arm correction)
        velocity_key : int
            Key for the velocity variable
        measured_velocity : np.ndarray
            Measured velocity vector [vx, vy, vz] in odometer frame (m/s)
        lever_arm : np.ndarray
            Lever arm from IMU/GNSS to odometer in body frame [forward, left, up] in meters
            Note: [-1.36, -0.79, -0.55] means odometer is 1.36m behind, 0.79m right, 0.55m down from IMU
        gyro_measurement : np.ndarray
            Gyroscope measurement [wx, wy, wz] in rad/s for lever arm correction
        noise_model : gtsam.noiseModel
            Noise model for the velocity measurement
        """
        # Store parameters before calling super().__init__
        self.measured_velocity = measured_velocity
        self.lever_arm_helper = LeverArm(lever_arm)
        self.gyro = gyro_measurement
        
        # Initialize base class with noise model, keys, and error function
        super().__init__(noise_model, [pose_key, velocity_key], lambda factor, v, jacobians: self.error(v))
    
    def error_func_unused(self, v: gtsam.Values, jacobians=None) -> np.ndarray:
        """
        Error function for CustomFactor interface.
        
        Parameters:
        -----------
        v : gtsam.Values
            Current values of variables
        jacobians : optional
            List to store Jacobians (not used in this implementation)
            
        Returns:
        --------
        np.ndarray
            Error vector
        """
        return self.error(v)
        
    def error(self, values: gtsam.Values) -> np.ndarray:
        """
        Calculate error between measured and estimated velocity with lever arm correction.
        
        Uses pyins LeverArm to properly compensate for the velocity difference between
        odometer and IMU/GNSS positions.
        
        Parameters:
        -----------
        values : gtsam.Values
            Current values of variables
            
        Returns:
        --------
        np.ndarray
            Error vector
        """
        # Get current pose and velocity estimate at IMU/GNSS position
        pose = values.atPose3(self.keys()[0])
        velocity_imu = values.atVector(self.keys()[1])
        
        # Get rotation matrix from body to world frame
        R_body_to_world = pose.rotation().matrix()
        
        # Compensate measured velocity from odometer to IMU/GNSS position
        # The odometer velocity needs to be corrected for lever arm effect
        # v_imu = v_odo + R_body * (omega × lever_arm)
        velocity_odo_compensated = self.lever_arm_helper.compensate_velocity(
            vel_body=self.measured_velocity,  # Odometer velocity in body frame
            omega_body=self.gyro,  # Angular velocity in body frame
            R_body=np.eye(3),  # Identity since we're already in body frame
            lever_arm=None  # Use the stored lever arm
        )
        
        # Transform compensated velocity to world frame
        velocity_odo_world = R_body_to_world @ velocity_odo_compensated
        
        # Calculate error
        error = velocity_imu - velocity_odo_world
        
        return error
    
    def clone(self):
        """Clone the factor"""
        return VelocityFactor(self.keys()[0], self.keys()[1], self.measured_velocity, 
                            self.lever_arm_helper.lever_arm, self.gyro, self.noiseModel())


class VelocityPoseFactor(gtsam.CustomFactor):
    """
    Factor for velocity measurements that constrains a Pose3 variable.
    This is useful when velocity is derived from pose changes.
    """
    
    def __init__(self, pose_key: int, velocity_key: int, dt: float,
                 noise_model: gtsam.noiseModel.Base):
        """
        Initialize velocity-pose factor.
        
        Parameters:
        -----------
        pose_key : int
            Key for the pose variable at current time
        velocity_key : int  
            Key for the velocity variable
        dt : float
            Time interval
        noise_model : gtsam.noiseModel
            Noise model for the constraint
        """
        self.dt = dt
        super().__init__(noise_model, [pose_key, velocity_key], lambda factor, v, jacobians: self.error(v))
    
        
    def error(self, values: gtsam.Values) -> np.ndarray:
        """
        Calculate error between pose-derived velocity and velocity state.
        
        This factor enforces consistency between the velocity state
        and the derivative of position.
        """
        pose = values.atPose3(self.keys()[0])
        velocity = values.atVector(self.keys()[1])
        
        # For now, we don't have previous pose, so this is a placeholder
        # In practice, this would compute velocity from pose change
        # and compare with velocity state
        
        return np.zeros(3)  # Placeholder
    
    def clone(self):
        """Clone the factor"""
        return VelocityPoseFactor(self.keys()[0], self.keys()[1], 
                                 self.dt, self.noiseModel())


class DopplerVelocityFactor(gtsam.CustomFactor):
    """
    Factor for Doppler-derived velocity measurements.
    Uses satellite line-of-sight vectors to constrain velocity.
    """
    
    def __init__(self, pose_key: int, velocity_key: int,
                 doppler_obs: float, sat_pos: np.ndarray, sat_vel: np.ndarray,
                 wavelength: float, noise_model: gtsam.noiseModel.Base):
        """
        Initialize Doppler velocity factor.
        
        Parameters:
        -----------
        pose_key : int
            Key for the pose variable
        velocity_key : int
            Key for the velocity variable  
        doppler_obs : float
            Doppler observation (Hz)
        sat_pos : np.ndarray
            Satellite position in ECEF
        sat_vel : np.ndarray
            Satellite velocity in ECEF
        wavelength : float
            Carrier wavelength (m)
        noise_model : gtsam.noiseModel
            Noise model for the measurement
        """
        self.doppler_obs = doppler_obs
        self.sat_pos = sat_pos
        self.sat_vel = sat_vel
        self.wavelength = wavelength
        super().__init__(noise_model, [pose_key, velocity_key], lambda factor, v, jacobians: self.error(v))
    
        
    def error(self, values: gtsam.Values) -> np.ndarray:
        """
        Calculate Doppler residual.
        
        Doppler shift relates to the relative velocity between
        receiver and satellite along the line-of-sight.
        """
        pose = values.atPose3(self.keys()[0])
        velocity = values.atVector(self.keys()[1])
        
        # Get receiver position
        rec_pos = pose.translation()
        
        # Line-of-sight unit vector
        los = self.sat_pos - rec_pos
        los_dist = np.linalg.norm(los)
        los_unit = los / los_dist
        
        # Relative velocity along line-of-sight
        rel_vel = self.sat_vel - velocity
        doppler_pred = -np.dot(rel_vel, los_unit) / self.wavelength
        
        # Error is difference between observed and predicted Doppler
        error = np.array([self.doppler_obs - doppler_pred])
        
        return error
    
    def clone(self):
        """Clone the factor"""
        return DopplerVelocityFactor(self.keys()[0], self.keys()[1],
                                    self.doppler_obs, self.sat_pos, self.sat_vel,
                                    self.wavelength, self.noiseModel())