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

"""Extended Kalman Filter for GNSS/IMU fusion"""

import numpy as np
from typing import Optional, List, Tuple
from ..sensors.imu import IMUData, IMUPreintegration
from ..core.data_structures import Observation, Ephemeris
from ..observation.pseudorange import PseudorangeProcessor
from ..observation.carrier_phase import CarrierPhaseProcessor
from .state import NavigationState, StateError, StateManager
from ..coordinate.geodetic import gravity_model
from ..coordinate.transforms import ecef2llh

class GNSSIMUFilter:
    """Extended Kalman Filter for tightly-coupled GNSS/IMU integration"""
    
    def __init__(self, initial_state: NavigationState):
        """
        Initialize filter
        
        Parameters:
        -----------
        initial_state : NavigationState
            Initial navigation state
        """
        self.state_manager = StateManager(initial_state)
        
        # Initialize covariance
        self._initialize_covariance()
        
        # IMU preintegration
        self.imu_preint = IMUPreintegration()
        
        # GNSS processors
        self.pr_processor = PseudorangeProcessor()
        self.cp_processor = CarrierPhaseProcessor()
        
        # Process noise parameters
        self.acc_noise = 0.1      # m/s^2
        self.gyro_noise = 0.01    # rad/s
        self.acc_bias_walk = 0.001   # m/s^3
        self.gyro_bias_walk = 0.0001  # rad/s^2
        self.clock_noise = 1.0     # m/s
        self.clock_drift_noise = 0.1  # m/s^2
        
    def _initialize_covariance(self):
        """Initialize state covariance matrix"""
        n = self.state_manager.state_dim
        P = np.eye(n)
        
        # Position uncertainty (m)
        P[0:3, 0:3] *= 10.0**2
        
        # Velocity uncertainty (m/s)
        P[3:6, 3:6] *= 1.0**2
        
        # Attitude uncertainty (rad)
        P[6:9, 6:9] *= (0.1)**2
        
        # Accelerometer bias (m/s^2)
        P[10:13, 10:13] *= 0.05**2
        
        # Gyroscope bias (rad/s)
        P[13:16, 13:16] *= 0.001**2
        
        # Clock bias (m)
        P[16, 16] = 30.0**2
        
        # Clock drift (m/s)
        P[17, 17] = 1.0**2
        
        self.P = P
        
    def predict_imu(self, imu_data: IMUData, dt: float):
        """
        Predict state using IMU measurement
        
        Parameters:
        -----------
        imu_data : IMUData
            IMU measurement
        dt : float
            Time step (s)
        """
        state = self.state_manager.state
        
        # Remove biases from measurements
        acc = imu_data.acceleration - state.acc_bias
        gyro = imu_data.angular_velocity - state.gyro_bias
        
        # Current rotation matrix
        R = state.dcm
        
        # Gravity in ECEF
        llh = ecef2llh(state.position)
        g_local = gravity_model(llh[0], llh[2])
        g_ecef = np.array([0, 0, -g_local])  # Simplified
        
        # State prediction
        # Position
        state.position += state.velocity * dt + 0.5 * (R @ acc + g_ecef) * dt**2
        
        # Velocity
        state.velocity += (R @ acc + g_ecef) * dt
        
        # Attitude
        from ..coordinate.rotation import RotationIntegrator
        rot_int = RotationIntegrator('quaternion')
        state.quaternion = rot_int.integrate(state.quaternion, gyro, dt)
        
        # Clock
        state.clock_bias += state.clock_drift * dt
        
        # Covariance prediction
        self._predict_covariance(acc, gyro, dt)
        
        # Update IMU preintegration
        self.imu_preint.integrate(acc, gyro, dt)
        
    def _predict_covariance(self, acc: np.ndarray, gyro: np.ndarray, dt: float):
        """Predict covariance matrix"""
        n = self.state_manager.state_dim
        
        # State transition matrix
        F = np.eye(n)
        
        # Position-velocity coupling
        F[0:3, 3:6] = np.eye(3) * dt
        
        # Velocity-attitude coupling
        R = self.state_manager.state.dcm
        F[3:6, 6:9] = -R @ skew_symmetric(acc) * dt
        
        # Velocity-acc_bias coupling
        F[3:6, 10:13] = -R * dt
        
        # Attitude-gyro_bias coupling
        F[6:9, 13:16] = -np.eye(3) * dt
        
        # Clock coupling
        F[16, 17] = dt
        
        # Process noise
        Q = np.zeros((n, n))
        
        # IMU noise
        Q[3:6, 3:6] = (self.acc_noise * dt)**2 * np.eye(3)
        Q[6:9, 6:9] = (self.gyro_noise * dt)**2 * np.eye(3)
        
        # Bias random walk
        Q[10:13, 10:13] = (self.acc_bias_walk * dt)**2 * np.eye(3)
        Q[13:16, 13:16] = (self.gyro_bias_walk * dt)**2 * np.eye(3)
        
        # Clock noise
        Q[16, 16] = (self.clock_noise * dt)**2
        Q[17, 17] = (self.clock_drift_noise * dt)**2
        
        # Predict covariance
        self.P = F @ self.P @ F.T + Q
        
    def update_gnss(self, observations: List[Observation], 
                    ephemerides: List[Ephemeris],
                    use_carrier: bool = True):
        """
        Update state with GNSS observations
        
        Parameters:
        -----------
        observations : List[Observation]
            GNSS observations
        ephemerides : List[Ephemeris]
            Satellite ephemerides
        use_carrier : bool
            Whether to use carrier phase measurements
        """
        state = self.state_manager.state
        
        # Process pseudorange
        pr_res, pr_H, pr_R = self.pr_processor.process_observations(
            observations, ephemerides, 
            state.position, state.clock_bias[0] / CLIGHT
        )
        
        if len(pr_res) > 0:
            self._measurement_update(pr_res, pr_H, pr_R)
            
        # Process carrier phase if requested
        if use_carrier:
            cp_res, cp_H, cp_R = self.cp_processor.process_observations(
                observations, ephemerides,
                state.position, state.clock_bias[0] / CLIGHT,
                state.ambiguities
            )
            
            if len(cp_res) > 0:
                # Expand H matrix for ambiguities
                cp_H_full = self._expand_carrier_jacobian(cp_H, observations)
                self._measurement_update(cp_res, cp_H_full, cp_R)
                
    def _expand_carrier_jacobian(self, H: np.ndarray, 
                                observations: List[Observation]) -> np.ndarray:
        """Expand carrier phase Jacobian for full state"""
        n_obs = H.shape[0]
        n_states = self.state_manager.state_dim
        
        H_full = np.zeros((n_obs, n_states))
        
        # Copy position and clock columns
        H_full[:, 0:3] = H[:, 0:3]  # Position
        H_full[:, 16] = H[:, 3] / CLIGHT  # Clock (convert to meters)
        
        # Ambiguity columns
        for i, obs in enumerate(observations):
            if obs.sat in self.state_manager.idx_amb:
                amb_idx = self.state_manager.idx_amb[obs.sat]
                H_full[i, amb_idx] = H[i, 4]  # Ambiguity term
                
        return H_full
        
    def _measurement_update(self, residuals: np.ndarray,
                          H: np.ndarray, R: np.ndarray):
        """Perform measurement update"""
        # Innovation covariance
        S = H @ self.P @ H.T + R
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # State update
        dx = K @ residuals
        
        # Apply state correction
        error = StateError()
        error.from_vector(dx)
        self.state_manager.update(error)
        
        # Covariance update
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T
        
    def get_state(self) -> NavigationState:
        """Get current navigation state"""
        return self.state_manager.state.copy()
    
    def get_covariance(self) -> np.ndarray:
        """Get state covariance matrix"""
        return self.P.copy()


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Create skew-symmetric matrix"""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


class AdaptiveGNSSIMUFilter(GNSSIMUFilter):
    """Adaptive EKF with online noise estimation"""
    
    def __init__(self, initial_state: NavigationState):
        super().__init__(initial_state)
        
        # Innovation monitoring
        self.innovation_window = 10
        self.innovations = []
        
    def _measurement_update(self, residuals: np.ndarray,
                          H: np.ndarray, R: np.ndarray):
        """Adaptive measurement update"""
        # Store innovation
        S = H @ self.P @ H.T + R
        normalized_innovation = residuals.T @ np.linalg.inv(S) @ residuals
        self.innovations.append(normalized_innovation)
        
        # Keep window size
        if len(self.innovations) > self.innovation_window:
            self.innovations.pop(0)
            
        # Check for outliers
        if len(self.innovations) >= 5:
            mean_innovation = np.mean(self.innovations)
            if normalized_innovation > 3 * mean_innovation:
                # Inflate measurement noise for this update
                R = R * (normalized_innovation / mean_innovation)
                
        # Perform standard update
        super()._measurement_update(residuals, H, R)