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

"""State representation for sensor fusion"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from ..coordinate.rotation import quaternion2dcm, dcm2quaternion
from ..sensors.imu import IMUBias

@dataclass
class NavigationState:
    """Navigation state for GNSS/IMU fusion"""
    
    # Time
    time: float = 0.0
    
    # Position, velocity, attitude (PVA)
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))      # ECEF position (m)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))      # ECEF velocity (m/s)
    quaternion: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))  # Body to ECEF quaternion
    
    # IMU biases
    acc_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))      # Accelerometer bias (m/s^2)
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))     # Gyroscope bias (rad/s)
    
    # Clock states
    clock_bias: np.ndarray = field(default_factory=lambda: np.zeros(1))    # Receiver clock bias (m)
    clock_drift: np.ndarray = field(default_factory=lambda: np.zeros(1))   # Receiver clock drift (m/s)
    
    # Inter-system biases (ISB)
    isb: Dict[str, float] = field(default_factory=dict)                    # System -> ISB (m)
    
    # Ambiguities
    ambiguities: Dict[int, float] = field(default_factory=dict)            # Satellite -> ambiguity (cycles)
    
    # Covariance
    covariance: Optional[np.ndarray] = None                                # State covariance matrix
    
    @property
    def dcm(self) -> np.ndarray:
        """Get DCM from quaternion"""
        return quaternion2dcm(self.quaternion)
    
    @dcm.setter
    def dcm(self, R: np.ndarray):
        """Set quaternion from DCM"""
        self.quaternion = dcm2quaternion(R)
        
    @property
    def attitude_euler(self) -> np.ndarray:
        """Get Euler angles (roll, pitch, yaw)"""
        from ..coordinate.rotation import dcm2euler
        return dcm2euler(self.dcm)
    
    def get_imu_bias(self) -> IMUBias:
        """Get IMU bias object"""
        return IMUBias(acc_bias=self.acc_bias, gyro_bias=self.gyro_bias)
    
    def get_state_vector(self) -> np.ndarray:
        """Get state as vector for filtering"""
        # Basic states: pos(3) + vel(3) + quat(4) + ba(3) + bg(3) + clk(1) + clk_drift(1)
        state = np.concatenate([
            self.position,
            self.velocity,
            self.quaternion,
            self.acc_bias,
            self.gyro_bias,
            self.clock_bias,
            self.clock_drift
        ])
        
        # Add ISBs
        for system in sorted(self.isb.keys()):
            state = np.append(state, self.isb[system])
            
        # Add ambiguities
        for sat in sorted(self.ambiguities.keys()):
            state = np.append(state, self.ambiguities[sat])
            
        return state
    
    def set_state_vector(self, x: np.ndarray):
        """Set state from vector"""
        idx = 0
        
        # Position
        self.position = x[idx:idx+3]
        idx += 3
        
        # Velocity
        self.velocity = x[idx:idx+3]
        idx += 3
        
        # Quaternion (normalize)
        self.quaternion = x[idx:idx+4]
        self.quaternion /= np.linalg.norm(self.quaternion)
        idx += 4
        
        # IMU biases
        self.acc_bias = x[idx:idx+3]
        idx += 3
        self.gyro_bias = x[idx:idx+3]
        idx += 3
        
        # Clock
        self.clock_bias = x[idx:idx+1]
        idx += 1
        self.clock_drift = x[idx:idx+1]
        idx += 1
        
        # ISBs
        for system in sorted(self.isb.keys()):
            self.isb[system] = x[idx]
            idx += 1
            
        # Ambiguities
        for sat in sorted(self.ambiguities.keys()):
            self.ambiguities[sat] = x[idx]
            idx += 1
    
    def copy(self) -> 'NavigationState':
        """Create deep copy of state"""
        return NavigationState(
            time=self.time,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            quaternion=self.quaternion.copy(),
            acc_bias=self.acc_bias.copy(),
            gyro_bias=self.gyro_bias.copy(),
            clock_bias=self.clock_bias.copy(),
            clock_drift=self.clock_drift.copy(),
            isb=self.isb.copy(),
            ambiguities=self.ambiguities.copy(),
            covariance=self.covariance.copy() if self.covariance is not None else None
        )


@dataclass
class StateError:
    """Error state representation for error-state formulation"""
    
    # Position, velocity errors
    d_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    d_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Attitude error (rotation vector)
    d_theta: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Bias errors
    d_acc_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    d_gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Clock errors
    d_clock_bias: float = 0.0
    d_clock_drift: float = 0.0
    
    # ISB errors
    d_isb: Dict[str, float] = field(default_factory=dict)
    
    # Ambiguity errors
    d_ambiguities: Dict[int, float] = field(default_factory=dict)
    
    def to_vector(self) -> np.ndarray:
        """Convert to error vector"""
        dx = np.concatenate([
            self.d_position,
            self.d_velocity,
            self.d_theta,
            self.d_acc_bias,
            self.d_gyro_bias,
            [self.d_clock_bias],
            [self.d_clock_drift]
        ])
        
        # ISBs
        for system in sorted(self.d_isb.keys()):
            dx = np.append(dx, self.d_isb[system])
            
        # Ambiguities
        for sat in sorted(self.d_ambiguities.keys()):
            dx = np.append(dx, self.d_ambiguities[sat])
            
        return dx
    
    def from_vector(self, dx: np.ndarray):
        """Set from error vector"""
        idx = 0
        
        self.d_position = dx[idx:idx+3]
        idx += 3
        
        self.d_velocity = dx[idx:idx+3]
        idx += 3
        
        self.d_theta = dx[idx:idx+3]
        idx += 3
        
        self.d_acc_bias = dx[idx:idx+3]
        idx += 3
        
        self.d_gyro_bias = dx[idx:idx+3]
        idx += 3
        
        self.d_clock_bias = dx[idx]
        idx += 1
        
        self.d_clock_drift = dx[idx]
        idx += 1
        
        # ISBs
        for system in sorted(self.d_isb.keys()):
            self.d_isb[system] = dx[idx]
            idx += 1
            
        # Ambiguities
        for sat in sorted(self.d_ambiguities.keys()):
            self.d_ambiguities[sat] = dx[idx]
            idx += 1


class StateManager:
    """Manage navigation state and error state"""
    
    def __init__(self, initial_state: Optional[NavigationState] = None):
        """Initialize state manager"""
        if initial_state is None:
            self.state = NavigationState()
        else:
            self.state = initial_state
            
        self.error_state = StateError()
        
        # State indices for easy access
        self._update_indices()
        
    def _update_indices(self):
        """Update state vector indices"""
        self.idx_pos = slice(0, 3)
        self.idx_vel = slice(3, 6)
        self.idx_quat = slice(6, 10)
        self.idx_ba = slice(10, 13)
        self.idx_bg = slice(13, 16)
        self.idx_clk = 16
        self.idx_clk_drift = 17
        
        # Variable indices
        idx = 18
        self.idx_isb = {}
        for system in sorted(self.state.isb.keys()):
            self.idx_isb[system] = idx
            idx += 1
            
        self.idx_amb = {}
        for sat in sorted(self.state.ambiguities.keys()):
            self.idx_amb[sat] = idx
            idx += 1
            
        self.state_dim = idx
        
    def predict(self, imu_data, dt: float):
        """Predict state forward using IMU"""
        # Implementation would go here
        pass
    
    def update(self, error: StateError):
        """Update state with error state"""
        # Position and velocity
        self.state.position += error.d_position
        self.state.velocity += error.d_velocity
        
        # Attitude (quaternion update)
        from ..coordinate.rotation import axis_angle2dcm
        dR = axis_angle2dcm(error.d_theta, np.linalg.norm(error.d_theta))
        self.state.dcm = dR @ self.state.dcm
        
        # Biases
        self.state.acc_bias += error.d_acc_bias
        self.state.gyro_bias += error.d_gyro_bias
        
        # Clock
        self.state.clock_bias[0] += error.d_clock_bias
        self.state.clock_drift[0] += error.d_clock_drift
        
        # ISBs
        for system, d_isb in error.d_isb.items():
            self.state.isb[system] += d_isb
            
        # Ambiguities
        for sat, d_amb in error.d_ambiguities.items():
            self.state.ambiguities[sat] += d_amb
            
        # Reset error state
        self.error_state = StateError()
        
        # Update indices if state dimension changed
        self._update_indices()