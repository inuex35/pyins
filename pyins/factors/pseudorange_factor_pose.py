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
Pseudorange factor for Pose3 states in GTSAM.
Extracts position from Pose3 for pseudorange computation.
"""

import numpy as np
import gtsam
from typing import List, Optional


class PseudorangePoseFactor(gtsam.CustomFactor):
    """
    Custom factor for GNSS pseudorange measurements with Pose3 states.
    
    This factor relates a Pose3 state and clock bias to a pseudorange measurement.
    The position is extracted from the Pose3 for range computation.
    """
    
    def __init__(self, pose_key: int, clock_key: int, measured_range: float,
                 sat_pos: np.ndarray, sat_clock: float, noise_model):
        """
        Initialize pseudorange factor for Pose3.
        
        Parameters:
        -----------
        pose_key : int
            Key for the Pose3 state
        clock_key : int
            Key for the receiver clock bias
        measured_range : float
            Measured pseudorange (m)
        sat_pos : np.ndarray
            Satellite position in ECEF (m)
        sat_clock : float
            Satellite clock bias (m)
        noise_model : gtsam.noiseModel
            Noise model for the measurement
        """
        # Store measurement data
        self.measured_range = measured_range
        self.sat_pos = sat_pos
        self.sat_clock = sat_clock
        
        # Initialize CustomFactor with keys and noise model
        super().__init__(noise_model, [pose_key, clock_key], self.error_func)
        
    def error_func(self, this: gtsam.CustomFactor, 
                   values: gtsam.Values, 
                   jacobians: Optional[List[np.ndarray]]) -> np.ndarray:
        """
        Compute error and Jacobians for pseudorange measurement.
        
        Parameters:
        -----------
        this : gtsam.CustomFactor
            Reference to this factor
        values : gtsam.Values
            Current values
        jacobians : List[np.ndarray], optional
            Output Jacobians
            
        Returns:
        --------
        error : np.ndarray
            Error vector (1x1)
        """
        # Get keys
        pose_key = this.keys()[0]
        clock_key = this.keys()[1]
        
        # Extract pose and then position
        pose = values.atPose3(pose_key)
        receiver_pos = pose.translation()  # Extract position from Pose3
        
        # Extract clock bias
        clock_bias = values.atDouble(clock_key)
        
        # Compute geometric range
        delta = self.sat_pos - receiver_pos
        geometric_range = np.linalg.norm(delta)
        
        # Predicted pseudorange = geometric range + receiver clock - satellite clock
        predicted_range = geometric_range + clock_bias - self.sat_clock
        
        # Error = measured - predicted
        error = np.array([self.measured_range - predicted_range])
        
        # Compute Jacobians if requested
        if jacobians is not None:
            # Jacobian w.r.t. Pose3 (1x6: rotation and position)
            # Derivative of range w.r.t. position
            if geometric_range > 0:
                drange_dpos = -delta / geometric_range  # 1x3
            else:
                drange_dpos = np.zeros(3)
            
            # Full Jacobian for Pose3: [rotation(3) | position(3)]
            # Range doesn't depend on rotation, so first 3 columns are zero
            J_pose = np.zeros((1, 6))
            J_pose[0, 3:6] = -drange_dpos  # Negative because error = measured - predicted
            
            # Jacobian w.r.t. clock bias
            J_clock = np.array([[-1.0]])  # Negative because error = measured - predicted
            
            jacobians[0] = J_pose
            jacobians[1] = J_clock
            
        return error


class PseudorangeWithISBPoseFactor(gtsam.CustomFactor):
    """
    Pseudorange factor with inter-system bias support for Pose3 states.
    
    Handles different GNSS systems (GPS, GLONASS, Galileo, BeiDou) with ISBs.
    """
    
    def __init__(self, pose_key: int, clock_key: int, measured_range: float,
                 sat_pos: np.ndarray, sat_clock: float, system: str, noise_model):
        """
        Initialize pseudorange factor with ISB support.
        
        Parameters:
        -----------
        pose_key : int
            Key for the Pose3 state
        clock_key : int
            Key for the clock state vector [GPS_clock, GLO_ISB, GAL_ISB, BDS_ISB]
        measured_range : float
            Measured pseudorange (m)
        sat_pos : np.ndarray
            Satellite position in ECEF (m)
        sat_clock : float
            Satellite clock bias (m)
        system : str
            GNSS system ('G', 'R', 'E', 'C')
        noise_model : gtsam.noiseModel
            Noise model for the measurement
        """
        self.measured_range = measured_range
        self.sat_pos = sat_pos
        self.sat_clock = sat_clock
        self.system = system
        
        # System index for ISB
        self.system_idx = {'G': 0, 'R': 1, 'E': 2, 'C': 3}.get(system, 0)
        
        super().__init__(noise_model, [pose_key, clock_key], self.error_func)
        
    def error_func(self, this: gtsam.CustomFactor,
                   values: gtsam.Values,
                   jacobians: Optional[List[np.ndarray]]) -> np.ndarray:
        """
        Compute error and Jacobians for pseudorange with ISB.
        """
        # Get keys
        pose_key = this.keys()[0]
        clock_key = this.keys()[1]
        
        # Extract pose and position
        pose = values.atPose3(pose_key)
        receiver_pos = pose.translation()
        
        # Extract clock state vector [GPS_clock, GLO_ISB, GAL_ISB, BDS_ISB]
        clock_state = values.atVector(clock_key)
        
        # Get appropriate clock bias based on system
        if self.system == 'G':
            clock_bias = clock_state[0]  # GPS clock
        else:
            # Other systems: GPS clock + ISB
            clock_bias = clock_state[0] + clock_state[self.system_idx]
        
        # Compute geometric range
        delta = self.sat_pos - receiver_pos
        geometric_range = np.linalg.norm(delta)
        
        # Predicted pseudorange
        predicted_range = geometric_range + clock_bias - self.sat_clock
        
        # Error
        error = np.array([self.measured_range - predicted_range])
        
        # Compute Jacobians
        if jacobians is not None:
            # Jacobian w.r.t. Pose3
            if geometric_range > 0:
                drange_dpos = -delta / geometric_range
            else:
                drange_dpos = np.zeros(3)
            
            J_pose = np.zeros((1, 6))
            J_pose[0, 3:6] = -drange_dpos
            
            # Jacobian w.r.t. clock state
            J_clock = np.zeros((1, 4))  # [GPS, GLO, GAL, BDS]
            J_clock[0, 0] = -1.0  # GPS clock always contributes
            if self.system != 'G':
                J_clock[0, self.system_idx] = -1.0  # ISB for non-GPS
            
            jacobians[0] = J_pose
            jacobians[1] = J_clock
            
        return error