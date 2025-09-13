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
Double Difference Pseudorange Factor for Pose3.
Extracts position from Pose3 for DD computation.
"""

import gtsam
import numpy as np
from typing import List


class DDPseudorangePoseFactor(gtsam.CustomFactor):
    """
    Custom factor for double-differenced pseudorange measurements using Pose3.
    
    This factor extracts the position from a Pose3 state and applies
    the double difference pseudorange constraint.
    
    Error model:
    e = DD_measured - DD_predicted
    where DD = (ρ_rover_sat - ρ_rover_ref) - (ρ_base_sat - ρ_base_ref)
    """
    
    def __init__(self, pose_key: int, dd_measurement: float,
                 sat_pos: np.ndarray, ref_pos: np.ndarray,
                 base_pos: np.ndarray, noise_model):
        """
        Initialize DD pseudorange factor for Pose3.
        
        Parameters:
        -----------
        pose_key : int
            Key for the rover Pose3 state
        dd_measurement : float
            Double-differenced pseudorange measurement
        sat_pos : np.ndarray
            Satellite position ECEF [x, y, z]
        ref_pos : np.ndarray
            Reference satellite position ECEF [x, y, z]
        base_pos : np.ndarray
            Base station position ECEF [x, y, z]
        noise_model : gtsam.noiseModel
            Noise model for the measurement
        """
        super().__init__(noise_model, [pose_key])
        self.dd_measurement = dd_measurement
        self.sat_pos = sat_pos
        self.ref_pos = ref_pos
        self.base_pos = base_pos
        
    def error(self, values: gtsam.Values) -> np.ndarray:
        """
        Calculate error for the factor.
        
        Parameters:
        -----------
        values : gtsam.Values
            Current estimates
            
        Returns:
        --------
        error : np.ndarray
            Error vector [1]
        """
        # Extract pose and then position
        pose = values.atPose3(self.keys()[0])
        rover_pos = pose.translation()
        
        # Compute ranges
        range_rover_sat = np.linalg.norm(self.sat_pos - rover_pos)
        range_rover_ref = np.linalg.norm(self.ref_pos - rover_pos)
        range_base_sat = np.linalg.norm(self.sat_pos - self.base_pos)
        range_base_ref = np.linalg.norm(self.ref_pos - self.base_pos)
        
        # Double difference
        dd_predicted = (range_rover_sat - range_rover_ref) - (range_base_sat - range_base_ref)
        
        # Error
        error = self.dd_measurement - dd_predicted
        
        return np.array([error])
    
    def jacobian(self, values: gtsam.Values) -> List[np.ndarray]:
        """
        Calculate Jacobian for the factor.
        
        Returns Jacobian with respect to Pose3 (6x1).
        The first 3 rows are for rotation (zero), last 3 for position.
        """
        # Extract pose and position
        pose = values.atPose3(self.keys()[0])
        rover_pos = pose.translation()
        
        # Unit vectors
        vec_sat = self.sat_pos - rover_pos
        vec_ref = self.ref_pos - rover_pos
        range_sat = np.linalg.norm(vec_sat)
        range_ref = np.linalg.norm(vec_ref)
        
        if range_sat < 1e-6 or range_ref < 1e-6:
            return [np.zeros((1, 6))]
        
        unit_sat = vec_sat / range_sat
        unit_ref = vec_ref / range_ref
        
        # Jacobian with respect to position (1x3)
        J_pos = -(unit_sat - unit_ref).reshape(1, 3)
        
        # Full Jacobian with respect to Pose3 (1x6)
        # First 3 columns for rotation (zero), last 3 for position
        J_full = np.zeros((1, 6))
        J_full[0, 3:6] = J_pos[0, :]
        
        return [J_full]
    
    @staticmethod
    def Create(pose_key: int, dd_measurement: float,
               sat_pos: np.ndarray, ref_pos: np.ndarray,
               base_pos: np.ndarray, noise_model):
        """Factory method to create the factor."""
        return DDPseudorangePoseFactor(
            pose_key, dd_measurement, sat_pos, ref_pos, base_pos, noise_model
        )