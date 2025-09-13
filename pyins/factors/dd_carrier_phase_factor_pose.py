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
Double Difference Carrier Phase Factor for Pose3.
Extracts position from Pose3 for DD carrier phase computation.
"""

import gtsam
import numpy as np
from typing import List


class DDCarrierPhasePoseFactor(gtsam.CustomFactor):
    """
    Custom factor for double-differenced carrier phase measurements using Pose3.
    
    This factor extracts the position from a Pose3 state and applies
    the double difference carrier phase constraint with integer ambiguity.
    
    Error model:
    e = DD_phase_measured - (DD_range_predicted + λ * N_DD)
    where DD = (φ_rover_sat - φ_rover_ref) - (φ_base_sat - φ_base_ref)
    """
    
    def __init__(self, pose_key: int, ambiguity_key: int,
                 dd_phase: float, sat_pos: np.ndarray, ref_pos: np.ndarray,
                 base_pos: np.ndarray, wavelength: float, noise_model):
        """
        Initialize DD carrier phase factor for Pose3.
        
        Parameters:
        -----------
        pose_key : int
            Key for the rover Pose3 state
        ambiguity_key : int
            Key for the double-differenced integer ambiguity
        dd_phase : float
            Double-differenced carrier phase measurement (in meters)
        sat_pos : np.ndarray
            Satellite position ECEF [x, y, z]
        ref_pos : np.ndarray
            Reference satellite position ECEF [x, y, z]
        base_pos : np.ndarray
            Base station position ECEF [x, y, z]
        wavelength : float
            Carrier wavelength (meters)
        noise_model : gtsam.noiseModel
            Noise model for the measurement
        """
        super().__init__(noise_model, [pose_key, ambiguity_key])
        self.dd_phase = dd_phase
        self.sat_pos = sat_pos
        self.ref_pos = ref_pos
        self.base_pos = base_pos
        self.wavelength = wavelength
        
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
        # Extract pose and position
        pose = values.atPose3(self.keys()[0])
        rover_pos = pose.translation()
        
        # Extract ambiguity
        ambiguity = values.atDouble(self.keys()[1])
        
        # Compute ranges
        range_rover_sat = np.linalg.norm(self.sat_pos - rover_pos)
        range_rover_ref = np.linalg.norm(self.ref_pos - rover_pos)
        range_base_sat = np.linalg.norm(self.sat_pos - self.base_pos)
        range_base_ref = np.linalg.norm(self.ref_pos - self.base_pos)
        
        # Double difference range
        dd_range = (range_rover_sat - range_rover_ref) - (range_base_sat - range_base_ref)
        
        # Predicted phase with ambiguity
        dd_predicted = dd_range + self.wavelength * ambiguity
        
        # Error
        error = self.dd_phase - dd_predicted
        
        return np.array([error])
    
    def jacobian(self, values: gtsam.Values) -> List[np.ndarray]:
        """
        Calculate Jacobian for the factor.
        
        Returns Jacobians with respect to:
        1. Pose3 (1x6) - first 3 for rotation (zero), last 3 for position
        2. Ambiguity (1x1)
        """
        # Extract pose and position
        pose = values.atPose3(self.keys()[0])
        rover_pos = pose.translation()
        
        # Unit vectors
        vec_sat = self.sat_pos - rover_pos
        vec_ref = self.ref_pos - rover_pos
        range_sat = np.linalg.norm(vec_sat)
        range_ref = np.linalg.norm(vec_ref)
        
        jacobians = []
        
        # Jacobian with respect to Pose3
        if range_sat < 1e-6 or range_ref < 1e-6:
            J_pose = np.zeros((1, 6))
        else:
            unit_sat = vec_sat / range_sat
            unit_ref = vec_ref / range_ref
            
            # Jacobian with respect to position
            J_pos = -(unit_sat - unit_ref).reshape(1, 3)
            
            # Full Jacobian with respect to Pose3 (1x6)
            J_pose = np.zeros((1, 6))
            J_pose[0, 3:6] = J_pos[0, :]
        
        jacobians.append(J_pose)
        
        # Jacobian with respect to ambiguity
        J_ambiguity = np.array([[self.wavelength]])
        jacobians.append(J_ambiguity)
        
        return jacobians
    
    @staticmethod
    def Create(pose_key: int, ambiguity_key: int,
               dd_phase: float, sat_pos: np.ndarray, ref_pos: np.ndarray,
               base_pos: np.ndarray, wavelength: float, noise_model):
        """Factory method to create the factor."""
        return DDCarrierPhasePoseFactor(
            pose_key, ambiguity_key, dd_phase, sat_pos, ref_pos,
            base_pos, wavelength, noise_model
        )