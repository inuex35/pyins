#!/usr/bin/env python3
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
Double Difference Factor for GTSAM
"""

import gtsam
import numpy as np
from typing import Optional, List


def create_dd_error_func(dd_obs, sat_pos, ref_sat_pos, base_pos, sat_clk, ref_sat_clk):
    """
    Create error function for double difference factor with closure over parameters.
    """
    def error_func(measurement: gtsam.CustomFactor, values: gtsam.Values, jacobians: Optional[List[np.ndarray]]) -> np.ndarray:
        """
        Error function for double difference factor.
        
        The DD model is:
        DD_predicted = (||x - s_i|| - ||x - s_j||) - (||b - s_i|| - ||b - s_j||)
                     + (δt_i - δt_j)
        
        where:
        - x is rover position
        - b is base position
        - s_i, s_j are satellite positions
        - δt_i, δt_j are satellite clock biases
        """
        # Get rover position from Pose3
        pose = values.atPose3(measurement.keys()[0])
        rover_pos = pose.translation()
        
        # Compute ranges from rover to satellites
        vec_to_sat = sat_pos - rover_pos
        vec_to_ref = ref_sat_pos - rover_pos
        rover_to_sat = np.linalg.norm(vec_to_sat)
        rover_to_ref = np.linalg.norm(vec_to_ref)
        
        # Compute ranges from base to satellites
        base_to_sat = np.linalg.norm(sat_pos - base_pos)
        base_to_ref = np.linalg.norm(ref_sat_pos - base_pos)
        
        # Single differences
        sd_rover = rover_to_sat - rover_to_ref
        sd_base = base_to_sat - base_to_ref
        
        # Double difference prediction
        # DD = (range difference at rover) - (range difference at base)
        # Satellite clocks should cancel out in DD
        dd_pred = sd_rover - sd_base
        
        # Error (residual)
        error = np.array([dd_obs - dd_pred])
        
        # Compute Jacobian if requested
        if jacobians is not None:
            # Unit vectors (negated because we differentiate w.r.t. rover position)
            unit_to_sat = -vec_to_sat / rover_to_sat
            unit_to_ref = -vec_to_ref / rover_to_ref
            
            # DD Jacobian w.r.t. position (3x1)
            j_pos = unit_to_sat - unit_to_ref
            
            # Full Jacobian w.r.t. Pose3 (1x6: 3 for rotation, 3 for translation)
            # DD doesn't depend on rotation, so first 3 elements are zero
            J = np.zeros((1, 6))
            J[0, 3:6] = j_pos
            
            jacobians[0] = J
        
        return error
    
    return error_func


class DoubleDifferencePoseFactor(gtsam.CustomFactor):
    """
    Double Difference pseudorange factor for Pose3.
    
    This factor models the double difference observation:
    DD = (ρ_r^i - ρ_r^j) - (ρ_b^i - ρ_b^j)
    
    where:
    - ρ_r^i, ρ_r^j are ranges from rover to satellites i, j
    - ρ_b^i, ρ_b^j are ranges from base to satellites i, j
    """
    
    def __init__(self, pose_key: int, dd_obs: float, 
                 sat_pos: np.ndarray, ref_sat_pos: np.ndarray,
                 base_pos: np.ndarray, sat_clk: float, ref_sat_clk: float,
                 noise_model):
        """
        Initialize DD factor.
        
        Parameters:
        -----------
        pose_key : int
            Key for the Pose3 variable
        dd_obs : float
            Double difference observation (meters)
        sat_pos : np.ndarray
            Position of satellite i in ECEF
        ref_sat_pos : np.ndarray
            Position of reference satellite j in ECEF
        base_pos : np.ndarray
            Position of base station in ECEF
        sat_clk : float
            Satellite i clock bias (meters)
        ref_sat_clk : float
            Reference satellite j clock bias (meters)
        noise_model : gtsam.noiseModel
            Noise model for the measurement
        """
        # Create error function with closure over parameters
        error_func = create_dd_error_func(
            dd_obs, sat_pos, ref_sat_pos, base_pos, sat_clk, ref_sat_clk
        )
        
        # Initialize with error function
        super().__init__(noise_model, [pose_key], error_func)