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

"""Double Difference Carrier Phase Factor for GTSAM"""

import numpy as np
import gtsam
from pyins.core.stats import ERATIO_L1, ERATIO_L2, ERATIO_L5, ERR_CONSTANT


class DDCarrierPhaseFactor:
    """Double Difference Carrier Phase Factor with integer ambiguity
    
    Computes: DD_φ = (φ_r^j - φ_r^i) - (φ_b^j - φ_b^i) = DD_ρ/λ + DD_N
    where:
    - φ: carrier phase measurements (cycles)
    - DD_ρ: double difference geometric range (meters)
    - λ: wavelength (meters)
    - DD_N: double difference integer ambiguity (cycles)
    """
    
    def __init__(self, position_key, ambiguity_key, dd_phase_meters,
                 sat_pos_ref, sat_pos_other, base_pos_ecef, wavelength, 
                 noise_model, reference_llh, use_eratio=False):
        """Initialize DD Carrier Phase Factor
        
        Args:
            position_key: GTSAM symbol for rover position (ENU)
            ambiguity_key: GTSAM symbol for DD integer ambiguity
            dd_phase_meters: Double difference carrier phase measurement (meters)
            sat_pos_ref: Reference satellite ECEF position [x, y, z]
            sat_pos_other: Other satellite ECEF position [x, y, z]
            base_pos_ecef: Base station ECEF position [x, y, z]
            wavelength: Carrier wavelength (meters)
            noise_model: GTSAM noise model (can be None if use_eratio=True)
            reference_llh: Reference LLH for ENU conversion [lat, lon, height]
            use_eratio: If True, create noise model using RTKLIB phase error (3mm)
        """
        self.position_key = position_key
        self.ambiguity_key = ambiguity_key
        self.dd_phase_cycles = dd_phase_meters / wavelength  # Convert to cycles
        self.wavelength = wavelength
        self.sat_pos_ref = sat_pos_ref
        self.sat_pos_other = sat_pos_other
        self.base_pos_ecef = base_pos_ecef
        
        # Create noise model based on RTKLIB if requested
        if use_eratio:
            phase_sigma = ERR_CONSTANT  # 0.003m = 3mm
            # Convert to cycles for the noise model
            phase_sigma_cycles = phase_sigma / wavelength
            self.noise_model = gtsam.noiseModel.Isotropic.Sigma(1, phase_sigma_cycles)
        else:
            self.noise_model = noise_model
            
        self.reference_llh = reference_llh
        
        # Pre-compute rotation matrix
        from pyins.coordinate.dcm import enu2ecef_dcm
        # Ensure reference_llh is a proper numpy array
        if isinstance(reference_llh, np.ndarray):
            self.reference_llh = reference_llh.copy()
        else:
            self.reference_llh = np.array(reference_llh)
        
        
        self.R_enu2ecef = enu2ecef_dcm(self.reference_llh)
    
    def create_factor(self):
        """Create GTSAM CustomFactor"""
        # Define error function
        def error_func(this: gtsam.CustomFactor, v: gtsam.Values, H: list):
            # Get rover position in ENU (relative to base)
            rover_enu = v.atPoint3(self.position_key)
            
            # Get DD ambiguity
            dd_ambiguity = v.atDouble(self.ambiguity_key)
            
            # Convert to ECEF
            rover_ecef = self.base_pos_ecef + self.R_enu2ecef @ rover_enu
            
            # Compute geometric ranges
            range_rover_ref = np.linalg.norm(self.sat_pos_ref - rover_ecef)
            range_rover_other = np.linalg.norm(self.sat_pos_other - rover_ecef)
            range_base_ref = np.linalg.norm(self.sat_pos_ref - self.base_pos_ecef)
            range_base_other = np.linalg.norm(self.sat_pos_other - self.base_pos_ecef)
            
            # Compute double difference range
            dd_range = (range_rover_other - range_rover_ref) - (range_base_other - range_base_ref)
            
            # Convert to cycles
            dd_range_cycles = dd_range / self.wavelength
            
            # Compute error: measured - (computed + ambiguity)
            error = np.array([self.dd_phase_cycles - (dd_range_cycles + dd_ambiguity)], dtype=np.float64)
            
            # Compute Jacobians if requested
            if H is not None and len(H) > 0:
                # Unit vectors (line-of-sight)
                e_ref = (self.sat_pos_ref - rover_ecef) / range_rover_ref
                e_other = (self.sat_pos_other - rover_ecef) / range_rover_other
                
                # DD geometry matrix in ECEF
                H_ecef = -e_other + e_ref
                
                # Transform to ENU
                H_enu = H_ecef @ self.R_enu2ecef
                
                # Scale by wavelength to get cycles
                H_enu_cycles = H_enu / self.wavelength
                
                # Set Jacobians (same convention as pseudorange factor)
                H[0] = np.asarray(-H_enu_cycles.reshape(1, 3), dtype=np.float64)  # w.r.t. position
                if len(H) > 1:
                    H[1] = np.array([[-1.0]], dtype=np.float64)  # w.r.t. ambiguity
            
            return error
        
        # Create and return factor
        return gtsam.CustomFactor(self.noise_model, 
                                 [self.position_key, self.ambiguity_key], 
                                 error_func)


class DDCarrierPhaseFactorFixed:
    """Double Difference Carrier Phase Factor with fixed integer ambiguity
    
    Same as DDCarrierPhaseFactor but with known/fixed ambiguity
    """
    
    def __init__(self, position_key, dd_phase_cycles, dd_ambiguity_fixed, wavelength,
                 sat_pos_ref, sat_pos_other, base_pos_ecef, noise_model, reference_llh, use_eratio=False):
        """Initialize DD Carrier Phase Factor with fixed ambiguity
        
        Args:
            position_key: GTSAM symbol for rover position (ENU relative to base)
            dd_phase_cycles: Double difference carrier phase measurement (cycles)
            dd_ambiguity_fixed: Fixed DD integer ambiguity (cycles)
            wavelength: Carrier wavelength (meters)
            sat_pos_ref: Reference satellite ECEF position [x, y, z]
            sat_pos_other: Other satellite ECEF position [x, y, z]
            base_pos_ecef: Base station ECEF position [x, y, z]
            noise_model: GTSAM noise model (can be None if use_eratio=True)
            reference_llh: Reference LLH for ENU conversion [lat, lon, height]
            use_eratio: If True, create noise model using RTKLIB phase error (3mm)
        """
        self.position_key = position_key
        self.dd_phase_cycles = dd_phase_cycles
        self.dd_ambiguity_fixed = dd_ambiguity_fixed
        self.wavelength = wavelength
        self.sat_pos_ref = sat_pos_ref
        self.sat_pos_other = sat_pos_other
        self.base_pos_ecef = base_pos_ecef
        
        # Create noise model based on RTKLIB if requested
        if use_eratio:
            phase_sigma = ERR_CONSTANT  # 0.003m = 3mm
            # Already in cycles, so convert meters to cycles
            phase_sigma_cycles = phase_sigma / wavelength
            self.noise_model = gtsam.noiseModel.Isotropic.Sigma(1, phase_sigma_cycles)
        else:
            self.noise_model = noise_model
            
        self.reference_llh = reference_llh
        
        # Pre-compute rotation matrix
        from pyins.coordinate.dcm import enu2ecef_dcm
        # Ensure reference_llh is a proper numpy array
        if isinstance(reference_llh, np.ndarray):
            self.reference_llh = reference_llh.copy()
        else:
            self.reference_llh = np.array(reference_llh)
        
        
        self.R_enu2ecef = enu2ecef_dcm(self.reference_llh)
    
    def create_factor(self):
        """Create GTSAM CustomFactor"""
        # Define error function
        def error_func(this: gtsam.CustomFactor, v: gtsam.Values, H: list):
            # Get rover position in ENU (relative to base)
            rover_enu = v.atPoint3(self.position_key)
            
            # Convert to ECEF
            rover_ecef = self.base_pos_ecef + self.R_enu2ecef @ rover_enu
            
            # Compute geometric ranges
            range_rover_ref = np.linalg.norm(self.sat_pos_ref - rover_ecef)
            range_rover_other = np.linalg.norm(self.sat_pos_other - rover_ecef)
            range_base_ref = np.linalg.norm(self.sat_pos_ref - self.base_pos_ecef)
            range_base_other = np.linalg.norm(self.sat_pos_other - self.base_pos_ecef)
            
            # Compute double difference range
            dd_range = (range_rover_other - range_rover_ref) - (range_base_other - range_base_ref)
            
            # Convert to cycles
            dd_range_cycles = dd_range / self.wavelength
            
            # Compute error: measured - (computed + fixed_ambiguity)
            error = np.array([self.dd_phase_cycles - (dd_range_cycles + self.dd_ambiguity_fixed)], dtype=np.float64)
            
            # Compute Jacobian if requested
            if H is not None and len(H) > 0:
                # Unit vectors (line-of-sight)
                e_ref = (self.sat_pos_ref - rover_ecef) / range_rover_ref
                e_other = (self.sat_pos_other - rover_ecef) / range_rover_other
                
                # DD geometry matrix in ECEF
                H_ecef = -e_other + e_ref
                
                # Transform to ENU
                H_enu = H_ecef @ self.R_enu2ecef
                
                # Scale by wavelength to get cycles
                H_enu_cycles = H_enu / self.wavelength
                
                # Set Jacobian (same convention as pseudorange factor)
                H[0] = np.asarray(-H_enu_cycles.reshape(1, 3), dtype=np.float64)
            
            return error
        
        # Create and return factor
        return gtsam.CustomFactor(self.noise_model, 
                                 [self.position_key], 
                                 error_func)