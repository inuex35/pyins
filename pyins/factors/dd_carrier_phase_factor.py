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
    
    def __init__(self, position_key, ambiguity_key, dd_phase_cycles,
                 sat_pos_ref, sat_pos_other, base_pos_ecef, wavelength,
                 noise_model, reference_llh,
                 rover_obs_ref=None, rover_obs_other=None,
                 base_obs_ref=None, base_obs_other=None):
        """Initialize DD Carrier Phase Factor

        Args:
            position_key: GTSAM symbol for rover position (ENU)
            ambiguity_key: GTSAM symbol for DD integer ambiguity
            dd_phase_cycles: Double difference carrier phase measurement (cycles)
            sat_pos_ref: Reference satellite ECEF position [x, y, z]
            sat_pos_other: Other satellite ECEF position [x, y, z]
            base_pos_ecef: Base station ECEF position [x, y, z]
            wavelength: Carrier wavelength (meters)
            noise_model: GTSAM noise model (if None, defaults to RTKLIB 3mm phase error)
            reference_llh: Reference LLH for ENU conversion [lat, lon, height]
            rover_obs_ref: Rover observation to reference satellite (cycles) - for RTKLIB-style residual
            rover_obs_other: Rover observation to other satellite (cycles) - for RTKLIB-style residual
            base_obs_ref: Base observation to reference satellite (cycles) - for RTKLIB-style residual
            base_obs_other: Base observation to other satellite (cycles) - for RTKLIB-style residual
        """
        self.position_key = position_key
        self.ambiguity_key = ambiguity_key
        self.dd_phase_cycles = dd_phase_cycles  # Already in cycles
        self.wavelength = wavelength
        self.sat_pos_ref = np.array(sat_pos_ref)
        self.sat_pos_other = np.array(sat_pos_other)
        self.base_pos_ecef = np.array(base_pos_ecef)

        # Store individual observations for RTKLIB-style residual calculation
        # These are required for proper DD residual computation
        self.rover_obs_ref = rover_obs_ref      # in cycles
        self.rover_obs_other = rover_obs_other  # in cycles
        self.base_obs_ref = base_obs_ref        # in cycles
        self.base_obs_other = base_obs_other    # in cycles

        # Create noise model - default to RTKLIB 3mm phase error
        if noise_model is not None:
            # Use provided noise model
            self.noise_model = noise_model
        else:
            # Default to RTKLIB phase error (3mm) converted to cycles
            phase_sigma = ERR_CONSTANT  # 0.003m = 3mm
            phase_sigma_cycles = phase_sigma / wavelength
            self.noise_model = gtsam.noiseModel.Isotropic.Sigma(1, phase_sigma_cycles)
            
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
        """
        Create GTSAM CustomFactor for DD carrier phase with ambiguity estimation.

        This method creates a custom GTSAM factor that implements the double difference
        carrier phase measurement model. The factor estimates both the rover position
        and the integer ambiguity as part of the optimization.

        The measurement model is:
        DD_φ = (φ_r^j - φ_r^i) - (φ_b^j - φ_b^i) = DD_ρ/λ + DD_N

        Where:
        - DD_φ: Double difference carrier phase measurement (cycles)
        - DD_ρ: Double difference geometric range (meters)
        - λ: Carrier wavelength (meters)
        - DD_N: Double difference integer ambiguity (cycles)

        Returns:
            gtsam.CustomFactor: Factor object that can be added to a factor graph.
                               Contains error function and Jacobian computation.

        Notes:
            - Error is computed as: measured - (computed + ambiguity)
            - Jacobians are computed for both position (3x1) and ambiguity (1x1)
            - Position Jacobian is transformed from ECEF to ENU coordinates
            - All computations maintain cycle units for numerical stability
        """
        # Define error function
        def error_func(this: gtsam.CustomFactor, v: gtsam.Values, H: list):
            """
            Compute error and Jacobians for DD carrier phase factor.

            This inner function calculates the residual between measured and computed
            double difference carrier phase, including the integer ambiguity term.
            It also computes analytical Jacobians for efficient optimization.

            Parameters:
                this: Reference to the CustomFactor (required by GTSAM)
                v: Current variable values containing position and ambiguity
                H: List to store computed Jacobians (empty if not needed)

            Returns:
                np.ndarray: Error vector (1x1) in cycles

            Side Effects:
                - Populates H[0] with position Jacobian (1x3) if H is provided
                - Populates H[1] with ambiguity Jacobian (1x1) if H is provided
            """
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
            
            # RTKLIB-style: DD observation - (DD range + ambiguity)
            if self.rover_obs_ref is not None:
                # Form DD from observations (in cycles)
                sd_rover = self.rover_obs_other - self.rover_obs_ref
                sd_base = self.base_obs_other - self.base_obs_ref
                dd_obs = sd_rover - sd_base

                # Form DD from ranges (in cycles)
                dd_range = (range_rover_other - range_rover_ref) - (range_base_other - range_base_ref)
                dd_range_cycles = dd_range / self.wavelength

                # RTKLIB residual: v = L - (ρ + N)
                # where L is DD observation, ρ is DD range, N is ambiguity
                residual = dd_obs - (dd_range_cycles + dd_ambiguity)

                error = np.array([residual], dtype=np.float64)
            else:
                # Fallback: traditional method if observations not provided
                dd_range = (range_rover_other - range_rover_ref) - (range_base_other - range_base_ref)
                dd_range_cycles = dd_range / self.wavelength
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
        """
        Create GTSAM CustomFactor for DD carrier phase with fixed ambiguity.

        This method creates a custom GTSAM factor for cases where the integer
        ambiguity has been successfully resolved and fixed. Only the rover
        position is estimated, making the optimization more stable and faster.

        The measurement model with fixed ambiguity is:
        DD_φ = DD_ρ/λ + DD_N_fixed

        Where DD_N_fixed is the pre-resolved integer ambiguity.

        Returns:
            gtsam.CustomFactor: Factor object with fixed ambiguity constraint.
                               Only estimates position, not ambiguity.

        Notes:
            - More accurate than float solution when ambiguities are correctly fixed
            - Typically achieves cm-level positioning accuracy
            - Only computes position Jacobian (3x1), no ambiguity Jacobian
            - Used after successful ambiguity resolution validation
        """
        # Define error function
        def error_func(this: gtsam.CustomFactor, v: gtsam.Values, H: list):
            """
            Compute error and Jacobian for DD carrier phase with fixed ambiguity.

            This inner function calculates the residual using a known integer
            ambiguity value, simplifying the measurement model and improving
            solution stability.

            Parameters:
                this: Reference to the CustomFactor (required by GTSAM)
                v: Current variable values containing only position
                H: List to store computed Jacobian (empty if not needed)

            Returns:
                np.ndarray: Error vector (1x1) in cycles

            Side Effects:
                - Populates H[0] with position Jacobian (1x3) if H is provided
                - No ambiguity Jacobian since ambiguity is fixed
            """
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