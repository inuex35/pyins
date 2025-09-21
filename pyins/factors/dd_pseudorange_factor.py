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

"""Double Difference Pseudorange Factor for GTSAM - RTKLIB style implementation"""

import numpy as np
import gtsam
from pyins.core.constants import CLIGHT, FREQ_L1, FREQ_L2, FREQ_L5
from pyins.core.stats import ERATIO_L1 as ERRATIO_L1, ERATIO_L2 as ERRATIO_L2, ERATIO_L5 as ERRATIO_L5
from pyins.observation.measurement_model import troposphere_model, ionosphere_model
from pyins.coordinate.transforms import ecef2llh
from pyins.observation.pseudorange import elevation_angle

# Error parameters for RTK
ERR_BASE = 0.003  # Base error term (3mm) for carrier phase
ERR_EL = 0.003    # Elevation-dependent error term (3mm)
ERR_CONSTANT = 0.003  # Keep for compatibility


def compute_azimuth(sat_pos: np.ndarray, rcv_pos: np.ndarray) -> float:
    """Compute azimuth angle from receiver to satellite"""
    # Simple azimuth calculation (placeholder)
    # For atmospheric models, exact azimuth is not critical
    return 0.0


class DDPseudorangeFactor:
    """Double Difference Pseudorange Factor based on RTKLIB's ddres approach
    
    Following RTKLIB's implementation:
    - Works in ENU coordinates relative to base station
    - Compute line-of-sight vectors
    - Form design matrix H
    - Compute DD residuals
    - Uses RTKLIB eratio for automatic noise model generation
    """
    
    def __init__(self, position_key, dd_data, base_pos_ecef, reference_llh, 
                 noise_model=None, use_atmospheric=False, ion_params=None, gps_time=0.0):
        """Initialize DD Pseudorange Factor
        
        Args:
            position_key: GTSAM symbol for rover position (ENU relative to base)
            dd_data: Dictionary containing DD measurement data:
                     - 'dd_obs': Double difference pseudorange measurement (meters)
                     - 'ref_sat': Reference satellite number
                     - 'sat': Other satellite number  
                     - 'ref_sat_pos': Reference satellite ECEF position
                     - 'sat_pos': Other satellite ECEF position
                     - 'sat_clk': Satellite clock bias (seconds)
                     - 'ref_sat_clk': Reference satellite clock bias (seconds)
                     - 'freq_idx': Frequency index (0=L1, 1=L2, 2=L5)
            base_pos_ecef: Base station ECEF position [x, y, z]
            reference_llh: Reference LLH for ENU conversion [lat, lon, height]
            noise_model: GTSAM noise model (optional, auto-generated if None)
            use_atmospheric: Whether to apply atmospheric corrections (default: True)
            ion_params: Ionosphere parameters (8 values for Klobuchar model)
            gps_time: GPS time for ionospheric correction
        """
        # Extract data from dd_data dictionary
        self.dd_measurement = dd_data['dd_obs']
        self.sat_pos_ref = dd_data['ref_sat_pos'][:3] if 'ref_sat_pos' in dd_data else dd_data.get('sat_pos_ref', [0,0,0])[:3]
        self.sat_pos_other = dd_data['sat_pos'][:3] if 'sat_pos' in dd_data else dd_data.get('sat_pos_other', [0,0,0])[:3]
        self.sat_ref = dd_data.get('ref_sat', 0)
        self.sat_other = dd_data.get('sat', 0)
        self.freq_idx = dd_data.get('freq_idx', 0)  # Get frequency index from DD data
        
        # Store satellite clock biases - needed for residual-based DD
        self.sat_clk_other = dd_data.get('sat_clk', 0.0)
        self.sat_clk_ref = dd_data.get('ref_sat_clk', 0.0)
        
        # CRITICAL: Extract BASE satellite positions and clocks (at base time)
        # When there's a time difference between rover and base observations,
        # we MUST use base satellite positions/clocks for base calculations

        # Check if base satellite positions are provided (REQUIRED in v2.0)
        if 'base_sat_pos' not in dd_data and 'base_sat_pos_other' not in dd_data:
            raise ValueError(
                "DDPseudorangeFactor requires base satellite positions in v2.0+. "
                "Missing 'base_sat_pos' or 'base_sat_pos_other' in dd_data. "
                "This is likely because form_double_differences() was called from an older version. "
                "Please ensure you're using pyins v2.0+ for DD measurement formation."
            )
        if 'base_ref_sat_pos' not in dd_data and 'base_sat_pos_ref' not in dd_data:
            raise ValueError(
                "DDPseudorangeFactor requires base reference satellite position in v2.0+. "
                "Missing 'base_ref_sat_pos' or 'base_sat_pos_ref' in dd_data."
            )

        self.base_sat_pos_ref = dd_data.get('base_ref_sat_pos', dd_data.get('base_sat_pos_ref'))[:3]
        self.base_sat_pos_other = dd_data.get('base_sat_pos', dd_data.get('base_sat_pos_other'))[:3]
        self.base_sat_clk_ref = dd_data.get('base_ref_sat_clk', dd_data.get('base_sat_clk_ref', self.sat_clk_ref))
        self.base_sat_clk_other = dd_data.get('base_sat_clk', dd_data.get('base_sat_clk_other', self.sat_clk_other))
        
        # Extract raw observations for proper DD computation
        # When residual interpolation is used, we need raw observations
        self.use_residual_interp = dd_data.get('use_residual_interp', False)
        if self.use_residual_interp:
            self.rover_pr_other = dd_data.get('rover_pr', 0.0)
            self.rover_pr_ref = dd_data.get('rover_pr_ref', 0.0)
            self.base_pr_other = dd_data.get('base_pr', 0.0)
            self.base_pr_ref = dd_data.get('base_pr_ref', 0.0)
        
        # Store other parameters
        self.base_pos_ecef = base_pos_ecef
        self.reference_llh = reference_llh
        
        # Atmospheric correction parameters
        self.use_atmospheric = use_atmospheric
        self.ion_params = ion_params if ion_params is not None else np.zeros(8)
        self.gps_time = gps_time
        
        # Get frequency for ionospheric correction
        freq_map = [FREQ_L1, FREQ_L2, FREQ_L5]
        self.frequency = freq_map[min(self.freq_idx, 2)]
        
        # Pre-compute base station ranges WITH satellite clock corrections
        # CRITICAL: Use BASE satellite positions/clocks (at base time), not rover's!
        # This is essential when rover and base observations have different times
        self.range_base_ref = np.linalg.norm(self.base_sat_pos_ref - base_pos_ecef) - CLIGHT * self.base_sat_clk_ref
        self.range_base_other = np.linalg.norm(self.base_sat_pos_other - base_pos_ecef) - CLIGHT * self.base_sat_clk_other
        
        # Debug (commented out for performance)
        # print(f"Factor init: base-other={self.range_base_other:.3f}, base-ref={self.range_base_ref:.3f}")
        
        # Pre-compute atmospheric corrections for base station if enabled
        if use_atmospheric:
            base_llh = ecef2llh(base_pos_ecef)
            el_base_other = elevation_angle(self.sat_pos_other, base_pos_ecef)
            az_base_other = compute_azimuth(self.sat_pos_other, base_pos_ecef)
            el_base_ref = elevation_angle(self.sat_pos_ref, base_pos_ecef)
            az_base_ref = compute_azimuth(self.sat_pos_ref, base_pos_ecef)
            
            # Tropospheric delays for base
            self.trop_base_other = troposphere_model(el_base_other, base_llh[0], base_llh[2])
            self.trop_base_ref = troposphere_model(el_base_ref, base_llh[0], base_llh[2])
            
            # Ionospheric delays for base
            if np.any(ion_params):
                self.iono_base_other = ionosphere_model(
                    el_base_other, az_base_other, base_llh[0], base_llh[1],
                    gps_time, ion_params, self.frequency)
                self.iono_base_ref = ionosphere_model(
                    el_base_ref, az_base_ref, base_llh[0], base_llh[1],
                    gps_time, ion_params, self.frequency)
            else:
                self.iono_base_other = self.iono_base_ref = 0.0
            
            # Add atmospheric delays to base ranges
            self.range_base_ref += self.trop_base_ref + self.iono_base_ref
            self.range_base_other += self.trop_base_other + self.iono_base_other
        else:
            self.trop_base_other = self.trop_base_ref = 0.0
            self.iono_base_other = self.iono_base_ref = 0.0
        
        self.dd_base = self.range_base_other - self.range_base_ref
        
        # Pre-compute rotation matrix
        from pyins.coordinate.dcm import enu2ecef_dcm
        # Ensure reference_llh is a proper numpy array
        if isinstance(reference_llh, np.ndarray):
            self.reference_llh = reference_llh.copy()
        else:
            self.reference_llh = np.array(reference_llh)
        self.R_enu2ecef = enu2ecef_dcm(self.reference_llh)
        
        # Store parameters for error function
        self.position_key = position_key
        
        # Create noise model if not provided
        if noise_model is None:
            # Select ERATIO based on frequency index
            eratio_map = [ERRATIO_L1, ERRATIO_L2, ERRATIO_L5]
            eratio = eratio_map[min(self.freq_idx, 2)]  # Default to L5 ratio for higher frequencies
            
            # RTKLIB approach: variance = 2 * fact * (a^2 + b^2/sin^2(el))
            # where fact = eratio for pseudorange
            # Calculate elevation angle for the satellite
            el_deg = dd_data.get('elevation', 45.0)  # Default 45 degrees if not provided
            el_rad = np.radians(el_deg)
            sin_el = np.sin(el_rad)

            # Compute variance using RTKLIB formula
            # Note: RTKLIB uses factor of 2 for double difference
            a = ERR_BASE  # Base error term
            b = ERR_EL    # Elevation-dependent term
            var_base = 2.0 * eratio * (a*a + b*b/(sin_el*sin_el))

            # Convert variance to standard deviation
            code_sigma = np.sqrt(var_base)
            self.noise_model = gtsam.noiseModel.Isotropic.Sigma(1, code_sigma)
        else:
            self.noise_model = noise_model
        
    def create_factor(self):
        """Create GTSAM CustomFactor"""
        # Define error function
        def error_func(this: gtsam.CustomFactor, v: gtsam.Values, H: list):
            # Get rover position in ENU (relative to base)
            rover_enu_point = v.atPoint3(self.position_key)
            # Check if it's already a numpy array or a GTSAM Point3
            if isinstance(rover_enu_point, np.ndarray):
                rover_enu = rover_enu_point
            else:
                rover_enu = np.array([rover_enu_point.x(), rover_enu_point.y(), rover_enu_point.z()])
            
            # Convert to ECEF
            rover_ecef = self.base_pos_ecef + self.R_enu2ecef @ rover_enu
            
            # Compute geometric ranges (always needed for sanity check)
            vec_rover_other = self.sat_pos_other - rover_ecef
            geom_range_rover_other = np.linalg.norm(vec_rover_other)
            
            vec_rover_ref = self.sat_pos_ref - rover_ecef
            geom_range_rover_ref = np.linalg.norm(vec_rover_ref)
            
            # Compute rover ranges WITH satellite clock corrections
            range_rover_other = geom_range_rover_other - CLIGHT * self.sat_clk_other
            range_rover_ref = geom_range_rover_ref - CLIGHT * self.sat_clk_ref
            
            # Base ranges already pre-computed with satellite clock (use self. variables)
            range_base_other = self.range_base_other
            range_base_ref = self.range_base_ref
            
            # Check if we're using residual interpolation
            if self.use_residual_interp and self.rover_pr_other > 0:
                # PROPER DD COMPUTATION WITH RAW OBSERVATIONS
                # When we have raw observations, compute DD correctly:
                # DD = (rover_other - rover_ref) - (base_other - base_ref)

                # Single differences
                sd_rover = self.rover_pr_other - self.rover_pr_ref
                sd_base = self.base_pr_other - self.base_pr_ref

                # Double difference of observations
                dd_observations = sd_rover - sd_base

                # Compute base ranges using BASE time satellite positions
                # CRITICAL: Must use base satellite positions/clocks, not rover's!
                base_geom_other = np.linalg.norm(self.base_sat_pos_other - self.base_pos_ecef)
                base_geom_ref = np.linalg.norm(self.base_sat_pos_ref - self.base_pos_ecef)
                range_base_other_correct = base_geom_other - CLIGHT * self.base_sat_clk_other
                range_base_ref_correct = base_geom_ref - CLIGHT * self.base_sat_clk_ref

                # DD of computed ranges (using correct base ranges)
                dd_computed = (range_rover_other - range_base_other_correct) - (range_rover_ref - range_base_ref_correct)

                # Residual is the difference
                residual = dd_observations - dd_computed
                
            else:
                # FALLBACK: When raw observations not available
                # Assume dd_obs is DD of residuals (from residual interpolation)
                # In this case, dd_obs already IS the residual at the reference position
                # We need to adjust it based on position change
                
                # Compute DD geometric
                dd_geometric = (range_rover_other - range_base_other) - (range_rover_ref - range_base_ref)
                
                # For residual-based DD, the measurement IS the residual at reference position
                # We compute how the geometric DD changes from reference
                # But this is problematic without knowing the reference position...
                # For now, treat dd_obs as a direct measurement
                residual = self.dd_measurement - dd_geometric
            
            # Apply atmospheric corrections if enabled
            if self.use_atmospheric:
                # Convert rover position to LLH for atmospheric models
                rover_llh = ecef2llh(rover_ecef)
                
                # Compute elevation and azimuth for each satellite from rover
                el_rover_other = elevation_angle(self.sat_pos_other, rover_ecef)
                az_rover_other = compute_azimuth(self.sat_pos_other, rover_ecef)
                el_rover_ref = elevation_angle(self.sat_pos_ref, rover_ecef)
                az_rover_ref = compute_azimuth(self.sat_pos_ref, rover_ecef)
                
                # Tropospheric delays for rover (using Saastamoinen model)
                trop_rover_other = troposphere_model(el_rover_other, rover_llh[0], rover_llh[2])
                trop_rover_ref = troposphere_model(el_rover_ref, rover_llh[0], rover_llh[2])
                
                # Ionospheric delays for rover (using Klobuchar model if parameters available)
                if np.any(self.ion_params):
                    iono_rover_other = ionosphere_model(
                        el_rover_other, az_rover_other, rover_llh[0], rover_llh[1],
                        self.gps_time, self.ion_params, self.frequency)
                    iono_rover_ref = ionosphere_model(
                        el_rover_ref, az_rover_ref, rover_llh[0], rover_llh[1],
                        self.gps_time, self.ion_params, self.frequency)
                else:
                    iono_rover_other = iono_rover_ref = 0.0
                
                # Add atmospheric delays to rover ranges
                range_rover_other += trop_rover_other + iono_rover_other
                range_rover_ref += trop_rover_ref + iono_rover_ref
                
                # Base atmospheric corrections are pre-computed
                range_base_other += self.trop_base_other + self.iono_base_other
                range_base_ref += self.trop_base_ref + self.iono_base_ref
            
            # Sanity check for numerical stability (check geometric ranges)
            if geom_range_rover_ref < 1.0 or geom_range_rover_other < 1.0:
                # Too close to satellite (numerical issue)
                return np.array([0.0])
            
            # Residual has already been computed in the if-else block above
            
            # Debug output (commented out for performance)
            # if abs(residual) > 30:  # Log moderate residuals too
            #     print(f"DEBUG Large residual: meas={self.dd_measurement:.3f}, geom={dd_geometric:.3f}, res={residual:.3f}")
            #     print(f"      Rover ENU: {rover_enu}")
            #     print(f"      Rover ECEF: {rover_ecef}")
            #     print(f"      Base ECEF: {self.base_pos_ecef}")
            
            # Return error for GTSAM  
            error = np.array([residual], dtype=np.float64)
            
            # Compute Jacobian if requested
            if H is not None and len(H) > 0:
                # Analytical Jacobian computation
                # DD residual = measured - computed
                # computed DD = (rover_other - base_other) - (rover_ref - base_ref)
                # d(residual)/d(rover_pos) = -d(computed DD)/d(rover_pos)

                # Unit vectors from rover to satellites (in ECEF)
                e_other = vec_rover_other / geom_range_rover_other  # Unit vector to other satellite
                e_ref = vec_rover_ref / geom_range_rover_ref        # Unit vector to reference satellite

                # DD geometry matrix in ECEF
                # d(DD)/d(rover_ecef) = d(range_rover_other)/d(rover_ecef) - d(range_rover_ref)/d(rover_ecef)
                #                     = -e_other - (-e_ref) = e_ref - e_other
                H_ecef = e_ref - e_other

                # Convert Jacobian from ECEF to ENU coordinates
                # Since rover_ecef = base_ecef + R_enu2ecef @ rover_enu
                # d(rover_ecef)/d(rover_enu) = R_enu2ecef
                # Therefore: d(DD)/d(rover_enu) = d(DD)/d(rover_ecef) @ d(rover_ecef)/d(rover_enu)
                #                                = H_ecef @ R_enu2ecef
                H_enu = H_ecef @ self.R_enu2ecef

                # Since residual = measured - computed, and measured is constant:
                # d(residual)/d(rover_enu) = -d(computed)/d(rover_enu) = -H_enu
                H[0] = -H_enu.reshape(1, 3)
            
            return error
        
        # Create and return factor
        return gtsam.CustomFactor(self.noise_model, [self.position_key], error_func)