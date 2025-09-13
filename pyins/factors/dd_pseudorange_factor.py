"""Double Difference Pseudorange Factor for GTSAM - RTKLIB style implementation"""

import numpy as np
import gtsam
from pyins.core.constants import CLIGHT, FREQ_L1, FREQ_L2, FREQ_L5
from pyins.observation.measurement_model import troposphere_model, ionosphere_model
from pyins.coordinate.transforms import ecef2llh
from pyins.observation.pseudorange import elevation_angle

# RTKLIB-style error ratios
ERATIO_L1 = 100  # Error ratio for L1
ERATIO_L2 = 100  # Error ratio for L2  
ERATIO_L5 = 100  # Error ratio for L5
ERR_CONSTANT = 0.003  # 3mm constant error


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
        
        # DD cancels satellite clock biases, so we don't need them
        # self.sat_clk_other = dd_data.get('sat_clk', 0.0)
        # self.sat_clk_ref = dd_data.get('ref_sat_clk', 0.0)
        
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
        
        # Pre-compute base station ranges (satellite clock already applied in DD formation)
        # Note: DD observations already include satellite clock corrections
        self.range_base_ref = np.linalg.norm(self.sat_pos_ref - base_pos_ecef)
        self.range_base_other = np.linalg.norm(self.sat_pos_other - base_pos_ecef)
        
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
            eratio_map = [ERATIO_L1, ERATIO_L2, ERATIO_L5]
            eratio = eratio_map[min(self.freq_idx, 2)]  # Default to L5 ratio for higher frequencies
            
            # RTKLIB approach: code_sigma = phase_sigma * eratio
            phase_sigma = ERR_CONSTANT  # 0.003m = 3mm
            code_sigma = phase_sigma * eratio
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
            
            # Compute all 4 ranges (satellite clock already applied in DD observations)
            # rho_rov: rover to other satellite (geometric only)
            vec_rover_other = self.sat_pos_other - rover_ecef
            geom_range_rover_other = np.linalg.norm(vec_rover_other)
            range_rover_other = geom_range_rover_other
            
            # rho_ref: base to other satellite (pre-computed)
            range_base_other = self.range_base_other
            
            # rho_rov_base: rover to reference satellite (geometric only)
            vec_rover_ref = self.sat_pos_ref - rover_ecef
            geom_range_rover_ref = np.linalg.norm(vec_rover_ref)
            range_rover_ref = geom_range_rover_ref
            
            # rho_ref_base: base to reference satellite (pre-computed)
            range_base_ref = self.range_base_ref
            
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
            
            # Compute DD range: (rover-sat - base-sat) - (rover-ref - base-ref)
            # This matches how DD observations are formed
            dd_geometric = (range_rover_other - range_base_other) - (range_rover_ref - range_base_ref)
            
            # Residual (following RTKLIB convention)
            # v = measured - computed
            residual = self.dd_measurement - dd_geometric
            
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
                # Use numerical differentiation for now until analytical is fixed
                epsilon = 1e-6
                H_numerical = np.zeros((1, 3))
                
                for j in range(3):
                    # Perturb position
                    rover_enu_plus = rover_enu.copy()
                    rover_enu_minus = rover_enu.copy()
                    rover_enu_plus[j] += epsilon
                    rover_enu_minus[j] -= epsilon
                    
                    # Compute DD at perturbed positions
                    rover_ecef_plus = self.base_pos_ecef + self.R_enu2ecef @ rover_enu_plus
                    rover_ecef_minus = self.base_pos_ecef + self.R_enu2ecef @ rover_enu_minus
                    
                    # Ranges at plus position
                    range_rover_other_plus = np.linalg.norm(self.sat_pos_other - rover_ecef_plus)
                    range_rover_ref_plus = np.linalg.norm(self.sat_pos_ref - rover_ecef_plus)
                    dd_plus = (range_rover_other_plus - range_base_other) - (range_rover_ref_plus - range_base_ref)
                    
                    # Ranges at minus position
                    range_rover_other_minus = np.linalg.norm(self.sat_pos_other - rover_ecef_minus)
                    range_rover_ref_minus = np.linalg.norm(self.sat_pos_ref - rover_ecef_minus)
                    dd_minus = (range_rover_other_minus - range_base_other) - (range_rover_ref_minus - range_base_ref)
                    
                    # Numerical derivative
                    H_numerical[0, j] = -(dd_plus - dd_minus) / (2 * epsilon)  # Negative because error = measured - computed
                
                H[0] = H_numerical
            
            return error
        
        # Create and return factor
        return gtsam.CustomFactor(self.noise_model, [self.position_key], error_func)