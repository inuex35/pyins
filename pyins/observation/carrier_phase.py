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

"""Carrier phase measurement processing"""

import numpy as np
from typing import Tuple, Optional, List, Dict
from ..core.constants import *
from ..core.data_structures import Observation, Ephemeris
from ..satellite.satellite_position import compute_satellite_position
from .pseudorange import compute_range, sagnac_correction, elevation_angle

def compute_carrier_residual(obs: Observation,
                           eph: Ephemeris,
                           rcv_pos: np.ndarray,
                           rcv_clk: float,
                           ambiguity: float,
                           freq_idx: int = 0) -> Tuple[float, np.ndarray, float]:
    """
    Compute carrier phase residual
    
    Parameters:
    -----------
    obs : Observation
        GNSS observation
    eph : Ephemeris
        Satellite ephemeris
    rcv_pos : np.ndarray
        Receiver position in ECEF
    rcv_clk : float
        Receiver clock bias (s)
    ambiguity : float
        Carrier phase ambiguity (cycles)
    freq_idx : int
        Frequency index
        
    Returns:
    --------
    residual : float
        Carrier phase residual (m)
    H : np.ndarray
        Measurement Jacobian [dx, dy, dz, dt, N]
    var : float
        Measurement variance
    """
    # Check if carrier phase is available
    if obs.L[freq_idx] == 0.0:
        return 0.0, np.zeros(5), 0.0
    
    # Get wavelength
    wavelength = obs.get_wavelength(freq_idx)
    if wavelength == 0.0:
        return 0.0, np.zeros(5), 0.0
    
    # Signal transmission time (approximate)
    tau = 0.075  # Approximate travel time
    t_tx = obs.time - tau
    
    # Satellite position at transmission time
    sat_pos, sat_clk, _ = compute_satellite_position(eph, t_tx)
    
    # Geometric range
    rho, e = compute_range(sat_pos, rcv_pos)
    
    # Sagnac correction
    sagnac = sagnac_correction(sat_pos, rcv_pos)
    
    # Modeled carrier phase (in meters)
    phi_model = rho + sagnac + CLIGHT * (rcv_clk - sat_clk) + wavelength * ambiguity
    
    # Observed carrier phase in meters
    phi_obs = obs.L[freq_idx] * wavelength
    
    # Residual
    residual = phi_obs - phi_model
    
    # Jacobian
    H = np.zeros(5)
    H[:3] = -e  # Partial derivatives w.r.t position
    H[3] = CLIGHT  # Partial derivative w.r.t clock
    H[4] = wavelength  # Partial derivative w.r.t ambiguity
    
    # Measurement variance
    el = elevation_angle(sat_pos, rcv_pos)
    var = carrier_variance(obs.SNR[freq_idx], el, wavelength)
    
    return residual, H, var


def carrier_variance(snr: float, elevation: float, wavelength: float) -> float:
    """
    Compute carrier phase measurement variance
    
    Parameters:
    -----------
    snr : float
        Signal-to-noise ratio (dBHz)
    elevation : float
        Satellite elevation angle (rad)
    wavelength : float
        Carrier wavelength (m)
        
    Returns:
    --------
    var : float
        Measurement variance (m^2)
    """
    # Base standard deviation (cycles)
    sigma_base_cycles = 0.003  # 3mm at L1
    
    # Convert to meters
    sigma_base = sigma_base_cycles * wavelength
    
    # SNR-dependent factor
    if snr > 0:
        snr_factor = 10.0 ** (-snr / 40.0)  # Less sensitive than pseudorange
    else:
        snr_factor = 1.0
        
    # Elevation-dependent factor
    el_deg = np.rad2deg(elevation)
    if el_deg > 5.0:
        el_factor = 1.0 / np.sin(elevation)
    else:
        el_factor = 1.0 / np.sin(np.deg2rad(5.0))
        
    # Total variance
    sigma = sigma_base * snr_factor * el_factor
    return sigma ** 2


def detect_cycle_slip(obs_current: Observation,
                     obs_previous: Observation,
                     dt: float,
                     freq_idx: int = 0) -> bool:
    """
    Detect cycle slip using phase-code and Doppler methods
    
    Parameters:
    -----------
    obs_current : Observation
        Current observation
    obs_previous : Observation
        Previous observation
    dt : float
        Time difference (s)
    freq_idx : int
        Frequency index
        
    Returns:
    --------
    slip_detected : bool
        True if cycle slip detected
    """
    # Check if both observations are valid
    if (obs_current.L[freq_idx] == 0.0 or obs_previous.L[freq_idx] == 0.0 or
        obs_current.P[freq_idx] == 0.0 or obs_previous.P[freq_idx] == 0.0):
        return True
        
    # Get wavelength
    wavelength = obs_current.get_wavelength(freq_idx)
    if wavelength == 0.0:
        return True
    
    # Method 1: Phase-Code difference
    pc_current = obs_current.L[freq_idx] * wavelength - obs_current.P[freq_idx]
    pc_previous = obs_previous.L[freq_idx] * wavelength - obs_previous.P[freq_idx]
    pc_diff = pc_current - pc_previous
    
    # Threshold based on code noise (conservative)
    pc_threshold = 5.0  # meters
    if abs(pc_diff) > pc_threshold:
        return True
    
    # Method 2: Doppler prediction (if available)
    if obs_previous.D[freq_idx] != 0.0:
        # Predict phase change using Doppler
        predicted_phase_change = -obs_previous.D[freq_idx] * dt  # cycles
        actual_phase_change = obs_current.L[freq_idx] - obs_previous.L[freq_idx]
        
        # Check difference
        phase_diff = actual_phase_change - predicted_phase_change
        
        # Threshold (cycles)
        doppler_threshold = 0.5  # Half cycle
        if abs(phase_diff) > doppler_threshold:
            return True
    
    # Method 3: Loss of lock indicator
    if obs_current.LLI[freq_idx] != 0:
        return True
        
    return False


class AmbiguityEstimator:
    """Estimate and manage carrier phase ambiguities"""
    
    def __init__(self):
        self.ambiguities = {}  # (sat, freq) -> ambiguity
        self.fixed_ambiguities = {}  # (sat, freq) -> fixed ambiguity
        self.float_covariance = {}  # Covariance of float ambiguities
        
    def initialize_ambiguity(self, obs: Observation, 
                           rcv_pos: np.ndarray,
                           sat_pos: np.ndarray,
                           rcv_clk: float,
                           sat_clk: float,
                           freq_idx: int = 0) -> float:
        """
        Initialize ambiguity using pseudorange
        
        Parameters:
        -----------
        obs : Observation
            GNSS observation
        rcv_pos : np.ndarray
            Receiver position
        sat_pos : np.ndarray
            Satellite position
        rcv_clk : float
            Receiver clock bias (s)
        sat_clk : float
            Satellite clock bias (s)
        freq_idx : int
            Frequency index
            
        Returns:
        --------
        ambiguity : float
            Initial ambiguity estimate (cycles)
        """
        # Get wavelength
        wavelength = obs.get_wavelength(freq_idx)
        if wavelength == 0.0:
            return 0.0
            
        # Geometric range
        rho, _ = compute_range(sat_pos, rcv_pos)
        
        # Sagnac correction
        sagnac = sagnac_correction(sat_pos, rcv_pos)
        
        # Expected phase (without ambiguity)
        expected_range = rho + sagnac + CLIGHT * (rcv_clk - sat_clk)
        
        # Ambiguity = (observed phase - expected phase) / wavelength
        phase_obs_m = obs.L[freq_idx] * wavelength
        code_obs_m = obs.P[freq_idx]
        
        # Use combination of phase and code for initialization
        ambiguity = (phase_obs_m - code_obs_m) / wavelength
        
        return ambiguity
    
    def fix_ambiguities(self, float_ambiguities: np.ndarray,
                        covariance: np.ndarray,
                        method: str = 'lambda') -> Tuple[np.ndarray, float]:
        """
        Fix integer ambiguities
        
        Parameters:
        -----------
        float_ambiguities : np.ndarray
            Float ambiguity estimates
        covariance : np.ndarray
            Ambiguity covariance matrix
        method : str
            Fixing method ('lambda', 'rounding', 'bootstrapping')
            
        Returns:
        --------
        fixed_ambiguities : np.ndarray
            Fixed integer ambiguities
        ratio : float
            Ambiguity validation ratio
        """
        n = len(float_ambiguities)
        
        if method == 'rounding':
            # Simple rounding
            fixed_ambiguities = np.round(float_ambiguities)
            ratio = 1.0  # No validation
            
        elif method == 'bootstrapping':
            # Sequential fixing
            fixed_ambiguities = np.zeros(n)
            for i in range(n):
                # Conditional estimate
                if i == 0:
                    fixed_ambiguities[i] = np.round(float_ambiguities[i])
                else:
                    # Update based on previously fixed ambiguities
                    adjustment = 0.0
                    for j in range(i):
                        if covariance[i, i] > 0:
                            adjustment += (covariance[i, j] / covariance[j, j]) * \
                                        (fixed_ambiguities[j] - float_ambiguities[j])
                    fixed_ambiguities[i] = np.round(float_ambiguities[i] + adjustment)
            ratio = 1.0
            
        elif method == 'lambda':
            # LAMBDA method (simplified version)
            # In practice, would use full LAMBDA implementation
            from ..utils.ambiguity import lambda_reduction
            fixed_ambiguities, ratio = lambda_reduction(float_ambiguities, covariance)
            
        else:
            raise ValueError(f"Unknown ambiguity fixing method: {method}")
            
        return fixed_ambiguities, ratio
    
    def validate_fixed_ambiguities(self, ratio: float, 
                                 min_ratio: float = 3.0) -> bool:
        """
        Validate fixed ambiguities using ratio test
        
        Parameters:
        -----------
        ratio : float
            Ambiguity ratio (second-best/best)
        min_ratio : float
            Minimum acceptable ratio
            
        Returns:
        --------
        valid : bool
            True if ambiguities are valid
        """
        return ratio >= min_ratio


class CarrierPhaseProcessor:
    """Process carrier phase measurements"""
    
    def __init__(self):
        self.min_elevation = np.deg2rad(15.0)  # Higher than pseudorange
        self.cycle_slip_detector = {}  # sat -> previous observation
        self.ambiguity_estimator = AmbiguityEstimator()
        
    def process_observations(self,
                           observations: List[Observation],
                           ephemerides: List[Ephemeris],
                           rcv_pos: np.ndarray,
                           rcv_clk: float,
                           ambiguities: Dict[int, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process carrier phase observations
        
        Parameters:
        -----------
        observations : List[Observation]
            GNSS observations
        ephemerides : List[Ephemeris]
            Satellite ephemerides
        rcv_pos : np.ndarray
            Receiver position
        rcv_clk : float
            Receiver clock
        ambiguities : Dict[int, float]
            Carrier phase ambiguities
            
        Returns:
        --------
        residuals : np.ndarray
            Measurement residuals
        H : np.ndarray
            Jacobian matrix
        R : np.ndarray
            Covariance matrix
        """
        valid_obs = []
        eph_dict = {eph.sat: eph for eph in ephemerides}
        
        for obs in observations:
            if obs.sat not in eph_dict:
                continue
                
            eph = eph_dict[obs.sat]
            
            # Check elevation
            sat_pos, _, _ = compute_satellite_position(eph, obs.time)
            el = elevation_angle(sat_pos, rcv_pos)
            
            if el < self.min_elevation:
                continue
                
            # Check cycle slip
            if obs.sat in self.cycle_slip_detector:
                prev_obs = self.cycle_slip_detector[obs.sat]
                dt = obs.time - prev_obs.time
                
                if detect_cycle_slip(obs, prev_obs, dt):
                    # Reset ambiguity
                    if obs.sat in ambiguities:
                        del ambiguities[obs.sat]
                        
            # Update cycle slip detector
            self.cycle_slip_detector[obs.sat] = obs
            
            # Get or initialize ambiguity
            if obs.sat not in ambiguities:
                sat_pos, sat_clk, _ = compute_satellite_position(eph, obs.time)
                ambiguities[obs.sat] = self.ambiguity_estimator.initialize_ambiguity(
                    obs, rcv_pos, sat_pos, rcv_clk, sat_clk)
                    
            # Compute residual
            res, h, var = compute_carrier_residual(
                obs, eph, rcv_pos, rcv_clk, ambiguities[obs.sat])
                
            if res != 0.0:
                valid_obs.append((res, h, var))
                
        if not valid_obs:
            return np.array([]), np.array([]), np.array([])
            
        # Stack results
        residuals = np.array([res for res, _, _ in valid_obs])
        H = np.vstack([h for _, h, _ in valid_obs])
        R = np.diag([var for _, _, var in valid_obs])
        
        return residuals, H, R