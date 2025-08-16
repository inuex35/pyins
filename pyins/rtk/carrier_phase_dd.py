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

"""Carrier phase double difference processing for RTK positioning"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from ..core.data_structures import Observation
from ..core.constants import (
    CLIGHT, sat2sys, sys2char, 
    SYS_GPS, SYS_GLO, SYS_GAL, SYS_BDS, SYS_QZS,
    FREQ_L1, FREQ_E1, FREQ_B1I, FREQ_J1
)


class CarrierPhaseDD:
    """Carrier phase double difference processor for RTK"""
    
    def __init__(self):
        """Initialize carrier phase DD processor"""
        self.ambiguities = {}  # Store ambiguities for each satellite pair
        self.fixed_ambiguities = {}  # Fixed integer ambiguities
        self.cycle_slips = {}  # Track cycle slips
        
    def get_wavelength(self, sat_id: int) -> float:
        """Get carrier wavelength for satellite
        
        Parameters:
        -----------
        sat_id : int
            Satellite ID
            
        Returns:
        --------
        wavelength : float
            Carrier wavelength in meters
        """
        sys_id = sat2sys(sat_id)
        
        if sys_id == SYS_GPS:
            return CLIGHT / FREQ_L1
        elif sys_id == SYS_GLO:
            # GLONASS uses FDMA - each satellite has different frequency
            from ..gnss.glonass_ifb import get_glonass_channel
            from ..core.constants import FREQ_G1, DFREQ_G1
            
            # Get frequency channel for this satellite
            channel = get_glonass_channel(sat_id)
            
            # Calculate actual frequency for this satellite
            # GLONASS L1: f = f0 + k * df where k is channel number
            freq = FREQ_G1 + channel * DFREQ_G1
            
            return CLIGHT / freq
        elif sys_id == SYS_GAL:
            return CLIGHT / FREQ_E1
        elif sys_id == SYS_BDS:
            return CLIGHT / FREQ_B1I
        elif sys_id == SYS_QZS:
            return CLIGHT / FREQ_J1
        else:
            # Default to GPS L1
            return CLIGHT / FREQ_L1
    
    def form_carrier_dd(self,
                       rover_obs: List[Observation],
                       base_obs: List[Observation],
                       ref_sats: Dict[int, int]) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]], Dict]:
        """Form carrier phase double differences
        
        Parameters:
        -----------
        rover_obs : List[Observation]
            Rover observations
        base_obs : List[Observation]
            Base observations
        ref_sats : Dict[int, int]
            Reference satellite for each system
            
        Returns:
        --------
        dd_phase_m : np.ndarray
            Double difference carrier phase in meters
        dd_phase_cycles : np.ndarray
            Double difference carrier phase in cycles
        sat_pairs : List[Tuple[int, int]]
            Satellite pairs (ref_sat, other_sat)
        info : Dict
            Additional information
        """
        # Group by system
        rover_by_system = self._group_by_system(rover_obs)
        base_by_system = self._group_by_system(base_obs)
        
        all_dd_phase_m = []
        all_dd_phase_cycles = []
        all_sat_pairs = []
        wavelengths = []
        
        for sys_id, ref_sat in ref_sats.items():
            if sys_id not in rover_by_system or sys_id not in base_by_system:
                continue
                
            # Get observations
            rover_dict = {obs.sat: obs for obs in rover_by_system[sys_id]}
            base_dict = {obs.sat: obs for obs in base_by_system[sys_id]}
            
            # Find common satellites with carrier phase
            rover_sats = {obs.sat for obs in rover_by_system[sys_id] if obs.L[0] > 0}
            base_sats = {obs.sat for obs in base_by_system[sys_id] if obs.L[0] > 0}
            common_sats = rover_sats & base_sats
            
            if ref_sat not in common_sats:
                continue
                
            other_sats = [s for s in common_sats if s != ref_sat]
            
            for other_sat in other_sats:
                # Get wavelength
                wavelength = self.get_wavelength(other_sat)
                ref_wavelength = self.get_wavelength(ref_sat)
                
                # All systems: L is in cycles, convert to meters
                # This is consistent with RTKLIB and RINEX standard
                sd_ref_meters = rover_dict[ref_sat].L[0] * ref_wavelength - base_dict[ref_sat].L[0] * ref_wavelength
                sd_other_meters = rover_dict[other_sat].L[0] * wavelength - base_dict[other_sat].L[0] * wavelength
                
                # Double difference in meters
                dd_meters = sd_other_meters - sd_ref_meters
                
                # Convert to cycles for ambiguity resolution
                dd_cycles = dd_meters / wavelength
                
                all_dd_phase_m.append(dd_meters)
                all_dd_phase_cycles.append(dd_cycles)
                all_sat_pairs.append((ref_sat, other_sat))
                wavelengths.append(wavelength)
        
        info = {
            'wavelengths': np.array(wavelengths),
            'n_dd': len(all_sat_pairs)
        }
        
        return (np.array(all_dd_phase_m), 
                np.array(all_dd_phase_cycles),
                all_sat_pairs, 
                info)
    
    def estimate_float_ambiguities(self,
                                  dd_phase_cycles: np.ndarray,
                                  dd_range_m: np.ndarray,
                                  wavelengths: np.ndarray) -> np.ndarray:
        """Estimate float ambiguities from phase and range
        
        Parameters:
        -----------
        dd_phase_cycles : np.ndarray
            DD carrier phase in cycles
        dd_range_m : np.ndarray
            DD geometric range in meters
        wavelengths : np.ndarray
            Carrier wavelengths in meters
            
        Returns:
        --------
        float_ambiguities : np.ndarray
            Float ambiguity estimates in cycles
        """
        # Convert phase to meters
        dd_phase_m = dd_phase_cycles * wavelengths
        
        # Float ambiguity in cycles
        float_ambiguities = (dd_phase_m - dd_range_m) / wavelengths
        
        return float_ambiguities
    
    def fix_ambiguities_simple(self,
                              float_ambiguities: np.ndarray,
                              threshold: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
        """Simple ambiguity fixing by rounding
        
        Parameters:
        -----------
        float_ambiguities : np.ndarray
            Float ambiguity estimates
        threshold : float
            Threshold for accepting fixed ambiguity (cycles)
            
        Returns:
        --------
        fixed_ambiguities : np.ndarray
            Fixed integer ambiguities
        is_fixed : np.ndarray
            Boolean mask indicating which ambiguities were fixed
        """
        # Round to nearest integer
        fixed_ambiguities = np.round(float_ambiguities)
        
        # Check fractional part
        frac_part = np.abs(float_ambiguities - fixed_ambiguities)
        
        # Mark as fixed if close to integer
        is_fixed = frac_part < threshold
        
        return fixed_ambiguities.astype(int), is_fixed
    
    def detect_cycle_slips(self,
                          current_phase: np.ndarray,
                          previous_phase: np.ndarray,
                          threshold: float = 1.0) -> np.ndarray:
        """Detect cycle slips in carrier phase
        
        Parameters:
        -----------
        current_phase : np.ndarray
            Current epoch carrier phase (cycles)
        previous_phase : np.ndarray
            Previous epoch carrier phase (cycles)
        threshold : float
            Threshold for cycle slip detection (cycles)
            
        Returns:
        --------
        has_slip : np.ndarray
            Boolean mask indicating cycle slips
        """
        # Phase difference between epochs
        phase_diff = current_phase - previous_phase
        
        # Expected change (should be small)
        has_slip = np.abs(phase_diff) > threshold
        
        return has_slip
    
    def compute_phase_residuals(self,
                              dd_phase_m: np.ndarray,
                              dd_range_m: np.ndarray,
                              ambiguities: np.ndarray,
                              wavelengths: np.ndarray) -> np.ndarray:
        """Compute carrier phase residuals
        
        Parameters:
        -----------
        dd_phase_m : np.ndarray
            DD carrier phase in meters
        dd_range_m : np.ndarray
            DD geometric range in meters
        ambiguities : np.ndarray
            Integer ambiguities (cycles)
        wavelengths : np.ndarray
            Carrier wavelengths (meters)
            
        Returns:
        --------
        residuals : np.ndarray
            Phase residuals in meters
        """
        # Predicted phase with ambiguities
        dd_phase_pred = dd_range_m + ambiguities * wavelengths
        
        # Residuals
        residuals = dd_phase_m - dd_phase_pred
        
        return residuals
    
    def _group_by_system(self, observations: List[Observation]) -> Dict[int, List[Observation]]:
        """Group observations by system"""
        grouped = {}
        for obs in observations:
            sys_id = sat2sys(obs.sat)
            if sys_id not in grouped:
                grouped[sys_id] = []
            grouped[sys_id].append(obs)
        return grouped
    
    def apply_lambda_method(self,
                           float_ambiguities: np.ndarray,
                           covariance: np.ndarray,
                           n_candidates: int = 2) -> Tuple[np.ndarray, float]:
        """Apply simplified LAMBDA method for ambiguity resolution
        
        Parameters:
        -----------
        float_ambiguities : np.ndarray
            Float ambiguity estimates
        covariance : np.ndarray
            Covariance matrix of float ambiguities
        n_candidates : int
            Number of candidates to search
            
        Returns:
        --------
        fixed_ambiguities : np.ndarray
            Best integer ambiguity set
        ratio : float
            Ratio test value
        """
        n = len(float_ambiguities)
        
        # Simple integer search (not full LAMBDA)
        # Start with rounded values
        best_int = np.round(float_ambiguities).astype(int)
        
        # Compute residual
        residual = float_ambiguities - best_int
        best_cost = residual.T @ np.linalg.inv(covariance) @ residual
        
        # Search nearby integers
        second_best_cost = np.inf
        
        for i in range(n):
            # Try +1 and -1 for each ambiguity
            for delta in [-1, 1]:
                test_int = best_int.copy()
                test_int[i] += delta
                
                residual = float_ambiguities - test_int
                cost = residual.T @ np.linalg.inv(covariance) @ residual
                
                if cost < best_cost:
                    second_best_cost = best_cost
                    best_cost = cost
                    best_int = test_int
                elif cost < second_best_cost:
                    second_best_cost = cost
        
        # Ratio test
        ratio = second_best_cost / best_cost if best_cost > 0 else 0
        
        return best_int, ratio