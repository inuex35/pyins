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

"""Double difference processing for RTK with per-system reference satellites"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from ..core.data_structures import Observation
from ..core.constants import CLIGHT, sat2sys, sys2char, SYS_GPS, SYS_GLO, SYS_GAL, SYS_BDS, SYS_QZS


class DoubleDifferenceProcessor:
    """Process GNSS observations to form double differences with per-system reference satellites"""
    
    def __init__(self, reference_sats: Optional[Dict[int, int]] = None):
        """
        Initialize double difference processor
        
        Parameters:
        -----------
        reference_sats : Dict[int, int], optional
            Reference satellite for each system {system_id: sat_number}
        """
        self.reference_sats = reference_sats or {}
        self.last_ref_sats = {}
        
    def form_double_differences(self, 
                              rover_obs: List[Observation],
                              base_obs: List[Observation],
                              frequency_idx: int = 0) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]], Dict[int, int]]:
        """
        Form double difference observations with per-system reference satellites
        
        Parameters:
        -----------
        rover_obs : List[Observation]
            Rover observations
        base_obs : List[Observation]
            Base station observations
            
        Returns:
        --------
        dd_pseudorange : np.ndarray
            Double difference pseudorange (m)
        dd_carrier : np.ndarray
            Double difference carrier phase (cycles)
        sat_pairs : List[Tuple[int, int]]
            List of (reference_sat, other_sat) pairs
        ref_sats_used : Dict[int, int]
            Reference satellites used for each system
        """
        # Group satellites by system
        rover_by_system = self._group_by_system(rover_obs)
        base_by_system = self._group_by_system(base_obs)
        
        # Results containers
        all_dd_pr = []
        all_dd_cp = []
        all_sat_pairs = []
        ref_sats_used = {}
        
        # Process each system separately
        for sys_id in rover_by_system:
            if sys_id not in base_by_system:
                continue
                
            # Find common satellites for this system
            rover_sats = {obs.sat for obs in rover_by_system[sys_id] 
                         if obs.P[frequency_idx] > 0 and obs.L[frequency_idx] > 0}
            base_sats = {obs.sat for obs in base_by_system[sys_id] 
                        if obs.P[frequency_idx] > 0 and obs.L[frequency_idx] > 0}
            common_sats = rover_sats & base_sats
            
            if len(common_sats) < 2:
                continue
                
            # Select reference satellite for this system
            ref_sat = self._select_reference_satellite_for_system(
                common_sats, rover_by_system[sys_id], sys_id)
            ref_sats_used[sys_id] = ref_sat
            
            other_sats = [sat for sat in common_sats if sat != ref_sat]
            
            # Create observation dictionaries
            rover_dict = {obs.sat: obs for obs in rover_by_system[sys_id]}
            base_dict = {obs.sat: obs for obs in base_by_system[sys_id]}
            
            # Form double differences for this system
            for sat in other_sats:
                # Single differences
                rover_sd_pr = rover_dict[sat].P[frequency_idx] - rover_dict[ref_sat].P[frequency_idx]
                rover_sd_cp = rover_dict[sat].L[frequency_idx] - rover_dict[ref_sat].L[frequency_idx]
                base_sd_pr = base_dict[sat].P[frequency_idx] - base_dict[ref_sat].P[frequency_idx]
                base_sd_cp = base_dict[sat].L[frequency_idx] - base_dict[ref_sat].L[frequency_idx]
                
                # Double differences
                all_dd_pr.append(rover_sd_pr - base_sd_pr)
                all_dd_cp.append(rover_sd_cp - base_sd_cp)
                all_sat_pairs.append((ref_sat, sat))
                
        return (np.array(all_dd_pr), np.array(all_dd_cp), 
                all_sat_pairs, ref_sats_used)
    
    def form_multi_frequency_dd(self,
                                rover_obs: List[Observation],
                                base_obs: List[Observation]) -> Dict[int, Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]], Dict[int, int]]]:
        """
        Form double differences for all available frequencies
        
        Parameters:
        -----------
        rover_obs : List[Observation]
            Rover observations
        base_obs : List[Observation]
            Base station observations
            
        Returns:
        --------
        results : Dict[int, Tuple]
            Dictionary with frequency index as key and DD results as value
            Each tuple contains (dd_pseudorange, dd_carrier, sat_pairs, ref_sats_used)
        """
        results = {}
        
        # Check available frequencies
        for freq_idx in range(3):  # Support up to L1, L2, L5
            # Check if this frequency has valid observations
            has_freq = False
            for obs in rover_obs:
                if freq_idx < len(obs.P) and obs.P[freq_idx] > 0:
                    has_freq = True
                    break
            
            if has_freq:
                dd_pr, dd_cp, sat_pairs, ref_sats = self.form_double_differences(
                    rover_obs, base_obs, frequency_idx
                )
                
                if len(dd_pr) > 0:
                    results[freq_idx] = (dd_pr, dd_cp, sat_pairs, ref_sats)
        
        return results
    
    def compute_dd_residuals(self,
                           dd_observations: np.ndarray,
                           sat_positions: Dict[int, np.ndarray],
                           rover_position: np.ndarray,
                           base_position: np.ndarray,
                           sat_pairs: List[Tuple[int, int]],
                           sat_clocks: Optional[Dict[int, float]] = None,
                           ambiguities: Optional[np.ndarray] = None,
                           wavelength: float = CLIGHT / 1575.42e6,
                           use_carrier: bool = False) -> np.ndarray:
        """
        Compute double difference residuals
        
        Parameters:
        -----------
        dd_observations : np.ndarray
            Double difference observations (pseudorange in m or carrier in cycles)
        sat_positions : Dict[int, np.ndarray]
            Satellite positions in ECEF
        rover_position : np.ndarray
            Rover position in ECEF
        base_position : np.ndarray
            Base position in ECEF
        sat_pairs : List[Tuple[int, int]]
            Satellite pairs for double differences
        sat_clocks : Dict[int, float], optional
            Satellite clock biases in seconds
        ambiguities : np.ndarray, optional
            Integer ambiguities for carrier phase
        wavelength : float
            Carrier wavelength in meters
        use_carrier : bool
            If True, compute carrier phase residuals
            
        Returns:
        --------
        residuals : np.ndarray
            Observation residuals
        """
        n_dd = len(sat_pairs)
        predicted = np.zeros(n_dd)
        
        for i, (ref_sat, sat) in enumerate(sat_pairs):
            # Rover ranges
            rover_ref_range = np.linalg.norm(sat_positions[ref_sat] - rover_position)
            rover_sat_range = np.linalg.norm(sat_positions[sat] - rover_position)
            
            # Base ranges
            base_ref_range = np.linalg.norm(sat_positions[ref_sat] - base_position)
            base_sat_range = np.linalg.norm(sat_positions[sat] - base_position)
            
            # Single differences (geometric only)
            rover_sd = rover_sat_range - rover_ref_range
            base_sd = base_sat_range - base_ref_range
            
            # Double difference (satellite clocks cancel out in DD)
            predicted[i] = rover_sd - base_sd
            
            # Convert to cycles for carrier phase
            if use_carrier:
                predicted[i] = predicted[i] / wavelength
                if ambiguities is not None:
                    predicted[i] += ambiguities[i]
                    
        return dd_observations - predicted
    
    def compute_dd_geometry_matrix(self, 
                                 sat_positions: Dict[int, np.ndarray],
                                 user_position: np.ndarray,
                                 sat_pairs: List[Tuple[int, int]]) -> np.ndarray:
        """
        Compute double difference geometry matrix
        
        Parameters:
        -----------
        sat_positions : Dict[int, np.ndarray]
            Satellite positions in ECEF
        user_position : np.ndarray
            User position in ECEF
        sat_pairs : List[Tuple[int, int]]
            Satellite pairs for double differences
            
        Returns:
        --------
        H : np.ndarray
            Geometry matrix (n_dd x 3)
        """
        n_dd = len(sat_pairs)
        H = np.zeros((n_dd, 3))
        
        for i, (ref_sat, sat) in enumerate(sat_pairs):
            # Unit vectors
            ref_vec = sat_positions[ref_sat] - user_position
            ref_range = np.linalg.norm(ref_vec)
            ref_unit = ref_vec / ref_range
            
            sat_vec = sat_positions[sat] - user_position
            sat_range = np.linalg.norm(sat_vec)
            sat_unit = sat_vec / sat_range
            
            # Double difference geometry
            H[i, :] = sat_unit - ref_unit
            
        return H
    
    def get_reference_satellites(self) -> Dict[int, int]:
        """Get current reference satellites for each system"""
        return self.last_ref_sats.copy()
    
    def set_reference_satellite(self, system_id: int, sat_number: int):
        """Set reference satellite for a specific system"""
        self.reference_sats[system_id] = sat_number
        
    def _group_by_system(self, observations: List[Observation]) -> Dict[int, List[Observation]]:
        """Group observations by satellite system"""
        groups = {}
        for obs in observations:
            sys = sat2sys(obs.sat)
            if sys not in groups:
                groups[sys] = []
            groups[sys].append(obs)
        return groups
        
    def _select_reference_satellite_for_system(self, 
                                             common_sats: set, 
                                             observations: List[Observation],
                                             sys_id: int) -> int:
        """
        Select reference satellite for a specific system
        
        Parameters:
        -----------
        common_sats : set
            Set of common satellite numbers for this system
        observations : List[Observation]
            Observations for selection criteria
        sys_id : int
            System ID
            
        Returns:
        --------
        ref_sat : int
            Selected reference satellite
        """
        # Check if reference satellite is pre-specified
        if sys_id in self.reference_sats and self.reference_sats[sys_id] in common_sats:
            return self.reference_sats[sys_id]
            
        # Check if last reference satellite is still available
        if sys_id in self.last_ref_sats and self.last_ref_sats[sys_id] in common_sats:
            return self.last_ref_sats[sys_id]
            
        # Select satellite with highest SNR
        obs_dict = {obs.sat: obs for obs in observations}
        
        best_sat = None
        best_snr = 0
        
        for sat in common_sats:
            if sat in obs_dict:
                obs = obs_dict[sat]
                snr = obs.SNR[0] if obs.SNR[0] > 0 else 50
                    
                if snr > best_snr:
                    best_snr = snr
                    best_sat = sat
                    
        if best_sat is None:
            best_sat = min(common_sats)  # Fallback
            
        self.last_ref_sats[sys_id] = best_sat
        return best_sat
        
    @staticmethod
    def get_system_name(sys_id: int) -> str:
        """Get system name from ID"""
        names = {
            SYS_GPS: "GPS",
            SYS_GLO: "GLONASS",
            SYS_GAL: "Galileo",
            SYS_BDS: "BeiDou",
            SYS_QZS: "QZSS"
        }
        return names.get(sys_id, f"Unknown({sys_id})")


class SingleDifferenceProcessor:
    """Process single difference observations"""
    
    def __init__(self):
        self.reference_sat = None
        
    def form_single_differences(self, 
                              rover_obs: List[Observation],
                              base_obs: List[Observation]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Form single difference observations between rover and base
        
        Parameters:
        -----------
        rover_obs : List[Observation]
            Rover observations
        base_obs : List[Observation]
            Base station observations
            
        Returns:
        --------
        sd_pseudorange : np.ndarray
            Single difference pseudorange
        sd_carrier : np.ndarray
            Single difference carrier phase
        common_sats : List[int]
            Common satellites
        """
        # Find common satellites
        rover_sats = {obs.sat for obs in rover_obs if obs.P[0] > 0 and obs.L[0] > 0}
        base_sats = {obs.sat for obs in base_obs if obs.P[0] > 0 and obs.L[0] > 0}
        common_sats = sorted(rover_sats & base_sats)
        
        if len(common_sats) == 0:
            return np.array([]), np.array([]), []
            
        # Create observation dictionaries
        rover_dict = {obs.sat: obs for obs in rover_obs}
        base_dict = {obs.sat: obs for obs in base_obs}
        
        # Form single differences
        sd_pseudorange = []
        sd_carrier = []
        
        for sat in common_sats:
            sd_pseudorange.append(rover_dict[sat].P[0] - base_dict[sat].P[0])
            sd_carrier.append(rover_dict[sat].L[0] - base_dict[sat].L[0])
            
        return np.array(sd_pseudorange), np.array(sd_carrier), common_sats