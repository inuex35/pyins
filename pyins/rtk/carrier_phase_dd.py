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

from ..core.constants import (
    CLIGHT,
    FREQ_B1I,
    FREQ_B2a,
    FREQ_B3,
    FREQ_E1,
    FREQ_E5a,
    FREQ_E5b,
    FREQ_G1,
    FREQ_G2,
    DFREQ_G1,
    DFREQ_G2,
    FREQ_J1,
    FREQ_J2,
    FREQ_J5,
    FREQ_L1,
    FREQ_L2,
    FREQ_L5,
    SYS_BDS,
    SYS_GAL,
    SYS_GLO,
    SYS_GPS,
    SYS_QZS,
    sat2sys,
)
from ..core.data_structures import Observation


class CarrierPhaseDD:
    """Carrier phase double difference processor for RTK"""

    def __init__(self):
        """Initialize carrier phase DD processor"""
        self.ambiguities = {}  # Store ambiguities for each satellite pair
        self.fixed_ambiguities = {}  # Fixed integer ambiguities
        self.cycle_slips = {}  # Track cycle slips
        self.initial_dd = {}  # Store initial DD values for relative calculation

    def get_wavelength(self, sat_id: int, freq_idx: int = 0) -> float:
        """Get carrier wavelength for satellite and frequency

        Parameters:
        -----------
        sat_id : int
            Satellite ID
        freq_idx : int
            Frequency index (0=L1/E1/B1, 1=L2/E5b/B2, 2=L5/E5a/B2a)

        Returns:
        --------
        wavelength : float
            Carrier wavelength in meters
        """
        sys_id = sat2sys(sat_id)

        if sys_id == SYS_GPS:
            freqs = [FREQ_L1, FREQ_L2, FREQ_L5]
            return CLIGHT / freqs[min(freq_idx, 2)]
        elif sys_id == SYS_GLO:
            # GLONASS uses FDMA - each satellite has different frequency
            from ..gnss.glonass_ifb import get_glonass_channel
            channel = get_glonass_channel(sat_id)
            
            if freq_idx == 0:
                freq = FREQ_G1 + channel * DFREQ_G1
            elif freq_idx == 1:
                freq = FREQ_G2 + channel * DFREQ_G2
            else:
                freq = FREQ_G1  # No L3 for GLONASS
            
            return CLIGHT / freq
        elif sys_id == SYS_GAL:
            freqs = [FREQ_E1, FREQ_E5b, FREQ_E5a]
            return CLIGHT / freqs[min(freq_idx, 2)]
        elif sys_id == SYS_BDS:
            freqs = [FREQ_B1I, FREQ_B3, FREQ_B2a]
            return CLIGHT / freqs[min(freq_idx, 2)]
        elif sys_id == SYS_QZS:
            freqs = [FREQ_J1, FREQ_J2, FREQ_J5]
            return CLIGHT / freqs[min(freq_idx, 2)]
        else:
            # Default to GPS frequencies
            freqs = [FREQ_L1, FREQ_L2, FREQ_L5]
            return CLIGHT / freqs[min(freq_idx, 2)]

    def form_carrier_dd(self,
                       rover_obs: list[Observation],
                       base_obs: list[Observation],
                       ref_sats: dict[int, int],
                       freq_idx: int = None) -> tuple:
        """Form carrier phase double differences

        Parameters:
        -----------
        rover_obs : List[Observation]
            Rover observations
        base_obs : List[Observation]
            Base observations
        ref_sats : Dict[int, int]
            Reference satellite for each system
        freq_idx : int, optional
            Specific frequency index (0=L1/E1/B1, 1=L2/E5b/B2, 2=L5/E5a/B2a)
            If None, returns all available frequencies

        Returns:
        --------
        dd_phase : np.ndarray or dict
            Double difference carrier phase (cycles)
            If freq_idx is None, returns dict with keys as frequency indices
        dd_code : np.ndarray or dict
            Double difference pseudorange (meters)
            If freq_idx is None, returns dict with keys as frequency indices
        sat_pairs : List[Tuple[int, int]] or dict
            Satellite pairs (ref_sat, other_sat)
            If freq_idx is None, returns dict with keys as frequency indices
        info : Dict
            Additional information including wavelengths and frequency info
        """
        # Group by system
        rover_by_system = self._group_by_system(rover_obs)
        base_by_system = self._group_by_system(base_obs)

        if freq_idx is not None:
            # Single frequency mode
            all_dd_phase_m = []
            all_dd_phase_cycles = []
            all_dd_code = []
            all_sat_pairs = []
            wavelengths = []

            for sys_id, ref_sat in ref_sats.items():
                if sys_id not in rover_by_system or sys_id not in base_by_system:
                    continue

                # Get observations
                rover_dict = {obs.sat: obs for obs in rover_by_system[sys_id]}
                base_dict = {obs.sat: obs for obs in base_by_system[sys_id]}

                # Find common satellites with carrier phase at this frequency
                rover_sats = {obs.sat for obs in rover_by_system[sys_id] 
                             if freq_idx < len(obs.L) and obs.L[freq_idx] > 0}
                base_sats = {obs.sat for obs in base_by_system[sys_id] 
                            if freq_idx < len(obs.L) and obs.L[freq_idx] > 0}
                common_sats = rover_sats & base_sats

                if ref_sat not in common_sats:
                    continue

                other_sats = [s for s in common_sats if s != ref_sat]

                for other_sat in other_sats:
                    # Check if both have valid data at this frequency
                    rov_ref = rover_dict[ref_sat]
                    rov_other = rover_dict[other_sat]
                    base_ref = base_dict[ref_sat]
                    base_other = base_dict[other_sat]
                    
                    if (freq_idx >= len(rov_ref.L) or freq_idx >= len(rov_other.L) or
                        freq_idx >= len(base_ref.L) or freq_idx >= len(base_other.L)):
                        continue
                    
                    if (rov_ref.L[freq_idx] <= 0 or rov_other.L[freq_idx] <= 0 or
                        base_ref.L[freq_idx] <= 0 or base_other.L[freq_idx] <= 0):
                        continue

                    # Get wavelengths
                    wavelength = self.get_wavelength(other_sat, freq_idx)
                    ref_wavelength = self.get_wavelength(ref_sat, freq_idx)

                    # Carrier phase DD calculation (RTKLIB style)
                    # Assume values are now in cycles after RINEX reader fix
                    rov_ref_cycles = rov_ref.L[freq_idx]
                    rov_other_cycles = rov_other.L[freq_idx]
                    base_ref_cycles = base_ref.L[freq_idx]
                    base_other_cycles = base_other.L[freq_idx]
                    
                    # Form single differences in cycles
                    sd_ref_cycles = rov_ref_cycles - base_ref_cycles
                    sd_other_cycles = rov_other_cycles - base_other_cycles
                    
                    # Form double difference in cycles
                    dd_cycles = sd_other_cycles - sd_ref_cycles
                    
                    # Convert to meters for consistency
                    dd_meters = dd_cycles * wavelength
                    
                    # Debug values for both L1 and L2
                    if len(all_dd_phase_cycles) < 2:
                        print(f"DEBUG L{freq_idx+1} DD: sat pair {ref_sat}-{other_sat}")
                        print(f"  SD cycles: ref={sd_ref_cycles:.3f}, other={sd_other_cycles:.3f}")
                        print(f"  DD cycles={dd_cycles:.3f}, DD meters={dd_meters:.3f}")

                    # Pseudorange DD in meters
                    if (freq_idx < len(rov_ref.P) and freq_idx < len(rov_other.P) and
                        freq_idx < len(base_ref.P) and freq_idx < len(base_other.P) and
                        rov_ref.P[freq_idx] > 0 and rov_other.P[freq_idx] > 0 and
                        base_ref.P[freq_idx] > 0 and base_other.P[freq_idx] > 0):
                        dd_P = (rov_other.P[freq_idx] - rov_ref.P[freq_idx]) - \
                               (base_other.P[freq_idx] - base_ref.P[freq_idx])
                        all_dd_code.append(dd_P)
                    else:
                        all_dd_code.append(0.0)  # No code available

                    all_dd_phase_m.append(dd_meters)
                    all_dd_phase_cycles.append(dd_cycles)
                    all_sat_pairs.append((ref_sat, other_sat))
                    wavelengths.append(wavelength)

            info = {
                'wavelengths': np.array(wavelengths),
                'n_dd': len(all_sat_pairs),
                'freq_idx': freq_idx
            }

            # For backwards compatibility, return phase in both meters and cycles
            return (np.array(all_dd_phase_m),
                    np.array(all_dd_phase_cycles),
                    all_sat_pairs,
                    info)
        
        else:
            # Multi-frequency mode - return dict for each frequency
            results = {}
            
            # Try all possible frequencies
            for f_idx in range(3):
                dd_phase_m, dd_phase_cycles, sat_pairs, info = self.form_carrier_dd(
                    rover_obs, base_obs, ref_sats, freq_idx=f_idx
                )
                
                if info['n_dd'] > 0:  # Only include if we have observations
                    results[f_idx] = {
                        'dd_phase_m': dd_phase_m,
                        'dd_phase_cycles': dd_phase_cycles,
                        'sat_pairs': sat_pairs,
                        'wavelengths': info['wavelengths']
                    }
            
            # Return dict format for multi-frequency
            if results:
                # Return as dictionaries indexed by frequency
                all_dd_phase_m = {}
                all_dd_phase_cycles = {}
                all_sat_pairs = {}
                all_info = {
                    'n_freq': len(results),
                    'freq_indices': list(results.keys()),
                    'n_dd_per_freq': {f: len(results[f]['sat_pairs']) for f in results}
                }
                
                for f_idx, f_data in results.items():
                    all_dd_phase_m[f_idx] = f_data['dd_phase_m']
                    all_dd_phase_cycles[f_idx] = f_data['dd_phase_cycles']
                    all_sat_pairs[f_idx] = f_data['sat_pairs']
                
                return all_dd_phase_m, all_dd_phase_cycles, all_sat_pairs, all_info
            else:
                # No data available
                return {}, {}, {}, {'n_freq': 0, 'freq_indices': []}

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
                              threshold: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
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

    def _group_by_system(self, observations: list[Observation]) -> dict[int, list[Observation]]:
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
                           n_candidates: int = 2) -> tuple[np.ndarray, float]:
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
