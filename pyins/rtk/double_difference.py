#!/usr/bin/env python3
"""
Double Difference formation for GNSS observations
"""


import numpy as np

from ..core.constants import CLIGHT, SYS_BDS, SYS_GAL, SYS_GLO, SYS_GPS, SYS_QZS, sat2sys
from ..geometry.elevation import compute_elevation_angle
from ..gnss.ephemeris import satpos
from ..core.data_structures import Observation
from typing import Optional


class DoubleDifferenceProcessor:
    """Process GNSS observations to form double differences with per-system reference satellites"""

    def __init__(self, reference_sats: Optional[dict[int, int]] = None):
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
                              rover_obs: list[Observation],
                              base_obs: list[Observation],
                              frequency_idx: Optional[int] = None) -> tuple:
        """
        Form double difference observations with per-system reference satellites

        Parameters:
        -----------
        rover_obs : List[Observation]
            Rover observations
        base_obs : List[Observation]
            Base station observations
        frequency_idx : int, optional
            Frequency index to process. If None, process all available frequencies

        Returns:
        --------
        If frequency_idx is specified:
            dd_pseudorange : np.ndarray - Double difference pseudorange (m)
            dd_carrier : np.ndarray - Double difference carrier phase (cycles)
            sat_pairs : List[Tuple[int, int]] - List of (reference_sat, other_sat) pairs
            ref_sats_used : Dict[int, int] - Reference satellites used for each system
        
        If frequency_idx is None (all frequencies):
            dd_pseudorange : Dict[int, np.ndarray] - DD pseudorange per frequency
            dd_carrier : Dict[int, np.ndarray] - DD carrier phase per frequency  
            sat_pairs : Dict[int, List[Tuple[int, int]]] - Sat pairs per frequency
            ref_sats_used : Dict[int, Dict[int, int]] - Reference sats per frequency/system
        """
        # If frequency_idx is None, process all frequencies
        if frequency_idx is None:
            # Detect available frequencies
            max_freqs = 0
            for obs in rover_obs:
                if hasattr(obs, 'P') and len(obs.P) > max_freqs:
                    max_freqs = len(obs.P)
            
            # Process each frequency
            all_dd_pr = {}
            all_dd_cp = {}
            all_sat_pairs = {}
            all_ref_sats = {}
            
            for freq_idx in range(max_freqs):
                # Check if this frequency has valid observations
                has_freq = False
                for obs in rover_obs:
                    if freq_idx < len(obs.P) and obs.P[freq_idx] > 0:
                        has_freq = True
                        break
                
                if has_freq:
                    dd_pr, dd_cp, sat_pairs, ref_sats = self.form_double_differences(
                        rover_obs, base_obs, freq_idx
                    )
                    
                    if len(dd_pr) > 0:
                        all_dd_pr[freq_idx] = dd_pr
                        all_dd_cp[freq_idx] = dd_cp
                        all_sat_pairs[freq_idx] = sat_pairs
                        all_ref_sats[freq_idx] = ref_sats
            
            return (all_dd_pr, all_dd_cp, all_sat_pairs, all_ref_sats)
        
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
                                rover_obs: list[Observation],
                                base_obs: list[Observation]) -> dict[int, tuple[np.ndarray, np.ndarray, list[tuple[int, int]], dict[int, int]]]:
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
                           sat_positions: dict[int, np.ndarray],
                           rover_position: np.ndarray,
                           base_position: np.ndarray,
                           sat_pairs: list[tuple[int, int]],
                           sat_clocks: Optional[dict[int, float]] = None,
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
                                 sat_positions: dict[int, np.ndarray],
                                 user_position: np.ndarray,
                                 sat_pairs: list[tuple[int, int]]) -> np.ndarray:
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

    def get_reference_satellites(self) -> dict[int, int]:
        """Get current reference satellites for each system"""
        return self.last_ref_sats.copy()

    def set_reference_satellite(self, system_id: int, sat_number: int):
        """Set reference satellite for a specific system"""
        self.reference_sats[system_id] = sat_number

    def _group_by_system(self, observations: list[Observation]) -> dict[int, list[Observation]]:
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
                                             observations: list[Observation],
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
                              rover_obs: list[Observation],
                              base_obs: list[Observation]) -> tuple[np.ndarray, np.ndarray, list[int]]:
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


def form_double_differences(rover_obs, base_obs, nav_data, gps_time,
                           reference_ecef, reference_llh,
                           use_systems: list[str] = None,
                           cutoff_angle: float = 15.0) -> list[dict]:
    """
    Form double differences between rover and base observations.

    Double differences are formed independently for each satellite system
    to avoid inter-system biases.

    Parameters:
    -----------
    rover_obs : list
        Rover GNSS observations
    base_obs : list
        Base GNSS observations
    nav_data : dict
        Navigation data
    gps_time : float
        GPS time for this epoch
    reference_ecef : np.ndarray
        Reference position in ECEF (for elevation computation)
    reference_llh : np.ndarray
        Reference position in LLH (for elevation computation)
    use_systems : List[str], optional
        List of systems to use ('G', 'R', 'E', 'C', 'J')
        Default: ['G', 'E', 'C', 'J']
    cutoff_angle : float
        Elevation cutoff angle in degrees (default: 15.0)

    Returns:
    --------
    List of DD measurements with satellite positions
    Each measurement contains:
        - sat: satellite number
        - ref_sat: reference satellite number
        - dd_obs: double difference observation
        - sat_pos: satellite position
        - ref_sat_pos: reference satellite position
        - sat_clk: satellite clock bias
        - ref_sat_clk: reference satellite clock bias
        - elevation: satellite elevation angle
    """
    dd_measurements = []

    # Default systems to use
    if use_systems is None:
        use_systems = ['G', 'E', 'C', 'J']  # Exclude GLONASS by default

    # Get common satellites
    rover_sats = {obs.sat for obs in rover_obs}
    base_sats = {obs.sat for obs in base_obs}
    common_sats = rover_sats & base_sats

    if len(common_sats) < 2:
        return dd_measurements

    # Filter by system
    filtered_sats = []
    for sat in common_sats:
        sys_id = sat2sys(sat)
        sys_map = {SYS_GPS: 'G', SYS_GLO: 'R', SYS_GAL: 'E', SYS_BDS: 'C', SYS_QZS: 'J'}
        system_char = sys_map.get(sys_id, 'G')

        if system_char in use_systems:
            filtered_sats.append(sat)

    if len(filtered_sats) < 2:
        return dd_measurements

    # Compute satellite positions and elevations
    # First compute all satellite positions using satpos
    for obs in rover_obs:
        obs.time = gps_time
    sat_positions, sat_clocks, _, _ = satpos(rover_obs, nav_data)

    sat_data = {}
    for i, obs in enumerate(rover_obs):
        sat = obs.sat
        if sat not in filtered_sats:
            continue

        # Get satellite position
        sat_pos = sat_positions[i]
        sat_clk = sat_clocks[i]
        if sat_pos is None or np.any(np.isnan(sat_pos)):
            continue

        # Compute elevation angle
        elevation = compute_elevation_angle(sat_pos, reference_ecef, reference_llh)

        if elevation < cutoff_angle:
            continue

        # Get base observation
        base_obs_sat = next((o for o in base_obs if o.sat == obs.sat), None)
        if base_obs_sat is None:
            continue

        # Process all available frequencies
        freq_data = {}
        
        # Handle different observation data structures
        if hasattr(obs, 'P') and obs.P is not None:
            pseudorange_data = obs.P
        elif hasattr(obs, 'C') and obs.C is not None:
            pseudorange_data = obs.C
        elif hasattr(obs, 'code') and obs.code is not None:
            # Single frequency case
            rover_pr = obs.code
            base_pr = base_obs_sat.code if hasattr(base_obs_sat, 'code') else None
            
            # Single frequency carrier phase
            rover_cp = obs.phase if hasattr(obs, 'phase') else None
            base_cp = base_obs_sat.phase if hasattr(base_obs_sat, 'phase') else None
            
            if rover_pr is not None and base_pr is not None:
                rover_pr_corrected = rover_pr - sat_clk * CLIGHT
                base_pr_corrected = base_pr - sat_clk * CLIGHT
                freq_data[0] = {
                    'rover_pr': rover_pr_corrected,
                    'base_pr': base_pr_corrected
                }
                
                # Add carrier phase if available
                if rover_cp is not None and base_cp is not None:
                    freq_data[0]['rover_cp'] = rover_cp
                    freq_data[0]['base_cp'] = base_cp
        else:
            continue
        
        # Multi-frequency case
        if hasattr(obs, 'P') and obs.P is not None and len(obs.P) > 1:
            for freq_idx in range(len(obs.P)):
                rover_pr = obs.P[freq_idx]
                base_pr = base_obs_sat.P[freq_idx] if hasattr(base_obs_sat, 'P') and len(base_obs_sat.P) > freq_idx else None

                if rover_pr is None or base_pr is None:
                    continue
                
                # Skip zero/invalid pseudorange values
                if rover_pr <= 0 or base_pr <= 0:
                    continue

                # Apply satellite clock correction to pseudoranges
                rover_pr_corrected = rover_pr - sat_clk * CLIGHT
                base_pr_corrected = base_pr - sat_clk * CLIGHT
                
                freq_data[freq_idx] = {
                    'rover_pr': rover_pr_corrected,
                    'base_pr': base_pr_corrected
                }
                
                # Add carrier phase if available
                rover_cp = obs.L[freq_idx] if hasattr(obs, 'L') and obs.L is not None and len(obs.L) > freq_idx else None
                base_cp = base_obs_sat.L[freq_idx] if hasattr(base_obs_sat, 'L') and base_obs_sat.L is not None and len(base_obs_sat.L) > freq_idx else None
                
                if rover_cp is not None and base_cp is not None:
                    freq_data[freq_idx]['rover_cp'] = rover_cp
                    freq_data[freq_idx]['base_cp'] = base_cp

        if not freq_data:
            continue

        # Store satellite data with all frequencies
        sat_data[obs.sat] = {
            'sat_pos': sat_pos,
            'sat_clk': sat_clk,
            'elevation': elevation,
            'freq_data': freq_data,
            'system': sat2sys(obs.sat)  # Store system ID
        }

    # Group satellites by system
    sat_by_system = {}
    for sat, data in sat_data.items():
        sys_id = data['system']
        if sys_id not in sat_by_system:
            sat_by_system[sys_id] = {}
        sat_by_system[sys_id][sat] = data

    # Form DD for each system independently
    for sys_id, sys_sats in sat_by_system.items():
        if len(sys_sats) < 2:
            continue  # Need at least 2 satellites in the same system

        # Select reference satellite for this system (highest elevation)
        ref_sat = max(sys_sats.keys(), key=lambda s: sys_sats[s]['elevation'])
        ref_data = sys_sats[ref_sat]

        # Form double differences with respect to reference satellite
        for sat, data in sys_sats.items():
            if sat == ref_sat:
                continue

            # Form DD for each frequency that both satellites have
            common_freqs = set(data['freq_data'].keys()) & set(ref_data['freq_data'].keys())
            
            for freq_idx in common_freqs:
                # Pseudorange double difference
                sd_rover_pr = data['freq_data'][freq_idx]['rover_pr'] - ref_data['freq_data'][freq_idx]['rover_pr']
                sd_base_pr = data['freq_data'][freq_idx]['base_pr'] - ref_data['freq_data'][freq_idx]['base_pr']
                dd_pr = sd_rover_pr - sd_base_pr

                measurement = {
                    'sat': sat,
                    'ref_sat': ref_sat,
                    'dd_obs': dd_pr,
                    'freq_idx': freq_idx,
                    'sat_pos': data['sat_pos'],
                    'ref_sat_pos': ref_data['sat_pos'],
                    'sat_clk': data['sat_clk'],
                    'ref_sat_clk': ref_data['sat_clk'],
                    'elevation': data['elevation']
                }
                
                # Add carrier phase double difference if available
                if ('rover_cp' in data['freq_data'][freq_idx] and 
                    'base_cp' in data['freq_data'][freq_idx] and
                    'rover_cp' in ref_data['freq_data'][freq_idx] and
                    'base_cp' in ref_data['freq_data'][freq_idx]):
                    
                    sd_rover_cp = data['freq_data'][freq_idx]['rover_cp'] - ref_data['freq_data'][freq_idx]['rover_cp']
                    sd_base_cp = data['freq_data'][freq_idx]['base_cp'] - ref_data['freq_data'][freq_idx]['base_cp']
                    dd_cp = sd_rover_cp - sd_base_cp
                    
                    measurement['dd_phase_cycles'] = dd_cp

                dd_measurements.append(measurement)

    return dd_measurements
