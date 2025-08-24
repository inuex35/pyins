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

from typing import Optional

import numpy as np

from ..core.constants import SYS_BDS, SYS_GAL, SYS_GLO, SYS_GPS, SYS_QZS, sat2sys
from ..core.data_structures import Observation


class DoubleDifferenceProcessorPerSystem:
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
                              base_obs: list[Observation]) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], dict[int, int]]:
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
                         if obs.P[0] > 0 and obs.L[0] > 0}
            base_sats = {obs.sat for obs in base_by_system[sys_id]
                        if obs.P[0] > 0 and obs.L[0] > 0}
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
                rover_sd_pr = rover_dict[sat].P[0] - rover_dict[ref_sat].P[0]
                rover_sd_cp = rover_dict[sat].L[0] - rover_dict[ref_sat].L[0]
                base_sd_pr = base_dict[sat].P[0] - base_dict[ref_sat].P[0]
                base_sd_cp = base_dict[sat].L[0] - base_dict[ref_sat].L[0]

                # Double differences
                all_dd_pr.append(rover_sd_pr - base_sd_pr)
                all_dd_cp.append(rover_sd_cp - base_sd_cp)
                all_sat_pairs.append((ref_sat, sat))

        return (np.array(all_dd_pr), np.array(all_dd_cp),
                all_sat_pairs, ref_sats_used)

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

    def get_system_name(self, sys_id: int) -> str:
        """Get system name from ID"""
        names = {
            SYS_GPS: "GPS",
            SYS_GLO: "GLONASS",
            SYS_GAL: "Galileo",
            SYS_BDS: "BeiDou",
            SYS_QZS: "QZSS"
        }
        return names.get(sys_id, f"Unknown({sys_id})")
