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

"""Main RTK processor integrating all RTK components"""


import numpy as np

from ..core.data_structures import NavigationData, Observation
from ..satellite.ephemeris import select_ephemeris
from ..satellite.satellite_position import compute_satellite_position
from .ambiguity_resolution import RTKAmbiguityManager
from .cycle_slip import CycleSlipDetector
from .double_difference import form_double_differences


class RTKProcessor:
    """
    Complete RTK processing pipeline for Real-Time Kinematic positioning.

    This class integrates all RTK components to provide high-precision relative
    positioning between a rover and base station using carrier phase measurements.
    It handles cycle slip detection, double difference formation, ambiguity
    resolution, and position computation.

    Attributes:
        base_position: Base station position in ECEF coordinates (m)
        ambiguity_manager: Manager for integer ambiguity resolution
        cycle_slip_detector: Detector for carrier phase discontinuities
        last_fix_time: GPS time of last successful ambiguity fix
        continuous_fix_epochs: Number of consecutive fixed solutions
        baseline_length: Distance between rover and base (m)
        fix_ratio: Ambiguity resolution quality ratio
        position_accuracy: Estimated position accuracy (m)

    Methods:
        process_epoch: Process one epoch of RTK observations
        get_quality_metrics: Get current processing quality metrics
        reset_ambiguities: Reset all ambiguity tracking

    Examples:
        >>> processor = RTKProcessor(base_ecef)
        >>> position, info = processor.process_epoch(
        ...     rover_obs, base_obs, nav_data, gps_time, rover_approx)
        >>> print(f"Fix type: {info['fix_type']}")
        >>> print(f"Accuracy: {info['position_accuracy']:.3f} m")
    """

    def __init__(self, base_position: np.ndarray):
        """
        Initialize RTK processor

        Parameters:
        -----------
        base_position : np.ndarray
            Base station position in ECEF (m)
        """
        self.base_position = base_position

        # Processing components
        self.ambiguity_manager = RTKAmbiguityManager()
        self.cycle_slip_detector = CycleSlipDetector()

        # Processing state
        self.last_fix_time = 0.0
        self.continuous_fix_epochs = 0
        self.baseline_length = 0.0

        # Quality metrics
        self.fix_ratio = 0.0
        self.position_accuracy = 0.0

    def process_epoch(self,
                     rover_observations: list[Observation],
                     base_observations: list[Observation],
                     nav_data: NavigationData,
                     time: float,
                     rover_position_approx: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Process one epoch of RTK observations

        Parameters:
        -----------
        rover_observations : List[Observation]
            Rover observations
        base_observations : List[Observation]
            Base station observations
        nav_data : NavigationData
            Navigation data with ephemerides
        time : float
            Current time (GPS time)
        rover_position_approx : np.ndarray
            Approximate rover position (ECEF)

        Returns:
        --------
        position : np.ndarray
            Rover position (ECEF)
        info : Dict
            Processing information and quality metrics
        """
        info = {
            'fix_type': 'float',
            'num_satellites': 0,
            'baseline_length': 0.0,
            'ratio': 0.0,
            'position_accuracy': 0.0,
            'cycle_slips': {},
            'fixed_ambiguities': {}
        }

        try:
            # Detect cycle slips
            rover_slips = self.cycle_slip_detector.detect_cycle_slips(rover_observations, time)
            base_slips = self.cycle_slip_detector.detect_cycle_slips(base_observations, time)
            info['cycle_slips'] = {**rover_slips, **base_slips}

            # Reset ambiguities for satellites with cycle slips
            self._handle_cycle_slips(rover_slips, base_slips)

            # Form double differences (using function instead of class)
            dd_result = form_double_differences(
                rover_observations, base_observations, nav_data, time,
                self.base_position, np.array([0.0, 0.0, 0.0]),  # placeholder for base LLH
                use_systems=['G', 'E', 'C', 'J'], cutoff_angle=15.0
            )

            if not dd_result:
                return rover_position_approx, info

            # Extract DD measurements
            dd_pseudorange = np.array([dd['dd_obs'] for dd in dd_result])
            dd_carrier = np.array([dd['dd_carrier'] or 0.0 for dd in dd_result])
            sat_pairs = [(dd['ref_sat'], dd['sat']) for dd in dd_result]

            if len(sat_pairs) < 1:
                return rover_position_approx, info

            info['num_satellites'] = len(sat_pairs) + 1  # +1 for reference satellite

            # Get satellite positions and clocks
            sat_positions, sat_clocks = self._compute_satellite_states(
                sat_pairs, nav_data, time)

            # Compute geometry matrix
            H = self._compute_dd_geometry_matrix(sat_positions, rover_position_approx, sat_pairs)

            # Process measurements
            if len(dd_carrier) > 0:
                # RTK with carrier phase
                position, fix_info = self._process_rtk_carrier(
                    dd_pseudorange, dd_carrier, H, sat_pairs,
                    sat_positions, sat_clocks, rover_position_approx, time)
                info.update(fix_info)
            else:
                # Code-only differential positioning
                position = self._process_code_differential(
                    dd_pseudorange, H, rover_position_approx)
                info['fix_type'] = 'dgps'

            # Update quality metrics
            self.baseline_length = np.linalg.norm(position - self.base_position)
            info['baseline_length'] = self.baseline_length

            # Position accuracy estimation
            if info['fix_type'] == 'fixed':
                info['position_accuracy'] = 0.01 + 0.001 * self.baseline_length  # cm + ppm
            elif info['fix_type'] == 'float':
                info['position_accuracy'] = 0.1 + 0.01 * self.baseline_length   # dm + cm/km
            else:
                info['position_accuracy'] = 1.0 + 0.1 * self.baseline_length    # m + dm/km

            return position, info

        except Exception as e:
            print(f"RTK processing error: {e}")
            return rover_position_approx, info

    def _compute_satellite_states(self,
                                sat_pairs: list[tuple[int, int]],
                                nav_data: NavigationData,
                                time: float) -> tuple[dict[int, np.ndarray], dict[int, float]]:
        """
        Compute satellite positions and clock biases for all satellites in DD pairs.

        This method computes ECEF positions and clock biases for all satellites
        involved in double difference measurements using broadcast ephemerides.
        It handles multiple GNSS systems and selects the best available ephemeris
        for each satellite based on time validity.

        Parameters:
            sat_pairs: List of satellite pairs [(ref_sat, sat), ...]
                      Each tuple contains reference and non-reference satellite IDs
            nav_data: Navigation data containing broadcast ephemerides
            time: GPS time for position computation (seconds)

        Returns:
            Tuple containing:
            - sat_positions: Dictionary mapping satellite ID to ECEF position vector (m)
                           {sat_id: np.array([X, Y, Z])}
            - sat_clocks: Dictionary mapping satellite ID to clock bias (seconds)
                        {sat_id: clock_bias}

        Notes:
            - Returns empty dictionaries if ephemeris not available
            - Position accuracy depends on ephemeris age and quality
            - Clock biases include relativistic corrections
        """
        sat_positions = {}
        sat_clocks = {}

        # Get all unique satellites
        all_sats = set()
        for ref_sat, sat in sat_pairs:
            all_sats.add(ref_sat)
            all_sats.add(sat)

        for sat in all_sats:
            eph = select_ephemeris(nav_data, sat, time)
            if eph is not None:
                pos, clk, var = compute_satellite_position(eph, time)
                sat_positions[sat] = pos
                sat_clocks[sat] = clk

        return sat_positions, sat_clocks

    def _compute_dd_geometry_matrix(self, sat_positions, rover_position, sat_pairs):
        """
        Compute double difference geometry matrix

        Parameters:
        -----------
        sat_positions : dict
            Satellite positions {sat_id: position_vector}
        rover_position : np.ndarray
            Rover position (ECEF)
        sat_pairs : list
            List of satellite pairs [(ref_sat, sat), ...]

        Returns:
        --------
        np.ndarray : Geometry matrix (n_pairs x 3)
        """
        n_pairs = len(sat_pairs)
        H = np.zeros((n_pairs, 3))

        for i, (ref_sat, sat) in enumerate(sat_pairs):
            if ref_sat not in sat_positions or sat not in sat_positions:
                continue

            # Unit vectors from rover to satellites
            ref_pos = sat_positions[ref_sat]
            sat_pos = sat_positions[sat]

            ref_vec = ref_pos - rover_position
            sat_vec = sat_pos - rover_position

            ref_range = np.linalg.norm(ref_vec)
            sat_range = np.linalg.norm(sat_vec)

            if ref_range > 0 and sat_range > 0:
                ref_unit = ref_vec / ref_range
                sat_unit = sat_vec / sat_range

                # Double difference geometry: (sat - rover) - (ref_sat - rover)
                H[i, :] = sat_unit - ref_unit

        return H

    def _process_rtk_carrier(self,
                           dd_pseudorange: np.ndarray,
                           dd_carrier: np.ndarray,
                           H: np.ndarray,
                           sat_pairs: list[tuple[int, int]],
                           sat_positions: dict[int, np.ndarray],
                           sat_clocks: dict[int, float],
                           rover_pos_approx: np.ndarray,
                           time: float) -> tuple[np.ndarray, dict]:
        """
        Process RTK solution using both pseudorange and carrier phase double differences.

        This method performs weighted least squares estimation combining pseudorange
        and carrier phase DD measurements. It attempts integer ambiguity resolution
        using LAMBDA or similar algorithms and returns either fixed or float solution
        based on resolution success.

        The processing includes:
        - Float solution computation using weighted least squares
        - Integer ambiguity resolution attempt
        - Fixed solution computation if ambiguities resolved
        - Quality metrics calculation

        Parameters:
            dd_pseudorange: Array of DD pseudorange measurements (m)
            dd_carrier: Array of DD carrier phase measurements (cycles)
            H: Geometry matrix for DD measurements (n_pairs x 3)
            sat_pairs: List of satellite pairs [(ref_sat, sat), ...]
            sat_positions: Dictionary of satellite ECEF positions
            sat_clocks: Dictionary of satellite clock biases
            rover_pos_approx: Approximate rover position ECEF (m)
            time: Current GPS time (seconds)

        Returns:
            Tuple containing:
            - position: Computed rover position ECEF (m)
            - info: Dictionary with processing information:
                   - 'fix_type': 'fixed', 'partial', or 'float'
                   - 'ratio': Ambiguity resolution quality ratio
                   - 'fixed_ambiguities': Dictionary of fixed integer ambiguities

        Notes:
            - Uses L1 wavelength for single-frequency processing
            - Applies different weights to pseudorange (3m) and carrier phase (1cm)
            - Falls back to float solution if ambiguity resolution fails
        """
        n_dd = len(sat_pairs)
        wavelength = 299792458.0 / 1575.42e6  # L1 wavelength

        # Setup observation equation: y = H*dx + ambiguities
        # where y = [dd_pseudorange; dd_carrier_range]
        y = np.concatenate([dd_pseudorange, dd_carrier * wavelength])

        # Design matrix
        H_full = np.vstack([H, H])  # Same geometry for both pseudorange and carrier

        # Add ambiguity columns for carrier phase
        H_amb = np.zeros((2 * n_dd, n_dd))
        H_amb[n_dd:, :] = wavelength * np.eye(n_dd)  # Only carrier phase affected by ambiguities

        H_combined = np.hstack([H_full, H_amb])

        # Weight matrix
        sigma_pr = 3.0  # m
        sigma_cp = 0.01 * wavelength  # 1cm converted to meters
        W = np.diag(np.concatenate([
            np.full(n_dd, 1/sigma_pr**2),
            np.full(n_dd, 1/sigma_cp**2)
        ]))

        # Float solution
        N = H_combined.T @ W @ H_combined
        b = H_combined.T @ W @ y

        try:
            x_float = np.linalg.solve(N, b)
            P_float = np.linalg.inv(N)
        except np.linalg.LinAlgError:
            return rover_pos_approx, {'fix_type': 'float', 'ratio': 0.0}

        # Extract position and ambiguity estimates
        dx_float = x_float[:3]
        amb_float = x_float[3:]
        P_amb = P_float[3:, 3:]

        # Try integer ambiguity resolution
        satellites = [pair[1] for pair in sat_pairs]  # Non-reference satellites
        fixed_ambs, fix_status = self.ambiguity_manager.update_ambiguities(
            satellites, amb_float, P_amb)

        info = {}

        # Check if any ambiguities are fixed
        num_fixed = sum(fix_status.values())
        if num_fixed > 0:
            # Fixed solution
            amb_fixed = np.array([fixed_ambs.get(sat, amb_float[i])
                                for i, sat in enumerate(satellites)])

            # Recompute position with fixed ambiguities
            y_corrected = y.copy()
            y_corrected[n_dd:] -= wavelength * amb_fixed

            # Position-only least squares
            dx_fixed = np.linalg.lstsq(H, y_corrected[:n_dd], rcond=None)[0]

            info['fix_type'] = 'fixed' if num_fixed == len(satellites) else 'partial'
            info['fixed_ambiguities'] = fixed_ambs
            info['ratio'] = self.ambiguity_manager.resolver.ratio_threshold  # Placeholder

            self.continuous_fix_epochs += 1
            self.last_fix_time = time

            return rover_pos_approx + dx_fixed, info
        else:
            # Float solution
            info['fix_type'] = 'float'
            info['ratio'] = 0.0
            self.continuous_fix_epochs = 0

            return rover_pos_approx + dx_float, info

    def _process_code_differential(self,
                                 dd_pseudorange: np.ndarray,
                                 H: np.ndarray,
                                 rover_pos_approx: np.ndarray) -> np.ndarray:
        """
        Process differential positioning using only pseudorange double differences.

        This method computes DGPS (Differential GPS) solution when carrier phase
        measurements are unavailable or unreliable. It uses simple least squares
        estimation without ambiguity resolution, providing meter-level accuracy.

        Parameters:
            dd_pseudorange: Array of DD pseudorange measurements (m)
            H: Geometry matrix for DD measurements (n_pairs x 3)
            rover_pos_approx: Approximate rover position ECEF (m)

        Returns:
            Computed rover position in ECEF coordinates (m)
            Returns approximate position if least squares fails

        Notes:
            - Accuracy typically 1-3 meters depending on conditions
            - No ambiguity resolution required (code-only)
            - Suitable for applications not requiring cm-level precision
        """
        # Simple least squares
        try:
            dx = np.linalg.lstsq(H, dd_pseudorange, rcond=None)[0]
            return rover_pos_approx + dx
        except np.linalg.LinAlgError:
            return rover_pos_approx

    def _handle_cycle_slips(self,
                          rover_slips: dict[int, bool],
                          base_slips: dict[int, bool]):
        """
        Handle detected cycle slips by resetting affected ambiguities.

        When cycle slips are detected in carrier phase measurements, the integer
        ambiguities for affected satellites must be reset and re-estimated.
        This method identifies all satellites with cycle slips from both rover
        and base stations and clears their fixed ambiguity values.

        Parameters:
            rover_slips: Dictionary mapping satellite ID to cycle slip detection
                        {sat_id: True if slip detected, False otherwise}
            base_slips: Dictionary with same format for base station

        Returns:
            None (modifies internal ambiguity manager state)

        Side Effects:
            - Removes fixed ambiguities for affected satellites
            - Resets fix status flags for affected satellites
            - May cause temporary degradation to float solution

        Notes:
            - Cycle slips in either rover or base affect the DD measurement
            - Ambiguity re-initialization typically takes several epochs
        """
        all_slips = set()

        for sat, slip in rover_slips.items():
            if slip:
                all_slips.add(sat)

        for sat, slip in base_slips.items():
            if slip:
                all_slips.add(sat)

        # Reset ambiguities for affected satellites
        for sat in all_slips:
            if sat in self.ambiguity_manager.fixed_ambiguities:
                del self.ambiguity_manager.fixed_ambiguities[sat]
            if sat in self.ambiguity_manager.fix_status:
                self.ambiguity_manager.fix_status[sat] = False

    def get_quality_metrics(self) -> dict:
        """
        Get current RTK processing quality metrics.

        Returns a dictionary containing various quality indicators that can be
        used to assess the reliability and accuracy of the RTK solution.

        Returns:
            Dictionary containing:
            - 'baseline_length': Distance between rover and base (m)
            - 'continuous_fix_epochs': Number of consecutive fixed solutions
            - 'last_fix_time': GPS time of last successful fix (seconds)
            - 'position_accuracy': Estimated position accuracy (m)

        Notes:
            - Longer baselines generally have reduced accuracy
            - More continuous fix epochs indicate stable solution
            - Position accuracy is an estimate based on fix type and baseline
        """
        return {
            'baseline_length': self.baseline_length,
            'continuous_fix_epochs': self.continuous_fix_epochs,
            'last_fix_time': self.last_fix_time,
            'position_accuracy': self.position_accuracy
        }

    def reset_ambiguities(self):
        """
        Reset all integer ambiguity tracking and related state.

        This method completely clears all ambiguity-related state, forcing
        re-initialization of the RTK solution. Use when starting a new session,
        after extended signal loss, or when solution quality degrades.

        Returns:
            None

        Side Effects:
            - Clears all fixed ambiguities
            - Resets fix status for all satellites
            - Resets continuous fix epoch counter
            - Solution will revert to float until new ambiguities resolved

        Notes:
            - Call this method after major interruptions or antenna changes
            - Ambiguity re-convergence typically takes 10-30 seconds
        """
        self.ambiguity_manager.fixed_ambiguities.clear()
        self.ambiguity_manager.fix_status.clear()
        self.continuous_fix_epochs = 0
