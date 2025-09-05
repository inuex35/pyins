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

"""Cycle slip detection for RTK processing"""

import numpy as np
from typing import Dict, List

from ..core.constants import (
    CLIGHT,
    # GPS frequencies
    FREQ_L1, FREQ_L2,
    # GLONASS frequencies
    FREQ_G1, FREQ_G2,
    # Galileo frequencies
    FREQ_E1, FREQ_E5b,
    # BeiDou frequencies
    FREQ_B1I, FREQ_B3,
    # QZSS frequencies (same as GPS)
    FREQ_J1, FREQ_J2
)
from ..core.data_structures import Observation


class SimpleCycleSlipDetector:
    """
    Simple cycle slip detector using only LLI (Loss of Lock Indicator) flag
    Similar to GREAT-PVT implementation
    """

    def __init__(self):
        """Initialize simple cycle slip detector"""
        pass

    def detect_cycle_slips(self,
                          observations: List[Observation],
                          time: float) -> Dict[int, bool]:
        """
        Detect cycle slips using only LLI flag
        
        This is the same approach as GREAT-PVT:
        - Check LLI flag >= 1 for cycle slip
        - No complex algorithms like MW or GF
        
        Parameters:
        -----------
        observations : List[Observation]
            Current epoch observations
        time : float
            Current time (not used in simple detector)
            
        Returns:
        --------
        slip_flags : Dict[int, bool]
            Cycle slip flags by satellite (True if slip detected)
        """
        slip_flags = {}
        
        for obs in observations:
            # Skip invalid observations
            if obs.L[0] == 0.0:
                continue
                
            sat = obs.sat
            
            # Check LLI flag for L1
            # LLI >= 1 indicates loss of lock (cycle slip)
            slip_detected = obs.LLI[0] >= 1
            
            # If dual-frequency, also check L2 LLI
            if obs.L[1] > 0.0:
                slip_detected = slip_detected or (obs.LLI[1] >= 1)
            
            slip_flags[sat] = slip_detected
            
        return slip_flags


class CycleSlipDetector:
    """Detect cycle slips in carrier phase measurements"""

    def __init__(self,
                 lln_threshold: float = 0.25,  # cycles - relaxed for real data
                 mw_threshold: float = 4.0,    # cycles - relaxed for real data
                 gf_threshold: float = 0.05):   # meters
        """
        Initialize cycle slip detector

        Parameters:
        -----------
        lln_threshold : float
            Lock time normalized threshold (cycles)
        mw_threshold : float
            Melbourne-Wubbena threshold (cycles)
        gf_threshold : float
            Geometry-free threshold (meters)
        """
        self.lln_threshold = lln_threshold
        self.mw_threshold = mw_threshold
        self.gf_threshold = gf_threshold

        # History for detection
        self.phase_history = {}  # sat -> list of (time, phase)
        self.mw_history = {}     # sat -> list of (time, mw)
        self.gf_history = {}     # sat -> list of (time, gf)
        self.lock_time_history = {}  # sat -> list of (time, lock_time)

    def _get_frequencies(self, sat: int) -> tuple[float, float]:
        """Get L1/L2 frequencies for the satellite system"""
        # Determine system from satellite number
        if 1 <= sat <= 32:  # GPS
            return FREQ_L1, FREQ_L2
        elif 38 <= sat <= 69:  # GLONASS
            return FREQ_G1, FREQ_G2
        elif 71 <= sat <= 102:  # Galileo
            return FREQ_E1, FREQ_E5b
        elif 141 <= sat <= 175:  # BeiDou
            return FREQ_B1I, FREQ_B3
        elif 193 <= sat <= 202:  # QZSS
            return FREQ_J1, FREQ_J2
        else:
            # Default to GPS
            return FREQ_L1, FREQ_L2

    def detect_cycle_slips(self,
                          observations: List[Observation],
                          time: float) -> Dict[int, bool]:
        """
        Detect cycle slips for all satellites

        Parameters:
        -----------
        observations : List[Observation]
            Current epoch observations
        time : float
            Current time

        Returns:
        --------
        slip_flags : Dict[int, bool]
            Cycle slip flags by satellite
        """
        slip_flags = {}

        for obs in observations:
            # Check if L1 observation is valid (using array indexing)
            if obs.L[0] == 0.0:  # Skip invalid observations
                continue

            sat = obs.sat
            slip_detected = False
            
            # Skip cycle slip detection for BeiDou satellites
            # User feedback: cycle slips rarely occur in real data
            # BeiDou false positives are causing issues
            if 141 <= sat <= 175:  # BeiDou satellites
                slip_flags[sat] = False
                self._update_history(obs, time)
                continue

            # Lock time normalized detection using LLI array
            if obs.LLI[0] > 0:
                slip_detected = True

            # Dual-frequency detection if L2 available
            if obs.L[1] > 0.0:
                # Melbourne-Wubbena detection
                mw_slip = self._detect_mw_slip(obs, time)

                # Geometry-free detection
                gf_slip = self._detect_gf_slip(obs, time)

                slip_detected = slip_detected or mw_slip or gf_slip
            else:
                # Single-frequency detection
                phase_slip = self._detect_phase_slip(obs, time)
                slip_detected = slip_detected or phase_slip

            slip_flags[sat] = slip_detected

            # Update history
            self._update_history(obs, time)

        return slip_flags

    def _detect_mw_slip(self, obs: Observation, time: float) -> bool:
        """Detect cycle slip using Melbourne-Wubbena combination"""
        sat = obs.sat

        # Compute Melbourne-Wubbena combination
        mw = self._compute_melbourne_wubbena(obs)

        if sat not in self.mw_history:
            self.mw_history[sat] = []
            return False

        # Check against recent history
        recent_mw = [entry for entry in self.mw_history[sat] if time - entry[0] < 30.0]

        if len(recent_mw) < 3:
            return False

        # Compute mean and check for outliers
        mw_values = [entry[1] for entry in recent_mw]
        mean_mw = np.mean(mw_values)

        return abs(mw - mean_mw) > self.mw_threshold

    def _detect_gf_slip(self, obs: Observation, time: float) -> bool:
        """Detect cycle slip using geometry-free combination"""
        sat = obs.sat
        
        # Get system-specific frequencies
        l1_freq, l2_freq = self._get_frequencies(sat)
        l1_wavelength = CLIGHT / l1_freq
        l2_wavelength = CLIGHT / l2_freq

        # Compute geometry-free combination with correct wavelengths
        gf = obs.L[0] * l1_wavelength - obs.L[1] * l2_wavelength

        if sat not in self.gf_history:
            self.gf_history[sat] = []
            return False

        # Check recent history
        recent_gf = [entry for entry in self.gf_history[sat] if time - entry[0] < 30.0]

        if len(recent_gf) < 2:
            return False

        # Check for sudden jumps
        last_gf = recent_gf[-1][1]
        gf_diff = abs(gf - last_gf)

        return gf_diff > self.gf_threshold

    def _detect_phase_slip(self, obs: Observation, time: float) -> bool:
        """Detect cycle slip using single frequency phase"""
        sat = obs.sat

        if sat not in self.phase_history:
            self.phase_history[sat] = []
            return False

        # Check recent history
        recent_phase = [entry for entry in self.phase_history[sat] if time - entry[0] < 10.0]

        if len(recent_phase) < 3:
            return False

        # Compute phase rate
        times = [entry[0] for entry in recent_phase]
        phases = [entry[1] for entry in recent_phase]

        if len(times) >= 2:
            dt = times[-1] - times[-2]
            if dt > 0:
                phase_rate = (phases[-1] - phases[-2]) / dt

                # Check if current rate is consistent
                current_rate = (obs.L[0] - phases[-1]) / (time - times[-1])
                rate_diff = abs(current_rate - phase_rate)

                return rate_diff > 1.0  # cycles/s threshold

        return False

    def _compute_melbourne_wubbena(self, obs: Observation) -> float:
        """Compute Melbourne-Wubbena combination"""
        # Check if all required data is available
        if obs.L[1] == 0.0 or obs.P[0] == 0.0 or obs.P[1] == 0.0:
            return 0.0

        # Get system-specific frequencies
        l1_freq, l2_freq = self._get_frequencies(obs.sat)

        # MW combination with correct frequencies
        phase_term = obs.L[0] - obs.L[1]
        code_term = (l1_freq * obs.P[0] - l2_freq * obs.P[1]) / (
            CLIGHT * (l1_freq - l2_freq))

        return phase_term - code_term

    def _update_history(self, obs: Observation, time: float):
        """Update observation history"""
        sat = obs.sat
        
        # Get system-specific frequencies
        l1_freq, l2_freq = self._get_frequencies(sat)
        l1_wavelength = CLIGHT / l1_freq
        l2_wavelength = CLIGHT / l2_freq

        # Phase history
        if sat not in self.phase_history:
            self.phase_history[sat] = []
        self.phase_history[sat].append((time, obs.L[0]))

        # MW history
        if obs.L[1] > 0.0:
            mw = self._compute_melbourne_wubbena(obs)
            if sat not in self.mw_history:
                self.mw_history[sat] = []
            self.mw_history[sat].append((time, mw))

            # GF history with correct wavelengths
            gf = obs.L[0] * l1_wavelength - obs.L[1] * l2_wavelength
            if sat not in self.gf_history:
                self.gf_history[sat] = []
            self.gf_history[sat].append((time, gf))

        # Clean old history (keep 5 minutes)
        self._clean_history(time, 300.0)

    def _clean_history(self, current_time: float, max_age: float):
        """Clean old history entries"""
        cutoff_time = current_time - max_age

        for sat in list(self.phase_history.keys()):
            self.phase_history[sat] = [
                entry for entry in self.phase_history[sat]
                if entry[0] > cutoff_time
            ]
            if not self.phase_history[sat]:
                del self.phase_history[sat]

        for sat in list(self.mw_history.keys()):
            self.mw_history[sat] = [
                entry for entry in self.mw_history[sat]
                if entry[0] > cutoff_time
            ]
            if not self.mw_history[sat]:
                del self.mw_history[sat]

        for sat in list(self.gf_history.keys()):
            self.gf_history[sat] = [
                entry for entry in self.gf_history[sat]
                if entry[0] > cutoff_time
            ]
            if not self.gf_history[sat]:
                del self.gf_history[sat]


class TurboEditDetector:
    """TurboEdit cycle slip detection"""

    def __init__(self, threshold: float = 4.0):
        """
        Initialize TurboEdit detector

        Parameters:
        -----------
        threshold : float
            Detection threshold in standard deviations
        """
        self.threshold = threshold
        self.code_minus_phase_history = {}  # sat -> history

    def detect_slips(self,
                    observations: List[Observation],
                    time: float) -> Dict[int, bool]:
        """
        Detect cycle slips using TurboEdit algorithm

        Parameters:
        -----------
        observations : List[Observation]
            Current observations
        time : float
            Current time

        Returns:
        --------
        slip_flags : Dict[int, bool]
            Cycle slip detection flags
        """
        slip_flags = {}

        for obs in observations:
            if obs.L[0] == 0.0 or obs.P[0] == 0.0:
                continue

            sat = obs.sat

            # Get system-specific L1 frequency
            l1_freq, _ = self._get_frequencies(sat)
            wavelength = CLIGHT / l1_freq
            
            # Compute code-minus-phase
            cmp = obs.P[0] - obs.L[0] * wavelength

            slip_detected = self._check_turboedit(sat, cmp, time)
            slip_flags[sat] = slip_detected

            # Update history
            if sat not in self.code_minus_phase_history:
                self.code_minus_phase_history[sat] = []
            self.code_minus_phase_history[sat].append((time, cmp))

            # Clean old history
            self.code_minus_phase_history[sat] = [
                entry for entry in self.code_minus_phase_history[sat]
                if time - entry[0] < 60.0  # Keep 1 minute
            ]

        return slip_flags

    def _get_frequencies(self, sat: int) -> tuple[float, float]:
        """Get L1/L2 frequencies for the satellite system"""
        # Determine system from satellite number
        if 1 <= sat <= 32:  # GPS
            return FREQ_L1, FREQ_L2
        elif 38 <= sat <= 69:  # GLONASS
            return FREQ_G1, FREQ_G2
        elif 71 <= sat <= 102:  # Galileo
            return FREQ_E1, FREQ_E5b
        elif 141 <= sat <= 175:  # BeiDou
            return FREQ_B1I, FREQ_B3
        elif 193 <= sat <= 202:  # QZSS
            return FREQ_J1, FREQ_J2
        else:
            # Default to GPS
            return FREQ_L1, FREQ_L2

    def _check_turboedit(self, sat: int, cmp: float, time: float) -> bool:
        """Check for cycle slip using TurboEdit statistic"""
        if sat not in self.code_minus_phase_history:
            return False

        history = self.code_minus_phase_history[sat]
        if len(history) < 4:
            return False

        # Compute fourth-order difference
        values = [entry[1] for entry in history[-4:]]
        fourth_diff = values[3] - 4*values[2] + 6*values[1] - 4*values[0]

        # Estimate noise level from recent history
        if len(history) >= 10:
            recent_values = [entry[1] for entry in history[-10:]]
            noise_std = np.std(np.diff(recent_values))
        else:
            noise_std = 1.0  # Default noise level

        # TurboEdit test
        test_statistic = abs(fourth_diff) / (8 * noise_std)

        return test_statistic > self.threshold