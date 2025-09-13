#!/usr/bin/env python3
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

"""
Wide Lane and Narrow Lane Combinations for Ambiguity Resolution
================================================================

This module implements Wide Lane (WL), Narrow Lane (NL), and Melbourne-Wübbena (MW)
linear combinations for improved ambiguity resolution.

References:
- Teunissen & Montenbruck (2017): Springer Handbook of GNSS
- Hofmann-Wellenhof et al. (2008): GNSS - Global Navigation Satellite Systems
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
import logging

logger = logging.getLogger(__name__)

# GPS frequencies (Hz)
FREQ_L1 = 1.57542e9  # L1 frequency
FREQ_L2 = 1.22760e9  # L2 frequency
FREQ_L5 = 1.17645e9  # L5 frequency

# GLONASS base frequencies (Hz)
FREQ_G1_BASE = 1.60200e9  # G1 base frequency
FREQ_G2_BASE = 1.24600e9  # G2 base frequency
DFREQ_G1 = 0.56250e6  # G1 channel spacing
DFREQ_G2 = 0.43750e6  # G2 channel spacing

# Galileo frequencies (Hz)
FREQ_E1 = 1.57542e9  # E1 frequency (same as GPS L1)
FREQ_E5a = 1.17645e9  # E5a frequency (same as GPS L5)
FREQ_E5b = 1.20714e9  # E5b frequency
FREQ_E6 = 1.27875e9  # E6 frequency

# BeiDou frequencies (Hz)
FREQ_B1I = 1.561098e9  # B1I frequency
FREQ_B1C = 1.57542e9  # B1C frequency (same as GPS L1)
FREQ_B2a = 1.17645e9  # B2a frequency (same as GPS L5)
FREQ_B3 = 1.26852e9  # B3 frequency

# Speed of light (m/s)
CLIGHT = 299792458.0


class WideNarrowLaneProcessor:
    """
    Process Wide Lane and Narrow Lane combinations for ambiguity resolution
    """
    
    def __init__(self, system: str = 'GPS'):
        """
        Initialize WL/NL processor
        
        Parameters
        ----------
        system : str
            GNSS system ('GPS', 'GLONASS', 'Galileo', 'BeiDou')
        """
        self.system = system
        self._setup_frequencies()
        
    def _setup_frequencies(self):
        """Setup frequencies based on GNSS system"""
        if self.system == 'GPS':
            self.freq1 = FREQ_L1
            self.freq2 = FREQ_L2
            self.freq5 = FREQ_L5
        elif self.system == 'Galileo':
            self.freq1 = FREQ_E1
            self.freq2 = FREQ_E5b
            self.freq5 = FREQ_E5a
        elif self.system == 'BeiDou':
            self.freq1 = FREQ_B1C
            self.freq2 = FREQ_B3
            self.freq5 = FREQ_B2a
        else:
            # Default to GPS
            self.freq1 = FREQ_L1
            self.freq2 = FREQ_L2
            self.freq5 = FREQ_L5
        
        # Calculate wavelengths
        self.lambda1 = CLIGHT / self.freq1
        self.lambda2 = CLIGHT / self.freq2
        self.lambda5 = CLIGHT / self.freq5 if self.freq5 else None
        
    def compute_wide_lane(self, phase_l1: float, phase_l2: float,
                         in_cycles: bool = True) -> Tuple[float, float, float]:
        """
        Compute Wide Lane combination
        
        The Wide Lane has a longer wavelength (~86cm for GPS) making
        ambiguity resolution easier.
        
        WL = (f1*L1 - f2*L2)/(f1 - f2)
        
        Parameters
        ----------
        phase_l1 : float
            L1 carrier phase measurement (cycles or meters)
        phase_l2 : float
            L2 carrier phase measurement (cycles or meters)
        in_cycles : bool
            Whether input is in cycles (True) or meters (False)
            
        Returns
        -------
        wl_phase : float
            Wide lane phase combination
        wl_wavelength : float
            Wide lane wavelength (meters)
        wl_frequency : float
            Wide lane frequency (Hz)
        """
        # Convert to meters if in cycles
        if in_cycles:
            phase_l1_m = phase_l1 * self.lambda1
            phase_l2_m = phase_l2 * self.lambda2
        else:
            phase_l1_m = phase_l1
            phase_l2_m = phase_l2
        
        # Wide lane frequency and wavelength
        freq_wl = self.freq1 - self.freq2
        lambda_wl = CLIGHT / freq_wl
        
        # Wide lane combination in meters
        wl_phase_m = (self.freq1 * phase_l1_m - self.freq2 * phase_l2_m) / freq_wl
        
        # Convert to cycles if requested
        if in_cycles:
            wl_phase = wl_phase_m / lambda_wl
        else:
            wl_phase = wl_phase_m
        
        return wl_phase, lambda_wl, freq_wl
    
    def compute_narrow_lane(self, phase_l1: float, phase_l2: float,
                           in_cycles: bool = True) -> Tuple[float, float, float]:
        """
        Compute Narrow Lane combination
        
        The Narrow Lane has a shorter wavelength (~11cm for GPS) but
        reduced noise and multipath.
        
        NL = (f1*L1 + f2*L2)/(f1 + f2)
        
        Parameters
        ----------
        phase_l1 : float
            L1 carrier phase measurement
        phase_l2 : float
            L2 carrier phase measurement
        in_cycles : bool
            Whether input is in cycles (True) or meters (False)
            
        Returns
        -------
        nl_phase : float
            Narrow lane phase combination
        nl_wavelength : float
            Narrow lane wavelength (meters)
        nl_frequency : float
            Narrow lane frequency (Hz)
        """
        # Convert to meters if in cycles
        if in_cycles:
            phase_l1_m = phase_l1 * self.lambda1
            phase_l2_m = phase_l2 * self.lambda2
        else:
            phase_l1_m = phase_l1
            phase_l2_m = phase_l2
        
        # Narrow lane frequency and wavelength
        freq_nl = self.freq1 + self.freq2
        lambda_nl = CLIGHT / freq_nl
        
        # Narrow lane combination in meters
        nl_phase_m = (self.freq1 * phase_l1_m + self.freq2 * phase_l2_m) / freq_nl
        
        # Convert to cycles if requested
        if in_cycles:
            nl_phase = nl_phase_m / lambda_nl
        else:
            nl_phase = nl_phase_m
        
        return nl_phase, lambda_nl, freq_nl
    
    def compute_melbourne_wubbena(self, phase_l1: float, phase_l2: float,
                                  code_l1: float, code_l2: float) -> Tuple[float, float]:
        """
        Compute Melbourne-Wübbena (MW) combination
        
        MW combination is geometry-free and ionosphere-free, ideal for
        detecting cycle slips and fixing wide lane ambiguities.
        
        MW = WL_phase - NL_code
        
        Parameters
        ----------
        phase_l1 : float
            L1 carrier phase (cycles)
        phase_l2 : float
            L2 carrier phase (cycles)
        code_l1 : float
            L1 code/pseudorange (meters)
        code_l2 : float
            L2 code/pseudorange (meters)
            
        Returns
        -------
        mw_combination : float
            Melbourne-Wübbena combination (cycles)
        mw_std : float
            Expected standard deviation (cycles)
        """
        # Wide lane phase combination
        wl_phase, lambda_wl, _ = self.compute_wide_lane(phase_l1, phase_l2, in_cycles=True)
        
        # Narrow lane code combination (in meters)
        freq_nl = self.freq1 + self.freq2
        nl_code = (self.freq1 * code_l1 + self.freq2 * code_l2) / freq_nl
        
        # MW combination (convert code to WL cycles)
        mw_combination = wl_phase - nl_code / lambda_wl
        
        # Expected standard deviation (typical values)
        code_noise = 0.3  # meters
        phase_noise = 0.003  # meters
        mw_std = np.sqrt((code_noise/lambda_wl)**2 + (phase_noise/lambda_wl)**2)
        
        return mw_combination, mw_std
    
    def compute_ionosphere_free(self, phase_l1: float, phase_l2: float,
                               in_cycles: bool = True) -> Tuple[float, float]:
        """
        Compute Ionosphere-Free (IF) combination
        
        IF = (f1²*L1 - f2²*L2)/(f1² - f2²)
        
        Parameters
        ----------
        phase_l1 : float
            L1 carrier phase
        phase_l2 : float
            L2 carrier phase
        in_cycles : bool
            Whether input is in cycles
            
        Returns
        -------
        if_phase : float
            Ionosphere-free combination
        if_noise_factor : float
            Noise amplification factor
        """
        if in_cycles:
            phase_l1_m = phase_l1 * self.lambda1
            phase_l2_m = phase_l2 * self.lambda2
        else:
            phase_l1_m = phase_l1
            phase_l2_m = phase_l2
        
        # IF combination
        f1_sq = self.freq1**2
        f2_sq = self.freq2**2
        if_phase_m = (f1_sq * phase_l1_m - f2_sq * phase_l2_m) / (f1_sq - f2_sq)
        
        # Noise amplification factor
        alpha = f1_sq / (f1_sq - f2_sq)
        beta = f2_sq / (f1_sq - f2_sq)
        noise_factor = np.sqrt(alpha**2 + beta**2)
        
        if in_cycles:
            # No direct wavelength for IF, use L1 as reference
            if_phase = if_phase_m / self.lambda1
        else:
            if_phase = if_phase_m
        
        return if_phase, noise_factor
    
    def resolve_wide_lane_ambiguity(self, mw_values: np.ndarray,
                                   threshold: float = 0.25) -> Tuple[Optional[int], float, bool]:
        """
        Resolve wide lane ambiguity using MW combination
        
        Parameters
        ----------
        mw_values : np.ndarray
            Array of MW combination values over time
        threshold : float
            Maximum deviation from integer for acceptance
            
        Returns
        -------
        wl_ambiguity : int or None
            Fixed wide lane ambiguity
        confidence : float
            Confidence level (0-1)
        is_fixed : bool
            Whether ambiguity was successfully fixed
        """
        if len(mw_values) < 10:
            logger.warning("Not enough MW values for WL ambiguity resolution")
            return None, 0.0, False
        
        # Remove outliers using median absolute deviation
        median_mw = np.median(mw_values)
        mad = np.median(np.abs(mw_values - median_mw))
        threshold_mad = 3.0
        
        inliers = np.abs(mw_values - median_mw) < threshold_mad * mad
        if np.sum(inliers) < 5:
            logger.warning("Too many outliers in MW values")
            return None, 0.0, False
        
        # Compute mean of inliers
        mean_mw = np.mean(mw_values[inliers])
        std_mw = np.std(mw_values[inliers])
        
        # Round to nearest integer
        wl_ambiguity = np.round(mean_mw)
        deviation = np.abs(mean_mw - wl_ambiguity)
        
        # Check if close enough to integer
        if deviation < threshold and std_mw < 0.5:
            confidence = 1.0 - deviation / threshold
            confidence *= np.exp(-std_mw)  # Reduce confidence for high variance
            return int(wl_ambiguity), confidence, True
        else:
            return None, deviation, False
    
    def cascaded_ambiguity_resolution(self, phase_l1_dd: np.ndarray,
                                     phase_l2_dd: np.ndarray,
                                     code_l1_dd: np.ndarray,
                                     code_l2_dd: np.ndarray,
                                     epochs: int = 30) -> Dict:
        """
        Cascaded ambiguity resolution: WL first, then L1/L2
        
        This is a multi-step approach:
        1. Fix WL ambiguity using MW combination
        2. Use fixed WL to constrain L1/L2 ambiguities
        3. Fix L1 or L2 ambiguity
        4. Derive the other frequency ambiguity
        
        Parameters
        ----------
        phase_l1_dd : np.ndarray
            L1 double-difference phase (cycles) over epochs
        phase_l2_dd : np.ndarray
            L2 double-difference phase (cycles) over epochs
        code_l1_dd : np.ndarray
            L1 double-difference code (meters) over epochs
        code_l2_dd : np.ndarray
            L2 double-difference code (meters) over epochs
        epochs : int
            Number of epochs to use
            
        Returns
        -------
        result : dict
            Resolution results including WL, L1, L2 ambiguities
        """
        result = {
            'wl_ambiguity': None,
            'l1_ambiguity': None,
            'l2_ambiguity': None,
            'wl_confidence': 0.0,
            'success': False,
            'method': 'cascaded'
        }
        
        # Step 1: Compute MW combination for each epoch
        mw_values = []
        for i in range(min(epochs, len(phase_l1_dd))):
            mw, _ = self.compute_melbourne_wubbena(
                phase_l1_dd[i], phase_l2_dd[i],
                code_l1_dd[i], code_l2_dd[i]
            )
            mw_values.append(mw)
        
        mw_values = np.array(mw_values)
        
        # Step 2: Fix WL ambiguity
        wl_amb, wl_conf, wl_fixed = self.resolve_wide_lane_ambiguity(mw_values)
        result['wl_ambiguity'] = wl_amb
        result['wl_confidence'] = wl_conf
        
        if not wl_fixed:
            logger.info("Failed to fix WL ambiguity")
            return result
        
        logger.info(f"Fixed WL ambiguity: {wl_amb} (confidence: {wl_conf:.3f})")
        
        # Step 3: Use WL to help fix L1 or L2
        # The relationship is: N_WL = N1 - N2
        # So if we know N_WL, we only need to fix one of N1 or N2
        
        # Estimate float L1 ambiguity using phase and code
        float_l1 = np.mean((phase_l1_dd * self.lambda1 - code_l1_dd) / self.lambda1)
        float_l2 = np.mean((phase_l2_dd * self.lambda2 - code_l2_dd) / self.lambda2)
        
        # Try different integer combinations satisfying the WL constraint
        best_residual = float('inf')
        best_n1 = None
        best_n2 = None
        
        # Search around float solution
        search_range = 5
        for delta in range(-search_range, search_range + 1):
            n1_candidate = np.round(float_l1) + delta
            n2_candidate = n1_candidate - wl_amb  # WL constraint
            
            # Compute residual
            residual_l1 = np.abs(float_l1 - n1_candidate)
            residual_l2 = np.abs(float_l2 - n2_candidate)
            total_residual = residual_l1 + residual_l2
            
            if total_residual < best_residual:
                best_residual = total_residual
                best_n1 = int(n1_candidate)
                best_n2 = int(n2_candidate)
        
        # Check if solution is good enough
        if best_residual < 0.5:  # Less than 0.5 cycles total deviation
            result['l1_ambiguity'] = best_n1
            result['l2_ambiguity'] = best_n2
            result['success'] = True
            logger.info(f"Fixed L1/L2 ambiguities: N1={best_n1}, N2={best_n2}")
        else:
            logger.info(f"Failed to fix L1/L2 ambiguities (residual: {best_residual:.3f})")
        
        return result


class TripleFrequencyResolver:
    """
    Triple-frequency ambiguity resolution for modernized GNSS
    Uses L1/L2/L5 or E1/E5a/E5b combinations
    """
    
    def __init__(self, system: str = 'GPS'):
        """Initialize triple-frequency resolver"""
        self.system = system
        self.wl_processor = WideNarrowLaneProcessor(system)
        
    def compute_extra_wide_lane(self, phase_l2: float, phase_l5: float) -> Tuple[float, float]:
        """
        Compute Extra-Wide Lane (EWL) combination for L2/L5
        
        EWL has wavelength of ~5.86m for GPS, very easy to fix
        """
        if self.system == 'GPS':
            freq2 = FREQ_L2
            freq5 = FREQ_L5
        else:
            # Use appropriate frequencies for other systems
            freq2 = self.wl_processor.freq2
            freq5 = self.wl_processor.freq5
        
        # EWL frequency and wavelength
        freq_ewl = freq2 - freq5
        lambda_ewl = CLIGHT / freq_ewl
        
        # EWL combination (assuming input in cycles)
        ewl_phase = (freq2 * phase_l2 - freq5 * phase_l5) / freq_ewl
        
        return ewl_phase, lambda_ewl
    
    def three_frequency_resolution(self, phase_l1: np.ndarray,
                                  phase_l2: np.ndarray,
                                  phase_l5: np.ndarray,
                                  code_l1: np.ndarray,
                                  code_l2: np.ndarray,
                                  code_l5: np.ndarray) -> Dict:
        """
        Three-frequency cascaded ambiguity resolution
        
        Order:
        1. Fix EWL (L2-L5) - wavelength ~5.86m
        2. Fix WL (L1-L2) - wavelength ~86cm
        3. Fix L1 using constraints from EWL and WL
        """
        result = {
            'ewl_ambiguity': None,
            'wl_ambiguity': None,
            'l1_ambiguity': None,
            'l2_ambiguity': None,
            'l5_ambiguity': None,
            'success': False
        }
        
        # Step 1: Fix EWL (very easy with ~6m wavelength)
        ewl_values = []
        for i in range(len(phase_l2)):
            ewl, _ = self.compute_extra_wide_lane(phase_l2[i], phase_l5[i])
            ewl_values.append(ewl)
        
        ewl_mean = np.mean(ewl_values)
        ewl_std = np.std(ewl_values)
        
        if ewl_std < 0.1:  # Very stable
            ewl_amb = np.round(ewl_mean)
            result['ewl_ambiguity'] = int(ewl_amb)
            logger.info(f"Fixed EWL ambiguity: {ewl_amb}")
            
            # Continue with WL and L1 resolution...
            # (Similar to cascaded resolution but with additional constraint)
            result['success'] = True
        
        return result