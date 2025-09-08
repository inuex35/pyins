#!/usr/bin/env python
"""
Phase-Code Coherency Check for RTK
====================================

Implements RTKLIB-style phase-code coherency checking to ensure
consistency between carrier phase and pseudorange measurements.

The phase bias is estimated as:
    bias = (L - P * f / c)
where:
    L = carrier phase (cycles)
    P = pseudorange (meters)
    f = carrier frequency (Hz)
    c = speed of light (m/s)

References:
    RTKLIB rtkpos.c: udbias() function
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

from ..core.constants import CLIGHT, FREQ_L1, FREQ_L2, FREQ_L5
from ..core.constants import FREQ_E1, FREQ_E5a, FREQ_E5b
from ..core.constants import FREQ_B1I, FREQ_B2a, FREQ_B3
from ..core.stats import ERATIO_L1, DEFAULT_ERATIO

logger = logging.getLogger(__name__)


class PhaseCodeCoherency:
    """Check and maintain phase-code coherency in RTK processing"""
    
    def __init__(self, 
                 threshold: float = 0.3,
                 min_lock: int = 5,
                 enable_ionofree: bool = False,
                 eratio: float = 300.0):
        """
        Initialize phase-code coherency checker
        
        Parameters
        ----------
        threshold : float
            Maximum allowed phase-code difference in cycles (default: 0.3)
        min_lock : int
            Minimum lock count required (default: 5)
        enable_ionofree : bool
            Use ionosphere-free combination (default: False)
        eratio : float
            Code/phase error ratio (default: 300.0, from RTKLIB)
            This means code measurement error is ~300x larger than phase
        """
        self.threshold = threshold
        self.min_lock = min_lock
        self.enable_ionofree = enable_ionofree
        self.eratio = eratio  # Code/phase error ratio
        
        # Store phase biases for each satellite and frequency
        self.phase_biases = {}  # {(sat_id, freq): bias}
        self.lock_counts = {}   # {(sat_id, freq): count}
        
        logger.info(f"Phase-code coherency checker initialized (threshold={threshold} cycles, eratio={eratio})")
    
    def compute_phase_bias(self,
                           phase: float,
                           code: float,
                           frequency: float) -> float:
        """
        Compute phase bias from phase and code observations
        
        Parameters
        ----------
        phase : float
            Carrier phase observation in cycles
        code : float
            Pseudorange observation in meters
        frequency : float
            Carrier frequency in Hz
            
        Returns
        -------
        float
            Phase bias in cycles
        """
        if phase == 0.0 or code == 0.0 or frequency == 0.0:
            return 0.0
        
        # Phase bias = phase - (code * frequency / speed_of_light)
        bias = phase - (code * frequency / CLIGHT)
        
        return bias
    
    def compute_ionofree_bias(self,
                              phase1: float, phase2: float,
                              code1: float, code2: float,
                              freq1: float, freq2: float) -> float:
        """
        Compute ionosphere-free phase bias
        
        Parameters
        ----------
        phase1, phase2 : float
            Carrier phase observations for two frequencies (cycles)
        code1, code2 : float
            Pseudorange observations for two frequencies (meters)
        freq1, freq2 : float
            Carrier frequencies (Hz)
            
        Returns
        -------
        float
            Ionosphere-free phase bias in meters
        """
        if freq1 <= 0.0 or freq2 <= 0.0:
            return 0.0
        
        # Ionosphere-free combination coefficients
        C1 = freq1**2 / (freq1**2 - freq2**2)
        C2 = -freq2**2 / (freq1**2 - freq2**2)
        
        # Ionosphere-free phase in meters
        phase_if = C1 * phase1 * CLIGHT / freq1 + C2 * phase2 * CLIGHT / freq2
        
        # Ionosphere-free code in meters
        code_if = C1 * code1 + C2 * code2
        
        # Bias in meters
        bias = phase_if - code_if
        
        return bias
    
    def check_coherency(self,
                       sat_id: str,
                       freq_idx: int,
                       phase: float,
                       code: float,
                       frequency: float,
                       lock_count: int = 0) -> Tuple[bool, float, float]:
        """
        Check phase-code coherency for a satellite
        
        Parameters
        ----------
        sat_id : str
            Satellite ID (e.g., 'G01')
        freq_idx : int
            Frequency index (0=L1, 1=L2, etc.)
        phase : float
            Carrier phase observation (cycles)
        code : float
            Pseudorange observation (meters)
        frequency : float
            Carrier frequency (Hz)
        lock_count : int
            Current lock count
            
        Returns
        -------
        Tuple[bool, float, float]
            (is_coherent, phase_bias, residual)
        """
        key = (sat_id, freq_idx)
        
        # Compute current phase bias
        current_bias = self.compute_phase_bias(phase, code, frequency)
        
        if current_bias == 0.0:
            return False, 0.0, 0.0
        
        # Check if we have stored bias
        if key in self.phase_biases:
            stored_bias = self.phase_biases[key]
            
            # Compute residual (difference from stored bias)
            residual = current_bias - stored_bias
            
            # Check if coherent (within threshold)
            is_coherent = abs(residual) < self.threshold
            
            if is_coherent:
                # Update lock count
                if key in self.lock_counts:
                    self.lock_counts[key] += 1
                else:
                    self.lock_counts[key] = 1
            else:
                # Reset on loss of coherency
                logger.debug(f"Phase-code coherency lost for {sat_id} F{freq_idx+1}: residual={residual:.3f}")
                self.phase_biases[key] = current_bias
                self.lock_counts[key] = 0
                
            return is_coherent, stored_bias, residual
            
        else:
            # Initialize bias
            self.phase_biases[key] = current_bias
            self.lock_counts[key] = 0
            logger.debug(f"Initialize phase bias for {sat_id} F{freq_idx+1}: {current_bias:.3f}")
            return False, current_bias, 0.0
    
    def compute_dd_phase_bias(self,
                             base_phase: Dict[str, float],
                             base_code: Dict[str, float],
                             rover_phase: Dict[str, float],
                             rover_code: Dict[str, float],
                             frequencies: Dict[str, float],
                             ref_sat: str) -> Dict[str, float]:
        """
        Compute double-difference phase biases
        
        Parameters
        ----------
        base_phase : Dict[str, float]
            Base station carrier phase observations {sat_id: phase}
        base_code : Dict[str, float]
            Base station pseudorange observations {sat_id: code}
        rover_phase : Dict[str, float]
            Rover carrier phase observations
        rover_code : Dict[str, float]
            Rover pseudorange observations
        frequencies : Dict[str, float]
            Carrier frequencies for each satellite
        ref_sat : str
            Reference satellite ID
            
        Returns
        -------
        Dict[str, float]
            Double-difference phase biases {sat_id: dd_bias}
        """
        dd_biases = {}
        
        if ref_sat not in base_phase or ref_sat not in rover_phase:
            return dd_biases
        
        # Reference satellite single differences
        ref_sd_phase = rover_phase[ref_sat] - base_phase[ref_sat]
        ref_sd_code = rover_code[ref_sat] - base_code[ref_sat]
        ref_freq = frequencies[ref_sat]
        
        # Reference bias
        ref_bias = self.compute_phase_bias(ref_sd_phase, ref_sd_code, ref_freq)
        
        # Compute DD biases for other satellites
        for sat_id in base_phase:
            if sat_id == ref_sat or sat_id not in rover_phase:
                continue
            
            # Single differences
            sd_phase = rover_phase[sat_id] - base_phase[sat_id]
            sd_code = rover_code[sat_id] - base_code[sat_id]
            freq = frequencies[sat_id]
            
            # Single difference bias
            sd_bias = self.compute_phase_bias(sd_phase, sd_code, freq)
            
            # Double difference bias
            dd_biases[sat_id] = sd_bias - ref_bias
        
        return dd_biases
    
    def correct_phase_offset(self,
                            phase_states: np.ndarray,
                            new_biases: np.ndarray,
                            sat_indices: List[int]) -> np.ndarray:
        """
        Correct phase bias offset to ensure coherency (RTKLIB-style)
        
        Parameters
        ----------
        phase_states : np.ndarray
            Current phase bias states
        new_biases : np.ndarray
            Newly computed biases
        sat_indices : List[int]
            Satellite indices in state vector
            
        Returns
        -------
        np.ndarray
            Corrected phase states
        """
        # Compute average offset for initialized biases
        offset = 0.0
        count = 0
        
        for i, sat_idx in enumerate(sat_indices):
            if phase_states[sat_idx] != 0.0:
                offset += new_biases[i] - phase_states[sat_idx]
                count += 1
        
        # Apply average offset correction
        if count > 0:
            avg_offset = offset / count
            for sat_idx in sat_indices:
                if phase_states[sat_idx] != 0.0:
                    phase_states[sat_idx] += avg_offset
            
            logger.debug(f"Applied phase offset correction: {avg_offset:.3f} cycles")
        
        return phase_states
    
    def get_coherent_satellites(self, min_lock: Optional[int] = None) -> List[Tuple[str, int]]:
        """
        Get list of satellites with coherent phase-code observations
        
        Parameters
        ----------
        min_lock : int, optional
            Minimum lock count required (uses self.min_lock if None)
            
        Returns
        -------
        List[Tuple[str, int]]
            List of (sat_id, freq_idx) tuples with coherent observations
        """
        if min_lock is None:
            min_lock = self.min_lock
        
        coherent = []
        for key, count in self.lock_counts.items():
            if count >= min_lock:
                coherent.append(key)
        
        return coherent
    
    def reset_satellite(self, sat_id: str, freq_idx: Optional[int] = None):
        """
        Reset phase bias for a satellite
        
        Parameters
        ----------
        sat_id : str
            Satellite ID
        freq_idx : int, optional
            Frequency index (reset all if None)
        """
        if freq_idx is not None:
            key = (sat_id, freq_idx)
            if key in self.phase_biases:
                del self.phase_biases[key]
            if key in self.lock_counts:
                del self.lock_counts[key]
        else:
            # Reset all frequencies for this satellite
            keys_to_remove = [k for k in self.phase_biases if k[0] == sat_id]
            for key in keys_to_remove:
                del self.phase_biases[key]
                if key in self.lock_counts:
                    del self.lock_counts[key]
    
    def compute_measurement_weight(self, is_phase: bool, frequency_idx: int = 0) -> float:
        """
        Compute measurement weight based on eratio (RTKLIB-style)
        
        Parameters
        ----------
        is_phase : bool
            True for carrier phase, False for pseudorange
        frequency_idx : int
            Frequency index (0=L1, 1=L2, etc.)
            
        Returns
        -------
        float
            Weight factor for measurement variance
        """
        if not is_phase:
            # Pseudorange: multiply variance by eratio
            return self.eratio
        else:
            # Carrier phase: baseline weight is 1.0
            # Can adjust for different frequencies
            return 1.0
    
    def validate_phase_code_ratio(self, 
                                  phase_obs: np.ndarray,
                                  code_obs: np.ndarray,
                                  frequency: float) -> Tuple[bool, float]:
        """
        Validate phase/code measurements using error ratio
        
        Parameters
        ----------
        phase_obs : np.ndarray
            Carrier phase observations (cycles)
        code_obs : np.ndarray
            Pseudorange observations (meters)
        frequency : float
            Carrier frequency (Hz)
            
        Returns
        -------
        Tuple[bool, float]
            (is_valid, ratio_metric)
        """
        if len(phase_obs) == 0 or len(code_obs) == 0:
            return False, 0.0
        
        # Convert phase to meters for comparison
        wavelength = CLIGHT / frequency
        phase_meters = phase_obs * wavelength
        
        # Compute differences
        diff = phase_meters - code_obs
        
        # Expected std dev ratio based on eratio
        # Code std ~ eratio * phase std
        expected_code_std = np.std(phase_meters) * self.eratio
        actual_code_std = np.std(code_obs)
        
        # Check if ratio is reasonable
        if expected_code_std > 0:
            ratio_metric = actual_code_std / expected_code_std
            # Valid if ratio is between 0.1 and 10 (order of magnitude check)
            is_valid = 0.1 < ratio_metric < 10.0
        else:
            ratio_metric = 0.0
            is_valid = False
        
        logger.debug(f"Phase/code ratio validation: metric={ratio_metric:.2f}, valid={is_valid}")
        
        return is_valid, ratio_metric
    
    def get_statistics(self) -> Dict:
        """
        Get coherency statistics
        
        Returns
        -------
        Dict
            Statistics including number of coherent satellites, average lock counts, etc.
        """
        coherent_sats = self.get_coherent_satellites()
        
        stats = {
            'n_tracked': len(self.phase_biases),
            'n_coherent': len(coherent_sats),
            'avg_lock_count': np.mean(list(self.lock_counts.values())) if self.lock_counts else 0,
            'max_lock_count': max(self.lock_counts.values()) if self.lock_counts else 0,
            'coherent_satellites': coherent_sats
        }
        
        return stats