#!/usr/bin/env python
# Copyright 2024 pyins
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
Phase bias tracker for RTK positioning (RTKLIB-style)

Tracks single-differenced phase biases to handle the large cumulative
phase count that accumulates since GPS system start.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from ..core.constants import CLIGHT, sat2sys


class PhaseBiasTracker:
    """
    Track phase biases for each satellite and frequency
    
    This implements RTKLIB's approach where phase bias is:
    - Estimated as: bias = phase - pseudorange * freq / c
    - Stored as single-differenced values
    - Updated via Kalman filter or least squares
    
    The key insight is that DD biases between satellites of the same system
    should be relatively consistent, so we normalize biases per system.
    """
    
    def __init__(self):
        """Initialize bias tracker"""
        # Store biases as dict[freq_idx][sat] = bias_cycles
        self.biases = {}
        
        # Store bias variances for quality tracking
        self.variances = {}
        
        # Track which satellites have been initialized
        self.initialized = {}
        
        # Reference bias per system to reduce inter-system differences
        self.system_ref_bias = {}
        
    def estimate_initial_bias(self, rover_obs, base_obs, freq_idx=0):
        """
        Estimate initial single-differenced phase bias with system normalization
        
        This is equivalent to RTKLIB's initial bias estimation but with
        normalization per GNSS system to reduce variance.
        
        Parameters:
        -----------
        rover_obs : list
            Rover observations
        base_obs : list
            Base observations
        freq_idx : int
            Frequency index (0=L1, 1=L2)
            
        Returns:
        --------
        biases : dict
            Phase bias for each satellite in cycles
        """
        if freq_idx not in self.biases:
            self.biases[freq_idx] = {}
            self.variances[freq_idx] = {}
            self.initialized[freq_idx] = set()
            self.system_ref_bias[freq_idx] = {}
        
        # First pass: compute raw biases per satellite
        raw_biases = {}
        system_biases = {}  # Group by system
        
        for rov_obs in rover_obs:
            # Find matching base observation
            base_match = None
            for base_obs_item in base_obs:
                if base_obs_item.sat == rov_obs.sat:
                    base_match = base_obs_item
                    break
            
            if base_match is None:
                continue
            
            # Get phase and code observations
            if (freq_idx < len(rov_obs.L) and freq_idx < len(base_match.L) and
                freq_idx < len(rov_obs.P) and freq_idx < len(base_match.P)):
                
                rov_phase = rov_obs.L[freq_idx]  # cycles
                base_phase = base_match.L[freq_idx]  # cycles
                rov_code = rov_obs.P[freq_idx]  # meters
                base_code = base_match.P[freq_idx]  # meters
                
                if rov_phase == 0 or base_phase == 0 or rov_code == 0 or base_code == 0:
                    continue
                
                # Single-differenced observations
                sd_phase = rov_phase - base_phase  # cycles
                sd_code = rov_code - base_code  # meters
                
                # Get wavelength for this satellite and frequency
                from .carrier_phase_dd import CarrierPhaseDD
                cp_processor = CarrierPhaseDD()
                wavelength = cp_processor.get_wavelength(rov_obs.sat, freq_idx)
                
                # Convert code to cycles
                sd_code_cycles = sd_code / wavelength
                
                # Estimate bias (RTKLIB formula)
                bias = sd_phase - sd_code_cycles
                
                raw_biases[rov_obs.sat] = bias
                
                # Group by system
                sys_id = sat2sys(rov_obs.sat)
                if sys_id not in system_biases:
                    system_biases[sys_id] = []
                system_biases[sys_id].append(bias)
        
        # Second pass: normalize biases per system
        # Use median of each system as reference to reduce outlier effect
        for sys_id, biases_list in system_biases.items():
            if biases_list:
                # Use median as reference (more robust than mean)
                self.system_ref_bias[freq_idx][sys_id] = np.median(biases_list)
        
        # Third pass: store normalized biases
        normalized_biases = {}
        for sat, bias in raw_biases.items():
            sys_id = sat2sys(sat)
            
            # Normalize by system reference
            if sys_id in self.system_ref_bias[freq_idx]:
                # Store the difference from system median
                # This reduces the variance significantly
                normalized_bias = bias - self.system_ref_bias[freq_idx][sys_id]
            else:
                normalized_bias = bias
            
            normalized_biases[sat] = normalized_bias
            
            # Initialize or update stored bias
            if sat not in self.initialized[freq_idx]:
                self.biases[freq_idx][sat] = normalized_bias
                self.variances[freq_idx][sat] = 1.0  # Reduced initial variance
                self.initialized[freq_idx].add(sat)
            else:
                # Update with weighted average
                self.update_bias(sat, freq_idx, normalized_bias, 1.0)
        
        return normalized_biases
    
    def get_dd_bias(self, ref_sat, other_sat, freq_idx=0):
        """
        Get double-differenced bias with system normalization
        
        Parameters:
        -----------
        ref_sat : int
            Reference satellite ID
        other_sat : int
            Other satellite ID
        freq_idx : int
            Frequency index
            
        Returns:
        --------
        dd_bias : float
            Double-differenced bias in cycles
        """
        if freq_idx not in self.biases:
            return 0.0
        
        # Get normalized biases
        ref_bias = self.biases[freq_idx].get(ref_sat, 0.0)
        other_bias = self.biases[freq_idx].get(other_sat, 0.0)
        
        # Get system references
        ref_sys = sat2sys(ref_sat)
        other_sys = sat2sys(other_sat)
        
        # Add back system references for DD
        if freq_idx in self.system_ref_bias:
            if ref_sys in self.system_ref_bias[freq_idx]:
                ref_bias += self.system_ref_bias[freq_idx][ref_sys]
            if other_sys in self.system_ref_bias[freq_idx]:
                other_bias += self.system_ref_bias[freq_idx][other_sys]
        
        return other_bias - ref_bias
    
    def apply_dd_bias_correction(self, dd_phase_cycles, sat_pairs, freq_idx=0):
        """
        Apply bias correction to DD measurements
        
        Parameters:
        -----------
        dd_phase_cycles : np.ndarray
            Original DD phase measurements in cycles
        sat_pairs : list
            List of (ref_sat, other_sat) tuples
        freq_idx : int
            Frequency index
            
        Returns:
        --------
        corrected_dd : np.ndarray
            Bias-corrected DD phase in cycles
        """
        corrected_dd = np.zeros_like(dd_phase_cycles)
        
        for i, (ref_sat, other_sat) in enumerate(sat_pairs):
            dd_bias = self.get_dd_bias(ref_sat, other_sat, freq_idx)
            corrected_dd[i] = dd_phase_cycles[i] - dd_bias
        
        return corrected_dd
    
    def update_bias(self, sat, freq_idx, new_bias, variance=1.0):
        """
        Update bias estimate using weighted average
        
        Parameters:
        -----------
        sat : int
            Satellite ID
        freq_idx : int
            Frequency index
        new_bias : float
            New bias estimate (normalized)
        variance : float
            Variance of new estimate
        """
        if freq_idx not in self.biases:
            self.biases[freq_idx] = {}
            self.variances[freq_idx] = {}
        
        if sat in self.biases[freq_idx]:
            # Simple weighted average update
            old_bias = self.biases[freq_idx][sat]
            old_var = self.variances[freq_idx][sat]
            
            # Combine estimates
            w1 = 1.0 / old_var
            w2 = 1.0 / variance
            
            self.biases[freq_idx][sat] = (w1 * old_bias + w2 * new_bias) / (w1 + w2)
            self.variances[freq_idx][sat] = 1.0 / (w1 + w2)
        else:
            # First estimate
            self.biases[freq_idx][sat] = new_bias
            self.variances[freq_idx][sat] = variance
    
    def reset_satellite(self, sat, freq_idx=None):
        """
        Reset bias for a satellite (e.g., after cycle slip)
        
        Parameters:
        -----------
        sat : int
            Satellite ID
        freq_idx : int or None
            Frequency index (None = all frequencies)
        """
        if freq_idx is None:
            # Reset all frequencies
            for f in self.biases:
                if sat in self.biases[f]:
                    del self.biases[f][sat]
                if sat in self.variances[f]:
                    del self.variances[f][sat]
                if sat in self.initialized[f]:
                    self.initialized[f].remove(sat)
        else:
            # Reset specific frequency
            if freq_idx in self.biases and sat in self.biases[freq_idx]:
                del self.biases[freq_idx][sat]
            if freq_idx in self.variances and sat in self.variances[freq_idx]:
                del self.variances[freq_idx][sat]
            if freq_idx in self.initialized and sat in self.initialized[freq_idx]:
                self.initialized[freq_idx].remove(sat)
    
    def get_bias_stats(self, freq_idx=0):
        """
        Get statistics about current biases
        
        Parameters:
        -----------
        freq_idx : int
            Frequency index
            
        Returns:
        --------
        stats : dict
            Bias statistics
        """
        if freq_idx not in self.biases or not self.biases[freq_idx]:
            return {
                'n_satellites': 0,
                'mean_bias': 0.0,
                'std_bias': 0.0,
                'min_bias': 0.0,
                'max_bias': 0.0,
                'mean_system_ref': 0.0
            }
        
        biases = list(self.biases[freq_idx].values())
        
        # Get mean system reference
        mean_sys_ref = 0.0
        if freq_idx in self.system_ref_bias and self.system_ref_bias[freq_idx]:
            mean_sys_ref = np.mean(list(self.system_ref_bias[freq_idx].values()))
        
        return {
            'n_satellites': len(biases),
            'mean_bias': np.mean(biases),
            'std_bias': np.std(biases),
            'min_bias': np.min(biases),
            'max_bias': np.max(biases),
            'mean_system_ref': mean_sys_ref
        }
    
    def compute_corrected_float_ambiguities(self, dd_phase_cycles, dd_range_m, 
                                           wavelengths, sat_pairs, freq_idx=0):
        """
        Compute float ambiguities with bias correction applied
        
        Parameters:
        -----------
        dd_phase_cycles : np.ndarray
            DD carrier phase in cycles
        dd_range_m : np.ndarray
            DD geometric range in meters
        wavelengths : np.ndarray
            Carrier wavelengths in meters
        sat_pairs : list
            List of (ref_sat, other_sat) tuples
        freq_idx : int
            Frequency index
            
        Returns:
        --------
        float_ambiguities : np.ndarray
            Bias-corrected float ambiguity estimates in cycles
        """
        # Apply bias correction
        corrected_dd = self.apply_dd_bias_correction(dd_phase_cycles, sat_pairs, freq_idx)
        
        # Convert range to cycles
        dd_range_cycles = dd_range_m / wavelengths
        
        # Float ambiguities
        float_ambiguities = corrected_dd - dd_range_cycles
        
        return float_ambiguities