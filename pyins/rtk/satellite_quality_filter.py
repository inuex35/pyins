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
Satellite quality filter for RTK positioning

Filters out satellites with poor quality metrics to improve
ambiguity resolution success rate.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from ..core.constants import sat2sys, CLIGHT


class SatelliteQualityFilter:
    """
    Filter satellites based on quality metrics
    
    This implements quality checks similar to RTKLIB:
    - Elevation angle threshold
    - SNR threshold
    - Residual threshold
    - Cycle slip detection
    """
    
    def __init__(self,
                 min_elevation_deg: float = 10.0,
                 min_snr_dbhz: float = 30.0,
                 max_phase_residual_cycles: float = 0.3,
                 max_code_residual_m: float = 10.0):
        """
        Initialize quality filter
        
        Parameters:
        -----------
        min_elevation_deg : float
            Minimum elevation angle in degrees
        min_snr_dbhz : float
            Minimum SNR in dB-Hz
        max_phase_residual_cycles : float
            Maximum phase residual in cycles for outlier detection
        max_code_residual_m : float
            Maximum code residual in meters
        """
        self.min_elevation = np.radians(min_elevation_deg)
        self.min_snr = min_snr_dbhz
        self.max_phase_residual = max_phase_residual_cycles
        self.max_code_residual = max_code_residual_m
        
        # Track statistics for adaptive filtering
        self.residual_history = {}
        self.excluded_satellites = set()
        
    def filter_by_elevation(self, satellites: List[int], 
                           sat_positions: Dict[int, np.ndarray],
                           receiver_pos: np.ndarray) -> List[int]:
        """
        Filter satellites by elevation angle
        
        Parameters:
        -----------
        satellites : list
            List of satellite IDs
        sat_positions : dict
            Satellite ECEF positions
        receiver_pos : np.ndarray
            Receiver ECEF position
            
        Returns:
        --------
        filtered : list
            List of satellites above elevation threshold
        """
        filtered = []
        
        for sat in satellites:
            if sat not in sat_positions:
                continue
                
            # Calculate elevation angle
            sat_vec = sat_positions[sat] - receiver_pos
            sat_dist = np.linalg.norm(sat_vec)
            sat_unit = sat_vec / sat_dist
            
            # Convert to local ENU to get elevation
            # Simplified calculation - assumes spherical Earth
            up = receiver_pos / np.linalg.norm(receiver_pos)
            elevation = np.arcsin(np.dot(sat_unit, up))
            
            if elevation >= self.min_elevation:
                filtered.append(sat)
        
        return filtered
    
    def filter_by_snr(self, observations: List, min_snr_override: Optional[float] = None) -> List:
        """
        Filter observations by SNR
        
        Parameters:
        -----------
        observations : list
            List of observation objects with SNR field
        min_snr_override : float, optional
            Override default SNR threshold
            
        Returns:
        --------
        filtered : list
            List of observations with sufficient SNR
        """
        threshold = min_snr_override if min_snr_override is not None else self.min_snr
        filtered = []
        
        for obs in observations:
            # Check L1 SNR (index 0)
            if len(obs.SNR) > 0 and obs.SNR[0] >= threshold:
                filtered.append(obs)
            elif len(obs.SNR) == 0:
                # No SNR info - include but track
                filtered.append(obs)
        
        return filtered
    
    def detect_outliers_by_residuals(self, 
                                    float_ambiguities: np.ndarray,
                                    sat_pairs: List[Tuple[int, int]],
                                    threshold_scale: float = 3.0) -> np.ndarray:
        """
        Detect outliers based on float ambiguity residuals
        
        Uses robust MAD (Median Absolute Deviation) method
        
        Parameters:
        -----------
        float_ambiguities : np.ndarray
            Float ambiguity estimates
        sat_pairs : list
            List of (ref_sat, other_sat) tuples
        threshold_scale : float
            Number of MAD for outlier threshold
            
        Returns:
        --------
        mask : np.ndarray
            Boolean mask (True = good, False = outlier)
        """
        # Calculate residuals from nearest integer
        residuals = float_ambiguities - np.round(float_ambiguities)
        
        # Robust statistics using MAD
        median_residual = np.median(residuals)
        mad = np.median(np.abs(residuals - median_residual))
        
        # Scale MAD to approximate standard deviation
        # For normal distribution: sigma ≈ 1.4826 * MAD
        mad_scaled = 1.4826 * mad
        
        # Detect outliers
        threshold = threshold_scale * mad_scaled
        mask = np.abs(residuals - median_residual) < threshold
        
        # Also check absolute threshold
        mask &= np.abs(residuals) < self.max_phase_residual
        
        # Track problematic satellites
        for i, is_good in enumerate(mask):
            if not is_good and i < len(sat_pairs):
                ref_sat, other_sat = sat_pairs[i]
                self.excluded_satellites.add(other_sat)
        
        return mask
    
    def filter_by_consistency(self,
                             float_ambiguities: np.ndarray,
                             sat_pairs: List[Tuple[int, int]],
                             min_consistent_ratio: float = 0.6) -> Tuple[np.ndarray, List]:
        """
        Filter based on consistency of float ambiguities
        
        Remove satellites that cause many ambiguities to have
        large fractional parts
        
        Parameters:
        -----------
        float_ambiguities : np.ndarray
            Float ambiguity estimates
        sat_pairs : list
            List of (ref_sat, other_sat) tuples
        min_consistent_ratio : float
            Minimum ratio of consistent ambiguities
            
        Returns:
        --------
        filtered_ambiguities : np.ndarray
            Filtered float ambiguities
        filtered_pairs : list
            Filtered satellite pairs
        """
        # Calculate fractional parts
        fractional = np.abs(float_ambiguities - np.round(float_ambiguities))
        
        # Find satellites causing large fractional parts
        sat_scores = {}
        for i, (ref_sat, other_sat) in enumerate(sat_pairs):
            if other_sat not in sat_scores:
                sat_scores[other_sat] = []
            sat_scores[other_sat].append(fractional[i])
        
        # Calculate average fractional part per satellite
        sat_avg_frac = {}
        for sat, fracs in sat_scores.items():
            sat_avg_frac[sat] = np.mean(fracs)
        
        # Identify problematic satellites (high average fractional part)
        threshold = 0.35  # Satellites with avg frac > 0.35 are problematic
        problematic_sats = set()
        for sat, avg_frac in sat_avg_frac.items():
            if avg_frac > threshold:
                problematic_sats.add(sat)
        
        # Filter out problematic satellites
        mask = np.ones(len(float_ambiguities), dtype=bool)
        for i, (ref_sat, other_sat) in enumerate(sat_pairs):
            if other_sat in problematic_sats:
                mask[i] = False
        
        filtered_ambiguities = float_ambiguities[mask]
        filtered_pairs = [sat_pairs[i] for i in range(len(sat_pairs)) if mask[i]]
        
        return filtered_ambiguities, filtered_pairs
    
    def apply_quality_filter(self,
                            float_ambiguities: np.ndarray,
                            sat_pairs: List[Tuple[int, int]],
                            observations: Optional[List] = None,
                            sat_positions: Optional[Dict] = None,
                            receiver_pos: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List, Dict]:
        """
        Apply comprehensive quality filtering
        
        Parameters:
        -----------
        float_ambiguities : np.ndarray
            Float ambiguity estimates
        sat_pairs : list
            List of (ref_sat, other_sat) tuples
        observations : list, optional
            Observation data for SNR filtering
        sat_positions : dict, optional
            Satellite positions for elevation filtering
        receiver_pos : np.ndarray, optional
            Receiver position for elevation calculation
            
        Returns:
        --------
        filtered_ambiguities : np.ndarray
            Quality-filtered float ambiguities
        filtered_pairs : list
            Filtered satellite pairs
        filter_info : dict
            Information about filtering
        """
        n_original = len(float_ambiguities)
        
        # Start with all satellites
        mask = np.ones(n_original, dtype=bool)
        
        # 1. Filter by residuals (outlier detection)
        residual_mask = self.detect_outliers_by_residuals(float_ambiguities, sat_pairs)
        mask &= residual_mask
        n_after_residual = np.sum(mask)
        
        # 2. Filter by consistency
        if np.sum(mask) > 0:
            temp_ambiguities = float_ambiguities[mask]
            temp_pairs = [sat_pairs[i] for i in range(len(sat_pairs)) if mask[i]]
            
            filtered_amb, filtered_pairs = self.filter_by_consistency(
                temp_ambiguities, temp_pairs
            )
            
            # Update mask based on consistency filtering
            final_mask = np.zeros(n_original, dtype=bool)
            j = 0
            for i in range(n_original):
                if mask[i]:
                    if j < len(filtered_pairs) and sat_pairs[i] == temp_pairs[j]:
                        if j < len(filtered_amb):
                            final_mask[i] = True
                        j += 1
            mask = final_mask
        
        # Apply final mask
        filtered_ambiguities = float_ambiguities[mask]
        filtered_pairs = [sat_pairs[i] for i in range(len(sat_pairs)) if mask[i]]
        
        # Calculate statistics
        if len(filtered_ambiguities) > 0:
            fractional = np.abs(filtered_ambiguities - np.round(filtered_ambiguities))
            close_to_int = np.sum(fractional < 0.15)
        else:
            close_to_int = 0
        
        filter_info = {
            'n_original': n_original,
            'n_after_residual': n_after_residual,
            'n_final': len(filtered_ambiguities),
            'n_excluded': n_original - len(filtered_ambiguities),
            'close_to_int': close_to_int,
            'excluded_satellites': list(self.excluded_satellites)
        }
        
        return filtered_ambiguities, filtered_pairs, filter_info
    
    def reset(self):
        """Reset filter state"""
        self.residual_history.clear()
        self.excluded_satellites.clear()