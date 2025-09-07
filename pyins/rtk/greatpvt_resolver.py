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
GreatPVT-style cascaded multi-frequency ambiguity resolution

Implements the sequential fixing strategy:
1. Extra-Wide-Lane (EWL): L2-L5 or similar (~5.86m wavelength)
2. Wide-Lane (WL): L1-L2 (~86.2cm wavelength)  
3. Narrow-Lane (NL): (L1+L2)/2 (~10.7cm wavelength)

References:
    [1] Li et al. (2010) "Three carrier ambiguity resolution: distance-independent performance"
    [2] Feng (2008) "GNSS three carrier ambiguity resolution using ionosphere-reduced virtual signals"
    [3] GreatPVT implementation: https://github.com/great-pvt
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import logging

from ..core.constants import CLIGHT, FREQ_L1, FREQ_L2, FREQ_L5
from ..core.constants import FREQ_E1, FREQ_E5a, FREQ_E5b
from ..core.constants import FREQ_B1I, FREQ_B2a, FREQ_B3
from ..core.constants import sat2sys, SYS_GPS, SYS_GAL, SYS_BDS

logger = logging.getLogger(__name__)


@dataclass
class CascadedFixResult:
    """Result from cascaded ambiguity resolution"""
    fixed_L1: np.ndarray         # Fixed L1 ambiguities
    fixed_L2: np.ndarray         # Fixed L2 ambiguities
    fixed_L5: Optional[np.ndarray]  # Fixed L5 ambiguities (if available)
    is_fixed: np.ndarray         # Boolean mask of fixed ambiguities
    n_ewl_fixed: int            # Number of EWL fixed
    n_wl_fixed: int             # Number of WL fixed
    n_nl_fixed: int             # Number of NL fixed
    n_total: int                # Total number of ambiguities
    success_rate: float         # Overall success rate
    wl_residuals: np.ndarray    # WL residuals for quality check
    nl_residuals: np.ndarray    # NL residuals for quality check


class GreatPVTResolver:
    """
    GreatPVT-style cascaded multi-frequency ambiguity resolver
    
    Implements sequential fixing: EWL → WL → NL
    """
    
    def __init__(self,
                 wl_threshold: float = 0.25,
                 nl_threshold: float = 0.15,
                 ewl_threshold: float = 0.30,
                 min_satellites: int = 4,
                 enable_partial: bool = True):
        """
        Initialize GreatPVT resolver
        
        Parameters
        ----------
        wl_threshold : float
            Maximum fractional part for WL fixing (cycles)
        nl_threshold : float
            Maximum fractional part for NL fixing (cycles)
        ewl_threshold : float
            Maximum fractional part for EWL fixing (cycles)
        min_satellites : int
            Minimum satellites required for fixing
        enable_partial : bool
            Enable partial ambiguity resolution
        """
        self.wl_threshold = wl_threshold
        self.nl_threshold = nl_threshold
        self.ewl_threshold = ewl_threshold
        self.min_satellites = min_satellites
        self.enable_partial = enable_partial
        
        # Pre-compute wavelengths for different systems
        self._init_wavelengths()
        
        # Storage for fixed ambiguities (for Fix & Hold)
        self.fixed_storage = {}
        
    def _init_wavelengths(self):
        """Initialize wavelengths for different GNSS systems"""
        # GPS wavelengths
        self.wl_gps = {
            'L1': CLIGHT / FREQ_L1,           # ~19.0 cm
            'L2': CLIGHT / FREQ_L2,           # ~24.4 cm
            'L5': CLIGHT / FREQ_L5,           # ~25.5 cm
            'WL_L1L2': CLIGHT / (FREQ_L1 - FREQ_L2),  # ~86.2 cm
            'NL_L1L2': 2 * CLIGHT / (FREQ_L1 + FREQ_L2),  # ~10.7 cm
            'EWL_L2L5': CLIGHT / abs(FREQ_L2 - FREQ_L5) if FREQ_L5 > 0 else 0,  # ~5.86 m
        }
        
        # Galileo wavelengths
        self.wl_gal = {
            'E1': CLIGHT / FREQ_E1,
            'E5a': CLIGHT / FREQ_E5a,
            'E5b': CLIGHT / FREQ_E5b,
            'WL_E1E5a': CLIGHT / abs(FREQ_E1 - FREQ_E5a),
            'WL_E1E5b': CLIGHT / abs(FREQ_E1 - FREQ_E5b),
        }
        
        # BeiDou wavelengths
        self.wl_bds = {
            'B1': CLIGHT / FREQ_B1I,
            'B2': CLIGHT / FREQ_B2a,
            'B3': CLIGHT / FREQ_B3,
            'WL_B1B2': CLIGHT / abs(FREQ_B1I - FREQ_B2a),
            'WL_B1B3': CLIGHT / abs(FREQ_B1I - FREQ_B3),
        }
    
    def resolve(self,
                dd_observations: Dict[int, np.ndarray],
                dd_ranges: np.ndarray,
                sat_pairs: List[Tuple[int, int]],
                phase_biases: Optional[Dict] = None) -> CascadedFixResult:
        """
        Resolve ambiguities using GreatPVT cascaded approach
        
        Parameters
        ----------
        dd_observations : dict
            DD phase observations {freq_idx: cycles_array}
            freq_idx: 0=L1/E1/B1, 1=L2/E5a/B2, 2=L5/E5b/B3
        dd_ranges : np.ndarray
            DD geometric ranges in meters
        sat_pairs : list
            List of (ref_sat, other_sat) tuples
        phase_biases : dict, optional
            Phase bias corrections if available
            
        Returns
        -------
        result : CascadedFixResult
            Cascaded resolution result
        """
        n_dd = len(dd_ranges)
        
        # Check input
        if n_dd < self.min_satellites:
            logger.warning(f"Too few satellites: {n_dd} < {self.min_satellites}")
            return self._empty_result(n_dd)
        
        # Get L1 and L2 observations (required)
        if 0 not in dd_observations or 1 not in dd_observations:
            logger.warning("L1 and L2 observations required for cascaded resolution")
            return self._empty_result(n_dd)
        
        dd_L1 = dd_observations[0]
        dd_L2 = dd_observations[1]
        dd_L5 = dd_observations.get(2, None)  # Optional
        
        # Apply phase bias corrections if available
        if phase_biases:
            dd_L1 = self._apply_bias_correction(dd_L1, sat_pairs, phase_biases, 0)
            dd_L2 = self._apply_bias_correction(dd_L2, sat_pairs, phase_biases, 1)
            if dd_L5 is not None:
                dd_L5 = self._apply_bias_correction(dd_L5, sat_pairs, phase_biases, 2)
        
        # Determine wavelengths based on satellite systems
        wavelengths = self._get_wavelengths(sat_pairs)
        
        # Step 1: Extra-Wide-Lane (if L5 available)
        ewl_fixed = None
        n_ewl_fixed = 0
        if dd_L5 is not None and len(dd_L5) == n_dd:
            ewl_fixed, ewl_is_fixed = self._fix_extra_wide_lane(
                dd_L2, dd_L5, dd_ranges, wavelengths['EWL']
            )
            n_ewl_fixed = np.sum(ewl_is_fixed)
            logger.info(f"EWL fixed: {n_ewl_fixed}/{n_dd}")
        
        # Step 2: Wide-Lane (L1 - L2)
        wl_fixed, wl_is_fixed, wl_residuals = self._fix_wide_lane(
            dd_L1, dd_L2, dd_ranges, wavelengths['WL']
        )
        n_wl_fixed = np.sum(wl_is_fixed)
        logger.info(f"WL fixed: {n_wl_fixed}/{n_dd}")
        
        # Step 3: Narrow-Lane using WL constraint
        fixed_L1, fixed_L2, nl_is_fixed, nl_residuals = self._fix_narrow_lane(
            dd_L1, dd_L2, dd_ranges, wl_fixed, wl_is_fixed,
            wavelengths['L1'], wavelengths['L2']
        )
        n_nl_fixed = np.sum(nl_is_fixed)
        logger.info(f"NL fixed: {n_nl_fixed}/{n_dd}")
        
        # Handle partial resolution
        if self.enable_partial and n_nl_fixed < n_dd:
            fixed_L1, fixed_L2, nl_is_fixed = self._apply_partial_resolution(
                dd_L1, dd_L2, dd_ranges, fixed_L1, fixed_L2, 
                nl_is_fixed, wl_fixed, wl_is_fixed, wavelengths
            )
            n_nl_fixed = np.sum(nl_is_fixed)
            logger.info(f"After partial AR: {n_nl_fixed}/{n_dd}")
        
        # Store fixed ambiguities for Fix & Hold
        self._store_fixed_ambiguities(sat_pairs, fixed_L1, fixed_L2, nl_is_fixed)
        
        # Build result
        result = CascadedFixResult(
            fixed_L1=fixed_L1,
            fixed_L2=fixed_L2,
            fixed_L5=None,  # TODO: implement L5 fixing
            is_fixed=nl_is_fixed,
            n_ewl_fixed=n_ewl_fixed,
            n_wl_fixed=n_wl_fixed,
            n_nl_fixed=n_nl_fixed,
            n_total=n_dd,
            success_rate=n_nl_fixed / n_dd if n_dd > 0 else 0,
            wl_residuals=wl_residuals,
            nl_residuals=nl_residuals
        )
        
        return result
    
    def _fix_extra_wide_lane(self, dd_L2, dd_L5, dd_ranges, ewl_wavelength):
        """Fix Extra-Wide-Lane ambiguities (L2 - L5)"""
        # EWL combination
        ewl_float = dd_L2 - dd_L5
        ewl_range_cycles = dd_ranges / ewl_wavelength
        ewl_ambiguities = ewl_float - ewl_range_cycles
        
        # Fix by rounding (very long wavelength ~5.86m)
        ewl_fractional = np.abs(ewl_ambiguities - np.round(ewl_ambiguities))
        ewl_fixed = np.round(ewl_ambiguities).astype(int)
        ewl_is_fixed = ewl_fractional < self.ewl_threshold
        
        return ewl_fixed, ewl_is_fixed
    
    def _fix_wide_lane(self, dd_L1, dd_L2, dd_ranges, wl_wavelength):
        """Fix Wide-Lane ambiguities (L1 - L2)"""
        # WL combination
        wl_float = dd_L1 - dd_L2
        wl_range_cycles = dd_ranges / wl_wavelength
        wl_ambiguities = wl_float - wl_range_cycles
        
        # Fix by rounding (long wavelength ~86cm)
        wl_fractional = np.abs(wl_ambiguities - np.round(wl_ambiguities))
        wl_fixed = np.round(wl_ambiguities).astype(int)
        wl_is_fixed = wl_fractional < self.wl_threshold
        
        return wl_fixed, wl_is_fixed, wl_fractional
    
    def _fix_narrow_lane(self, dd_L1, dd_L2, dd_ranges, wl_fixed, wl_is_fixed,
                         L1_wavelengths, L2_wavelengths):
        """
        Fix Narrow-Lane ambiguities using WL constraint
        
        WL constraint: N1 - N2 = wl_fixed (known)
        Find best (N1, N2) pair that satisfies constraint
        """
        n_dd = len(dd_L1)
        fixed_L1 = np.full(n_dd, np.nan)
        fixed_L2 = np.full(n_dd, np.nan)
        is_fixed = np.zeros(n_dd, dtype=bool)
        residuals = np.full(n_dd, np.inf)
        
        for i in range(n_dd):
            if not wl_is_fixed[i]:
                continue
            
            # Float estimates
            L1_range_cycles = dd_ranges[i] / L1_wavelengths[i]
            L2_range_cycles = dd_ranges[i] / L2_wavelengths[i]
            
            N1_float = dd_L1[i] - L1_range_cycles
            N2_float = dd_L2[i] - L2_range_cycles
            
            # Search for best integer combination
            best_N1 = None
            best_N2 = None
            best_score = float('inf')
            
            # Search range
            search_range = 3
            for N1_test in range(int(N1_float - search_range), 
                                int(N1_float + search_range + 1)):
                # Apply WL constraint
                N2_test = N1_test - wl_fixed[i]
                
                # Compute residuals
                residual_L1 = abs(N1_test - N1_float)
                residual_L2 = abs(N2_test - N2_float)
                
                # Combined score
                score = residual_L1 + residual_L2
                
                if score < best_score:
                    best_score = score
                    best_N1 = N1_test
                    best_N2 = N2_test
            
            # Check if solution is acceptable
            if best_score < 2 * self.nl_threshold:
                fixed_L1[i] = best_N1
                fixed_L2[i] = best_N2
                is_fixed[i] = True
                residuals[i] = best_score
        
        return fixed_L1, fixed_L2, is_fixed, residuals
    
    def _apply_partial_resolution(self, dd_L1, dd_L2, dd_ranges, 
                                  fixed_L1, fixed_L2, is_fixed,
                                  wl_fixed, wl_is_fixed, wavelengths):
        """Apply partial ambiguity resolution for unfixed ambiguities"""
        if not self.enable_partial:
            return fixed_L1, fixed_L2, is_fixed
        
        # Find unfixed but WL-fixed ambiguities
        unfixed_indices = np.where(~is_fixed & wl_is_fixed)[0]
        
        if len(unfixed_indices) == 0:
            return fixed_L1, fixed_L2, is_fixed
        
        # Try to fix subset with relaxed threshold
        relaxed_threshold = self.nl_threshold * 1.5
        
        for i in unfixed_indices:
            L1_range_cycles = dd_ranges[i] / wavelengths['L1'][i]
            L2_range_cycles = dd_ranges[i] / wavelengths['L2'][i]
            
            N1_float = dd_L1[i] - L1_range_cycles
            N2_float = dd_L2[i] - L2_range_cycles
            
            # Use WL constraint
            N1_test = round(N1_float)
            N2_test = N1_test - wl_fixed[i]
            
            residual_L1 = abs(N1_test - N1_float)
            residual_L2 = abs(N2_test - N2_float)
            
            if residual_L1 < relaxed_threshold and residual_L2 < relaxed_threshold:
                fixed_L1[i] = N1_test
                fixed_L2[i] = N2_test
                is_fixed[i] = True
        
        return fixed_L1, fixed_L2, is_fixed
    
    def _get_wavelengths(self, sat_pairs):
        """Get wavelengths for satellite pairs"""
        n_pairs = len(sat_pairs)
        
        # Initialize wavelength arrays
        wavelengths = {
            'L1': np.zeros(n_pairs),
            'L2': np.zeros(n_pairs),
            'WL': np.zeros(n_pairs),
            'EWL': np.zeros(n_pairs)
        }
        
        for i, (ref_sat, other_sat) in enumerate(sat_pairs):
            # Determine system (use other_sat as it's not reference)
            sys_id = sat2sys(other_sat)
            
            if sys_id == SYS_GPS:
                wavelengths['L1'][i] = self.wl_gps['L1']
                wavelengths['L2'][i] = self.wl_gps['L2']
                wavelengths['WL'][i] = self.wl_gps['WL_L1L2']
                wavelengths['EWL'][i] = self.wl_gps.get('EWL_L2L5', 0)
            elif sys_id == SYS_GAL:
                wavelengths['L1'][i] = self.wl_gal['E1']
                wavelengths['L2'][i] = self.wl_gal['E5a']
                wavelengths['WL'][i] = self.wl_gal['WL_E1E5a']
            elif sys_id == SYS_BDS:
                wavelengths['L1'][i] = self.wl_bds['B1']
                wavelengths['L2'][i] = self.wl_bds['B2']
                wavelengths['WL'][i] = self.wl_bds['WL_B1B2']
            else:
                # Default to GPS wavelengths
                wavelengths['L1'][i] = self.wl_gps['L1']
                wavelengths['L2'][i] = self.wl_gps['L2']
                wavelengths['WL'][i] = self.wl_gps['WL_L1L2']
        
        return wavelengths
    
    def _apply_bias_correction(self, dd_phase, sat_pairs, phase_biases, freq_idx):
        """Apply phase bias correction if available"""
        # This should interface with PhaseBiasTracker
        # For now, return uncorrected
        return dd_phase
    
    def _store_fixed_ambiguities(self, sat_pairs, fixed_L1, fixed_L2, is_fixed):
        """Store fixed ambiguities for Fix & Hold strategy"""
        for i, (ref_sat, other_sat) in enumerate(sat_pairs):
            if is_fixed[i]:
                key = (ref_sat, other_sat)
                self.fixed_storage[key] = {
                    'N1': fixed_L1[i],
                    'N2': fixed_L2[i],
                    'epoch': 0  # TODO: add epoch tracking
                }
    
    def _empty_result(self, n_dd):
        """Return empty result when resolution fails"""
        return CascadedFixResult(
            fixed_L1=np.full(n_dd, np.nan),
            fixed_L2=np.full(n_dd, np.nan),
            fixed_L5=None,
            is_fixed=np.zeros(n_dd, dtype=bool),
            n_ewl_fixed=0,
            n_wl_fixed=0,
            n_nl_fixed=0,
            n_total=n_dd,
            success_rate=0.0,
            wl_residuals=np.full(n_dd, np.inf),
            nl_residuals=np.full(n_dd, np.inf)
        )