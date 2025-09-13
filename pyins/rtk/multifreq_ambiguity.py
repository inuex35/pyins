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
Multi-Frequency Cascaded Ambiguity Resolution
=============================================

Implements WL/NL/EWL cascaded ambiguity resolution for improved fixing rates.
Based on GreatPVT's multi-frequency approach.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FrequencyMode(Enum):
    """Frequency combination modes"""
    WL = "WL"    # Wide-Lane
    NL = "NL"    # Narrow-Lane
    EWL = "EWL"  # Extra-Wide-Lane
    EWL24 = "EWL24"  # EWL between L2 and L4
    EWL25 = "EWL25"  # EWL between L2 and L5
    L1 = "L1"    # L1 frequency
    L2 = "L2"    # L2 frequency
    L3 = "L3"    # L3 frequency (e.g., Galileo E6)
    L4 = "L4"    # L4 frequency
    L5 = "L5"    # L5 frequency


@dataclass
class MultiFreqAmbiguity:
    """Multi-frequency ambiguity container"""
    satellite: str
    wl_float: float  # Wide-Lane float value
    wl_fixed: Optional[int] = None  # Wide-Lane fixed value
    nl_float: Optional[float] = None  # Narrow-Lane float value
    nl_fixed: Optional[int] = None  # Narrow-Lane fixed value
    ewl_float: Optional[float] = None  # Extra-Wide-Lane float
    ewl_fixed: Optional[int] = None  # Extra-Wide-Lane fixed
    l1_float: Optional[float] = None  # L1 float ambiguity
    l1_fixed: Optional[int] = None  # L1 fixed ambiguity
    l2_float: Optional[float] = None  # L2 float ambiguity
    l2_fixed: Optional[int] = None  # L2 fixed ambiguity
    
    @property
    def is_wl_fixed(self) -> bool:
        return self.wl_fixed is not None
    
    @property
    def is_nl_fixed(self) -> bool:
        return self.nl_fixed is not None
    
    @property
    def is_fully_fixed(self) -> bool:
        return self.is_wl_fixed and self.is_nl_fixed


class MultiFrequencyResolver:
    """
    Multi-frequency cascaded ambiguity resolver
    
    Resolves ambiguities in order: EWL -> WL -> NL
    Uses fixed ambiguities from easier combinations to constrain harder ones.
    """
    
    def __init__(self, 
                 wl_threshold: float = 0.25,  # cycles
                 nl_threshold: float = 0.15,  # cycles
                 ewl_threshold: float = 0.30,  # cycles
                 min_common_time: float = 300.0):  # seconds
        """
        Initialize multi-frequency resolver
        
        Parameters
        ----------
        wl_threshold : float
            Threshold for WL ambiguity decision (cycles)
        nl_threshold : float
            Threshold for NL ambiguity decision (cycles)
        ewl_threshold : float
            Threshold for EWL ambiguity decision (cycles)
        min_common_time : float
            Minimum common observation time (seconds)
        """
        self.wl_threshold = wl_threshold
        self.nl_threshold = nl_threshold
        self.ewl_threshold = ewl_threshold
        self.min_common_time = min_common_time
        
        # Store fixed ambiguities
        self.fixed_wl: Dict[str, int] = {}
        self.fixed_nl: Dict[str, int] = {}
        self.fixed_ewl: Dict[str, int] = {}
        
        # Wavelengths for different combinations (GPS example)
        self.wavelengths = {
            'L1': 0.19029367,
            'L2': 0.24421021,
            'L5': 0.25480066,
            'WL': 0.86202774,   # L1-L2 Wide-Lane
            'NL': 0.10695182,   # L1-L2 Narrow-Lane
            'EWL': 5.861,       # L2-L5 Extra-Wide-Lane
        }
    
    def form_combinations(self, l1_obs: np.ndarray, l2_obs: np.ndarray,
                         l3_obs: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Form linear combinations of observations
        
        Parameters
        ----------
        l1_obs : np.ndarray
            L1 carrier phase observations (cycles)
        l2_obs : np.ndarray
            L2 carrier phase observations (cycles)
        l3_obs : np.ndarray, optional
            L3/L5 carrier phase observations (cycles)
            
        Returns
        -------
        combinations : Dict[str, np.ndarray]
            Dictionary of linear combinations
        """
        combinations = {}
        
        # GPS L1/L2 frequencies
        f1 = 1575.42e6  # Hz
        f2 = 1227.60e6  # Hz
        
        # Wide-Lane combination: N_WL = N1 - N2
        combinations['WL'] = l1_obs - l2_obs
        
        # Narrow-Lane combination: N_NL = N1
        # (after WL is fixed, NL can be derived)
        combinations['NL'] = l1_obs
        
        # Extra-Wide-Lane if L5 available
        if l3_obs is not None:
            f5 = 1176.45e6  # Hz
            combinations['EWL'] = l2_obs - l3_obs
        
        return combinations
    
    def resolve_ewl(self, ewl_float: np.ndarray, ewl_sigma: np.ndarray,
                   satellites: List[str]) -> Dict[str, int]:
        """
        Resolve Extra-Wide-Lane ambiguities (easiest due to long wavelength)
        
        Parameters
        ----------
        ewl_float : np.ndarray
            EWL float ambiguities
        ewl_sigma : np.ndarray
            EWL standard deviations
        satellites : List[str]
            Satellite PRNs
            
        Returns
        -------
        fixed_ewl : Dict[str, int]
            Fixed EWL ambiguities
        """
        fixed = {}
        
        for i, sat in enumerate(satellites):
            if ewl_sigma[i] < self.ewl_threshold:
                # Simple rounding for EWL (very long wavelength)
                fixed_val = round(ewl_float[i])
                
                # Check residual
                residual = abs(ewl_float[i] - fixed_val)
                if residual < self.ewl_threshold:
                    fixed[sat] = fixed_val
                    logger.debug(f"Fixed EWL for {sat}: {fixed_val} (residual: {residual:.3f})")
        
        self.fixed_ewl.update(fixed)
        return fixed
    
    def resolve_wl(self, wl_float: np.ndarray, wl_sigma: np.ndarray,
                  satellites: List[str],
                  mw_values: Optional[Dict[str, float]] = None) -> Dict[str, int]:
        """
        Resolve Wide-Lane ambiguities using MW combination if available
        
        Parameters
        ----------
        wl_float : np.ndarray
            WL float ambiguities
        wl_sigma : np.ndarray
            WL standard deviations
        satellites : List[str]
            Satellite PRNs
        mw_values : Dict[str, float], optional
            Melbourne-Wübbena values for validation
            
        Returns
        -------
        fixed_wl : Dict[str, int]
            Fixed WL ambiguities
        """
        fixed = {}
        
        for i, sat in enumerate(satellites):
            # Use EWL to constrain WL if available
            constraint = 0.0
            if sat in self.fixed_ewl:
                # WL can be constrained by EWL
                constraint = self.fixed_ewl[sat] * 0.1  # Scaling factor
            
            adjusted_float = wl_float[i] - constraint
            
            if wl_sigma[i] < self.wl_threshold:
                fixed_val = round(adjusted_float)
                
                # Validate with MW if available
                if mw_values and sat in mw_values:
                    mw_val = mw_values[sat]
                    if abs(fixed_val - mw_val) > 1.0:
                        logger.warning(f"WL fix for {sat} rejected by MW validation")
                        continue
                
                # Check residual
                residual = abs(adjusted_float - fixed_val)
                if residual < self.wl_threshold:
                    fixed[sat] = fixed_val
                    logger.debug(f"Fixed WL for {sat}: {fixed_val} (residual: {residual:.3f})")
        
        self.fixed_wl.update(fixed)
        return fixed
    
    def resolve_nl(self, nl_float: np.ndarray, nl_sigma: np.ndarray,
                  satellites: List[str],
                  l1_float: np.ndarray, l2_float: np.ndarray) -> Dict[str, int]:
        """
        Resolve Narrow-Lane ambiguities using fixed WL as constraint
        
        Parameters
        ----------
        nl_float : np.ndarray
            NL float ambiguities
        nl_sigma : np.ndarray
            NL standard deviations
        satellites : List[str]
            Satellite PRNs
        l1_float : np.ndarray
            L1 float ambiguities
        l2_float : np.ndarray
            L2 float ambiguities
            
        Returns
        -------
        fixed_nl : Dict[str, int]
            Fixed NL ambiguities
        """
        fixed = {}
        
        # GPS frequencies for NL calculation
        f1 = 1575.42e6
        f2 = 1227.60e6
        alpha = f1 / (f1 + f2)
        beta = f2 / (f1 + f2)
        
        for i, sat in enumerate(satellites):
            if sat not in self.fixed_wl:
                continue  # Need WL fixed first
            
            # Use fixed WL to constrain NL
            # N1 = alpha * N_WL + N_NL
            # N2 = beta * N_WL - N_NL
            n_wl = self.fixed_wl[sat]
            
            # Calculate constrained NL
            n1_constrained = l1_float[i] - alpha * n_wl
            n2_constrained = l2_float[i] - beta * n_wl
            
            # Average for better estimate
            nl_constrained = (n1_constrained - n2_constrained) / 2
            
            if nl_sigma[i] < self.nl_threshold:
                fixed_val = round(nl_constrained)
                
                # Check residual
                residual = abs(nl_constrained - fixed_val)
                if residual < self.nl_threshold:
                    fixed[sat] = fixed_val
                    logger.debug(f"Fixed NL for {sat}: {fixed_val} (residual: {residual:.3f})")
        
        self.fixed_nl.update(fixed)
        return fixed
    
    def cascaded_resolution(self, observations: Dict[str, np.ndarray],
                          covariances: Dict[str, np.ndarray],
                          satellites: List[str]) -> Dict[str, MultiFreqAmbiguity]:
        """
        Perform cascaded multi-frequency ambiguity resolution
        
        Parameters
        ----------
        observations : Dict[str, np.ndarray]
            Dictionary with keys 'L1', 'L2', 'L3' (optional) containing observations
        covariances : Dict[str, np.ndarray]
            Covariance matrices for each frequency
        satellites : List[str]
            Satellite PRNs
            
        Returns
        -------
        resolved : Dict[str, MultiFreqAmbiguity]
            Resolved multi-frequency ambiguities
        """
        results = {}
        
        # Extract observations
        l1_obs = observations.get('L1', np.array([]))
        l2_obs = observations.get('L2', np.array([]))
        l3_obs = observations.get('L3', None)
        
        # Form combinations
        combos = self.form_combinations(l1_obs, l2_obs, l3_obs)
        
        # Extract standard deviations
        l1_sigma = np.sqrt(np.diag(covariances.get('L1', np.eye(len(l1_obs)))))
        l2_sigma = np.sqrt(np.diag(covariances.get('L2', np.eye(len(l2_obs)))))
        
        # Calculate combination sigmas (simplified)
        wl_sigma = np.sqrt(l1_sigma**2 + l2_sigma**2)
        nl_sigma = l1_sigma  # Simplified
        ewl_sigma = np.sqrt(l2_sigma**2 + (l3_obs is not None) * 0.01)
        
        # Step 1: Resolve EWL (if L3/L5 available)
        if 'EWL' in combos:
            self.resolve_ewl(combos['EWL'], ewl_sigma, satellites)
        
        # Step 2: Resolve WL
        self.resolve_wl(combos['WL'], wl_sigma, satellites)
        
        # Step 3: Resolve NL using fixed WL
        self.resolve_nl(combos['NL'], nl_sigma, satellites, l1_obs, l2_obs)
        
        # Build results
        for i, sat in enumerate(satellites):
            amb = MultiFreqAmbiguity(
                satellite=sat,
                wl_float=combos['WL'][i] if i < len(combos['WL']) else None,
                wl_fixed=self.fixed_wl.get(sat),
                nl_float=combos['NL'][i] if i < len(combos['NL']) else None,
                nl_fixed=self.fixed_nl.get(sat),
                ewl_float=combos['EWL'][i] if 'EWL' in combos and i < len(combos['EWL']) else None,
                ewl_fixed=self.fixed_ewl.get(sat),
                l1_float=l1_obs[i] if i < len(l1_obs) else None,
                l2_float=l2_obs[i] if i < len(l2_obs) else None
            )
            results[sat] = amb
        
        # Log summary
        n_wl_fixed = len(self.fixed_wl)
        n_nl_fixed = len(self.fixed_nl)
        n_ewl_fixed = len(self.fixed_ewl)
        logger.info(f"Cascaded resolution: EWL={n_ewl_fixed}, WL={n_wl_fixed}, NL={n_nl_fixed}")
        
        return results
    
    def get_fixed_l1_l2(self, sat: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Calculate fixed L1/L2 ambiguities from fixed WL/NL
        
        Parameters
        ----------
        sat : str
            Satellite PRN
            
        Returns
        -------
        n1_fixed : int or None
            Fixed L1 ambiguity
        n2_fixed : int or None
            Fixed L2 ambiguity
        """
        if sat not in self.fixed_wl or sat not in self.fixed_nl:
            return None, None
        
        n_wl = self.fixed_wl[sat]
        n_nl = self.fixed_nl[sat]
        
        # Back-calculate L1/L2
        # N1 - N2 = N_WL
        # N1 = N_NL (simplified for GPS L1/L2)
        n1_fixed = n_nl
        n2_fixed = n1_fixed - n_wl
        
        return int(n1_fixed), int(n2_fixed)