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
Enhanced Multi-Frequency Cascaded Ambiguity Resolution V2
==========================================================

Full implementation of multi-frequency GNSS ambiguity resolution
following the mandatory multi-frequency policy.

Supports: L1+L2+L5, EWL→WL→NL cascaded fixing, IF combinations
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class GNSSFrequency(Enum):
    """GNSS frequency definitions (Hz)"""
    # GPS
    GPS_L1 = 1575.42e6
    GPS_L2 = 1227.60e6
    GPS_L5 = 1176.45e6
    
    # Galileo
    GAL_E1 = 1575.42e6
    GAL_E5a = 1176.45e6
    GAL_E5b = 1207.14e6
    GAL_E6 = 1278.75e6
    
    # BeiDou
    BDS_B1I = 1561.098e6
    BDS_B1C = 1575.42e6
    BDS_B2a = 1176.45e6
    BDS_B2b = 1207.14e6
    BDS_B3 = 1268.52e6
    
    # GLONASS (center frequencies)
    GLO_L1 = 1602.0e6
    GLO_L2 = 1246.0e6
    
    # QZSS
    QZS_L1 = 1575.42e6
    QZS_L2 = 1227.60e6
    QZS_L5 = 1176.45e6
    QZS_L6 = 1278.75e6


@dataclass
class MultiFreqObservation:
    """Multi-frequency observation container (MANDATORY structure)"""
    satellite: str
    
    # Phase observations (cycles) - REQUIRED
    L1: float  # MANDATORY
    L2: float  # MANDATORY
    
    # Code observations (meters) - REQUIRED
    P1: float  # MANDATORY
    P2: float  # MANDATORY
    
    # Optional phase observations
    L3: Optional[float] = None  # L5/E5a/B2a
    L4: Optional[float] = None  # L6/E6/B3
    L5: Optional[float] = None  # E5b/B2b
    
    # Optional code observations
    P3: Optional[float] = None
    P4: Optional[float] = None
    P5: Optional[float] = None
    
    # Signal strength (dB-Hz) - with defaults
    S1: float = 45.0  # MANDATORY with default
    S2: float = 42.0  # MANDATORY with default
    S3: Optional[float] = None
    S4: Optional[float] = None
    S5: Optional[float] = None
    
    # Doppler (Hz) - Optional but useful
    D1: Optional[float] = None
    D2: Optional[float] = None
    D3: Optional[float] = None
    
    @property
    def has_l5(self) -> bool:
        """Check if L5/E5a observation available"""
        return self.L3 is not None and self.P3 is not None
    
    @property
    def is_valid(self) -> bool:
        """Validate minimum requirements (L1+L2)"""
        return (self.L1 is not None and self.L2 is not None and 
                self.P1 is not None and self.P2 is not None)


@dataclass
class LinearCombination:
    """Linear combination of observations"""
    name: str
    value: float
    wavelength: float
    noise: float  # Standard deviation
    iono_free: bool = False


class MultiFrequencyResolverV2:
    """
    Enhanced multi-frequency cascaded ambiguity resolver
    
    MANDATORY: Always uses minimum L1+L2, preferably L1+L2+L5
    Implements EWL→WL→NL cascaded resolution
    """
    
    def __init__(self, system: str = 'GPS'):
        """
        Initialize multi-frequency resolver
        
        Parameters
        ----------
        system : str
            GNSS system ('GPS', 'GAL', 'BDS', 'GLO', 'QZS')
        """
        self.system = system
        self.c = 299792458.0  # Speed of light (m/s)
        
        # Set frequencies based on system
        self._setup_frequencies()
        
        # Thresholds for each combination
        self.ewl_threshold = 0.20  # cycles (5.86m wavelength)
        self.wl_threshold = 0.15   # cycles (86cm wavelength)
        self.nl_threshold = 0.10   # cycles (10.7cm wavelength)
        
        # Fixed ambiguities storage
        self.fixed_ewl: Dict[str, int] = {}
        self.fixed_wl: Dict[str, int] = {}
        self.fixed_nl: Dict[str, int] = {}
        self.fixed_l1: Dict[str, int] = {}
        self.fixed_l2: Dict[str, int] = {}
        self.fixed_l5: Dict[str, int] = {}
        
        logger.info(f"Initialized MultiFrequencyResolverV2 for {system}")
        logger.info(f"  L1: {self.f1/1e9:.3f} GHz, λ={self.lambda1:.3f}m")
        logger.info(f"  L2: {self.f2/1e9:.3f} GHz, λ={self.lambda2:.3f}m")
        if hasattr(self, 'f3'):
            logger.info(f"  L5: {self.f3/1e9:.3f} GHz, λ={self.lambda3:.3f}m")
    
    def _setup_frequencies(self):
        """Setup system-specific frequencies"""
        if self.system == 'GPS':
            self.f1 = GNSSFrequency.GPS_L1.value
            self.f2 = GNSSFrequency.GPS_L2.value
            self.f3 = GNSSFrequency.GPS_L5.value
        elif self.system == 'GAL':
            self.f1 = GNSSFrequency.GAL_E1.value
            self.f2 = GNSSFrequency.GAL_E5b.value
            self.f3 = GNSSFrequency.GAL_E5a.value
        elif self.system == 'BDS':
            self.f1 = GNSSFrequency.BDS_B1C.value
            self.f2 = GNSSFrequency.BDS_B2b.value
            self.f3 = GNSSFrequency.BDS_B2a.value
        else:  # Default to GPS
            self.f1 = GNSSFrequency.GPS_L1.value
            self.f2 = GNSSFrequency.GPS_L2.value
            self.f3 = GNSSFrequency.GPS_L5.value
        
        # Calculate wavelengths
        self.lambda1 = self.c / self.f1
        self.lambda2 = self.c / self.f2
        self.lambda3 = self.c / self.f3 if hasattr(self, 'f3') else None
        
        # Calculate combination wavelengths
        self.lambda_wl = self.c / (self.f1 - self.f2)  # Wide-Lane
        self.lambda_nl = self.c / (self.f1 + self.f2)  # Narrow-Lane
        if self.lambda3:
            self.lambda_ewl = self.c / abs(self.f2 - self.f3)  # Extra-Wide-Lane
    
    def compute_combinations(self, obs: MultiFreqObservation) -> Dict[str, LinearCombination]:
        """
        Compute all linear combinations (MANDATORY)
        
        Parameters
        ----------
        obs : MultiFreqObservation
            Multi-frequency observations
            
        Returns
        -------
        combinations : Dict[str, LinearCombination]
            All computed combinations
        """
        if not obs.is_valid:
            raise ValueError(f"Invalid observation for {obs.satellite}: missing L1 or L2")
        
        combinations = {}
        
        # 1. Wide-Lane (L1-L2) - MANDATORY
        wl_phase = obs.L1 - obs.L2
        wl_code = (self.f1 * obs.P1 - self.f2 * obs.P2) / (self.f1 - self.f2) / self.lambda_wl
        combinations['WL'] = LinearCombination(
            name='Wide-Lane',
            value=wl_phase,
            wavelength=self.lambda_wl,
            noise=0.01,  # cycles
            iono_free=False
        )
        
        # 2. Narrow-Lane - MANDATORY
        nl_phase = (self.f1 * obs.L1 + self.f2 * obs.L2) / (self.f1 + self.f2)
        nl_code = (self.f1 * obs.P1 + self.f2 * obs.P2) / (self.f1 + self.f2) / self.lambda_nl
        combinations['NL'] = LinearCombination(
            name='Narrow-Lane',
            value=nl_phase,
            wavelength=self.lambda_nl,
            noise=0.005,
            iono_free=False
        )
        
        # 3. Ionosphere-Free - MANDATORY
        if_phase = (self.f1**2 * obs.L1 * self.lambda1 - self.f2**2 * obs.L2 * self.lambda2) / (self.f1**2 - self.f2**2)
        if_code = (self.f1**2 * obs.P1 - self.f2**2 * obs.P2) / (self.f1**2 - self.f2**2)
        combinations['IF'] = LinearCombination(
            name='Ionosphere-Free',
            value=if_phase,
            wavelength=0.0,  # Not used for ambiguity
            noise=0.003,
            iono_free=True
        )
        
        # 4. Geometry-Free (for ionosphere estimation)
        gf_phase = obs.L1 * self.lambda1 - obs.L2 * self.lambda2
        combinations['GF'] = LinearCombination(
            name='Geometry-Free',
            value=gf_phase,
            wavelength=0.0,
            noise=0.01,
            iono_free=False
        )
        
        # 5. Melbourne-Wübbena - MANDATORY for WL validation
        mw = wl_phase - wl_code
        combinations['MW'] = LinearCombination(
            name='Melbourne-Wübbena',
            value=mw,
            wavelength=self.lambda_wl,
            noise=0.3,  # Higher noise due to code
            iono_free=True  # Both iono and geometry free
        )
        
        # 6. Extra-Wide-Lane (if L5 available)
        if obs.has_l5 and self.lambda3:
            ewl_phase = obs.L2 - obs.L3
            ewl_code = (self.f2 * obs.P2 - self.f3 * obs.P3) / abs(self.f2 - self.f3) / self.lambda_ewl
            combinations['EWL'] = LinearCombination(
                name='Extra-Wide-Lane',
                value=ewl_phase,
                wavelength=self.lambda_ewl,
                noise=0.02,
                iono_free=False
            )
            
            # EWL Melbourne-Wübbena
            ewl_mw = ewl_phase - ewl_code
            combinations['EWL_MW'] = LinearCombination(
                name='EWL-MW',
                value=ewl_mw,
                wavelength=self.lambda_ewl,
                noise=0.4,
                iono_free=True
            )
        
        return combinations
    
    def cascaded_resolution(self, observations: List[MultiFreqObservation],
                          covariances: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
        """
        MANDATORY cascaded multi-frequency ambiguity resolution
        
        Order: EWL → WL → NL → L1/L2/L5
        
        Parameters
        ----------
        observations : List[MultiFreqObservation]
            Multi-frequency observations for all satellites
        covariances : Dict[str, np.ndarray]
            Covariance matrices for each frequency
            
        Returns
        -------
        results : Dict[str, Dict[str, Any]]
            Resolution results for each satellite
        """
        results = {}
        
        logger.info(f"Starting cascaded resolution for {len(observations)} satellites")
        
        for obs in observations:
            if not obs.is_valid:
                logger.warning(f"Skipping {obs.satellite}: invalid observations")
                continue
            
            sat = obs.satellite
            result = {
                'satellite': sat,
                'ewl_fixed': None,
                'wl_fixed': None,
                'nl_fixed': None,
                'l1_fixed': None,
                'l2_fixed': None,
                'l5_fixed': None,
                'success': False
            }
            
            # Compute all combinations
            combos = self.compute_combinations(obs)
            
            # Step 1: Fix EWL (if L5 available)
            if 'EWL' in combos and obs.has_l5:
                ewl_fixed = self._fix_ewl(combos['EWL'], combos.get('EWL_MW'))
                if ewl_fixed is not None:
                    self.fixed_ewl[sat] = ewl_fixed
                    result['ewl_fixed'] = ewl_fixed
                    logger.debug(f"{sat}: EWL fixed = {ewl_fixed}")
            
            # Step 2: Fix WL (using MW for validation)
            wl_fixed = self._fix_wl(combos['WL'], combos['MW'], 
                                   ewl_constraint=self.fixed_ewl.get(sat))
            if wl_fixed is not None:
                self.fixed_wl[sat] = wl_fixed
                result['wl_fixed'] = wl_fixed
                logger.debug(f"{sat}: WL fixed = {wl_fixed}")
            
            # Step 3: Fix NL (using WL constraint)
            if sat in self.fixed_wl:
                nl_fixed = self._fix_nl(combos['NL'], obs.L1, obs.L2, 
                                       wl_fixed=self.fixed_wl[sat])
                if nl_fixed is not None:
                    self.fixed_nl[sat] = nl_fixed
                    result['nl_fixed'] = nl_fixed
                    logger.debug(f"{sat}: NL fixed = {nl_fixed}")
            
            # Step 4: Recover L1/L2/L5 from WL/NL
            if sat in self.fixed_wl and sat in self.fixed_nl:
                l1_fixed, l2_fixed = self._recover_l1_l2(
                    self.fixed_wl[sat], self.fixed_nl[sat]
                )
                result['l1_fixed'] = l1_fixed
                result['l2_fixed'] = l2_fixed
                self.fixed_l1[sat] = l1_fixed
                self.fixed_l2[sat] = l2_fixed
                result['success'] = True
                
                # Fix L5 if available
                if obs.has_l5 and sat in self.fixed_ewl:
                    l5_fixed = l2_fixed - self.fixed_ewl[sat]
                    result['l5_fixed'] = l5_fixed
                    self.fixed_l5[sat] = l5_fixed
                
                logger.info(f"{sat}: Successfully fixed L1={l1_fixed}, L2={l2_fixed}")
            
            results[sat] = result
        
        # Summary
        n_success = sum(1 for r in results.values() if r['success'])
        logger.info(f"Cascaded resolution complete: {n_success}/{len(results)} satellites fixed")
        
        return results
    
    def _fix_ewl(self, ewl_combo: LinearCombination, 
                ewl_mw: Optional[LinearCombination] = None) -> Optional[int]:
        """Fix Extra-Wide-Lane ambiguity (easiest due to 5.86m wavelength)"""
        
        # Simple rounding for EWL
        ewl_float = ewl_combo.value
        ewl_int = round(ewl_float)
        
        # Check residual
        residual = abs(ewl_float - ewl_int)
        
        # Validate with MW if available
        if ewl_mw and abs(ewl_mw.value - ewl_int) > 0.5:
            logger.debug(f"EWL rejected by MW validation: {ewl_float:.3f}")
            return None
        
        if residual < self.ewl_threshold:
            return ewl_int
        
        return None
    
    def _fix_wl(self, wl_combo: LinearCombination,
               mw_combo: LinearCombination,
               ewl_constraint: Optional[int] = None) -> Optional[int]:
        """Fix Wide-Lane ambiguity using Melbourne-Wübbena"""
        
        # Use MW combination (more reliable for WL)
        wl_float = mw_combo.value
        
        # Apply EWL constraint if available
        if ewl_constraint is not None:
            # EWL provides additional constraint on WL
            wl_float += ewl_constraint * 0.05  # Small adjustment
        
        wl_int = round(wl_float)
        
        # Validate with phase WL
        phase_wl = wl_combo.value
        if abs(phase_wl - wl_int) > 1.0:
            logger.debug(f"WL rejected: MW={wl_float:.3f}, Phase={phase_wl:.3f}")
            return None
        
        # Check residual
        residual = abs(wl_float - wl_int)
        if residual < self.wl_threshold:
            return wl_int
        
        return None
    
    def _fix_nl(self, nl_combo: LinearCombination,
               l1_phase: float, l2_phase: float,
               wl_fixed: int) -> Optional[int]:
        """Fix Narrow-Lane ambiguity using WL constraint"""
        
        # Use WL to constrain NL
        # N1 = N_WL + N_NL
        # N2 = N_NL
        
        # Estimate NL from L1 with WL constraint
        nl_from_l1 = l1_phase - wl_fixed
        nl_from_l2 = l2_phase
        
        # Average for better estimate
        nl_float = (nl_from_l1 + nl_from_l2) / 2.0
        nl_int = round(nl_float)
        
        # Validate consistency
        if abs(nl_from_l1 - nl_from_l2) > 1.0:
            logger.debug(f"NL inconsistent: from_L1={nl_from_l1:.3f}, from_L2={nl_from_l2:.3f}")
            return None
        
        # Check residual
        residual = abs(nl_float - nl_int)
        if residual < self.nl_threshold:
            return nl_int
        
        return None
    
    def _recover_l1_l2(self, wl_fixed: int, nl_fixed: int) -> Tuple[int, int]:
        """
        Recover L1/L2 integer ambiguities from WL/NL
        
        N1 - N2 = N_WL
        N1 + N2 = 2*N_NL (approximately)
        
        Therefore:
        N1 = N_WL + N2
        N2 = N_NL (approximately)
        """
        n2_fixed = nl_fixed
        n1_fixed = wl_fixed + n2_fixed
        
        return n1_fixed, n2_fixed
    
    def compute_ionosphere_free_solution(self, obs: MultiFreqObservation,
                                        n1_fixed: int, n2_fixed: int) -> float:
        """
        Compute ionosphere-free carrier phase solution
        
        This is the most precise positioning solution
        """
        # Ionosphere-free combination with fixed ambiguities
        if_phase = (self.f1**2 * (obs.L1 - n1_fixed) * self.lambda1 - 
                    self.f2**2 * (obs.L2 - n2_fixed) * self.lambda2) / (self.f1**2 - self.f2**2)
        
        return if_phase
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get resolution statistics"""
        stats = {
            'n_ewl_fixed': len(self.fixed_ewl),
            'n_wl_fixed': len(self.fixed_wl),
            'n_nl_fixed': len(self.fixed_nl),
            'n_l1_fixed': len(self.fixed_l1),
            'n_l2_fixed': len(self.fixed_l2),
            'n_l5_fixed': len(self.fixed_l5),
            'success_rate': len(self.fixed_l1) / max(1, len(self.fixed_wl)) if self.fixed_wl else 0
        }
        
        return stats