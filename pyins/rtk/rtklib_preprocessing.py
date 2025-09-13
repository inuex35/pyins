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
RTKLIB-compatible preprocessing for carrier phase observations
==============================================================

Based on RTKLIB's approach for handling carrier phase data:
1. detslp_ll() - Cycle slip detection using LLI flag  
2. detslp_gf() - Geometry-free combination detection
3. detslp_mw() - Melbourne-Wubbena combination detection
4. Initial carrier phase ambiguity estimation
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from ..core.constants import CLIGHT


@dataclass
class RTKLIBCarrierState:
    """State for carrier phase tracking (equivalent to RTKLIB's ambc_t)"""
    epoch: int = 0          # Epoch number
    n: np.ndarray = None    # Ambiguity array
    LC: np.ndarray = None   # Carrier phase in cycles
    LQ: np.ndarray = None   # Geometry-free phase combination
    lock: np.ndarray = None # Lock time counter
    slip: np.ndarray = None # Cycle slip flags
    
    def __init__(self, max_sat: int = 64):
        """Initialize state arrays"""
        self.n = np.zeros((max_sat, 2))     # L1, L2 ambiguities
        self.LC = np.zeros((max_sat, 2))    # L1, L2 carrier phase
        self.LQ = np.zeros(max_sat)         # Geometry-free combination
        self.lock = np.zeros(max_sat, dtype=int)
        self.slip = np.zeros(max_sat, dtype=bool)


class RTKLIBPreprocessor:
    """
    RTKLIB-compatible carrier phase preprocessor
    
    Implements RTKLIB's approach for:
    - Cycle slip detection
    - Initial ambiguity estimation
    - Carrier phase bias handling
    """
    
    # RTKLIB constants
    THRES_MW_JUMP = 10.0    # MW jump threshold (cycles)
    THRES_GF_JUMP = 0.05    # GF jump threshold (m)
    THRES_SLIP = 0.05       # Cycle slip threshold
    MAX_STD_FIX = 0.15      # Max std for fix (m)
    MIN_LOCK_TO_FIX = 5     # Min lock count to fix
    
    def __init__(self):
        """Initialize preprocessor"""
        self.f1 = 1575.42e6  # L1 frequency (Hz)
        self.f2 = 1227.60e6  # L2 frequency (Hz)
        self.lam1 = CLIGHT / self.f1  # L1 wavelength
        self.lam2 = CLIGHT / self.f2  # L2 wavelength
        self.lam_wl = CLIGHT / (self.f1 - self.f2)  # Wide-lane wavelength
        
        # State storage
        self.states = {}  # sat_id -> RTKLIBCarrierState
        
    def detect_cycle_slip(self, 
                         sat: str,
                         L1: float, L2: float,
                         P1: float, P2: float,
                         lli1: int = 0, lli2: int = 0,
                         lock_time: float = 0) -> bool:
        """
        Detect cycle slip using RTKLIB method
        
        Parameters:
        -----------
        sat : str
            Satellite ID
        L1, L2 : float
            Carrier phase observations (cycles)
        P1, P2 : float
            Pseudorange observations (m)
        lli1, lli2 : int
            Loss of lock indicators
        lock_time : float
            Lock time (seconds)
            
        Returns:
        --------
        slip_detected : bool
            True if cycle slip detected
        """
        
        # Initialize state if needed
        if sat not in self.states:
            self.states[sat] = RTKLIBCarrierState()
            return True  # First observation
        
        state = self.states[sat]
        
        # Method 1: Check LLI (Loss of Lock Indicator)
        if lli1 & 1 or lli2 & 1:
            state.slip[0] = True
            return True
            
        # Method 2: Check lock time
        if lock_time > 0 and state.lock[0] > 0:
            if lock_time < state.lock[0]:
                state.slip[0] = True
                return True
        
        # Method 3: Geometry-free combination (if dual-freq)
        if L2 > 0 and P2 > 0:
            gf_current = L1 * self.lam1 - L2 * self.lam2
            
            if state.LQ[0] != 0:
                gf_diff = gf_current - state.LQ[0]
                if abs(gf_diff) > self.THRES_GF_JUMP:
                    state.slip[0] = True
                    state.LQ[0] = gf_current
                    return True
            
            state.LQ[0] = gf_current
            
            # Method 4: Melbourne-Wubbena combination
            mw_current = self._compute_mw(L1, L2, P1, P2)
            
            if state.LC[0, 0] != 0:  # Previous MW stored
                mw_diff = mw_current - state.LC[0, 0]
                if abs(mw_diff) > self.THRES_MW_JUMP:
                    state.slip[0] = True
                    state.LC[0, 0] = mw_current
                    return True
            
            state.LC[0, 0] = mw_current
        
        # Update lock time
        state.lock[0] = lock_time if lock_time > 0 else state.lock[0] + 1
        
        return False
        
    def _compute_mw(self, L1: float, L2: float, P1: float, P2: float) -> float:
        """
        Compute Melbourne-Wubbena combination
        
        RTKLIB formula: 
        MW = (L1 - L2) - (f1*P1 + f2*P2)/(f1-f2)/λ_WL
        """
        if L1 == 0 or L2 == 0 or P1 == 0 or P2 == 0:
            return 0.0
            
        # Wide-lane phase (cycles)
        Lwl = L1 - L2
        
        # Narrow-lane code (converted to cycles)
        Pnl = (self.f1 * P1 + self.f2 * P2) / (self.f1 + self.f2)
        Pwl_cycles = Pnl / self.lam_wl
        
        return Lwl - Pwl_cycles
        
    def init_phase_bias(self,
                       rover_L1: float, base_L1: float,
                       rover_P1: float, base_P1: float) -> float:
        """
        Initialize carrier phase bias using RTKLIB method
        
        RTKLIB uses the code-phase difference to estimate initial bias:
        N = (P - Φ*λ) / λ
        
        Parameters:
        -----------
        rover_L1, base_L1 : float
            Carrier phase observations (cycles)
        rover_P1, base_P1 : float
            Code observations (m)
            
        Returns:
        --------
        bias : float
            Initial phase bias (cycles)
        """
        if rover_L1 == 0 or base_L1 == 0:
            return 0.0
            
        # Single difference
        sd_phase = rover_L1 - base_L1  # cycles
        sd_code = rover_P1 - base_P1    # meters
        
        # Convert code to cycles
        sd_code_cycles = sd_code / self.lam1
        
        # Initial ambiguity estimate
        N_float = sd_code_cycles - sd_phase
        
        # RTKLIB rounds to nearest integer for initial estimate
        N_init = np.round(N_float)
        
        return N_init
        
    def udbias(self, 
              rover_obs: Dict, base_obs: Dict,
              ref_sat: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update double-difference biases (RTKLIB's udbias function)
        
        Parameters:
        -----------
        rover_obs : dict
            Rover observations {sat: {'L1': ..., 'P1': ...}}
        base_obs : dict
            Base observations
        ref_sat : str
            Reference satellite (auto-select if None)
            
        Returns:
        --------
        dd_bias : np.ndarray
            Double-difference biases
        sats : np.ndarray
            Satellite IDs
        """
        # Find common satellites
        common = set(rover_obs.keys()) & set(base_obs.keys())
        valid = []
        
        for sat in common:
            if 'L1' in rover_obs[sat] and 'L1' in base_obs[sat]:
                if rover_obs[sat]['L1'] > 0 and base_obs[sat]['L1'] > 0:
                    if 'P1' in rover_obs[sat] and 'P1' in base_obs[sat]:
                        if rover_obs[sat]['P1'] > 0 and base_obs[sat]['P1'] > 0:
                            valid.append(sat)
        
        if len(valid) < 2:
            return np.array([]), np.array([])
            
        # Select reference satellite (highest elevation/SNR)
        if ref_sat is None or ref_sat not in valid:
            # Use first satellite as reference (should use elevation in practice)
            ref_sat = valid[0]
            
        # Remove reference from list
        other_sats = [s for s in valid if s != ref_sat]
        
        # Compute double-difference biases
        dd_biases = []
        
        # Reference satellite single difference
        ref_sd_phase = rover_obs[ref_sat]['L1'] - base_obs[ref_sat]['L1']
        ref_sd_code = rover_obs[ref_sat]['P1'] - base_obs[ref_sat]['P1']
        ref_N = (ref_sd_code / self.lam1) - ref_sd_phase
        
        for sat in other_sats:
            # Single differences
            sd_phase = rover_obs[sat]['L1'] - base_obs[sat]['L1']
            sd_code = rover_obs[sat]['P1'] - base_obs[sat]['P1']
            N = (sd_code / self.lam1) - sd_phase
            
            # Double difference bias
            dd_bias = N - ref_N
            dd_biases.append(dd_bias)
            
        return np.array(dd_biases), np.array([ref_sat] + other_sats)
        
    def fix_amb_ROUND(self, amb_float: np.ndarray) -> np.ndarray:
        """
        Fix ambiguities by simple rounding (RTKLIB ROUND method)
        
        Parameters:
        -----------
        amb_float : np.ndarray
            Float ambiguities
            
        Returns:
        --------
        amb_fixed : np.ndarray
            Fixed integer ambiguities
        """
        return np.round(amb_float).astype(int)
        
    def valpos(self, 
              dd_residuals: np.ndarray,
              threshold: float = 0.1) -> bool:
        """
        Validate position by residuals (RTKLIB's valpos function)
        
        Parameters:
        -----------
        dd_residuals : np.ndarray
            Double-difference residuals (m)
        threshold : float
            Validation threshold (m)
            
        Returns:
        --------
        valid : bool
            True if position is valid
        """
        if len(dd_residuals) == 0:
            return False
            
        # Check RMS of residuals
        rms = np.sqrt(np.mean(dd_residuals**2))
        
        # RTKLIB also checks ratio of largest residual to RMS
        max_res = np.max(np.abs(dd_residuals))
        
        return rms < threshold and max_res < 3 * rms