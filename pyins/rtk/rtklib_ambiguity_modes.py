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
RTKLIB Ambiguity Resolution Modes
==================================

Implements RTKLIB's ambiguity resolution strategies:
1. INSTANTANEOUS - Fix ambiguities independently each epoch
2. CONTINUOUS - Keep ambiguities if validation passes
3. FIX AND HOLD - Hold fixed ambiguities across epochs
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from ..core.constants import CLIGHT


class AmbiguityMode(Enum):
    """RTKLIB ambiguity resolution modes"""
    INSTANTANEOUS = 0  # Fix independently each epoch
    CONTINUOUS = 1      # Keep if validation passes
    FIX_AND_HOLD = 2   # Hold fixed values


@dataclass
class AmbiguityState:
    """State for tracking ambiguities"""
    satellites: List[str]           # Satellite IDs
    ambiguities: np.ndarray         # Float ambiguities
    fixed_ambiguities: np.ndarray   # Fixed integer ambiguities
    covariance: np.ndarray          # Covariance matrix
    is_fixed: bool                  # Fix status
    ratio: float                    # Ratio test value
    epoch: int                      # Epoch number
    lock_count: int                 # Continuous lock count
    

class RTKLIBAmbiguityModes:
    """
    RTKLIB-style ambiguity resolution with different modes
    
    Based on RTKLIB's resamb_LAMBDA function
    """
    
    def __init__(self, mode: AmbiguityMode = AmbiguityMode.CONTINUOUS):
        """
        Initialize ambiguity resolver
        
        Parameters:
        -----------
        mode : AmbiguityMode
            Resolution mode (INSTANTANEOUS, CONTINUOUS, FIX_AND_HOLD)
        """
        self.mode = mode
        
        # State tracking
        self.prev_state = None
        self.fixed_ambiguities = {}  # sat_pair -> fixed value
        self.lock_counts = {}        # sat_pair -> lock count
        self.hold_ambiguities = {}   # FIX AND HOLD storage
        
        # RTKLIB parameters
        self.ratio_threshold = 3.0    # Ratio test threshold
        self.min_lock_count = 5       # Min epochs for CONTINUOUS
        self.min_fix_elevation = 15.0 # Min elevation for fixing (degrees)
        self.max_age = 30             # Max age for held ambiguities (epochs)
        
    def resolve(self,
               float_amb: np.ndarray,
               covariance: np.ndarray,
               satellites: List[str],
               elevations: Optional[np.ndarray] = None,
               epoch: int = 0) -> Tuple[np.ndarray, bool, float]:
        """
        Resolve ambiguities based on mode
        
        Parameters:
        -----------
        float_amb : np.ndarray
            Float ambiguities
        covariance : np.ndarray
            Covariance matrix
        satellites : List[str]
            Satellite pair identifiers
        elevations : np.ndarray, optional
            Satellite elevations (degrees)
        epoch : int
            Current epoch number
            
        Returns:
        --------
        fixed_amb : np.ndarray
            Fixed ambiguities
        is_fixed : bool
            Whether solution is fixed
        ratio : float
            Ratio test value
        """
        
        if self.mode == AmbiguityMode.INSTANTANEOUS:
            return self._resolve_instantaneous(float_amb, covariance, satellites, elevations)
            
        elif self.mode == AmbiguityMode.CONTINUOUS:
            return self._resolve_continuous(float_amb, covariance, satellites, elevations, epoch)
            
        elif self.mode == AmbiguityMode.FIX_AND_HOLD:
            return self._resolve_fix_and_hold(float_amb, covariance, satellites, elevations, epoch)
            
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
            
    def _resolve_instantaneous(self,
                              float_amb: np.ndarray,
                              covariance: np.ndarray,
                              satellites: List[str],
                              elevations: Optional[np.ndarray]) -> Tuple[np.ndarray, bool, float]:
        """
        INSTANTANEOUS mode - fix independently each epoch
        
        This is the simplest mode:
        - Try to fix ambiguities using LAMBDA
        - No memory between epochs
        - Each epoch is independent
        """
        
        # Import LAMBDA solver
        from .mlambda import mlambda, ratio_test
        
        # Apply LAMBDA
        try:
            F, s = mlambda(float_amb, covariance, m=2)
            is_fixed, ratio = ratio_test(s, threshold=self.ratio_threshold)
            
            if is_fixed:
                # Apply elevation mask if provided
                if elevations is not None:
                    mask = elevations >= self.min_fix_elevation
                    if not np.all(mask):
                        # Partial fixing based on elevation
                        fixed_amb = float_amb.copy()
                        fixed_amb[mask] = F[mask, 0]
                        return fixed_amb, False, ratio  # Not fully fixed
                
                return F[:, 0], True, ratio
            else:
                return float_amb, False, ratio
                
        except Exception:
            return float_amb, False, 0.0
            
    def _resolve_continuous(self,
                           float_amb: np.ndarray,
                           covariance: np.ndarray,
                           satellites: List[str],
                           elevations: Optional[np.ndarray],
                           epoch: int) -> Tuple[np.ndarray, bool, float]:
        """
        CONTINUOUS mode - keep ambiguities if validation passes
        
        RTKLIB logic:
        - Try to fix with LAMBDA
        - If fixed, validate against previous epoch
        - Keep lock count for each satellite pair
        - Require minimum lock count for fixing
        """
        
        from .mlambda import mlambda, ratio_test
        
        # Try to fix current epoch
        try:
            F, s = mlambda(float_amb, covariance, m=2)
            is_fixed, ratio = ratio_test(s, threshold=self.ratio_threshold)
        except Exception:
            is_fixed = False
            ratio = 0.0
            F = None
        
        # Check continuity with previous epoch
        if self.prev_state is not None:
            # Find common satellites
            prev_sats = set(self.prev_state.satellites)
            curr_sats = set(satellites)
            common = prev_sats & curr_sats
            
            if len(common) > 0 and is_fixed:
                # Check consistency
                consistent = True
                for sat in common:
                    prev_idx = self.prev_state.satellites.index(sat)
                    curr_idx = satellites.index(sat)
                    
                    if self.prev_state.is_fixed:
                        # Check if ambiguity jumped
                        prev_fixed = self.prev_state.fixed_ambiguities[prev_idx]
                        curr_fixed = F[curr_idx, 0]
                        
                        if abs(prev_fixed - curr_fixed) > 0.5:  # Half cycle threshold
                            consistent = False
                            break
                
                # Update lock counts
                for sat in satellites:
                    if sat in common and consistent:
                        self.lock_counts[sat] = self.lock_counts.get(sat, 0) + 1
                    else:
                        self.lock_counts[sat] = 1
                        
                # Check if we have enough lock count
                min_lock = min(self.lock_counts.get(sat, 0) for sat in satellites)
                
                if min_lock >= self.min_lock_count and is_fixed:
                    fixed_amb = F[:, 0]
                else:
                    # Not enough continuity
                    fixed_amb = float_amb
                    is_fixed = False
            else:
                # No common satellites or not fixed
                fixed_amb = F[:, 0] if is_fixed else float_amb
                # Reset lock counts
                for sat in satellites:
                    self.lock_counts[sat] = 1 if is_fixed else 0
        else:
            # First epoch
            fixed_amb = F[:, 0] if is_fixed else float_amb
            for sat in satellites:
                self.lock_counts[sat] = 1 if is_fixed else 0
        
        # Update state
        self.prev_state = AmbiguityState(
            satellites=satellites.copy(),
            ambiguities=float_amb.copy(),
            fixed_ambiguities=fixed_amb.copy(),
            covariance=covariance.copy(),
            is_fixed=is_fixed,
            ratio=ratio,
            epoch=epoch,
            lock_count=min(self.lock_counts.get(sat, 0) for sat in satellites) if satellites else 0
        )
        
        return fixed_amb, is_fixed, ratio
        
    def _resolve_fix_and_hold(self,
                              float_amb: np.ndarray,
                              covariance: np.ndarray,
                              satellites: List[str],
                              elevations: Optional[np.ndarray],
                              epoch: int) -> Tuple[np.ndarray, bool, float]:
        """
        FIX AND HOLD mode - hold fixed ambiguities across epochs
        
        RTKLIB logic:
        - Once fixed, hold the integer values
        - Only update if cycle slip detected
        - Constrain float solution with fixed values
        """
        
        from .mlambda import mlambda, ratio_test
        
        # Check for held ambiguities
        held_amb = np.full_like(float_amb, np.nan)
        n_held = 0
        
        for i, sat_pair in enumerate(satellites):
            if sat_pair in self.hold_ambiguities:
                held_epoch, held_value = self.hold_ambiguities[sat_pair]
                
                # Check age
                if epoch - held_epoch < self.max_age:
                    held_amb[i] = held_value
                    n_held += 1
                else:
                    # Too old, remove
                    del self.hold_ambiguities[sat_pair]
        
        # If we have held values, use them
        if n_held > 0:
            # Constrained solution
            fixed_amb = float_amb.copy()
            
            # Apply held values
            mask = ~np.isnan(held_amb)
            fixed_amb[mask] = held_amb[mask]
            
            # For new satellites, try to fix
            new_mask = np.isnan(held_amb)
            if np.any(new_mask):
                # Partial LAMBDA for new satellites
                new_float = float_amb[new_mask]
                new_cov = covariance[np.ix_(new_mask, new_mask)]
                
                try:
                    F_new, s_new = mlambda(new_float, new_cov, m=2)
                    is_fixed_new, ratio_new = ratio_test(s_new, threshold=self.ratio_threshold)
                    
                    if is_fixed_new:
                        fixed_amb[new_mask] = F_new[:, 0]
                        # Add to held ambiguities
                        new_sats = [satellites[i] for i in range(len(satellites)) if new_mask[i]]
                        for i, sat in enumerate(new_sats):
                            self.hold_ambiguities[sat] = (epoch, F_new[i, 0])
                        
                        return fixed_amb, True, ratio_new
                except Exception:
                    pass
            
            # Return with held values (partial fix)
            return fixed_amb, n_held == len(satellites), 999.0  # High ratio for held
            
        else:
            # No held values, try normal fixing
            try:
                F, s = mlambda(float_amb, covariance, m=2)
                is_fixed, ratio = ratio_test(s, threshold=self.ratio_threshold)
                
                if is_fixed:
                    # Store in hold dictionary
                    for i, sat in enumerate(satellites):
                        self.hold_ambiguities[sat] = (epoch, F[i, 0])
                    
                    return F[:, 0], True, ratio
                else:
                    return float_amb, False, ratio
                    
            except Exception:
                return float_amb, False, 0.0
    
    def reset(self):
        """Reset all states"""
        self.prev_state = None
        self.fixed_ambiguities.clear()
        self.lock_counts.clear()
        self.hold_ambiguities.clear()
        
    def handle_cycle_slip(self, satellite: str):
        """
        Handle cycle slip for a satellite
        
        Parameters:
        -----------
        satellite : str
            Satellite with cycle slip
        """
        # Remove from held ambiguities
        if satellite in self.hold_ambiguities:
            del self.hold_ambiguities[satellite]
            
        # Reset lock count
        if satellite in self.lock_counts:
            self.lock_counts[satellite] = 0
            
        # Remove from fixed ambiguities
        if satellite in self.fixed_ambiguities:
            del self.fixed_ambiguities[satellite]


class PartialAmbiguityResolution:
    """
    Partial Ambiguity Resolution (PAR)
    Fix subset of ambiguities with high confidence
    """
    
    def __init__(self, 
                 min_ratio: float = 3.0,
                 min_elevation: float = 30.0):
        """
        Initialize PAR
        
        Parameters:
        -----------
        min_ratio : float
            Minimum ratio for subset
        min_elevation : float
            Minimum elevation for fixing (degrees)
        """
        self.min_ratio = min_ratio
        self.min_elevation = min_elevation
        
    def resolve_partial(self,
                       float_amb: np.ndarray,
                       covariance: np.ndarray,
                       elevations: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Resolve partial set of ambiguities
        
        Parameters:
        -----------
        float_amb : np.ndarray
            Float ambiguities
        covariance : np.ndarray
            Covariance matrix
        elevations : np.ndarray
            Satellite elevations
            
        Returns:
        --------
        fixed_amb : np.ndarray
            Fixed ambiguities (NaN for unfixed)
        fix_mask : np.ndarray
            Boolean mask of fixed ambiguities
        ratio : float
            Ratio test value
        """
        from .mlambda import mlambda, ratio_test
        
        # Sort by elevation
        elev_order = np.argsort(elevations)[::-1]  # Highest first
        
        fixed_amb = np.full_like(float_amb, np.nan)
        fix_mask = np.zeros(len(float_amb), dtype=bool)
        best_ratio = 0.0
        
        # Try fixing subsets starting with highest elevation
        for n in range(len(float_amb), 0, -1):
            subset = elev_order[:n]
            
            # Check minimum elevation
            if elevations[subset[-1]] < self.min_elevation:
                continue
                
            # Extract subset
            sub_float = float_amb[subset]
            sub_cov = covariance[np.ix_(subset, subset)]
            
            try:
                F, s = mlambda(sub_float, sub_cov, m=2)
                is_fixed, ratio = ratio_test(s, threshold=self.min_ratio)
                
                if is_fixed:
                    # Fix this subset
                    fixed_amb[subset] = F[:, 0]
                    fix_mask[subset] = True
                    best_ratio = ratio
                    break
                    
            except Exception:
                continue
                
        return fixed_amb, fix_mask, best_ratio