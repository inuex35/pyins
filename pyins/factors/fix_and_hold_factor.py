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
Fix-and-Hold Factor for GNSS Ambiguity Resolution
==================================================

This factor implements the Fix-and-Hold strategy used in RTKLIB to improve
ambiguity resolution success rate. Once ambiguities are fixed, they are
constrained to remain fixed as long as the carrier phase observations
remain continuous (no cycle slips).

The Fix-and-Hold approach:
1. Once ambiguities are fixed with sufficient confidence, add strong constraints
2. Hold the fixed values until a cycle slip is detected
3. Allow small variations to account for noise but prevent large jumps
"""

import numpy as np
from typing import Optional, Dict, List
try:
    import gtsam
except ImportError:
    # Handle case where gtsam is imported later
    gtsam = None


class FixAndHoldFactor:
    """
    Factor that constrains previously fixed ambiguities to remain fixed
    
    This implements RTKLIB's Fix-and-Hold strategy where once ambiguities
    are reliably fixed, they are strongly constrained in subsequent epochs.
    """
    
    def __init__(self, 
                 ambiguity_key: int,
                 fixed_value: float,
                 noise_model: Optional[object] = None,  # gtsam.noiseModel.Base
                 confidence: float = 1.0):
        """
        Initialize Fix-and-Hold factor
        
        Parameters
        ----------
        ambiguity_key : int
            GTSAM key for the ambiguity variable
        fixed_value : float
            The fixed integer value for the ambiguity
        noise_model : gtsam.noiseModel.Base, optional
            Noise model for the constraint (default: very tight constraint)
        confidence : float
            Confidence level in the fixed solution (0-1)
        """
        self.ambiguity_key = ambiguity_key
        self.fixed_value = fixed_value
        self.confidence = confidence
        
        if noise_model is None:
            # Import gtsam here to avoid circular import
            import gtsam
            # RTKLIB-style: very tight constraint for fixed ambiguities
            # Once fixed, hold strongly unless cycle slip detected
            if confidence > 0.5:  # Considered fixed
                sigma = 0.001  # Very tight: 0.001 cycles (essentially fixed)
            else:
                sigma = 0.1 / confidence  # Looser for low confidence
            self.noise_model = gtsam.noiseModel.Isotropic.Sigma(1, sigma)
        else:
            self.noise_model = noise_model
    
    def create_factor(self):
        """
        Create GTSAM prior factor for the fixed ambiguity
        
        Returns
        -------
        gtsam.PriorFactorDouble
            Prior factor constraining the ambiguity to its fixed value
        """
        import gtsam
        return gtsam.PriorFactorDouble(
            self.ambiguity_key,
            self.fixed_value,
            self.noise_model
        )


class FixAndHoldManager:
    """
    Manages Fix-and-Hold constraints across multiple epochs
    
    This class tracks fixed ambiguities and maintains their constraints,
    detecting cycle slips and managing the lifecycle of fixes.
    """
    
    def __init__(self,
                 hold_threshold: float = 0.15,
                 confidence_decay: float = 0.95,
                 min_confidence: float = 0.5,
                 cycle_slip_threshold: float = 0.5):
        """
        Initialize Fix-and-Hold manager
        
        Parameters
        ----------
        hold_threshold : float
            Maximum deviation from integer to maintain hold (cycles)
        confidence_decay : float
            Factor to decay confidence each epoch (for aging fixes)
        min_confidence : float
            Minimum confidence to maintain a fix
        cycle_slip_threshold : float
            Threshold for detecting cycle slips (cycles)
        """
        self.hold_threshold = hold_threshold
        self.confidence_decay = confidence_decay
        self.min_confidence = min_confidence
        self.cycle_slip_threshold = cycle_slip_threshold
        
        # Track fixed ambiguities: {(ref_sat, other_sat): (fixed_value, confidence, epoch_fixed)}
        self.fixed_ambiguities: Dict[tuple, tuple] = {}
        self.epoch_count = 0
        
    def update_fixes(self,
                     new_fixes: Dict[tuple, float],
                     ratio: float,
                     current_floats: Dict[tuple, float]) -> Dict[tuple, tuple]:
        """
        Update fixed ambiguities with new fixes and check existing ones
        
        Parameters
        ----------
        new_fixes : dict
            Newly fixed ambiguities {(ref_sat, other_sat): fixed_value}
        ratio : float
            Ratio test value for the new fixes
        current_floats : dict
            Current float solutions for comparison
            
        Returns
        -------
        dict
            Updated fixed ambiguities to apply
        """
        self.epoch_count += 1
        fixes_to_apply = {}
        
        # Add new fixes with confidence based on ratio
        # Lower threshold to 2.0 to allow partial fixes to bootstrap
        if new_fixes and ratio > 0:
            # Confidence mapping: ratio 2.0 -> 0.3, ratio 3.0 -> 0.5, ratio 10.0 -> 1.0
            confidence = min(1.0, max(0.3, (ratio - 2.0) / 8.0))
            
            for sat_pair, fixed_val in new_fixes.items():
                if sat_pair not in self.fixed_ambiguities:
                    # New fix
                    self.fixed_ambiguities[sat_pair] = (
                        fixed_val, confidence, self.epoch_count
                    )
                    fixes_to_apply[sat_pair] = (fixed_val, confidence)
                else:
                    # Check if new fix agrees with old
                    old_fix, old_conf, _ = self.fixed_ambiguities[sat_pair]
                    if abs(fixed_val - old_fix) < self.hold_threshold:
                        # Agrees - boost confidence
                        new_conf = min(1.0, old_conf + 0.1)
                        self.fixed_ambiguities[sat_pair] = (
                            old_fix, new_conf, self.epoch_count
                        )
                        fixes_to_apply[sat_pair] = (old_fix, new_conf)
        
        # Check existing fixes for cycle slips
        to_remove = []
        for sat_pair, (fixed_val, confidence, epoch_fixed) in self.fixed_ambiguities.items():
            if sat_pair in current_floats:
                float_val = current_floats[sat_pair]
                deviation = abs(float_val - fixed_val)
                
                if deviation > self.cycle_slip_threshold:
                    # Cycle slip detected
                    to_remove.append(sat_pair)
                elif deviation < self.hold_threshold:
                    # Still valid - apply with decayed confidence
                    age = self.epoch_count - epoch_fixed
                    decayed_conf = confidence * (self.confidence_decay ** age)
                    
                    if decayed_conf >= self.min_confidence:
                        fixes_to_apply[sat_pair] = (fixed_val, decayed_conf)
                    else:
                        to_remove.append(sat_pair)
        
        # Remove invalid fixes
        for sat_pair in to_remove:
            del self.fixed_ambiguities[sat_pair]
        
        return fixes_to_apply
    
    def create_factors(self,
                      fixes: Dict[tuple, tuple],
                      ambiguity_symbols: Dict[tuple, int]) -> List:
        """
        Create Fix-and-Hold factors for the given fixes
        
        Parameters
        ----------
        fixes : dict
            Fixes to apply {(ref_sat, other_sat): (value, confidence)}
        ambiguity_symbols : dict
            Mapping from satellite pairs to GTSAM symbols
            
        Returns
        -------
        list
            List of prior factors to add to the graph
        """
        factors = []
        
        for sat_pair, (fixed_val, confidence) in fixes.items():
            if sat_pair in ambiguity_symbols:
                symbol = ambiguity_symbols[sat_pair]
                factor = FixAndHoldFactor(
                    symbol, fixed_val, confidence=confidence
                )
                factors.append(factor.create_factor())
        
        return factors
    
    def get_status(self) -> Dict:
        """
        Get current status of Fix-and-Hold manager
        
        Returns
        -------
        dict
            Status information including number of fixes, average confidence, etc.
        """
        if not self.fixed_ambiguities:
            return {
                'n_fixed': 0,
                'avg_confidence': 0.0,
                'oldest_fix_age': 0
            }
        
        confidences = [conf for _, (_, conf, _) in self.fixed_ambiguities.items()]
        ages = [self.epoch_count - epoch for _, (_, _, epoch) in self.fixed_ambiguities.items()]
        
        return {
            'n_fixed': len(self.fixed_ambiguities),
            'avg_confidence': np.mean(confidences),
            'max_confidence': np.max(confidences),
            'min_confidence': np.min(confidences),
            'oldest_fix_age': np.max(ages) if ages else 0,
            'newest_fix_age': np.min(ages) if ages else 0
        }