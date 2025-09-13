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
Carrier phase processor that converts accumulated phase to relative phase
Similar to RTKLIB's approach
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class CarrierPhaseProcessor:
    """
    Processes carrier phase measurements by converting accumulated phase
    to relative phase (similar to RTKLIB)
    """
    
    def __init__(self):
        """Initialize the carrier phase processor"""
        # Store initial phase for each satellite-frequency pair
        # Key: (sat, freq_idx), Value: initial phase in cycles
        self.initial_phases: Dict[Tuple[int, int], float] = {}
        
        # Store previous phase for cycle slip detection
        # Key: (sat, freq_idx), Value: previous phase in cycles
        self.prev_phases: Dict[Tuple[int, int], float] = {}
        
        # Cycle slip threshold (cycles)
        # Set very high to avoid false detections with accumulated phase
        # Real cycle slips will be handled separately
        self.cycle_slip_threshold = 1e9
        
        # Track which satellites have been initialized
        self.initialized_sats = set()
        
    def process_observation(self, sat: int, freq_idx: int, 
                           accumulated_phase: float,
                           reset: bool = False) -> Optional[float]:
        """
        Convert accumulated carrier phase to relative phase
        
        Args:
            sat: Satellite ID
            freq_idx: Frequency index (0=L1, 1=L2, etc.)
            accumulated_phase: Accumulated carrier phase in cycles from RINEX
            reset: Force reset of initial phase (for cycle slip)
            
        Returns:
            Relative phase in cycles (from initial epoch), or None if reset
        """
        key = (sat, freq_idx)
        
        # Check for cycle slip
        if key in self.prev_phases and not reset:
            phase_diff = accumulated_phase - self.prev_phases[key]
            if abs(phase_diff) > self.cycle_slip_threshold:
                logger.warning(f"Cycle slip detected for sat {sat} freq {freq_idx}: "
                             f"diff = {phase_diff:.1f} cycles")
                reset = True
        
        # Initialize or reset initial phase
        if key not in self.initial_phases or reset:
            self.initial_phases[key] = accumulated_phase
            self.prev_phases[key] = accumulated_phase
            logger.debug(f"Initialized sat {sat} freq {freq_idx} with phase {accumulated_phase:.1f}")
            return 0.0  # First epoch is reference (0 relative phase)
        
        # Calculate relative phase
        relative_phase = accumulated_phase - self.initial_phases[key]
        
        # Update previous phase for next cycle slip check
        self.prev_phases[key] = accumulated_phase
        
        return relative_phase
    
    def process_epoch(self, observations: List) -> Dict:
        """
        Process all observations in an epoch
        
        Args:
            observations: List of observation objects with L and P attributes
            
        Returns:
            Dictionary with processed phases for each satellite
        """
        processed = {}
        
        for obs in observations:
            if not hasattr(obs, 'L') or not hasattr(obs, 'P'):
                continue
                
            sat = obs.sat
            
            # Process each frequency
            if isinstance(obs.L, (list, np.ndarray)):
                for freq_idx in range(len(obs.L)):
                    if obs.L[freq_idx] != 0:  # Valid phase measurement
                        accumulated_phase = obs.L[freq_idx]
                        relative_phase = self.process_observation(
                            sat, freq_idx, accumulated_phase
                        )
                        
                        if relative_phase is not None:
                            key = f"{sat}_L{freq_idx+1}"
                            processed[key] = {
                                'accumulated': accumulated_phase,
                                'relative': relative_phase,
                                'pseudorange': obs.P[freq_idx] if isinstance(obs.P, (list, np.ndarray)) else obs.P
                            }
            else:
                # Single frequency
                if obs.L != 0:
                    relative_phase = self.process_observation(sat, 0, obs.L)
                    if relative_phase is not None:
                        processed[f"{sat}_L1"] = {
                            'accumulated': obs.L,
                            'relative': relative_phase,
                            'pseudorange': obs.P
                        }
        
        return processed
    
    def reset_satellite(self, sat: int, freq_idx: Optional[int] = None):
        """
        Reset initial phase for a satellite (e.g., after loss of lock)
        
        Args:
            sat: Satellite ID
            freq_idx: Frequency index, or None to reset all frequencies
        """
        if freq_idx is not None:
            key = (sat, freq_idx)
            if key in self.initial_phases:
                del self.initial_phases[key]
                del self.prev_phases[key]
                logger.info(f"Reset sat {sat} freq {freq_idx}")
        else:
            # Reset all frequencies for this satellite
            keys_to_remove = [k for k in self.initial_phases.keys() if k[0] == sat]
            for key in keys_to_remove:
                del self.initial_phases[key]
                del self.prev_phases[key]
            logger.info(f"Reset all frequencies for sat {sat}")
    
    def get_float_ambiguity(self, sat: int, freq_idx: int,
                           relative_phase: float, 
                           pseudorange: float,
                           wavelength: float) -> float:
        """
        Calculate float ambiguity using relative phase
        
        Args:
            sat: Satellite ID
            freq_idx: Frequency index
            relative_phase: Relative carrier phase in cycles
            pseudorange: Pseudorange in meters
            wavelength: Carrier wavelength in meters
            
        Returns:
            Float ambiguity in cycles
        """
        # Convert pseudorange to cycles
        pseudorange_cycles = pseudorange / wavelength
        
        # Float ambiguity = phase - range (both in cycles)
        # Using relative phase makes this manageable
        ambiguity = relative_phase - pseudorange_cycles
        
        return ambiguity
    
    def get_stats(self) -> Dict:
        """Get statistics about processed satellites"""
        return {
            'n_initialized': len(self.initial_phases),
            'satellites': list(set(k[0] for k in self.initial_phases.keys())),
            'frequencies': list(set(k[1] for k in self.initial_phases.keys()))
        }