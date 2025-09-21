#!/usr/bin/env python3
"""
Carrier phase ambiguity initialization for RTK/DD positioning.

This module provides functions for initializing and managing carrier phase
ambiguities in double-difference GNSS processing.
"""

import numpy as np
import gtsam
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def initialize_dd_ambiguity(dd_pseudorange: float, dd_carrier_phase: float,
                           wavelength: float) -> float:
    """
    Initialize double-difference carrier phase ambiguity.

    Uses the measurement equation: CP = ρ/λ - N
    Therefore: N = ρ/λ - CP ≈ PR/λ - CP

    Parameters:
    -----------
    dd_pseudorange : float
        Double-difference pseudorange measurement in meters
    dd_carrier_phase : float
        Double-difference carrier phase measurement in cycles
    wavelength : float
        Carrier wavelength in meters

    Returns:
    --------
    float
        Initial ambiguity estimate in cycles
    """
    # Convert pseudorange to cycles
    pr_cycles = dd_pseudorange / wavelength

    # Calculate ambiguity: N = PR/λ - CP
    ambiguity = pr_cycles - dd_carrier_phase

    return ambiguity


def round_to_integer(ambiguity: float, confidence_threshold: float = 0.2) -> Tuple[float, bool]:
    """
    Round ambiguity to nearest integer if confidence is high enough.

    Parameters:
    -----------
    ambiguity : float
        Float ambiguity value
    confidence_threshold : float
        Maximum distance from integer to consider rounding (default 0.2 cycles)

    Returns:
    --------
    Tuple[float, bool]
        (rounded_value, was_rounded)
    """
    nearest_int = round(ambiguity)
    distance = abs(ambiguity - nearest_int)

    if distance < confidence_threshold:
        return nearest_int, True
    else:
        return ambiguity, False


def check_cycle_slip(prev_ambiguity: float, curr_ambiguity: float,
                    threshold: float = 0.5) -> bool:
    """
    Check for cycle slip by comparing consecutive ambiguity estimates.

    Parameters:
    -----------
    prev_ambiguity : float
        Previous epoch ambiguity estimate
    curr_ambiguity : float
        Current epoch ambiguity estimate
    threshold : float
        Threshold for detecting cycle slip (default 0.5 cycles)

    Returns:
    --------
    bool
        True if cycle slip detected
    """
    diff = abs(curr_ambiguity - prev_ambiguity)
    return diff > threshold


class AmbiguityManager:
    """
    Manages carrier phase ambiguities for DD processing WITHOUT continuity constraints.

    This version does NOT add BetweenFactor constraints between epochs.
    Each epoch's ambiguity is treated independently with only a prior constraint.
    """

    def __init__(self, prior_sigma: float = 100.0):
        """
        Initialize ambiguity manager without continuity constraints.

        Parameters:
        -----------
        prior_sigma : float
            Standard deviation for ambiguity prior (cycles)
        """
        self.prior_sigma = prior_sigma
        self.ambiguity_tracking = {}  # (sat, ref_sat, freq) -> current_key
        self.ambiguity_history = {}   # (sat, ref_sat, freq) -> list of values
        self.cycle_slip_count = {}    # (sat, ref_sat, freq) -> count

    def process_ambiguity(self, dd_data: Dict, wavelength: float, epoch_idx: int,
                         graph: gtsam.NonlinearFactorGraph,
                         initial_values: gtsam.Values,
                         N: callable) -> gtsam.Symbol:
        """
        Process ambiguity for a DD measurement WITHOUT continuity constraints.

        Parameters:
        -----------
        dd_data : dict
            DD measurement data with keys 'sat', 'ref_sat', 'dd_obs', 'dd_carrier'
        wavelength : float
            Carrier wavelength in meters
        epoch_idx : int
            Current epoch index
        graph : gtsam.NonlinearFactorGraph
            Factor graph to add constraints to
        initial_values : gtsam.Values
            Initial values container
        N : callable
            Symbol generator for ambiguities N(sat, ref_sat, freq, epoch)

        Returns:
        --------
        gtsam.Symbol
            Ambiguity symbol for this measurement
        """
        # Extract data
        sat = dd_data['sat']
        ref_sat = dd_data['ref_sat']
        freq_idx = dd_data.get('freq_idx', 0)

        # Create ambiguity key
        ambiguity_id = N(sat, ref_sat, freq_idx, epoch_idx)
        ambiguity_key_base = (sat, ref_sat, freq_idx)

        # Calculate current ambiguity estimate
        current_ambiguity = initialize_dd_ambiguity(
            dd_data['dd_obs'],
            dd_data['dd_carrier'],
            wavelength
        )

        if ambiguity_key_base not in self.ambiguity_tracking:
            # New ambiguity - first observation of this satellite pair
            logger.debug(f"New ambiguity for {ambiguity_key_base}: {current_ambiguity:.3f} cycles")

            # Initialize tracking
            self.ambiguity_tracking[ambiguity_key_base] = ambiguity_id
            self.ambiguity_history[ambiguity_key_base] = [current_ambiguity]
            self.cycle_slip_count[ambiguity_key_base] = 0

            # Add to initial values
            initial_values.insert(ambiguity_id, current_ambiguity)

            # Add prior constraint
            prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([self.prior_sigma]))
            graph.push_back(gtsam.PriorFactorDouble(
                ambiguity_id, current_ambiguity, prior_noise
            ))

        else:
            # Continuing existing ambiguity
            prev_key = self.ambiguity_tracking[ambiguity_key_base]
            prev_values = self.ambiguity_history[ambiguity_key_base]

            # Check for cycle slip (for logging purposes only)
            if len(prev_values) > 0:
                prev_ambiguity = prev_values[-1]
                if check_cycle_slip(prev_ambiguity, current_ambiguity):
                    logger.warning(f"Cycle slip detected for {ambiguity_key_base}: "
                                 f"prev={prev_ambiguity:.3f}, curr={current_ambiguity:.3f}")
                    self.cycle_slip_count[ambiguity_key_base] += 1

            # IMPORTANT CHANGE: NO CONTINUITY CONSTRAINT
            # Each epoch gets its own independent ambiguity with only a prior
            initial_values.insert(ambiguity_id, current_ambiguity)

            # Add prior constraint (same for all epochs, no continuity)
            prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([self.prior_sigma]))
            graph.push_back(gtsam.PriorFactorDouble(
                ambiguity_id, current_ambiguity, prior_noise
            ))

            logger.debug(f"Ambiguity for {ambiguity_key_base} epoch {epoch_idx}: "
                        f"{current_ambiguity:.3f} cycles (no continuity constraint)")

            # Update tracking
            self.ambiguity_tracking[ambiguity_key_base] = ambiguity_id
            self.ambiguity_history[ambiguity_key_base].append(current_ambiguity)

            # Keep history limited
            if len(self.ambiguity_history[ambiguity_key_base]) > 10:
                self.ambiguity_history[ambiguity_key_base].pop(0)

        return ambiguity_id

    def get_statistics(self) -> Dict:
        """
        Get ambiguity statistics.

        Returns:
        --------
        dict
            Statistics including cycle slip counts and ambiguity convergence
        """
        stats = {
            'total_ambiguities': len(self.ambiguity_tracking),
            'total_cycle_slips': sum(self.cycle_slip_count.values()),
            'ambiguity_convergence': {}
        }

        # Check convergence (how close to integers)
        for key, history in self.ambiguity_history.items():
            if history:
                latest = history[-1]
                distance_to_int = abs(latest - round(latest))
                stats['ambiguity_convergence'][key] = {
                    'value': latest,
                    'nearest_int': round(latest),
                    'distance': distance_to_int,
                    'converged': distance_to_int < 0.1
                }

        return stats


def validate_ambiguity_fix(float_solution: np.ndarray, fixed_solution: np.ndarray,
                           ratio_threshold: float = 3.0) -> bool:
    """
    Validate ambiguity fixing using ratio test (simplified LAMBDA-style).

    Parameters:
    -----------
    float_solution : np.ndarray
        Float ambiguity solution
    fixed_solution : np.ndarray
        Fixed (integer) ambiguity solution
    ratio_threshold : float
        Minimum ratio for validation (default 3.0)

    Returns:
    --------
    bool
        True if fix is validated
    """
    # Simplified ratio test
    # In practice, would use second-best integer solution
    residual_float = np.linalg.norm(float_solution - fixed_solution)

    # Simple heuristic: check if close enough to integer
    if residual_float < 0.15:  # Within 0.15 cycles
        return True

    return False