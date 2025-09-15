"""
Ambiguity offset correction for RTK processing.

This module implements RTKLIB-style phase-bias offset correction to maintain
phase-code coherency across multiple satellites and epochs.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_ambiguity_offset(new_estimates: Dict[Tuple, float],
                              existing_estimates: Dict[Tuple, float],
                              offset_threshold: float = 0.1) -> Optional[float]:
    """
    Calculate the mean offset between new ambiguity estimates and existing values.

    This implements the RTKLIB approach: computing the average difference
    between newly calculated ambiguities (L - P/λ) and current estimates.

    Args:
        new_estimates: Dictionary of (sat, ref_sat, freq) -> new ambiguity estimate
        existing_estimates: Dictionary of (sat, ref_sat, freq) -> existing ambiguity
        offset_threshold: Minimum offset magnitude to apply correction (cycles)

    Returns:
        Mean offset in cycles, or None if no common satellites or offset too small
    """
    if not existing_estimates:
        return None

    offset_sum = 0.0
    offset_count = 0

    # Calculate differences for common satellite pairs
    for key, existing_val in existing_estimates.items():
        if key in new_estimates:
            diff = new_estimates[key] - existing_val
            offset_sum += diff
            offset_count += 1

    if offset_count == 0:
        return None

    mean_offset = offset_sum / offset_count

    # Only return significant offsets
    if abs(mean_offset) < offset_threshold:
        return None

    return mean_offset


def apply_ambiguity_offset(ambiguities: Dict[Tuple, float],
                          offset: float) -> Dict[Tuple, float]:
    """
    Apply offset correction to all ambiguities.

    Args:
        ambiguities: Dictionary of (sat, ref_sat, freq) -> ambiguity value
        offset: Offset to apply (cycles)

    Returns:
        Corrected ambiguities
    """
    corrected = {}
    for key, value in ambiguities.items():
        corrected[key] = value + offset
    return corrected


def compute_ambiguity_from_measurements(dd_carrier_cycles: float,
                                      dd_pseudorange_m: float,
                                      wavelength: float) -> float:
    """
    Compute initial ambiguity estimate from DD measurements.

    Implements: N = L - P/λ
    where L is DD carrier phase in cycles, P is DD pseudorange in meters

    Args:
        dd_carrier_cycles: Double-differenced carrier phase (cycles)
        dd_pseudorange_m: Double-differenced pseudorange (meters)
        wavelength: Carrier wavelength (meters)

    Returns:
        Ambiguity estimate in cycles
    """
    dd_pseudorange_cycles = dd_pseudorange_m / wavelength
    return dd_carrier_cycles - dd_pseudorange_cycles


def correct_ambiguity_offsets(dd_measurements: List[Dict],
                              existing_ambiguities: Dict[Tuple, float],
                              wavelengths: Dict[int, float]) -> Tuple[Dict[Tuple, float], float]:
    """
    Convenience function to apply RTKLIB-style offset correction.

    Args:
        dd_measurements: List of DD measurement dictionaries
        existing_ambiguities: Current ambiguity estimates
        wavelengths: Wavelength for each frequency index

    Returns:
        Tuple of (corrected_ambiguities, mean_offset)
    """
    # Calculate new estimates from measurements
    new_estimates = {}
    for dd in dd_measurements:
        if dd.get('dd_carrier') is not None:
            sat = dd['sat']
            ref_sat = dd['ref_sat']
            freq_idx = dd.get('freq_idx', 0)
            key = (sat, ref_sat, freq_idx)

            wavelength = dd.get('wavelength', wavelengths.get(freq_idx, 0.19))
            new_estimates[key] = compute_ambiguity_from_measurements(
                dd['dd_carrier'],
                dd['dd_obs'],
                wavelength
            )

    # Calculate and apply offset
    offset = calculate_ambiguity_offset(new_estimates, existing_ambiguities)

    if offset is not None:
        corrected = apply_ambiguity_offset(existing_ambiguities, offset)
        logger.info(f"Applied ambiguity offset correction: {offset:.3f} cycles")
        return corrected, offset
    else:
        return existing_ambiguities, 0.0