#!/usr/bin/env python3
"""
Ionosphere-free combination for dual-frequency GNSS processing.

This module provides functions to form ionosphere-free combinations
of pseudorange and carrier phase measurements using L1/L2 frequencies.
"""

import numpy as np
from ..core.constants import FREQ_L1, FREQ_L2, FREQ_L5, CLIGHT

def ionosphere_free_combination(obs_L1, obs_L2, freq1=FREQ_L1, freq2=FREQ_L2):
    """
    Form ionosphere-free linear combination of observations.

    The ionosphere-free combination eliminates first-order ionospheric delay:
    IF = (f1²*obs1 - f2²*obs2) / (f1² - f2²)

    Parameters
    ----------
    obs_L1 : float
        L1 observation (pseudorange in meters or carrier phase in cycles)
    obs_L2 : float
        L2 observation (same units as obs_L1)
    freq1 : float
        L1 frequency in Hz (default: 1575.42 MHz)
    freq2 : float
        L2 frequency in Hz (default: 1227.60 MHz)

    Returns
    -------
    float
        Ionosphere-free combination in same units as input

    Notes
    -----
    For GPS L1/L2:
    - Coefficient for L1: 2.546
    - Coefficient for L2: -1.546
    """
    f1_sq = freq1 * freq1
    f2_sq = freq2 * freq2
    denominator = f1_sq - f2_sq

    if abs(denominator) < 1e-10:
        return obs_L1  # Fallback to L1 if frequencies are identical

    # Ionosphere-free combination
    coeff_L1 = f1_sq / denominator
    coeff_L2 = -f2_sq / denominator

    return coeff_L1 * obs_L1 + coeff_L2 * obs_L2


def ionosphere_free_variance(sigma_L1, sigma_L2, freq1=FREQ_L1, freq2=FREQ_L2):
    """
    Calculate variance of ionosphere-free combination.

    Parameters
    ----------
    sigma_L1 : float
        Standard deviation of L1 observation
    sigma_L2 : float
        Standard deviation of L2 observation
    freq1 : float
        L1 frequency in Hz
    freq2 : float
        L2 frequency in Hz

    Returns
    -------
    float
        Standard deviation of ionosphere-free combination

    Notes
    -----
    The variance is amplified by approximately 3x compared to single frequency.
    """
    f1_sq = freq1 * freq1
    f2_sq = freq2 * freq2
    denominator = f1_sq - f2_sq

    if abs(denominator) < 1e-10:
        return sigma_L1

    coeff_L1 = f1_sq / denominator
    coeff_L2 = f2_sq / denominator

    # Error propagation
    var_IF = (coeff_L1 * sigma_L1) ** 2 + (coeff_L2 * sigma_L2) ** 2
    return np.sqrt(var_IF)


def wide_lane_combination(L1_cycles, L2_cycles, C1_meters, C2_meters,
                         freq1=FREQ_L1, freq2=FREQ_L2):
    """
    Form wide-lane combination for ambiguity resolution.

    Wide-lane has longer wavelength (~86cm) making ambiguity resolution easier.

    Parameters
    ----------
    L1_cycles : float
        L1 carrier phase in cycles
    L2_cycles : float
        L2 carrier phase in cycles
    C1_meters : float
        L1 pseudorange in meters
    C2_meters : float
        L2 pseudorange in meters
    freq1 : float
        L1 frequency in Hz
    freq2 : float
        L2 frequency in Hz

    Returns
    -------
    tuple
        (wide_lane_phase_cycles, wide_lane_wavelength_meters)
    """
    # Wide-lane frequency and wavelength
    f_WL = freq1 - freq2
    lambda_WL = CLIGHT / f_WL

    # Convert carrier phases to meters
    lambda1 = CLIGHT / freq1
    lambda2 = CLIGHT / freq2
    L1_meters = L1_cycles * lambda1
    L2_meters = L2_cycles * lambda2

    # Wide-lane phase combination
    WL_phase_meters = (freq1 * L1_meters - freq2 * L2_meters) / (freq1 - freq2)
    WL_phase_cycles = WL_phase_meters / lambda_WL

    # Wide-lane ambiguity from Melbourne-Wübbena combination
    MW_combination = (freq1 * L1_meters - freq2 * L2_meters) / (freq1 - freq2) - \
                     (freq1 * C1_meters + freq2 * C2_meters) / (freq1 + freq2)
    WL_ambiguity = MW_combination / lambda_WL

    return WL_phase_cycles, lambda_WL, WL_ambiguity


def narrow_lane_combination(L1_cycles, L2_cycles, freq1=FREQ_L1, freq2=FREQ_L2):
    """
    Form narrow-lane combination.

    Narrow-lane has shorter wavelength (~10.7cm) but less noise.

    Parameters
    ----------
    L1_cycles : float
        L1 carrier phase in cycles
    L2_cycles : float
        L2 carrier phase in cycles
    freq1 : float
        L1 frequency in Hz
    freq2 : float
        L2 frequency in Hz

    Returns
    -------
    tuple
        (narrow_lane_phase_cycles, narrow_lane_wavelength_meters)
    """
    # Narrow-lane frequency and wavelength
    f_NL = freq1 + freq2
    lambda_NL = CLIGHT / f_NL

    # Convert to meters
    lambda1 = CLIGHT / freq1
    lambda2 = CLIGHT / freq2
    L1_meters = L1_cycles * lambda1
    L2_meters = L2_cycles * lambda2

    # Narrow-lane phase
    NL_phase_meters = (freq1 * L1_meters + freq2 * L2_meters) / (freq1 + freq2)
    NL_phase_cycles = NL_phase_meters / lambda_NL

    return NL_phase_cycles, lambda_NL


def geometry_free_combination(L1_cycles, L2_cycles, freq1=FREQ_L1, freq2=FREQ_L2):
    """
    Form geometry-free combination for ionosphere estimation.

    Geometry-free combination eliminates geometric range and clock errors,
    leaving only ionospheric delay (and ambiguities).

    Parameters
    ----------
    L1_cycles : float
        L1 carrier phase in cycles
    L2_cycles : float
        L2 carrier phase in cycles
    freq1 : float
        L1 frequency in Hz
    freq2 : float
        L2 frequency in Hz

    Returns
    -------
    float
        Geometry-free combination in meters (proportional to TEC)
    """
    # Convert to meters
    lambda1 = CLIGHT / freq1
    lambda2 = CLIGHT / freq2
    L1_meters = L1_cycles * lambda1
    L2_meters = L2_cycles * lambda2

    # Geometry-free: L1 - L2
    # This is proportional to ionospheric delay
    return L1_meters - L2_meters


def get_ionosphere_free_coefficients(freq1=FREQ_L1, freq2=FREQ_L2):
    """
    Get coefficients for ionosphere-free combination.

    Parameters
    ----------
    freq1 : float
        First frequency in Hz
    freq2 : float
        Second frequency in Hz

    Returns
    -------
    tuple
        (coeff1, coeff2) coefficients for IF = coeff1*obs1 + coeff2*obs2
    """
    f1_sq = freq1 * freq1
    f2_sq = freq2 * freq2
    denominator = f1_sq - f2_sq

    coeff1 = f1_sq / denominator
    coeff2 = -f2_sq / denominator

    return coeff1, coeff2


# Pre-computed coefficients for common frequency pairs
IF_COEFFS_L1_L2 = get_ionosphere_free_coefficients(FREQ_L1, FREQ_L2)  # (2.546, -1.546)
IF_COEFFS_L1_L5 = get_ionosphere_free_coefficients(FREQ_L1, FREQ_L5)  # (2.260, -1.260)