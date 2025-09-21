"""Zero-differenced residual computation for RTK processing.

This module computes zero-differenced residuals (observation - computed range)
with all corrections applied, similar to RTKLIB's zdres function.
Double differences are then formed from these corrected residuals.
"""

import numpy as np
from ..core.constants import CLIGHT
from ..gnss.troposphere import troposphere_correction
from ..gnss.ionosphere import ionosphere_correction
from ..geometry.elevation import compute_elevation_angle
from ..coordinate.transforms import ecef2llh


def compute_zero_diff_residual(obs_value, sat_pos, sat_clk, receiver_pos,
                              elevation_deg=None, azimuth_deg=None,
                              apply_troposphere=True, apply_ionosphere=True,
                              iono_model='simple', frequency_idx=0):
    """Compute zero-differenced residual with corrections.

    Computes: residual = observation - (range + corrections)

    Parameters
    ----------
    obs_value : float
        Observed pseudorange or carrier phase in meters
    sat_pos : array_like
        Satellite position in ECEF (meters)
    sat_clk : float
        Satellite clock bias in seconds
    receiver_pos : array_like
        Receiver position in ECEF (meters)
    elevation_deg : float, optional
        Satellite elevation angle in degrees
    azimuth_deg : float, optional
        Satellite azimuth angle in degrees
    apply_troposphere : bool
        Whether to apply tropospheric correction
    apply_ionosphere : bool
        Whether to apply ionospheric correction (L1 only)
    iono_model : str
        Ionosphere model to use
    frequency_idx : int
        Frequency index (0=L1, 1=L2, etc.)

    Returns
    -------
    float
        Zero-differenced residual in meters
    """
    # Geometric range
    geometric_range = np.linalg.norm(sat_pos[:3] - receiver_pos)

    # Initialize corrected range
    corrected_range = geometric_range

    # Apply satellite clock correction
    corrected_range -= CLIGHT * sat_clk

    # Get receiver LLH for atmospheric models
    receiver_llh = ecef2llh(receiver_pos)

    # Compute elevation if not provided
    if elevation_deg is None:
        from ..coordinate.transforms import ecef2llh
        receiver_llh_deg = np.degrees(receiver_llh)
        elevation_deg = compute_elevation_angle(sat_pos[:3], receiver_pos, receiver_llh_deg)

    # Apply tropospheric correction
    if apply_troposphere and elevation_deg > 0:
        trop_delay = troposphere_correction(elevation_deg, receiver_llh)
        corrected_range += trop_delay

    # Apply ionospheric correction (only for L1)
    if apply_ionosphere and frequency_idx == 0 and elevation_deg > 0:
        if azimuth_deg is None:
            # Simple azimuth computation (can be improved)
            azimuth_deg = 0

        iono_delay = ionosphere_correction(
            elevation_deg, receiver_llh,
            az_deg=azimuth_deg,
            model=iono_model
        )
        corrected_range += iono_delay

    # Compute residual
    residual = obs_value - corrected_range

    return residual


def form_dd_from_zero_diff(rover_residuals, base_residuals, ref_idx, other_idx):
    """Form double-differenced residual from zero-diff residuals.

    DD = (rover_other - rover_ref) - (base_other - base_ref)

    Parameters
    ----------
    rover_residuals : array_like
        Zero-diff residuals at rover for all satellites
    base_residuals : array_like
        Zero-diff residuals at base for all satellites
    ref_idx : int
        Index of reference satellite
    other_idx : int
        Index of other satellite

    Returns
    -------
    float
        Double-differenced residual in meters
    """
    # Single differences
    sd_rover = rover_residuals[other_idx] - rover_residuals[ref_idx]
    sd_base = base_residuals[other_idx] - base_residuals[ref_idx]

    # Double difference
    dd = sd_rover - sd_base

    return dd


def compute_dd_residuals_corrected(dd_data, rover_pos, base_pos,
                                  apply_troposphere=True,
                                  apply_ionosphere=True,
                                  iono_model='simple'):
    """Compute DD residuals using zero-diff-then-DD approach with corrections.

    This follows the RTKLIB approach:
    1. Compute zero-diff residuals with all corrections
    2. Form DD from corrected residuals

    Parameters
    ----------
    dd_data : dict
        DD measurement data from form_double_differences
    rover_pos : array_like
        Rover position in ECEF (meters)
    base_pos : array_like
        Base position in ECEF (meters)
    apply_troposphere : bool
        Whether to apply tropospheric correction
    apply_ionosphere : bool
        Whether to apply ionospheric correction
    iono_model : str
        Ionosphere model to use

    Returns
    -------
    dict
        Dictionary containing:
        - 'residual': DD residual in meters
        - 'residual_no_corr': DD residual without corrections
        - 'trop_correction': Applied tropospheric correction
        - 'iono_correction': Applied ionospheric correction
    """
    # Get satellite data
    ref_sat_pos = dd_data['ref_sat_pos'][:3]
    ref_sat_clk = dd_data['ref_sat_clk']
    other_sat_pos = dd_data['sat_pos'][:3]
    other_sat_clk = dd_data['sat_clk']

    # Get elevations
    elevation_ref = dd_data.get('ref_elevation', 70.0)
    elevation_other = dd_data['elevation']

    # Get frequency index
    freq_idx = dd_data.get('freq_idx', 0)

    # Reconstruct approximate observations from DD
    # This is approximate since we don't store original observations
    dd_obs = dd_data['dd_obs']

    # Compute rover LLH
    rover_llh = ecef2llh(rover_pos)
    base_llh = ecef2llh(base_pos)

    # === Without corrections (for comparison) ===
    # Simple geometric ranges
    range_rover_ref = np.linalg.norm(ref_sat_pos - rover_pos)
    range_rover_other = np.linalg.norm(other_sat_pos - rover_pos)
    range_base_ref = np.linalg.norm(ref_sat_pos - base_pos)
    range_base_other = np.linalg.norm(other_sat_pos - base_pos)

    dd_range_no_corr = (range_rover_other - range_rover_ref) - \
                       (range_base_other - range_base_ref)
    residual_no_corr = dd_obs - dd_range_no_corr

    # === With corrections (RTKLIB approach) ===
    # Compute corrected ranges for each satellite-receiver pair

    # Rover to reference satellite
    range_rover_ref_corr = np.linalg.norm(ref_sat_pos - rover_pos)
    range_rover_ref_corr -= CLIGHT * ref_sat_clk

    # Rover to other satellite
    range_rover_other_corr = np.linalg.norm(other_sat_pos - rover_pos)
    range_rover_other_corr -= CLIGHT * other_sat_clk

    # Base to reference satellite
    range_base_ref_corr = np.linalg.norm(ref_sat_pos - base_pos)
    range_base_ref_corr -= CLIGHT * ref_sat_clk

    # Base to other satellite
    range_base_other_corr = np.linalg.norm(other_sat_pos - base_pos)
    range_base_other_corr -= CLIGHT * other_sat_clk

    # Apply tropospheric corrections
    trop_correction_total = 0
    if apply_troposphere:
        trop_rover_ref = troposphere_correction(elevation_ref, rover_llh)
        trop_rover_other = troposphere_correction(elevation_other, rover_llh)
        trop_base_ref = troposphere_correction(elevation_ref, base_llh)
        trop_base_other = troposphere_correction(elevation_other, base_llh)

        range_rover_ref_corr += trop_rover_ref
        range_rover_other_corr += trop_rover_other
        range_base_ref_corr += trop_base_ref
        range_base_other_corr += trop_base_other

        # DD tropospheric correction
        trop_correction_total = (trop_rover_other - trop_rover_ref) - \
                               (trop_base_other - trop_base_ref)

    # Apply ionospheric corrections (L1 only)
    iono_correction_total = 0
    if apply_ionosphere and freq_idx == 0:
        # Calculate baseline for spatial gradient
        baseline_km = np.linalg.norm(rover_pos - base_pos) / 1000

        # Use spatial gradient model
        from ..gnss.ionosphere import compute_dd_ionosphere_correction
        iono_correction_total = compute_dd_ionosphere_correction(
            elevation_ref, elevation_other,
            rover_llh, base_llh,
            model=iono_model,
            baseline_km=baseline_km
        )

        # Apply to individual ranges for consistency
        iono_rover_ref = ionosphere_correction(elevation_ref, rover_llh, model=iono_model)
        iono_rover_other = ionosphere_correction(elevation_other, rover_llh, model=iono_model)
        # Base with gradient
        gradient_factor = 1.0 - 0.15 * (baseline_km / 10.0) / 10.0
        from ..gnss.ionosphere import simple_ionosphere_model
        iono_base_ref = simple_ionosphere_model(elevation_ref, 10.0 * gradient_factor)
        iono_base_other = simple_ionosphere_model(elevation_other, 10.0 * gradient_factor)

        range_rover_ref_corr += iono_rover_ref
        range_rover_other_corr += iono_rover_other
        range_base_ref_corr += iono_base_ref
        range_base_other_corr += iono_base_other

    # Form DD from corrected ranges
    dd_range_corr = (range_rover_other_corr - range_rover_ref_corr) - \
                    (range_base_other_corr - range_base_ref_corr)

    # Residual with corrections
    residual_corr = dd_obs - dd_range_corr

    return {
        'residual': residual_corr,
        'residual_no_corr': residual_no_corr,
        'trop_correction': trop_correction_total,
        'iono_correction': iono_correction_total,
        'dd_range_corr': dd_range_corr,
        'dd_range_no_corr': dd_range_no_corr
    }