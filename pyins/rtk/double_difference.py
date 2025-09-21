#!/usr/bin/env python3
"""
Double difference formation following RTKLIB-py approach.
Computes zero-difference residuals first, then forms double differences.
"""

import numpy as np
from pyins.core.constants import CLIGHT
from pyins.gnss.satposs import satposs
import logging

logger = logging.getLogger(__name__)


def compute_zero_diff_residual(obs, sat_pos, sat_clk, receiver_pos, freq_idx=0):
    """
    Compute zero-difference residual for carrier phase like RTKLIB-py.

    RTKLIB: y = L * c/f - r
    where:
    - L: carrier phase observation (cycles)
    - c: speed of light
    - f: frequency
    - r: geometric range with clock corrections

    Returns:
        residual_m: Zero-difference residual in meters
        residual_cycles: Zero-difference residual in cycles
        range_m: Geometric range in meters
    """
    # Get frequency (GPS L1 default)
    if hasattr(obs, 'freq') and obs.freq[freq_idx] > 0:
        freq = obs.freq[freq_idx]
    else:
        freq = 1575.42e6  # GPS L1

    wavelength = CLIGHT / freq

    # Get carrier phase observation (cycles)
    cp_cycles = obs.L[freq_idx]
    if cp_cycles == 0:
        return None, None, None

    # Convert to meters
    cp_meters = cp_cycles * wavelength

    # Compute geometric range
    range_m = np.linalg.norm(sat_pos - receiver_pos)

    # Apply satellite clock correction
    range_m += CLIGHT * sat_clk

    # Zero-difference residual (meters)
    residual_m = cp_meters - range_m
    residual_cycles = residual_m / wavelength

    return residual_m, residual_cycles, range_m


def form_double_differences(rover_obs, base_obs, nav_data, rover_time,
                           base_position, base_llh_deg,
                           rover_position=None,
                           use_systems=None, use_frequencies=None,
                           cutoff_angle=10.0,
                           base_obs_list=None, base_obs_index=None):
    """
    Form double differences using RTKLIB-py approach.

    Steps:
    1. Compute satellite positions for rover and base times
    2. Compute zero-difference residuals for each satellite
    3. Form single differences between rover and base
    4. Form double differences using reference satellite

    Parameters
    ----------
    rover_obs : list
        Rover observations
    base_obs : list
        Base observations
    nav_data : dict
        Navigation data
    rover_time : float
        Rover observation time (GPS seconds)
    base_position : array
        Base station ECEF position
    base_llh_deg : array
        Base station LLH position in degrees
    rover_position : array, optional
        Rover ECEF position (for satellite selection)
    use_systems : list, optional
        GNSS systems to use (default: ['G'])
    use_frequencies : list, optional
        Frequency indices to use (default: [0])
    cutoff_angle : float
        Elevation cutoff angle in degrees

    Returns
    -------
    list
        List of DD measurements with residuals
    """

    if use_systems is None:
        use_systems = ['G']
    if use_frequencies is None:
        use_frequencies = [0]

    # Get base observation time
    base_time = base_obs[0].time if base_obs else rover_time

    # Compute satellite positions at rover and base times
    rover_sat_pos, rover_sat_clk, _, _ = satposs(rover_obs, nav_data, rover_time)
    base_sat_pos, base_sat_clk, _, _ = satposs(base_obs, nav_data, base_time)

    # If no rover position provided, use base position as approximation
    if rover_position is None:
        rover_position = base_position + np.array([100, 100, 10])  # Rough offset

    dd_measurements = []

    # Process each frequency
    for freq_idx in use_frequencies:
        # Compute zero-difference residuals for all satellites
        rover_residuals = {}
        base_residuals = {}

        # Process rover observations
        for i, obs in enumerate(rover_obs):
            # Check system
            system = 'G' if obs.system == 1 else 'E' if obs.system == 4 else 'C' if obs.system == 5 else 'R'
            if system not in use_systems:
                continue

            sat_id = f"{system}{obs.sat:02d}"

            # Compute residual at rover position
            residual_m, residual_cycles, range_m = compute_zero_diff_residual(
                obs, rover_sat_pos[i], rover_sat_clk[i], rover_position, freq_idx
            )

            if residual_cycles is not None:
                rover_residuals[sat_id] = {
                    'residual_m': residual_m,
                    'residual_cycles': residual_cycles,
                    'range_m': range_m,
                    'sat_pos': rover_sat_pos[i],
                    'sat_clk': rover_sat_clk[i],
                    'sat_num': obs.sat,
                    'obs': obs
                }

        # Process base observations
        for i, obs in enumerate(base_obs):
            # Check system
            system = 'G' if obs.system == 1 else 'E' if obs.system == 4 else 'C' if obs.system == 5 else 'R'
            if system not in use_systems:
                continue

            sat_id = f"{system}{obs.sat:02d}"

            # Compute residual at base position
            residual_m, residual_cycles, range_m = compute_zero_diff_residual(
                obs, base_sat_pos[i], base_sat_clk[i], base_position, freq_idx
            )

            if residual_cycles is not None:
                base_residuals[sat_id] = {
                    'residual_m': residual_m,
                    'residual_cycles': residual_cycles,
                    'range_m': range_m,
                    'sat_pos': base_sat_pos[i],
                    'sat_clk': base_sat_clk[i],
                    'sat_num': obs.sat,
                    'obs': obs
                }

        # Find common satellites
        common_sats = set(rover_residuals.keys()) & set(base_residuals.keys())

        if len(common_sats) < 2:
            continue  # Need at least 2 satellites for DD

        # Select reference satellite (highest elevation or first available)
        # For simplicity, use first satellite
        ref_sat_id = sorted(common_sats)[0]
        ref_rover = rover_residuals[ref_sat_id]
        ref_base = base_residuals[ref_sat_id]

        # Form double differences
        for sat_id in common_sats:
            if sat_id == ref_sat_id:
                continue

            rover_data = rover_residuals[sat_id]
            base_data = base_residuals[sat_id]

            # Single difference residuals (rover - base)
            sd_residual_ref = ref_rover['residual_cycles'] - ref_base['residual_cycles']
            sd_residual_sat = rover_data['residual_cycles'] - base_data['residual_cycles']

            # Double difference residual
            dd_residual_cycles = sd_residual_sat - sd_residual_ref

            # Get frequency and wavelength
            if hasattr(rover_data['obs'], 'freq') and rover_data['obs'].freq[freq_idx] > 0:
                freq = rover_data['obs'].freq[freq_idx]
            else:
                freq = 1575.42e6
            wavelength = CLIGHT / freq

            # Also compute DD pseudorange
            rover_pr = rover_data['obs'].P[freq_idx]
            base_pr = base_data['obs'].P[freq_idx]
            ref_rover_pr = ref_rover['obs'].P[freq_idx]
            ref_base_pr = ref_base['obs'].P[freq_idx]

            if rover_pr > 0 and base_pr > 0 and ref_rover_pr > 0 and ref_base_pr > 0:
                dd_pr = (rover_pr - ref_rover_pr) - (base_pr - ref_base_pr)
            else:
                dd_pr = None

            # Get original carrier phase observations for verification
            rover_cp = rover_data['obs'].L[freq_idx]
            base_cp = base_data['obs'].L[freq_idx]
            ref_rover_cp = ref_rover['obs'].L[freq_idx]
            ref_base_cp = ref_base['obs'].L[freq_idx]

            # DD carrier phase (raw observation)
            dd_cp_obs = (rover_cp - ref_rover_cp) - (base_cp - ref_base_cp)

            dd_measurements.append({
                'sat': rover_data['sat_num'],
                'ref_sat': ref_rover['sat_num'],
                'sys': sat_id[0],
                'freq_idx': freq_idx,
                'dd_obs': dd_pr,  # DD pseudorange in meters
                'dd_carrier': dd_cp_obs,  # DD carrier phase in cycles (raw observation)
                'dd_residual': dd_residual_cycles,  # DD residual in cycles (after removing range)
                'wavelength': wavelength,
                'sat_pos': rover_data['sat_pos'],
                'ref_sat_pos': ref_rover['sat_pos'],
                # Add base satellite positions for DDPseudorangeFactor v2.0+
                'base_sat_pos': base_data['sat_pos'],  # Base time satellite position
                'base_ref_sat_pos': ref_base['sat_pos'],  # Base time ref satellite position
                'base_sat_clk': base_data['sat_clk'],  # Base time satellite clock
                'base_ref_sat_clk': ref_base['sat_clk'],  # Base time ref satellite clock
                'rover_residual': rover_data['residual_cycles'],
                'base_residual': base_data['residual_cycles'],
                'ref_rover_residual': ref_rover['residual_cycles'],
                'ref_base_residual': ref_base['residual_cycles'],
                # Also include raw carrier phase observations for DDCarrierPhaseFactor
                'rover_cp_ref': ref_rover_cp,  # Rover obs to ref sat (cycles)
                'rover_cp_other': rover_cp,     # Rover obs to other sat (cycles)
                'base_cp_ref': ref_base_cp,     # Base obs to ref sat (cycles)
                'base_cp_other': base_cp        # Base obs to other sat (cycles)
            })

    return dd_measurements