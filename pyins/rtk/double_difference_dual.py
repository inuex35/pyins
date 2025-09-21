#!/usr/bin/env python3
"""
Enhanced double difference formation with dual-frequency support.

This module extends the basic DD formation to support L1/L2 dual-frequency
processing with ionosphere-free combinations.
"""

import numpy as np
from collections import defaultdict
import logging

from ..core.constants import CLIGHT, FREQ_L1, FREQ_L2, SYS_GPS, sat2sys
from ..gnss.ephemeris import satpos
from ..gnss.ionosphere_free import (
    ionosphere_free_combination,
    ionosphere_free_variance,
    get_ionosphere_free_coefficients
)
from .double_difference import form_double_differences

logger = logging.getLogger(__name__)


def form_double_differences_dual(rover_obs, base_obs, nav_data, gps_time,
                                 reference_ecef=None, reference_llh=None,
                                 use_systems=['G', 'E', 'J', 'C'],
                                 cutoff_angle=10.0,
                                 rover_position=None,
                                 base_obs_list=None,
                                 base_obs_index=None,
                                 use_ionosphere_free=True,
                                 frequencies=['L1', 'L2']):
    """
    Form double differences with dual-frequency support.

    Parameters
    ----------
    rover_obs : list
        Rover observations
    base_obs : list
        Base observations (can be None if using base_obs_list)
    nav_data : dict
        Navigation data
    gps_time : float
        GPS time of epoch
    reference_ecef : array-like
        Base station ECEF position
    reference_llh : array-like
        Base station LLH position [deg, deg, m]
    use_systems : list
        GNSS systems to use
    cutoff_angle : float
        Elevation cutoff angle in degrees
    rover_position : array-like
        Approximate rover position for elevation calculation
    base_obs_list : list
        List of base observations for interpolation
    base_obs_index : int
        Index for base observation selection
    use_ionosphere_free : bool
        If True, form ionosphere-free combinations
    frequencies : list
        Frequencies to use ['L1', 'L2'] or ['L1', 'L5']

    Returns
    -------
    list
        List of DD measurements with dual-frequency data
    """
    # First get single-frequency DD measurements
    dd_measurements = form_double_differences(
        rover_obs, base_obs, nav_data, gps_time,
        reference_ecef, reference_llh,
        rover_position,
        use_systems, None,  # use_systems, use_frequencies
        cutoff_angle,
        base_obs_list, base_obs_index
    )

    if not dd_measurements or not use_ionosphere_free:
        return dd_measurements

    # Enhance with dual-frequency data
    enhanced_dd = []

    for dd in dd_measurements:
        # Check if we have L2 data
        has_l2_pr = False
        has_l2_cp = False

        # Get observation attributes based on frequency request
        if 'L2' in frequencies:
            # Check for L2 pseudorange
            # Note: These object references might not be stored in the DD dict
            # We need to check the observation objects from the original data
            rover_obs_ref = dd.get('rover_obs_ref_obj')
            rover_obs_other = dd.get('rover_obs_other_obj')
            base_obs_ref = dd.get('base_obs_ref_obj')
            base_obs_other = dd.get('base_obs_other_obj')

            # Debug: Print what we have
            if rover_obs_ref is None:
                # Try to get from the original observation lists if not stored
                # This is a limitation - we'd need to modify form_double_differences
                # to store the observation objects
                pass

            # Check if L2 data exists (try different codes)
            l2_codes = ['C2C', 'C2W', 'C2L', 'C2P']
            l2_phase_codes = ['L2C', 'L2W', 'L2L', 'L2P']

            # Find available L2 pseudorange
            for code in l2_codes:
                if (rover_obs_ref and hasattr(rover_obs_ref, code) and
                    rover_obs_other and hasattr(rover_obs_other, code) and
                    base_obs_ref and hasattr(base_obs_ref, code) and
                    base_obs_other and hasattr(base_obs_other, code)):

                    # Get L2 measurements
                    rover_pr_ref_l2 = getattr(rover_obs_ref, code)
                    rover_pr_other_l2 = getattr(rover_obs_other, code)
                    base_pr_ref_l2 = getattr(base_obs_ref, code)
                    base_pr_other_l2 = getattr(base_obs_other, code)

                    if all(v > 0 for v in [rover_pr_ref_l2, rover_pr_other_l2,
                                           base_pr_ref_l2, base_pr_other_l2]):
                        has_l2_pr = True
                        dd['l2_pr_code'] = code

                        # Form L2 DD pseudorange
                        sd_rover_l2 = rover_pr_other_l2 - rover_pr_ref_l2
                        sd_base_l2 = base_pr_other_l2 - base_pr_ref_l2
                        dd['dd_obs_l2'] = sd_rover_l2 - sd_base_l2
                        break

            # Find available L2 carrier phase
            for code in l2_phase_codes:
                if (rover_obs_ref and hasattr(rover_obs_ref, code) and
                    rover_obs_other and hasattr(rover_obs_other, code) and
                    base_obs_ref and hasattr(base_obs_ref, code) and
                    base_obs_other and hasattr(base_obs_other, code)):

                    rover_cp_ref_l2 = getattr(rover_obs_ref, code)
                    rover_cp_other_l2 = getattr(rover_obs_other, code)
                    base_cp_ref_l2 = getattr(base_obs_ref, code)
                    base_cp_other_l2 = getattr(base_obs_other, code)

                    if all(v != 0 for v in [rover_cp_ref_l2, rover_cp_other_l2,
                                            base_cp_ref_l2, base_cp_other_l2]):
                        has_l2_cp = True
                        dd['l2_cp_code'] = code

                        # Form L2 DD carrier phase
                        sd_rover_cp_l2 = rover_cp_other_l2 - rover_cp_ref_l2
                        sd_base_cp_l2 = base_cp_ref_l2 - base_cp_other_l2
                        dd['dd_carrier_l2'] = sd_rover_cp_l2 - sd_base_cp_l2
                        dd['wavelength_l2'] = CLIGHT / FREQ_L2
                        break

        # If we have dual-frequency data, compute ionosphere-free combinations
        if has_l2_pr:
            # Ionosphere-free pseudorange
            dd['dd_obs_if'] = ionosphere_free_combination(
                dd['dd_obs'], dd['dd_obs_l2'], FREQ_L1, FREQ_L2
            )

            # Update noise model for IF combination
            # Note: IF combination has ~3x worse noise
            coeff1, coeff2 = get_ionosphere_free_coefficients(FREQ_L1, FREQ_L2)
            dd['if_coeff_l1'] = coeff1
            dd['if_coeff_l2'] = coeff2

            # Mark as dual-frequency
            dd['dual_freq'] = True
            dd['freq_type'] = 'L1/L2'

            if has_l2_cp and dd.get('dd_carrier'):
                # Ionosphere-free carrier phase
                dd['dd_carrier_if'] = ionosphere_free_combination(
                    dd['dd_carrier'], dd['dd_carrier_l2'], FREQ_L1, FREQ_L2
                )

                # IF wavelength (not physical, but for scaling)
                # IF carrier is in cycles of L1, needs proper scaling
                dd['wavelength_if'] = CLIGHT / FREQ_L1  # Approximate

        else:
            dd['dual_freq'] = False
            dd['freq_type'] = 'L1'

        enhanced_dd.append(dd)

    # Log dual-frequency statistics
    dual_freq_count = sum(1 for dd in enhanced_dd if dd.get('dual_freq'))
    logger.info(f"  Dual-frequency DD: {dual_freq_count}/{len(enhanced_dd)}")

    return enhanced_dd


def form_wide_lane_dd(dd_measurement):
    """
    Form wide-lane combination from dual-frequency DD.

    Parameters
    ----------
    dd_measurement : dict
        DD measurement with L1 and L2 data

    Returns
    -------
    dict
        Wide-lane DD measurement
    """
    if not dd_measurement.get('dual_freq'):
        return None

    # Wide-lane frequency and wavelength
    f_wl = FREQ_L1 - FREQ_L2
    lambda_wl = CLIGHT / f_wl  # ~86 cm

    # Wide-lane combination
    # WL = (f1*L1 - f2*L2)/(f1-f2) - (f1*P1 + f2*P2)/(f1+f2)
    if dd_measurement.get('dd_carrier') and dd_measurement.get('dd_carrier_l2'):
        # Melbourne-Wübbena combination for ambiguity
        L1 = dd_measurement['dd_carrier'] * dd_measurement['wavelength']
        L2 = dd_measurement['dd_carrier_l2'] * dd_measurement['wavelength_l2']
        P1 = dd_measurement['dd_obs']
        P2 = dd_measurement['dd_obs_l2']

        wl_phase = (FREQ_L1 * L1 - FREQ_L2 * L2) / (FREQ_L1 - FREQ_L2)
        narrow_lane_pr = (FREQ_L1 * P1 + FREQ_L2 * P2) / (FREQ_L1 + FREQ_L2)

        # MW combination
        mw_combination = wl_phase - narrow_lane_pr
        wl_ambiguity = mw_combination / lambda_wl

        return {
            'wl_phase': wl_phase / lambda_wl,  # In cycles
            'wl_ambiguity': wl_ambiguity,
            'wl_wavelength': lambda_wl,
            'mw_combination': mw_combination
        }

    return None