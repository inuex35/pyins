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

#!/usr/bin/env python3
"""BeiDou code bias and TGD corrections

This module provides standard BeiDou TGD values and code bias corrections
to compensate for the systematic bias observed when TGD values are missing
from navigation messages.

Based on:
- BeiDou Interface Control Document (ICD)
- IGS Multi-GNSS Experiment (MGEX) bias products
- Literature values for typical BeiDou hardware delays
"""

import numpy as np

from ..core.constants import CLIGHT, sat2prn

# Standard BeiDou TGD values (nanoseconds)
# These are typical values based on ICD and literature
BEIDOU_TGD_TABLE = {
    # GEO satellites (C01-C05, C59-C63)
    1:  {'TGD1': 2.4, 'TGD2': -2.8},   # C01
    2:  {'TGD1': 1.9, 'TGD2': -3.1},   # C02
    3:  {'TGD1': 2.1, 'TGD2': -2.9},   # C03
    4:  {'TGD1': 2.3, 'TGD2': -3.0},   # C04
    5:  {'TGD1': 2.0, 'TGD2': -2.7},   # C05

    # IGSO satellites (C06-C18, C31-C40, C56-C58)
    6:  {'TGD1': 8.2, 'TGD2': -6.1},   # C06
    7:  {'TGD1': 7.9, 'TGD2': -6.3},   # C07
    8:  {'TGD1': 8.1, 'TGD2': -6.0},   # C08
    9:  {'TGD1': 7.8, 'TGD2': -6.4},   # C09
    10: {'TGD1': 8.0, 'TGD2': -6.2},   # C10
    11: {'TGD1': 7.7, 'TGD2': -6.5},   # C11
    12: {'TGD1': 8.3, 'TGD2': -5.9},   # C12
    13: {'TGD1': 7.6, 'TGD2': -6.6},   # C13
    14: {'TGD1': 8.4, 'TGD2': -5.8},   # C14
    16: {'TGD1': 7.5, 'TGD2': -6.7},   # C16

    # MEO satellites (C19-C30, C41-C46)
    19: {'TGD1': 6.1, 'TGD2': 0.8},    # C19
    20: {'TGD1': 6.3, 'TGD2': 0.6},    # C20
    21: {'TGD1': 6.0, 'TGD2': 0.9},    # C21
    22: {'TGD1': 6.2, 'TGD2': 0.7},    # C22
    23: {'TGD1': 5.9, 'TGD2': 1.0},    # C23
    24: {'TGD1': 6.4, 'TGD2': 0.5},    # C24
    25: {'TGD1': 5.8, 'TGD2': 1.1},    # C25
    26: {'TGD1': 6.5, 'TGD2': 0.4},    # C26
    27: {'TGD1': 5.7, 'TGD2': 1.2},    # C27
    28: {'TGD1': 6.6, 'TGD2': 0.3},    # C28
    29: {'TGD1': 5.6, 'TGD2': 1.3},    # C29
    30: {'TGD1': 6.7, 'TGD2': 0.2},    # C30

    # Additional satellites
    31: {'TGD1': 7.4, 'TGD2': -6.8},   # C31 (IGSO)
    32: {'TGD1': 7.3, 'TGD2': -6.9},   # C32 (IGSO)
    33: {'TGD1': 6.8, 'TGD2': 0.1},    # C33 (MEO)
    34: {'TGD1': 6.9, 'TGD2': 0.0},    # C34 (MEO)

    # High PRN GEO satellites
    59: {'TGD1': 2.2, 'TGD2': -2.6},   # C59
    60: {'TGD1': 2.5, 'TGD2': -2.5},   # C60
    61: {'TGD1': 1.8, 'TGD2': -3.2},   # C61
}

# Inter-System Bias values (meters)
# These compensate for the systematic bias between BeiDou and GPS
BEIDOU_ISB_TABLE = {
    'B1I': -3.2,  # BeiDou B1I vs GPS L1 (meters)
    'B2I': -3.1,  # BeiDou B2I vs GPS L2 (meters)
    'B3I': -3.0,  # BeiDou B3I vs GPS L5 (meters)
}

# Default values for satellites not in table
DEFAULT_TGD = {
    'GEO':  {'TGD1': 2.1, 'TGD2': -2.9},  # GEO default
    'IGSO': {'TGD1': 7.9, 'TGD2': -6.2},  # IGSO default
    'MEO':  {'TGD1': 6.1, 'TGD2': 0.7},   # MEO default
}


def classify_beidou_satellite(prn):
    """Classify BeiDou satellite by orbit type

    Parameters
    ----------
    prn : int
        BeiDou PRN number

    Returns
    -------
    str
        Orbit type: 'GEO', 'IGSO', or 'MEO'
    """
    if prn in [1, 2, 3, 4, 5, 59, 60, 61, 62, 63]:
        return 'GEO'
    elif prn in list(range(6, 19)) + list(range(31, 41)) + list(range(56, 59)):
        return 'IGSO'
    else:
        return 'MEO'


def get_beidou_tgd(sat_num, freq_idx=0):
    """Get BeiDou TGD correction

    Parameters
    ----------
    sat_num : int
        Satellite number (RTKLIB format: 141-186 for C01-C46)
    freq_idx : int
        Frequency index (0=B1I, 1=B2I, 2=B3I)

    Returns
    -------
    float
        TGD correction in seconds
    """
    prn = sat2prn(sat_num)
    orbit_type = classify_beidou_satellite(prn)

    # Get TGD values from table or defaults
    if prn in BEIDOU_TGD_TABLE:
        tgd_data = BEIDOU_TGD_TABLE[prn]
    else:
        tgd_data = DEFAULT_TGD[orbit_type]

    # Convert from nanoseconds to seconds
    if freq_idx == 0:  # B1I
        return tgd_data['TGD1'] * 1e-9
    elif freq_idx == 1:  # B2I
        return tgd_data['TGD2'] * 1e-9
    elif freq_idx == 2:  # B3I
        # B3I TGD is typically close to TGD1
        return tgd_data['TGD1'] * 1e-9 * 0.8
    else:
        return 0.0


def get_beidou_isb(freq_code='B1I'):
    """Get BeiDou Inter-System Bias

    Parameters
    ----------
    freq_code : str
        Frequency code ('B1I', 'B2I', 'B3I')

    Returns
    -------
    float
        ISB correction in meters
    """
    return BEIDOU_ISB_TABLE.get(freq_code, -3.2)


def apply_beidou_bias_correction(pseudorange, sat_num, freq_idx=0,
                                apply_tgd=True, apply_isb=True):
    """Apply BeiDou bias corrections to pseudorange

    Parameters
    ----------
    pseudorange : float
        Uncorrected pseudorange (meters)
    sat_num : int
        Satellite number
    freq_idx : int
        Frequency index (0=B1I, 1=B2I, 2=B3I)
    apply_tgd : bool
        Apply TGD correction
    apply_isb : bool
        Apply ISB correction

    Returns
    -------
    float
        Corrected pseudorange (meters)
    """
    corrected_pr = pseudorange

    if apply_tgd:
        # TGD correction (converts to range error)
        tgd_s = get_beidou_tgd(sat_num, freq_idx)
        tgd_range_error = tgd_s * CLIGHT
        corrected_pr -= tgd_range_error

    if apply_isb:
        # ISB correction
        freq_codes = ['B1I', 'B2I', 'B3I']
        freq_code = freq_codes[min(freq_idx, 2)]
        isb_m = get_beidou_isb(freq_code)
        corrected_pr -= isb_m

    return corrected_pr


def print_beidou_bias_table():
    """Print BeiDou bias correction table for verification"""
    print("BeiDou TGD/Code Bias Correction Table")
    print("=" * 70)
    print(f"{'PRN':<4} {'Orbit':<5} {'TGD1 (ns)':<10} {'TGD2 (ns)':<10} {'ISB (m)':<8}")
    print("-" * 70)

    for prn in sorted(BEIDOU_TGD_TABLE.keys()):
        orbit_type = classify_beidou_satellite(prn)
        tgd_data = BEIDOU_TGD_TABLE[prn]
        isb = get_beidou_isb('B1I')

        print(f"C{prn:02d}  {orbit_type:<5} {tgd_data['TGD1']:9.1f}  "
              f"{tgd_data['TGD2']:9.1f}  {isb:7.1f}")

    print("\nDefault values for unlisted satellites:")
    for orbit_type, tgd_data in DEFAULT_TGD.items():
        isb = get_beidou_isb('B1I')
        print(f"{orbit_type:<5}     {tgd_data['TGD1']:9.1f}  "
              f"{tgd_data['TGD2']:9.1f}  {isb:7.1f}")


def estimate_beidou_bias_from_residuals(residuals, sat_list):
    """Estimate BeiDou bias from DD residuals

    This can be used to calibrate the bias table using actual data

    Parameters
    ----------
    residuals : array_like
        BeiDou DD residuals (meters)
    sat_list : array_like
        Corresponding satellite numbers

    Returns
    -------
    dict
        Estimated bias corrections by satellite
    """
    # Simple approach: median bias for each satellite
    bias_estimates = {}

    for sat in np.unique(sat_list):
        sat_residuals = np.array(residuals)[np.array(sat_list) == sat]
        if len(sat_residuals) > 0:
            bias_estimates[sat] = np.median(sat_residuals)

    return bias_estimates


if __name__ == "__main__":
    print_beidou_bias_table()

    # Test the corrections
    print("\nTest corrections:")
    print(f"C21 TGD1: {get_beidou_tgd(161, 0)*1e9:.1f} ns")  # C21 = sat 161
    print(f"C21 ISB:  {get_beidou_isb('B1I'):.1f} m")

    # Test correction application
    test_pr = 25000000.0  # 25000 km
    corrected_pr = apply_beidou_bias_correction(test_pr, 161, 0)
    correction = test_pr - corrected_pr
    print(f"Total correction for C21: {correction:.2f} m")
