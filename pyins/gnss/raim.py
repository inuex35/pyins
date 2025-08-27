# Copyright 2024 The PyIns Authors. All Rights Reserved.
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

"""RAIM (Receiver Autonomous Integrity Monitoring) implementation"""

import numpy as np
from pyins.core.constants import sat2sys, sys2char, SYS_GPS, SYS_GLO


def raim_fde(H, v, var, used_sats, threshold=30.0, min_sats=4):
    """
    RAIM Fault Detection and Exclusion
    
    Parameters:
    -----------
    H : np.ndarray
        Design matrix
    v : np.ndarray
        Residual vector
    var : np.ndarray
        Variance of measurements
    used_sats : list
        List of satellite numbers
    threshold : float
        Residual threshold in meters
    min_sats : int
        Minimum number of satellites required
        
    Returns:
    --------
    H : np.ndarray
        Updated design matrix
    v : np.ndarray
        Updated residual vector
    var : np.ndarray
        Updated variance vector
    used_sats : list
        Updated satellite list
    excluded : list
        List of excluded satellites
    """
    if len(v) <= min_sats:
        return H, v, var, used_sats, []
    
    excluded = []
    threshold_m = threshold * 1000  # Convert to meters
    
    while len(v) > min_sats:
        # Find satellite with largest absolute residual
        abs_residuals = np.abs(v)
        max_idx = np.argmax(abs_residuals)
        max_residual = abs_residuals[max_idx]
        
        # Check if this residual is an outlier using MAD
        median_res = np.median(abs_residuals)
        mad = np.median(np.abs(abs_residuals - median_res))
        
        # Modified Z-score using MAD
        if mad > 0:
            z_score = (max_residual - median_res) / (1.4826 * mad)
        else:
            z_score = 0
        
        # Check if we should exclude this satellite
        if max_residual > threshold_m and z_score > 3.5:
            # Exclude the satellite
            excluded_sat = used_sats[max_idx]
            excluded.append(excluded_sat)
            
            # Report exclusion
            sys = sat2sys(excluded_sat)
            sys_char = sys2char(sys)
            sat_id = excluded_sat if sys == SYS_GPS else excluded_sat - 64 if sys == SYS_GLO else excluded_sat
            print(f"    RAIM: Excluding {sys_char}{sat_id:02d} (residual = {v[max_idx]/1000:.1f} km, z-score = {z_score:.1f})")
            
            # Remove the faulty satellite
            mask = np.ones(len(v), dtype=bool)
            mask[max_idx] = False
            H = H[mask]
            v = v[mask]
            var = var[mask]
            used_sats = [used_sats[i] for i in range(len(used_sats)) if mask[i]]
        else:
            # No more outliers
            break
    
    if len(excluded) > 0:
        print(f"    RAIM: {len(excluded)} satellites excluded, {len(v)} remaining")
    
    return H, v, var, used_sats, excluded


def check_gdop(H, threshold=300.0):
    """
    Check Geometric Dilution of Precision
    
    Parameters:
    -----------
    H : np.ndarray
        Design matrix
    threshold : float
        Maximum acceptable GDOP
        
    Returns:
    --------
    gdop : float
        Computed GDOP value
    valid : bool
        True if GDOP is below threshold
    """
    try:
        # Compute covariance matrix
        Q = np.linalg.inv(H.T @ H)
        # GDOP is square root of trace
        gdop = np.sqrt(np.trace(Q))
        valid = gdop < threshold
        return gdop, valid
    except np.linalg.LinAlgError:
        return np.inf, False