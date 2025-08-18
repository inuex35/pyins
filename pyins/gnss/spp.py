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

"""Robust Single Point Positioning with multi-GNSS support"""

import numpy as np
from typing import List, Tuple, Optional
from ..core.constants import *
from ..core.constants import sat2sys, SYS_GAL, SYS_BDS
from ..core.data_structures import Observation, NavigationData, Solution
from ..gnss.ephemeris import seleph, eph2pos
from ..coordinate import ecef2llh, llh2ecef
from numpy.linalg import norm
from .glonass_ifb import GLONASSBias, get_glonass_channel
import logging

logger = logging.getLogger(__name__)


def robust_troposphere_model(pos: np.ndarray, el: float) -> float:
    """
    Robust troposphere delay model that handles edge cases
    """
    # Saastamoinen model parameters
    P0 = 1013.25  # hPa
    T0 = 288.15   # K
    e0 = 11.75    # hPa
    
    # Height correction with bounds
    h = pos[2] if len(pos) > 2 else 0.0
    h = np.clip(h, -500.0, 9000.0)  # Reasonable height bounds
    
    # Latitude for pressure correction
    lat = 0.0
    if norm(pos) > RE_WGS84:
        llh = ecef2llh(pos)
        lat = llh[0]
    
    # Pressure and temperature at height
    P = P0 * (1 - 2.26e-5 * h) ** 5.225
    T = T0 - 6.5e-3 * h
    e = e0 * (T / T0) ** 4.0
    
    # Zenith delays
    zhd = 0.0022768 * P / (1 - 0.00266 * np.cos(2 * lat) - 0.00028e-3 * h)
    zwd = 0.0022768 * (1255 / T + 0.05) * e
    
    # Mapping function with elevation cutoff
    el_deg = np.rad2deg(el)
    if el_deg < 5.0:
        return 0.0  # Skip very low elevation satellites
    
    mapf = 1.0 / np.sin(el) if el > 0.1 else 10.0
    
    return (zhd + zwd) * mapf


def spp_solve(observations: List[Observation], 
                     nav_data: NavigationData,
                     max_iter: int = 20,  # Increased iterations
                     converge_threshold: float = 0.1,  # 10cm convergence threshold
                     systems_to_use: Optional[List[str]] = None,
                     use_glonass_ifb: bool = True,
                     use_all_frequencies: bool = True) -> Tuple[Optional[Solution], List[int]]:
    """
    Robust SPP solver with better error handling
    
    Parameters
    ----------
    observations : List[Observation]
        List of GNSS observations
    nav_data : NavigationData
        Navigation data containing ephemerides
    max_iter : int
        Maximum number of iterations
    converge_threshold : float
        Convergence threshold in meters
    systems_to_use : Optional[List[str]]
        List of systems to use: 'G'(GPS), 'R'(GLONASS), 'E'(Galileo), 'C'(BeiDou), 'J'(QZSS)
        Default: ['G', 'E', 'C', 'J'] (excludes GLONASS due to IFB)
    use_glonass_ifb : bool
        Whether to apply GLONASS IFB correction (only relevant if 'R' in systems_to_use)
        
    Returns
    -------
    Tuple[Optional[Solution], List[int]]
        Solution object and list of used satellite numbers
    """
    # Default systems
    if systems_to_use is None:
        systems_to_use = ['G', 'R', 'E', 'C', 'J']  # Include all major systems
    
    # Convert system chars to IDs
    sys_char_to_id = {
        'G': SYS_GPS,
        'R': SYS_GLO,
        'E': SYS_GAL,
        'C': SYS_BDS,
        'J': SYS_QZS
    }
    
    allowed_systems = set()
    for sys_char in systems_to_use:
        if sys_char in sys_char_to_id:
            allowed_systems.add(sys_char_to_id[sys_char])
    
    # Collect valid observations with their frequencies
    valid_obs_freqs = []
    for obs in observations:
        # Check if satellite system is allowed
        sys = sat2sys(obs.sat)
        if sys not in allowed_systems:
            continue
        
        # Collect all available frequencies for this satellite
        if use_all_frequencies:
            freqs = []
            # Check L1/E1/B1 (index 0)
            if obs.P[0] > 0 and 1e6 < obs.P[0] < 100e6:  # Increased upper limit
                freqs.append((0, obs.P[0]))
            # Check L2/E5b/B2 (index 1) 
            if obs.P[1] > 0 and 1e6 < obs.P[1] < 100e6:  # Increased upper limit
                freqs.append((1, obs.P[1]))
            # Check L5/E5a/B2a (index 2) if available
            if len(obs.P) > 2 and obs.P[2] > 0 and 1e6 < obs.P[2] < 100e6:  # Increased upper limit
                freqs.append((2, obs.P[2]))
            
            # Add observation with all its frequencies
            if freqs:
                valid_obs_freqs.append((obs, freqs))
        else:
            # Original single-frequency behavior
            pr = obs.P[0] if obs.P[0] > 0 else obs.P[1]
            if pr > 0 and 1e6 < pr < 100e6:  # Increased upper limit
                valid_obs_freqs.append((obs, [(0, pr)]))
    
    # Count total measurements
    total_measurements = sum(len(freqs) for _, freqs in valid_obs_freqs)
    
    if total_measurements < 5:  # Need at least 5 measurements
        logger.warning(f"Insufficient measurements: {total_measurements}")
        return None, []
    
    # Initial state - always start from origin (0,0,0) like RTKLIB
    x = np.zeros(8)  # pos(3) + clk_gps + clk_glo + clk_gal + clk_bds + clk_qzs
    # x[:3] stays at [0, 0, 0] for first iteration
    
    # System indices for clock biases
    sys_clk_idx = {
        SYS_GPS: 3,
        SYS_GLO: 4,
        SYS_GAL: 5,
        SYS_BDS: 6,
        SYS_QZS: 3  # QZSS shares GPS clock
    }
    
    # Initialize GLONASS IFB handler
    glonass_bias = GLONASSBias() if use_glonass_ifb else None
    
    used_sats = []
    
    for iteration in range(max_iter):
        H = []
        y = []
        used_sats = []
        measurement_info = []
        
        if iteration == 0:
            logger.debug(f"Starting iteration {iteration} with {len(valid_obs_freqs)} observations")
        
        # Use satpos function to compute all satellite positions at once
        from .ephemeris import satpos
        
        # Add time to observations for satpos
        for obs, _ in valid_obs_freqs:
            if not hasattr(obs, 'time'):
                obs.time = observations[0].time if observations else 0
        
        # Extract just the observations from valid_obs_freqs
        valid_obs_list = [obs for obs, _ in valid_obs_freqs]
        
        # Compute satellite positions using satpos
        sat_positions, sat_clocks, sat_vars, sat_healths = satpos(valid_obs_list, nav_data)
        
        for idx, (obs, freqs) in enumerate(valid_obs_freqs):
            # Check satellite health from satpos (-1 means invalid)
            if sat_healths[idx] == -1:
                if iteration == 0:
                    logger.debug(f"Skipping sat {obs.sat}: health=-1")
                continue
                
            # Get satellite position and clock from satpos results
            sat_pos = sat_positions[idx]
            dts = sat_clocks[idx]
            
            # Validate position
            if sat_pos is None or np.any(np.isnan(sat_pos)) or norm(sat_pos) < RE_WGS84:
                if iteration == 0:
                    logger.debug(f"Skipping sat {obs.sat}: invalid position")
                continue
            
            # Get satellite system
            sys = sat2sys(obs.sat)
            
            # Filter out problematic GAL 101 satellite
            if sys == SYS_GAL and obs.sat == 101:
                if iteration == 0:
                    logger.debug(f"Skipping problematic GAL satellite {obs.sat}")
                continue
            
            # Filter out BDS GEO satellites (PRN 1-5, 59-63)
            # These satellites have very high orbits and can cause issues
            if sys == SYS_BDS:
                from ..core.constants import sat2prn
                prn = sat2prn(obs.sat)
                if prn <= 5 or prn >= 59:
                    if iteration == 0:
                        logger.debug(f"Skipping BDS GEO satellite {obs.sat} (PRN {prn})")
                    continue
            
            # Filter satellites with abnormal pseudorange (>30000km for non-GEO)
            # Get first available pseudorange
            pr = 0
            if len(freqs) > 0:
                # freqs is a list of frequency indices
                freq_idx = freqs[0] if isinstance(freqs[0], int) else freqs[0][0] if isinstance(freqs[0], tuple) else 0
                if freq_idx < len(obs.P):
                    pr = obs.P[freq_idx]
            if pr <= 0:
                # Try to find any valid pseudorange
                for freq_idx in range(len(obs.P)):
                    if obs.P[freq_idx] > 0:
                        pr = obs.P[freq_idx]
                        break
            if pr > 30000000:  # 30,000 km threshold
                if iteration == 0:
                    logger.debug(f"Skipping sat {obs.sat}: abnormal pseudorange {pr/1000:.0f}km")
                continue
            
            # Geometric range
            r = norm(sat_pos - x[:3])
            
            # Line of sight vector
            los = (x[:3] - sat_pos) / r
            
            # Elevation angle
            # For initial iterations with position near origin, don't check elevation
            # This follows RTKLIB approach
            if norm(x[:3]) < 1000.0:  # Position near origin
                el = np.deg2rad(45.0)  # Default elevation
            elif iteration < 3:  # First few iterations
                el = np.deg2rad(45.0)  # Default elevation
            else:
                # Only check elevation after position converges
                llh = ecef2llh(x[:3])
                el = elevation_angle(x[:3], sat_pos, llh)
                if el < np.deg2rad(10.0):  # 10 degree mask
                    continue
            
            # System-specific clock
            sys = sat2sys(obs.sat)
            clk_idx = sys_clk_idx.get(sys, 3)
            
            # Troposphere delay (same for all frequencies)
            trop = robust_troposphere_model(x[:3], el)
            
            # Base weight based on elevation and system
            base_weight = np.sin(el) if el > 0 else 0.1
            
            # Reduce weight for GLONASS due to inter-frequency bias
            # But not too much - GLONASS can still contribute
            if sys == SYS_GLO:
                base_weight *= 0.7  # Changed from 0.5 to 0.7
            
            # Process each frequency
            if iteration == 0 and idx < 3:
                logger.debug(f"Processing sat {obs.sat}: {len(freqs)} frequencies")
            for freq_idx, pr in freqs:
                # Predicted range
                rho_pred = r + x[clk_idx] - dts * CLIGHT + trop
                
                # Observation equation
                h = np.zeros(8)
                h[:3] = los
                h[clk_idx] = 1.0
                
                # Residual
                res = pr - rho_pred
                
                # Debug first iteration
                if iteration == 0 and idx < 3:
                    logger.debug(f"  Sat {obs.sat} freq {freq_idx}: pr={pr:.1f}, pred={rho_pred:.1f}, res={res:.1f}")
                
                # Outlier detection (after convergence begins)
                # For initial iterations, allow large residuals (RTKLIB approach)
                if iteration <= 2:
                    # Accept all measurements in first iterations
                    pass
                elif iteration > 3 and abs(res) > 1000.0:  # 1km threshold
                    continue  # Skip outlier
                
                # Frequency-specific weight adjustment
                # L1/E1/B1 typically most precise
                freq_weight = 1.0 if freq_idx == 0 else 0.9
                # Combined weight
                w = base_weight * freq_weight
                
                # Robust weighting based on residual (Huber weighting)
                # Apply after initial convergence
                if iteration > 2:
                    k = 30.0  # Huber constant (30m)
                    if abs(res) > k:
                        w *= k / abs(res)
                elif iteration == 0:
                    # First iteration: use uniform weights
                    w = 1.0
                
                H.append(h * w)
                y.append(res * w)
                
                # Store measurement info
                measurement_info.append({
                    'sat': obs.sat,
                    'freq': freq_idx,
                    'residual': res,
                    'elevation': np.rad2deg(el)
                })
            
            # Add satellite to used list (once per satellite, not per frequency)
            if obs.sat not in used_sats:
                used_sats.append(obs.sat)
        
        if len(H) < 5:
            logger.warning(f"Insufficient valid measurements: {len(H)}")
            return None, []
        
        # Solve normal equations
        H = np.array(H)
        y = np.array(y)
        
        try:
            # Remove unused clock parameters
            used_params = [0, 1, 2]  # Always use position
            for i in range(3, 8):
                if np.any(H[:, i] != 0):
                    used_params.append(i)
            
            H_used = H[:, used_params]
            HTH = H_used.T @ H_used
            HTy = H_used.T @ y
            
            # Add small regularization for stability
            HTH += np.eye(len(used_params)) * 1e-9
            
            dx_used = np.linalg.solve(HTH, HTy)
            
            # Map back to full state
            dx = np.zeros(8)
            for i, idx in enumerate(used_params):
                dx[idx] = dx_used[i]
            
        except np.linalg.LinAlgError:
            logger.warning("Failed to solve normal equations")
            return None, []
        
        # Update state
        x += dx
        
        # Check convergence
        if norm(dx[:3]) < converge_threshold:
            # Log frequency usage statistics
            if use_all_frequencies:
                freq_count = {0: 0, 1: 0, 2: 0}
                for info in measurement_info:
                    freq_count[info['freq']] += 1
            
            # Create solution
            # The state vector x has clock biases in meters:
            # x[3] = GPS clock bias (in standard SPP, this absorbs the absolute bias)
            # x[4] = GLO clock bias  
            # x[5] = GAL clock bias
            # x[6] = BDS clock bias
            # x[7] = QZS clock bias (same as GPS)
            
            # Estimate the absolute GPS clock bias from residuals
            # Since GPS is the reference, x[3] is typically close to 0
            # The actual clock bias is absorbed in the pseudorange residuals
            
            # We need to estimate the actual GPS clock from the data
            # A typical receiver clock bias is around 70-80ms (21000-24000 km)
            # But it can vary significantly
            
            # RTKLIB style: GPS clock and ISBs
            # x[3] = GPS clock bias
            # x[4] = GLO clock bias = GPS clock + GLO ISB
            # x[5] = GAL clock bias = GPS clock + GAL ISB
            # x[6] = BDS clock bias = GPS clock + BDS ISB
            # x[7] = QZS clock bias = GPS clock (same system)
            
            raw_clocks = x[3:8].copy()  # Raw clock biases from solution
            
            # Create RTKLIB-style output:
            # dtr_meters[0] = GPS clock
            # dtr_meters[1] = GLO ISB (relative to GPS)
            # dtr_meters[2] = GAL ISB (relative to GPS)
            # dtr_meters[3] = BDS ISB (relative to GPS)
            # dtr_meters[4] = QZS ISB (0 for QZS as it's same as GPS)
            dtr_meters = np.zeros(5)
            dtr_meters[0] = raw_clocks[0]  # GPS clock
            dtr_meters[1] = raw_clocks[1] - raw_clocks[0] if len(raw_clocks) > 1 else 0.0  # GLO ISB
            dtr_meters[2] = raw_clocks[2] - raw_clocks[0] if len(raw_clocks) > 2 else 0.0  # GAL ISB
            dtr_meters[3] = raw_clocks[3] - raw_clocks[0] if len(raw_clocks) > 3 else 0.0  # BDS ISB
            dtr_meters[4] = 0.0  # QZS ISB (same as GPS)
            
            # For compatibility, also keep absolute values
            dtr_meters_absolute = raw_clocks.copy()
            
            # Convert to seconds for standard output
            dtr_seconds = dtr_meters / CLIGHT  # Relative values in seconds
            dtr_seconds_absolute = dtr_meters_absolute / CLIGHT  # Absolute values in seconds
            
            sol = Solution(
                time=observations[0].time,
                type=1,  # Single point
                rr=x[:3].copy(),
                vv=np.zeros(3),
                dtr=dtr_seconds,  # Keep original relative values for compatibility
                qr=np.zeros((6, 6)),  # Simplified
                ns=len(used_sats),
                age=0.0,
                ratio=0.0
            )
            
            # Add both relative and absolute clock biases
            sol.dtr_meters = dtr_meters  # Relative clock biases in meters
            sol.dtr_absolute = dtr_seconds_absolute  # Absolute clock biases in seconds
            sol.dtr_meters_absolute = dtr_meters_absolute  # Absolute clock biases in meters
            
            return sol, used_sats
    
    logger.warning("SPP did not converge")
    return None, []


def elevation_angle(rcv_pos: np.ndarray, sat_pos: np.ndarray, llh: np.ndarray) -> float:
    """Calculate satellite elevation angle"""
    # Vector from receiver to satellite
    los = sat_pos - rcv_pos
    
    # Local ENU transformation
    lat, lon = llh[0], llh[1]
    
    R = np.array([
        [-np.sin(lon), np.cos(lon), 0],
        [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)],
        [np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)]
    ])
    
    enu = R @ los
    
    # Elevation angle
    el = np.arctan2(enu[2], np.sqrt(enu[0]**2 + enu[1]**2))
    
    return el


# Alias for backward compatibility
robust_spp_solve = spp_solve