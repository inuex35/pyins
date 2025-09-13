#!/usr/bin/env python3
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
Double Difference formation for GNSS observations
"""


import numpy as np

from ..core.constants import CLIGHT, SYS_BDS, SYS_GAL, SYS_GLO, SYS_GPS, SYS_QZS, sat2sys
from ..geometry.elevation import compute_elevation_angle
from ..gnss.ephemeris import satpos


def interpolate_observation(obs1, obs2, target_time):
    """
    Linear interpolation of GNSS observations based on RTKLIB approach
    
    Parameters:
    -----------
    obs1, obs2 : observation objects
        Bracketing observations for interpolation
    target_time : float
        Target time for interpolation
        
    Returns:
    --------
    interpolated observation object
    """
    if not hasattr(obs1, 'time') or not hasattr(obs2, 'time'):
        return obs1  # Return first observation if no time info
    
    dt = obs2.time - obs1.time
    if abs(dt) < 1e-6:  # Times are essentially equal
        return obs1
    
    # Interpolation factor
    alpha = (target_time - obs1.time) / dt
    alpha = max(0.0, min(1.0, alpha))  # Clamp to [0,1]
    
    # Create interpolated observation (copy from obs1)
    interp_obs = type(obs1).__new__(type(obs1))
    for attr in dir(obs1):
        if not attr.startswith('_'):
            setattr(interp_obs, attr, getattr(obs1, attr))
    
    # Interpolate pseudoranges
    if hasattr(obs1, 'P') and hasattr(obs2, 'P'):
        interp_obs.P = [0.0] * len(obs1.P)
        for i in range(len(obs1.P)):
            if obs1.P[i] != 0 and obs2.P[i] != 0:
                interp_obs.P[i] = obs1.P[i] + alpha * (obs2.P[i] - obs1.P[i])
            else:
                interp_obs.P[i] = obs1.P[i] if obs1.P[i] != 0 else obs2.P[i]
    
    interp_obs.time = target_time
    return interp_obs


# Storage for previous base observations (module-level for persistence)
_prev_base_obs = {}
_prev_base_time = None

def synchronize_observations(rover_obs, base_obs, gps_time, max_tdiff=30.0):
    """
    Synchronize base observations to rover time using RTKLIB-style approach
    with 2-epoch interpolation
    
    Parameters:
    -----------
    rover_obs : list
        Rover observations
    base_obs : list  
        Base observations
    gps_time : float
        GPS time for this epoch
    max_tdiff : float
        Maximum time difference for interpolation (default: 30.0s)
        
    Returns:
    --------
    list of synchronized base observations
    """
    global _prev_base_obs, _prev_base_time
    
    DTTOL = 0.005  # RTKLIB time difference tolerance (5ms)
    
    synchronized_base_obs = []
    
    if not rover_obs or not base_obs:
        return base_obs
        
    rover_time = getattr(rover_obs[0], 'time', gps_time)
    
    # Build current base observations by satellite
    curr_base_obs = {}
    curr_base_time = None
    for obs in base_obs:
        obs_time = getattr(obs, 'time', gps_time)
        if curr_base_time is None:
            curr_base_time = obs_time
        curr_base_obs[obs.sat] = obs
    
    if curr_base_time is None:
        return []
    
    # Time difference between rover and current base
    tt = rover_time - curr_base_time
    
    # Check if within tolerance for exact match (RTKLIB DTTOL)
    if abs(tt) <= DTTOL:
        # Exact time match - use current base observations
        synchronized_base_obs = list(curr_base_obs.values())
        print(f"INFO: Exact time match (dt={tt*1000:.1f}ms)")
        # Update previous observations for next epoch
        _prev_base_obs = curr_base_obs.copy()
        _prev_base_time = curr_base_time
        return synchronized_base_obs
    
    # Check if we have previous base observations for interpolation
    if _prev_base_obs and _prev_base_time is not None:
        # Time difference between rover and previous base
        ttb = rover_time - _prev_base_time
        
        # RTKLIB-style interpolation conditions
        # 1. Previous and current bracket the rover time
        # 2. Time differences are within max_tdiff
        # 3. Not the same time difference
        
        if (abs(ttb) <= max_tdiff * 2.0 and 
            abs(tt) <= max_tdiff * 2.0 and 
            ttb != tt and
            ttb * tt < 0):  # Different signs = bracketing
            
            # Perform 2-epoch linear interpolation
            print(f"INFO: Interpolating between base epochs (ttb={ttb:.3f}s, tt={tt:.3f}s)")
            
            # Find common satellites between previous and current base
            common_sats = set(_prev_base_obs.keys()) & set(curr_base_obs.keys())
            
            for sat in common_sats:
                prev_obs = _prev_base_obs[sat]
                curr_obs = curr_base_obs[sat]
                
                # Create interpolated observation
                interp_obs = type(curr_obs).__new__(type(curr_obs))
                # Copy only writable attributes
                for attr in ['P', 'L', 'C', 'D', 'S', 'LLI', 'SSI', 'time', 'sat']:
                    if hasattr(curr_obs, attr):
                        try:
                            setattr(interp_obs, attr, getattr(curr_obs, attr))
                        except AttributeError:
                            pass  # Skip read-only attributes
                
                # RTKLIB-style interpolation formula:
                # y_interp = (ttb * y_curr - tt * y_prev) / (ttb - tt)
                # where tt = t_rover - t_curr, ttb = t_rover - t_prev
                
                if hasattr(curr_obs, 'P') and hasattr(prev_obs, 'P'):
                    interp_obs.P = [0.0] * len(curr_obs.P)
                    for i in range(len(curr_obs.P)):
                        if (i < len(prev_obs.P) and 
                            curr_obs.P[i] != 0 and prev_obs.P[i] != 0):
                            # Linear interpolation using RTKLIB formula
                            interp_obs.P[i] = (ttb * curr_obs.P[i] - tt * prev_obs.P[i]) / (ttb - tt)
                        elif curr_obs.P[i] != 0:
                            interp_obs.P[i] = curr_obs.P[i]
                        elif i < len(prev_obs.P) and prev_obs.P[i] != 0:
                            interp_obs.P[i] = prev_obs.P[i]
                
                # Interpolate carrier phase if available
                if hasattr(curr_obs, 'L') and hasattr(prev_obs, 'L'):
                    interp_obs.L = [0.0] * len(curr_obs.L)
                    for i in range(len(curr_obs.L)):
                        if (i < len(prev_obs.L) and 
                            curr_obs.L[i] != 0 and prev_obs.L[i] != 0):
                            # Check for cycle slip
                            if not ((curr_obs.LLI[i] if i < len(curr_obs.LLI) else 0) & 1 or
                                   (prev_obs.LLI[i] if i < len(prev_obs.LLI) else 0) & 1):
                                interp_obs.L[i] = (ttb * curr_obs.L[i] - tt * prev_obs.L[i]) / (ttb - tt)
                            else:
                                interp_obs.L[i] = 0.0  # Skip if cycle slip detected
                        elif curr_obs.L[i] != 0:
                            interp_obs.L[i] = curr_obs.L[i]
                        elif i < len(prev_obs.L) and prev_obs.L[i] != 0:
                            interp_obs.L[i] = prev_obs.L[i]
                
                interp_obs.time = rover_time
                interp_obs.sat = sat
                synchronized_base_obs.append(interp_obs)
            
            # Update previous observations for next epoch
            _prev_base_obs = curr_base_obs.copy()
            _prev_base_time = curr_base_time
            
            return synchronized_base_obs
    
    # No interpolation possible - use nearest neighbor
    if abs(tt) <= 1.0:
        # Use current base if close enough
        synchronized_base_obs = list(curr_base_obs.values())
        print(f"INFO: Using nearest base epoch (dt={tt:.3f}s)")
    elif abs(tt) <= max_tdiff:
        # Still use current base but warn about time difference
        synchronized_base_obs = list(curr_base_obs.values())
        print(f"WARNING: Large time difference but within limit (dt={tt:.3f}s)")
    else:
        # Time difference too large
        print(f"WARNING: Base time gap too large ({abs(tt):.1f}s) - skipping epoch")
        return []
    
    # Update previous observations for next epoch
    _prev_base_obs = curr_base_obs.copy()
    _prev_base_time = curr_base_time
    
    return synchronized_base_obs


def form_double_differences(rover_obs, base_obs, nav_data, gps_time,
                           reference_ecef, reference_llh,
                           use_systems: list[str] = None,
                           cutoff_angle: float = 15.0) -> list[dict]:
    """
    Form double differences between rover and base observations.

    Double differences are formed independently for each satellite system
    to avoid inter-system biases.

    Parameters:
    -----------
    rover_obs : list
        Rover GNSS observations
    base_obs : list
        Base GNSS observations
    nav_data : dict
        Navigation data
    gps_time : float
        GPS time for this epoch
    reference_ecef : np.ndarray
        Reference position in ECEF (for elevation computation)
    reference_llh : np.ndarray
        Reference position in LLH (for elevation computation)
    use_systems : List[str], optional
        List of systems to use ('G', 'R', 'E', 'C', 'J')
        Default: ['G', 'E', 'C', 'J']
    cutoff_angle : float
        Elevation cutoff angle in degrees (default: 15.0)

    Returns:
    --------
    List of DD measurements with satellite positions
    Each measurement contains:
        - sat: satellite number
        - ref_sat: reference satellite number
        - dd_obs: double difference observation
        - sat_pos: satellite position
        - ref_sat_pos: reference satellite position
        - sat_clk: satellite clock bias
        - ref_sat_clk: reference satellite clock bias
        - elevation: satellite elevation angle
    """
    dd_measurements = []

    # Use separate synchronization function 
    synchronized_base_obs = synchronize_observations(rover_obs, base_obs, gps_time)
    
    if not synchronized_base_obs:
        return dd_measurements  # Return empty if synchronization failed

    # Default systems to use
    if use_systems is None:
        use_systems = ['G', 'E', 'C', 'J']  # Exclude GLONASS by default

    # Get common satellites
    rover_sats = {obs.sat for obs in rover_obs}
    base_sats = {obs.sat for obs in synchronized_base_obs}
    common_sats = rover_sats & base_sats

    if len(common_sats) < 2:
        return dd_measurements

    # Filter by system
    filtered_sats = []
    for sat in common_sats:
        sys_id = sat2sys(sat)
        sys_map = {SYS_GPS: 'G', SYS_GLO: 'R', SYS_GAL: 'E', SYS_BDS: 'C', SYS_QZS: 'J'}
        system_char = sys_map.get(sys_id, 'G')

        if system_char in use_systems:
            filtered_sats.append(sat)

    if len(filtered_sats) < 2:
        return dd_measurements

    # Compute satellite positions and elevations
    # First compute all satellite positions using satpos
    for obs in rover_obs:
        obs.time = gps_time
    sat_positions, sat_clocks, _, _ = satpos(rover_obs, nav_data)

    sat_data = {}
    for i, obs in enumerate(rover_obs):
        sat = obs.sat
        if sat not in filtered_sats:
            continue

        # Get satellite position
        sat_pos = sat_positions[i]
        sat_clk = sat_clocks[i]
        if sat_pos is None or np.any(np.isnan(sat_pos)):
            continue

        # Compute elevation angle
        elevation = compute_elevation_angle(sat_pos, reference_ecef, reference_llh)

        if elevation < cutoff_angle:
            continue

        # Get base observation
        base_obs_sat = next((o for o in synchronized_base_obs if o.sat == obs.sat), None)
        if base_obs_sat is None:
            continue

        # Get pseudoranges for multiple frequencies
        # Check L1 and L2 frequencies
        freqs_data = []
        for freq_idx in [0, 1]:  # L1 and L2
            rover_pr = obs.P[freq_idx] if freq_idx < len(obs.P) and obs.P[freq_idx] != 0 else None
            base_pr = base_obs_sat.P[freq_idx] if freq_idx < len(base_obs_sat.P) and base_obs_sat.P[freq_idx] != 0 else None
            
            if rover_pr is not None and base_pr is not None:
                # Apply satellite clock correction to pseudoranges
                # PR_corrected = PR_raw - c*sat_clk
                rover_pr_corrected = rover_pr - sat_clk * CLIGHT
                base_pr_corrected = base_pr - sat_clk * CLIGHT
                freqs_data.append({
                    'freq_idx': freq_idx,
                    'rover_pr': rover_pr_corrected,
                    'base_pr': base_pr_corrected
                })
        
        if not freqs_data:
            continue  # No valid frequency data
        
        # Store satellite data (with multiple frequencies)
        sat_data[obs.sat] = {
            'sat_pos': sat_pos,
            'sat_clk': sat_clk,
            'elevation': elevation,
            'freqs': freqs_data,  # Store all frequency data
            'system': sat2sys(obs.sat)  # Store system ID
        }

    # Group satellites by system
    sat_by_system = {}
    for sat, data in sat_data.items():
        sys_id = data['system']
        if sys_id not in sat_by_system:
            sat_by_system[sys_id] = {}
        sat_by_system[sys_id][sat] = data

    # Form DD for each system independently
    for sys_id, sys_sats in sat_by_system.items():
        if len(sys_sats) < 2:
            continue  # Need at least 2 satellites in the same system

        # Select reference satellite for this system (highest elevation)
        ref_sat = max(sys_sats.keys(), key=lambda s: sys_sats[s]['elevation'])
        ref_data = sys_sats[ref_sat]

        # Form double differences with respect to reference satellite
        for sat, data in sys_sats.items():
            if sat == ref_sat:
                continue

            # Form DD for each frequency
            for freq_data in data['freqs']:
                freq_idx = freq_data['freq_idx']
                
                # Find matching frequency in reference satellite
                ref_freq_data = next((f for f in ref_data['freqs'] if f['freq_idx'] == freq_idx), None)
                if ref_freq_data is None:
                    continue
                
                # Single difference at rover
                sd_rover = freq_data['rover_pr'] - ref_freq_data['rover_pr']

                # Single difference at base
                sd_base = freq_data['base_pr'] - ref_freq_data['base_pr']

                # Double difference
                dd = sd_rover - sd_base

                dd_measurements.append({
                    'sat': sat,
                    'ref_sat': ref_sat,
                    'dd_obs': dd,
                    'sat_pos': data['sat_pos'],
                    'ref_sat_pos': ref_data['sat_pos'],
                    'sat_clk': data['sat_clk'],
                    'ref_sat_clk': ref_data['sat_clk'],
                    'elevation': data['elevation'],
                    'freq_idx': freq_idx  # Add frequency index
                })

    return dd_measurements
