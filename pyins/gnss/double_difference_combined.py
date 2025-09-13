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
Double Difference formation with combined base/rover observations for better time matching
"""

import numpy as np
from collections import defaultdict

from ..core.constants import CLIGHT, SYS_BDS, SYS_GAL, SYS_GLO, SYS_GPS, SYS_QZS, sat2sys
from ..geometry.elevation import compute_elevation_angle
from ..gnss.ephemeris import satpos


def combine_and_synchronize_observations(rover_obs_list, base_obs_list, max_time_diff=0.5):
    """
    Combine base and rover observations into a time-sorted list for better matching
    
    Parameters:
    -----------
    rover_obs_list : list
        List of rover observations (list of epochs with 'observations' dict)
    base_obs_list : list
        List of base observations (list of epochs with 'observations' dict)
    max_time_diff : float
        Maximum time difference for pairing (default: 0.5 seconds)
        
    Returns:
    --------
    List of synchronized observation pairs (rover_obs, base_obs, time_diff)
    """
    
    # Helper to get time from observation
    def get_time(obs):
        """Get time from observation, preferring gps_time over time"""
        if isinstance(obs, dict):
            # Prefer gps_time if available
            if 'gps_time' in obs:
                return obs['gps_time']
            elif 'time' in obs:
                return obs['time']
        return None
    
    # Check if we have valid data
    if not rover_obs_list or not base_obs_list:
        return []
    
    synchronized_pairs = []
    
    # For each rover observation, find the closest base observation
    for rover_obs in rover_obs_list:
        rover_time = get_time(rover_obs)
        if rover_time is None:
            continue
            
        # Find closest base observation
        best_base = None
        best_time_diff = float('inf')
        
        for base_obs in base_obs_list:
            base_time = get_time(base_obs)
            if base_time is None:
                continue
                
            time_diff = abs(rover_time - base_time)
            
            if time_diff < best_time_diff and time_diff <= max_time_diff:
                best_time_diff = time_diff
                best_base = base_obs
        
        if best_base is not None:
            synchronized_pairs.append({
                'rover_obs': rover_obs,
                'base_obs': best_base,
                'rover_time': rover_time,
                'base_time': get_time(best_base),
                'time_diff': rover_time - get_time(best_base)
            })
            if best_time_diff < 0.001:
                print(f"INFO: Exact time match (dt={best_time_diff*1000:.1f}ms)")
            else:
                print(f"INFO: Paired rover {rover_time:.1f} with base {get_time(best_base):.1f} (diff={best_time_diff:.3f}s)")
    
    return synchronized_pairs


def form_double_differences_combined(synchronized_pairs, nav_data,
                                    reference_ecef, reference_llh,
                                    use_systems: list[str] = None,
                                    cutoff_angle: float = 15.0) -> list[dict]:
    """
    Form double differences from pre-synchronized observation pairs
    
    Parameters:
    -----------
    synchronized_pairs : list
        List of synchronized observation pairs from combine_and_synchronize_observations
    nav_data : dict
        Navigation data
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
    List of DD measurements for all epochs
    """
    
    all_dd_measurements = []
    
    # Default systems to use
    if use_systems is None:
        use_systems = ['G', 'E', 'C', 'J']  # Exclude GLONASS by default
    
    for pair_idx, pair in enumerate(synchronized_pairs):
        rover_epoch = pair['rover_obs']
        base_epoch = pair['base_obs']
        rover_time = pair['rover_time']
        time_diff = pair['time_diff']
        
        # Extract observations from epoch data
        rover_obs_raw = rover_epoch.get('observations', {})
        base_obs_raw = base_epoch.get('observations', {})
        
        # Convert to dictionary if in list format
        if isinstance(rover_obs_raw, list):
            rover_obs = {obs.sat: obs for obs in rover_obs_raw}
        else:
            rover_obs = rover_obs_raw
            
        if isinstance(base_obs_raw, list):
            base_obs = {obs.sat: obs for obs in base_obs_raw}
        else:
            base_obs = base_obs_raw
        
        dd_measurements = []
        
        # Get common satellites from observation dictionaries
        rover_sats = set(rover_obs.keys())
        base_sats = set(base_obs.keys())
        common_sats = rover_sats & base_sats
        
        if len(common_sats) < 2:
            print(f"  Only {len(common_sats)} common satellites - skipping")
            continue
        
        # Filter by system
        filtered_sats = []
        for sat in common_sats:
            # Get system from satellite
            if isinstance(sat, str):
                # String format like 'G01', 'E05', etc.
                system_char = sat[0] if sat else ''
            else:
                # Integer format - need to get system from observation
                # Get the observation to check system
                obs = rover_obs.get(sat)
                if obs and hasattr(obs, 'system'):
                    sys_map = {SYS_GPS: 'G', SYS_GLO: 'R', SYS_GAL: 'E', SYS_BDS: 'C', SYS_QZS: 'J'}
                    system_char = sys_map.get(obs.system, 'G')
                else:
                    # Default to GPS if can't determine
                    system_char = 'G'
            
            if system_char in use_systems:
                filtered_sats.append(sat)
        
        if len(filtered_sats) < 2:
            print(f"  Only {len(filtered_sats)} satellites after filtering - skipping")
            continue
        
        # Convert observations to list format for satpos
        rover_obs_list = []
        for sat in filtered_sats:
            if sat in rover_obs:
                obs = rover_obs[sat]
                obs.sat = sat
                obs.time = rover_time
                rover_obs_list.append(obs)
        
        if not rover_obs_list:
            continue
            
        # Compute satellite positions using rover time
        sat_positions, sat_clocks, _, _ = satpos(rover_obs_list, nav_data)
        
        sat_data = {}
        for i, obs in enumerate(rover_obs_list):
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
            
            # Get base observation for same satellite
            base_obs_sat = base_obs.get(sat)
            if base_obs_sat is None:
                continue
            
            # Get pseudoranges for multiple frequencies
            freqs_data = []
            for freq_idx in [0, 1]:  # L1 and L2
                rover_pr = obs.P[freq_idx] if freq_idx < len(obs.P) and obs.P[freq_idx] != 0 else None
                base_pr = base_obs_sat.P[freq_idx] if freq_idx < len(base_obs_sat.P) and base_obs_sat.P[freq_idx] != 0 else None
                
                if rover_pr is not None and base_pr is not None:
                    # Apply satellite clock correction
                    rover_pr_corrected = rover_pr - sat_clk * CLIGHT
                    base_pr_corrected = base_pr - sat_clk * CLIGHT
                    
                    # Adjust base pseudorange for time difference
                    # Simple approximation: range rate ~ 800 m/s max for LEO
                    # For GNSS satellites, max is ~800 m/s radial velocity
                    # This is a first-order correction
                    if abs(time_diff) > 0.001:  # Only apply if > 1ms
                        range_rate_est = 800.0  # m/s, conservative estimate
                        base_pr_corrected += range_rate_est * time_diff
                    
                    freqs_data.append({
                        'freq_idx': freq_idx,
                        'rover_pr': rover_pr_corrected,
                        'base_pr': base_pr_corrected
                    })
            
            if not freqs_data:
                continue
            
            # Store satellite data
            # Get system from observation
            if hasattr(obs, 'system'):
                sys_id = obs.system
            elif isinstance(obs.sat, str):
                # String format like 'G01' - get system from first character
                sys_char = obs.sat[0] if obs.sat else 'G'
                sys_map = {'G': SYS_GPS, 'R': SYS_GLO, 'E': SYS_GAL, 'C': SYS_BDS, 'J': SYS_QZS}
                sys_id = sys_map.get(sys_char, SYS_GPS)
            else:
                # Default to GPS if can't determine
                sys_id = SYS_GPS
            
            sat_data[obs.sat] = {
                'sat_pos': sat_pos,
                'sat_clk': sat_clk,
                'elevation': elevation,
                'freqs': freqs_data,
                'system': sys_id
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
                continue
            
            # Select reference satellite (highest elevation)
            ref_sat = max(sys_sats.keys(), key=lambda s: sys_sats[s]['elevation'])
            ref_data = sys_sats[ref_sat]
            
            # Form double differences
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
                        'freq_idx': freq_idx,
                        'time': rover_time,
                        'time_diff': time_diff
                    })
        
        print(f"  Formed {len(dd_measurements)} DD measurements")
        all_dd_measurements.append({
            'epoch': pair_idx,
            'time': rover_time,
            'dd_measurements': dd_measurements
        })
    
    return all_dd_measurements