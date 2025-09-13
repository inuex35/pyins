#!/usr/bin/env python3
"""
Double Difference formation with combined base/rover observations for better time matching
Now includes RTKLIB-style residual interpolation for handling time differences
"""

import numpy as np
from collections import defaultdict

from ..core.constants import CLIGHT, SYS_BDS, SYS_GAL, SYS_GLO, SYS_GPS, SYS_QZS, sat2sys
from ..geometry.elevation import compute_elevation_angle
from ..gnss.ephemeris import satpos
from ..gnss.rtklib_interp import interpolate_base_epoch, DTTOL, DTTOL_LOWRATE
from .residual_interpolation import compute_residual, interpolate_residual, compute_residuals_for_epoch


def combine_and_synchronize_observations(rover_obs_list, base_obs_list, max_time_diff=0.5, interpolate=True):
    """
    Combine base and rover observations with RTKLIB-style interpolation
    
    Parameters:
    -----------
    rover_obs_list : list
        List of rover observations (list of epochs with 'observations' dict)
    base_obs_list : list
        List of base observations (list of epochs with 'observations' dict)
    max_time_diff : float
        Maximum time difference for pairing (default: 0.5 seconds)
    interpolate : bool
        Whether to interpolate base observations (default: True)
        
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
    
    # Sort base observations by time for interpolation
    base_sorted = sorted(base_obs_list, key=lambda x: get_time(x) or 0)
    
    synchronized_pairs = []
    
    # For each rover observation, find or interpolate base observation
    for rover_obs in rover_obs_list:
        rover_time = get_time(rover_obs)
        if rover_time is None:
            continue
            
        # Find bracketing base observations for interpolation
        base_before = None
        base_after = None
        base_exact = None
        
        for i, base_obs in enumerate(base_sorted):
            base_time = get_time(base_obs)
            if base_time is None:
                continue
                
            if abs(base_time - rover_time) < DTTOL:  # RTKLIBの許容値を使用
                base_exact = base_obs
                break
            elif base_time < rover_time:
                base_before = (i, base_obs, base_time)
            elif base_time > rover_time and base_after is None:
                base_after = (i, base_obs, base_time)
                break
        
        # Use exact match if available
        if base_exact is not None:
            synchronized_pairs.append({
                'rover_obs': rover_obs,
                'base_obs': base_exact,
                'rover_time': rover_time,
                'base_time': get_time(base_exact),
                'time_diff': 0.0,
                'interpolated': False
            })
            print(f"INFO: Exact time match (dt=0.0ms)")
            
        # Otherwise try interpolation if enabled (RTKLIBスタイル)
        elif interpolate and base_before is not None and base_after is not None:
            idx_before, obs_before, time_before = base_before
            idx_after, obs_after, time_after = base_after
            
            # Check if interpolation span is reasonable
            # Allow up to 10 seconds for low-rate base stations (e.g., 5-second intervals)
            if time_after - time_before <= 10.0:  # Max 10 second span for low-rate base
                # RTKLIBスタイルの補間を実行
                base_interp = interpolate_base_epoch(
                    obs_before, obs_after,
                    time_before, time_after,
                    rover_time
                )
                
                synchronized_pairs.append({
                    'rover_obs': rover_obs,
                    'base_obs': base_interp,
                    'rover_time': rover_time,
                    'base_time': rover_time,  # 補間後は同じ時刻
                    'time_diff': 0.0,  # 補間により時刻差はゼロ
                    'interpolated': True
                })
                print(f"INFO: RTKLIB-interpolated for rover time {rover_time:.1f} (base: {time_before:.1f}-{time_after:.1f})")
        
        # Fall back to closest observation
        else:
            best_base = None
            best_time_diff = float('inf')
            
            for base_obs in base_sorted:
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
                    'time_diff': rover_time - get_time(best_base),
                    'interpolated': False
                })
                print(f"INFO: Paired rover {rover_time:.1f} with base {get_time(best_base):.1f} (diff={best_time_diff:.3f}s)")
    
    return synchronized_pairs


def form_double_differences_combined(synchronized_pairs, nav_data,
                                    reference_ecef, reference_llh,
                                    use_systems: list[str] = None,
                                    cutoff_angle: float = 15.0,
                                    use_residual_interp: bool = True,  # Enable by default for better accuracy
                                    rover_position: np.ndarray = None) -> list[dict]:
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
            # Debug output
            print(f"  DEBUG: Rover sats: {len(rover_sats)}, Base sats: {len(base_sats)}, Common: {len(common_sats)}")
            if len(common_sats) > 0:
                print(f"    Common satellites: {list(common_sats)[:5]}")
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
            print(f"  DEBUG: Common={len(common_sats)}, After filter={len(filtered_sats)}")
            print(f"  Only {len(filtered_sats)} satellites after filtering - skipping")
            continue
        
        # Convert observations to list format for satpos  
        rover_obs_list = []
        base_obs_list = []
        for sat in filtered_sats:
            if sat in rover_obs:
                obs = rover_obs[sat]
                obs.sat = sat
                obs.time = rover_time
                rover_obs_list.append(obs)
            
            if sat in base_obs:
                obs = base_obs[sat]
                obs.sat = sat
                obs.time = pair['base_time']  # Use actual base observation time
                base_obs_list.append(obs)
        
        if not rover_obs_list or not base_obs_list:
            continue
            
        # FIXED: Compute satellite positions at correct times
        # Rover satellite positions at rover time
        rover_sat_positions, rover_sat_clocks, _, _ = satpos(rover_obs_list, nav_data)
        
        # Base satellite positions at base time (different if time offset exists)
        base_sat_positions, base_sat_clocks, _, _ = satpos(base_obs_list, nav_data)
        
        # Debug satellite position computation
        valid_rover_positions = sum(1 for pos in rover_sat_positions if pos is not None and not np.any(np.isnan(pos)))
        valid_base_positions = sum(1 for pos in base_sat_positions if pos is not None and not np.any(np.isnan(pos)))
        print(f"  DEBUG: Rover positions {valid_rover_positions}/{len(rover_obs_list)}, Base positions {valid_base_positions}/{len(base_obs_list)}")
        
        sat_data = {}
        for i, obs in enumerate(rover_obs_list):
            sat = obs.sat
            if sat not in filtered_sats:
                continue
            
            # Get rover satellite position (at rover time)
            rover_sat_pos = rover_sat_positions[i]
            rover_sat_clk = rover_sat_clocks[i]
            if rover_sat_pos is None or np.any(np.isnan(rover_sat_pos)):
                continue
                
            # Find corresponding base satellite position (at base time)
            base_sat_idx = None
            for j, base_obs_item in enumerate(base_obs_list):
                if base_obs_item.sat == sat:
                    base_sat_idx = j
                    break
            
            if base_sat_idx is None:
                continue
                
            base_sat_pos = base_sat_positions[base_sat_idx]
            base_sat_clk = base_sat_clocks[base_sat_idx]
            if base_sat_pos is None or np.any(np.isnan(base_sat_pos)):
                continue
            
            # Compute elevation angle using rover satellite position
            elevation = compute_elevation_angle(rover_sat_pos, reference_ecef, reference_llh)
            
            if elevation < cutoff_angle:
                if i < 3:  # Debug first few satellites
                    print(f"    DEBUG: Sat {sat} elevation {elevation:.1f}° < {cutoff_angle}° cutoff")
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
                    # DD cancels satellite clock errors, so we don't apply them here
                    # The satellite clock correction cancels in the double difference:
                    # DD = (ρ_r^s - ρ_b^s) - (ρ_r^ref - ρ_b^ref)
                    # Both satellites' clock errors cancel out
                    
                    # However, with large time differences, receiver clock bias may not cancel
                    # completely due to clock drift. We estimate and remove receiver clock bias.
                    rover_pr_corrected = rover_pr
                    base_pr_corrected = base_pr
                    
                    # IMPORTANT: After RTKLIB-style interpolation, base observations are
                    # interpolated to rover time, so we use the same satellite position
                    # and clock for both receivers in DD computation
                    
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
                'rover_sat_pos': rover_sat_pos,
                'rover_sat_clk': rover_sat_clk,
                'base_sat_pos': base_sat_pos,
                'base_sat_clk': base_sat_clk,
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
        
        # Debug: Show satellite data collected
        if len(sat_data) == 0:
            print(f"  DEBUG: No satellite data collected after elevation/position filtering")
            print(f"    Filtered sats: {len(filtered_sats)}, Sat data: {len(sat_data)}")
        
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
                    
                    # Check if we should use residual interpolation
                    if use_residual_interp and rover_position is not None and abs(time_diff) > 0.001:
                        # RTKLIB-style residual interpolation for large time differences
                        # Compute residuals (observation - geometric range - sat clock)
                        # Residuals include receiver clock bias
                        
                        # Rover residuals (at rover time)
                        rover_res_other = compute_residual(
                            freq_data['rover_pr'],
                            data['rover_sat_pos'],
                            data['rover_sat_clk'],
                            rover_position
                        )
                        rover_res_ref = compute_residual(
                            ref_freq_data['rover_pr'],
                            ref_data['rover_sat_pos'],
                            ref_data['rover_sat_clk'],
                            rover_position
                        )
                        
                        # Base residuals (at base time)
                        base_res_other = compute_residual(
                            freq_data['base_pr'],
                            data['base_sat_pos'],
                            data['base_sat_clk'],
                            reference_ecef  # base position
                        )
                        base_res_ref = compute_residual(
                            ref_freq_data['base_pr'],
                            ref_data['base_sat_pos'],
                            ref_data['base_sat_clk'],
                            reference_ecef  # base position
                        )
                        
                        # Single differences from residuals
                        sd_rover = rover_res_other - rover_res_ref
                        sd_base = base_res_other - base_res_ref
                        
                        # Double difference
                        dd = sd_rover - sd_base
                        
                        if pair_idx == 0 and freq_idx == 0 and sat == sys_sats[ref_sat]:
                            # Debug first DD
                            print(f"    DEBUG: Using residual interpolation for time diff={time_diff:.1f}s")
                            print(f"      Rover residuals: other={rover_res_other:.1f}, ref={rover_res_ref:.1f}")
                            print(f"      Base residuals: other={base_res_other:.1f}, ref={base_res_ref:.1f}")
                            print(f"      DD from residuals: {dd:.1f} m")
                    else:
                        # Standard DD calculation (direct observation difference)
                        # Single difference at rover
                        sd_rover = freq_data['rover_pr'] - ref_freq_data['rover_pr']
                        
                        # Single difference at base
                        sd_base = freq_data['base_pr'] - ref_freq_data['base_pr']
                        
                        # Double difference
                        dd = sd_rover - sd_base
                    
                    # Get system character
                    sys_map = {SYS_GPS: 'G', SYS_GLO: 'R', SYS_GAL: 'E', SYS_BDS: 'C', SYS_QZS: 'J'}
                    system_char = sys_map.get(sys_id, 'U')
                    
                    dd_measurements.append({
                        'sat': sat,
                        'ref_sat': ref_sat,
                        'dd_obs': dd,
                        'sat_pos': data['rover_sat_pos'],  # Use rover satellite position for DD factor
                        'ref_sat_pos': ref_data['rover_sat_pos'],  # Use rover ref satellite position for DD factor
                        'sat_clk': data['rover_sat_clk'],
                        'ref_sat_clk': ref_data['rover_sat_clk'],
                        'base_sat_pos': data['base_sat_pos'],  # Include base satellite positions for validation
                        'base_ref_sat_pos': ref_data['base_sat_pos'],
                        'base_sat_clk': data['base_sat_clk'],
                        'base_ref_sat_clk': ref_data['base_sat_clk'],
                        # Include raw observations for proper factor computation
                        'rover_pr': freq_data['rover_pr'],  # Rover pseudorange to other sat
                        'rover_pr_ref': ref_freq_data['rover_pr'],  # Rover pseudorange to ref sat
                        'base_pr': freq_data['base_pr'],  # Base pseudorange to other sat
                        'base_pr_ref': ref_freq_data['base_pr'],  # Base pseudorange to ref sat
                        'elevation': data['elevation'],
                        'freq_idx': freq_idx,
                        'system': system_char,
                        'time': rover_time,
                        'time_diff': time_diff,
                        'use_residual_interp': use_residual_interp  # Flag indicating computation method
                    })
        
        print(f"  Formed {len(dd_measurements)} DD measurements")
        all_dd_measurements.append({
            'epoch': pair_idx,
            'time': rover_time,
            'dd_measurements': dd_measurements
        })
    
    return all_dd_measurements


def form_double_differences(rover_obs, base_obs, nav_data, gps_time,
                          reference_ecef, reference_llh,
                          use_systems=None, cutoff_angle=15.0,
                          use_residual_interp=True, rover_position=None):
    """
    Legacy API for forming double differences - single epoch
    
    This wraps the combined implementation for backward compatibility
    """
    # Extract actual time from observations if available
    # This handles cases where rover and base have different times
    rover_time = gps_time
    base_time = gps_time
    
    # Check if observations have time information
    if rover_obs and hasattr(rover_obs[0], 'time'):
        rover_time = rover_obs[0].time
    elif rover_obs and isinstance(rover_obs, dict):
        # If observations are in dict format
        first_obs = next(iter(rover_obs.values()), None)
        if first_obs and hasattr(first_obs, 'time'):
            rover_time = first_obs.time
    
    if base_obs and hasattr(base_obs[0], 'time'):
        base_time = base_obs[0].time
    elif base_obs and isinstance(base_obs, dict):
        # If observations are in dict format
        first_obs = next(iter(base_obs.values()), None)
        if first_obs and hasattr(first_obs, 'time'):
            base_time = first_obs.time
    
    # Create single-epoch lists with actual times
    rover_epoch = {
        'gps_time': rover_time,
        'observations': rover_obs
    }
    base_epoch = {
        'gps_time': base_time,
        'observations': base_obs
    }
    
    # Synchronize with interpolation for large time differences (like opensky2 6-second offset)
    sync_pairs = combine_and_synchronize_observations(
        [rover_epoch], [base_epoch], max_time_diff=10.0, interpolate=True
    )
    
    if not sync_pairs:
        return []
    
    # Form DD
    dd_results = form_double_differences_combined(
        sync_pairs, nav_data,
        reference_ecef, reference_llh,
        use_systems, cutoff_angle,
        use_residual_interp, rover_position
    )
    
    # Return just the DD measurements for the single epoch
    if dd_results and dd_results[0]['dd_measurements']:
        return dd_results[0]['dd_measurements']
    return []


# Keep old synchronize_observations for compatibility  
def synchronize_observations(rover_obs, base_obs, gps_time, max_tdiff=30.0):
    """Legacy synchronization function - just returns base_obs if time close enough"""
    # This is a simplified version for backward compatibility
    # The real synchronization happens in combine_and_synchronize_observations
    return base_obs