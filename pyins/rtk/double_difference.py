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
Now includes RTKLIB-style residual interpolation for handling time differences
"""

import numpy as np
from collections import defaultdict

from ..core.constants import CLIGHT, SYS_BDS, SYS_GAL, SYS_GLO, SYS_GPS, SYS_QZS, sat2sys
from ..geometry.elevation import compute_elevation_angle
from ..gnss.ephemeris import satpos
from ..gnss.rtklib_interp import interpolate_base_epoch, DTTOL, DTTOL_LOWRATE
from ..gnss.residual_interpolation import compute_residual, interpolate_residual, compute_residuals_for_epoch


class DoubleDifferenceProcessor:
    """Process GNSS observations to form double differences"""

    def __init__(self):
        """Initialize double difference processor"""
        pass

    def form_double_differences(self, rover_obs, base_obs, frequency_idx=0):
        """
        Form double difference observations

        Parameters:
        -----------
        rover_obs : list
            Rover observations
        base_obs : list
            Base observations
        frequency_idx : int
            Frequency index (default: 0 for L1)

        Returns:
        --------
        tuple : (dd_pseudorange, dd_carrier, pairs, reference_sats) or None
        """
        # Convert observations to dictionary format if needed
        if isinstance(rover_obs, list):
            rover_obs_dict = {obs.sat: obs for obs in rover_obs}
        else:
            rover_obs_dict = rover_obs

        if isinstance(base_obs, list):
            base_obs_dict = {obs.sat: obs for obs in base_obs}
        else:
            base_obs_dict = base_obs

        # Find common satellites
        common_sats = set(rover_obs_dict.keys()) & set(base_obs_dict.keys())
        if len(common_sats) < 2:
            return None

        # Group by system and find reference satellite (highest PRN for simplicity)
        from collections import defaultdict
        system_sats = defaultdict(list)
        sys_map = {SYS_GPS: 'G', SYS_GLO: 'R', SYS_GAL: 'E', SYS_BDS: 'C', SYS_QZS: 'J'}

        for sat in common_sats:
            sys_id = sat2sys(sat)
            sys_char = sys_map.get(sys_id, 'U')
            system_sats[sys_char].append(sat)

        dd_pseudorange = []
        dd_carrier = []
        pairs = []
        reference_sats = {}

        for system, sats in system_sats.items():
            if len(sats) < 2:
                continue

            # Select reference satellite (highest PRN)
            ref_sat = max(sats)
            reference_sats[system] = ref_sat

            ref_rover_obs = rover_obs_dict[ref_sat]
            ref_base_obs = base_obs_dict[ref_sat]

            # Form DDs for each satellite vs reference
            for sat in sats:
                if sat == ref_sat:
                    continue

                rover_obs_sat = rover_obs_dict[sat]
                base_obs_sat = base_obs_dict[sat]

                # Check if observations have required frequency
                if (len(rover_obs_sat.P) <= frequency_idx or
                    len(base_obs_sat.P) <= frequency_idx or
                    len(ref_rover_obs.P) <= frequency_idx or
                    len(ref_base_obs.P) <= frequency_idx):
                    continue

                # Get pseudorange observations
                rover_pr = rover_obs_sat.P[frequency_idx]
                base_pr = base_obs_sat.P[frequency_idx]
                ref_rover_pr = ref_rover_obs.P[frequency_idx]
                ref_base_pr = ref_base_obs.P[frequency_idx]

                if 0 in [rover_pr, base_pr, ref_rover_pr, ref_base_pr]:
                    continue

                # Form double difference pseudorange
                sd_rover = rover_pr - ref_rover_pr
                sd_base = base_pr - ref_base_pr
                dd_pr = sd_rover - sd_base
                dd_pseudorange.append(dd_pr)

                # Form double difference carrier phase if available
                dd_cp = 0.0
                if (hasattr(rover_obs_sat, 'L') and hasattr(base_obs_sat, 'L') and
                    hasattr(ref_rover_obs, 'L') and hasattr(ref_base_obs, 'L')):

                    if (len(rover_obs_sat.L) > frequency_idx and
                        len(base_obs_sat.L) > frequency_idx and
                        len(ref_rover_obs.L) > frequency_idx and
                        len(ref_base_obs.L) > frequency_idx):

                        rover_cp = rover_obs_sat.L[frequency_idx]
                        base_cp = base_obs_sat.L[frequency_idx]
                        ref_rover_cp = ref_rover_obs.L[frequency_idx]
                        ref_base_cp = ref_base_obs.L[frequency_idx]

                        if 0 not in [rover_cp, base_cp, ref_rover_cp, ref_base_cp]:
                            sd_rover_cp = rover_cp - ref_rover_cp
                            sd_base_cp = base_cp - ref_base_cp
                            dd_cp = sd_rover_cp - sd_base_cp

                dd_carrier.append(dd_cp)
                pairs.append((ref_sat, sat))

        if not dd_pseudorange:
            return None

        return (np.array(dd_pseudorange), np.array(dd_carrier), pairs, reference_sats)


def combine_and_synchronize_observations(rover_obs_list, base_obs_list, max_time_diff=30.0, interpolate=True):
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
        
        # Try extrapolation for data beyond last base epoch (only for 0.2Hz base)
        elif interpolate and base_before is not None and base_after is None:
            # Extrapolate using last two base epochs if rover is beyond last base
            if len(base_sorted) >= 2:
                # Get last two base epochs for extrapolation
                last_base = base_sorted[-1]
                second_last_base = base_sorted[-2]
                last_time = get_time(last_base)
                second_last_time = get_time(second_last_base)
                
                # Only extrapolate if within reasonable range (e.g., 30 seconds)
                if rover_time - last_time <= 30.0 and last_time - second_last_time <= 10.0:
                    # Simple hold-last-value extrapolation (safest approach)
                    synchronized_pairs.append({
                        'rover_obs': rover_obs,
                        'base_obs': last_base,  # Use last base observation
                        'rover_time': rover_time,
                        'base_time': last_time,
                        'time_diff': rover_time - last_time,
                        'interpolated': False,
                        'extrapolated': True
                    })
                    print(f"INFO: Extrapolated for rover time {rover_time:.1f} using last base {last_time:.1f} (diff={rover_time - last_time:.1f}s)")
        
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


def form_double_differences(rover_obs, base_obs, nav_data, gps_time,
                          reference_ecef, reference_llh,
                          use_systems=None, cutoff_angle=15.0,
                          use_residual_interp=True, rover_position=None,
                          base_obs_list=None, base_obs_index=None):
    """
    Form double differences for single epoch - direct implementation
    
    Parameters:
    -----------
    rover_obs : list or dict
        Rover observations for single epoch
    base_obs : list or dict  
        Base observations for single epoch (used when base_obs_list is None)
    nav_data : dict
        Navigation data
    gps_time : float
        GPS time
    reference_ecef : np.ndarray
        Reference position in ECEF (base station)
    reference_llh : np.ndarray
        Reference position in LLH [lat(deg), lon(deg), height(m)]
    use_systems : list[str], optional
        Systems to use ('G', 'E', 'C', 'J'). Default: ['G', 'E', 'C', 'J']
    cutoff_angle : float
        Elevation cutoff angle in degrees
    use_residual_interp : bool
        Whether to use residual interpolation
    rover_position : np.ndarray
        Rover position for residual interpolation
    base_obs_list : list, optional
        Complete list of base station epochs for interpolation
    base_obs_index : int, optional
        Index of current base epoch in base_obs_list
        
    Returns:
    --------
    list : List of DD measurements with freq_idx included
    """
    
    # Default systems
    if use_systems is None:
        use_systems = ['G', 'E', 'C', 'J']  # Exclude GLONASS
    
    # Extract actual time from observations if available
    rover_time = gps_time
    base_time = gps_time
    
    # Check if observations have time information
    if rover_obs and hasattr(rover_obs[0], 'time'):
        rover_time = rover_obs[0].time
    elif rover_obs and isinstance(rover_obs, dict):
        first_obs = next(iter(rover_obs.values()), None)
        if first_obs and hasattr(first_obs, 'time'):
            rover_time = first_obs.time
    
    if base_obs and hasattr(base_obs[0], 'time'):
        base_time = base_obs[0].time
    elif base_obs and isinstance(base_obs, dict):
        first_obs = next(iter(base_obs.values()), None)
        if first_obs and hasattr(first_obs, 'time'):
            base_time = first_obs.time
    
    # Create base epochs list for interpolation
    if base_obs_list is not None and base_obs_index is not None:
        # Use provided base station list for better interpolation
        base_epochs = base_obs_list
    else:
        # Fall back to single epoch (backward compatibility)
        base_epoch = {'gps_time': base_time, 'observations': base_obs}
        base_epochs = [base_epoch]
    
    # Create rover epoch
    rover_epoch = {'gps_time': rover_time, 'observations': rover_obs}
    
    # Synchronize observations with interpolation support
    sync_pairs = combine_and_synchronize_observations(
        [rover_epoch], base_epochs, max_time_diff=30.0, interpolate=True
    )
    
    if not sync_pairs:
        return []
    
    # Process the synchronized pair directly
    pair = sync_pairs[0]
    rover_epoch = pair['rover_obs']
    base_epoch = pair['base_obs']
    time_diff = pair['time_diff']
    
    # Extract observations
    rover_obs_raw = rover_epoch.get('observations', {})
    base_obs_raw = base_epoch.get('observations', {})
    
    # Convert to dictionary if in list format
    if isinstance(rover_obs_raw, list):
        rover_obs_dict = {obs.sat: obs for obs in rover_obs_raw}
    else:
        rover_obs_dict = rover_obs_raw
        
    if isinstance(base_obs_raw, list):
        base_obs_dict = {obs.sat: obs for obs in base_obs_raw}
    else:
        base_obs_dict = base_obs_raw
    
    # Find common satellites
    common_sats = set(rover_obs_dict.keys()) & set(base_obs_dict.keys())
    if len(common_sats) < 2:
        return []
    
    # Group satellites by system
    system_sats = defaultdict(list)
    sys_map = {SYS_GPS: 'G', SYS_GLO: 'R', SYS_GAL: 'E', SYS_BDS: 'C', SYS_QZS: 'J'}
    
    for sat in common_sats:
        sys_id = sat2sys(sat)
        sys_char = sys_map.get(sys_id, 'U')
        if sys_char in use_systems:
            system_sats[sys_char].append(sat)
    
    # Compute satellite positions using satpos function
    all_obs = list(rover_obs_dict.values()) + list(base_obs_dict.values())
    sat_positions, sat_clocks, _, _ = satpos(all_obs, nav_data)
    
    dd_measurements = []
    
    for system, sats in system_sats.items():
        if len(sats) < 2:
            continue
        
        # Prepare satellite data with positions and elevations
        sat_data = {}
        for sat in sats:
            # Find the satellite in the observation list to get the index
            sat_idx = None
            for i, obs in enumerate(all_obs):
                if obs.sat == sat:
                    sat_idx = i
                    break
            
            if sat_idx is None:
                continue
                
            # Get satellite position and clock from satpos results
            sat_pos = sat_positions[sat_idx] if sat_idx < len(sat_positions) else None
            sat_clk = sat_clocks[sat_idx] if sat_idx < len(sat_clocks) else 0.0
            
            if sat_pos is None or np.any(np.isnan(sat_pos)):
                continue
                
            # Compute elevation angle
            elevation = compute_elevation_angle(sat_pos, reference_ecef, reference_llh)
            
            if elevation < cutoff_angle:
                continue
                
            sat_data[sat] = {
                'pos': sat_pos,
                'clk': sat_clk,
                'elevation': elevation,
                'rover_obs': rover_obs_dict[sat],
                'base_obs': base_obs_dict[sat]
            }
        
        if len(sat_data) < 2:
            continue
            
        # Select reference satellite (highest elevation)
        ref_sat = max(sat_data.keys(), key=lambda s: sat_data[s]['elevation'])
        ref_data = sat_data[ref_sat]
        
        # Form double differences for each satellite vs reference
        for sat in sat_data:
            if sat == ref_sat:
                continue
                
            data = sat_data[sat]
            
            # Process each frequency
            rover_freqs = {}
            base_freqs = {}
            
            # Extract frequency data using .P attribute
            rover_obs_sat = data['rover_obs']
            base_obs_sat = data['base_obs'] 
            ref_rover_obs = ref_data['rover_obs']
            ref_base_obs = ref_data['base_obs']
            
            # Check available frequencies
            max_freqs = min(len(rover_obs_sat.P), len(base_obs_sat.P), 
                           len(ref_rover_obs.P), len(ref_base_obs.P))
            
            for freq_idx in range(max_freqs):
                # Check if all pseudoranges are valid
                rover_pr = rover_obs_sat.P[freq_idx] if rover_obs_sat.P[freq_idx] != 0 else None
                base_pr = base_obs_sat.P[freq_idx] if base_obs_sat.P[freq_idx] != 0 else None
                ref_rover_pr = ref_rover_obs.P[freq_idx] if ref_rover_obs.P[freq_idx] != 0 else None
                ref_base_pr = ref_base_obs.P[freq_idx] if ref_base_obs.P[freq_idx] != 0 else None
                
                if None in [rover_pr, base_pr, ref_rover_pr, ref_base_pr]:
                    continue
                
                # Apply satellite clock corrections  
                rover_corrected = rover_pr - data['clk'] * CLIGHT
                base_corrected = base_pr - data['clk'] * CLIGHT
                ref_rover_corrected = ref_rover_pr - ref_data['clk'] * CLIGHT
                ref_base_corrected = ref_base_pr - ref_data['clk'] * CLIGHT
                
                # Form single differences
                sd_rover = rover_corrected - ref_rover_corrected
                sd_base = base_corrected - ref_base_corrected
                
                # Form double difference
                dd_obs = sd_rover - sd_base
                
                # Also extract carrier phase if available
                dd_carrier = None
                wavelength = None
                
                # Check carrier phase availability (L attribute)
                if (hasattr(rover_obs_sat, 'L') and hasattr(base_obs_sat, 'L') and 
                    hasattr(ref_rover_obs, 'L') and hasattr(ref_base_obs, 'L')):
                    
                    # Check if all have enough frequencies
                    if (len(rover_obs_sat.L) > freq_idx and len(base_obs_sat.L) > freq_idx and
                        len(ref_rover_obs.L) > freq_idx and len(ref_base_obs.L) > freq_idx):
                        
                        # Extract carrier phase measurements (in cycles)
                        rover_cp = rover_obs_sat.L[freq_idx] if rover_obs_sat.L[freq_idx] != 0 else None
                        base_cp = base_obs_sat.L[freq_idx] if base_obs_sat.L[freq_idx] != 0 else None
                        ref_rover_cp = ref_rover_obs.L[freq_idx] if ref_rover_obs.L[freq_idx] != 0 else None
                        ref_base_cp = ref_base_obs.L[freq_idx] if ref_base_obs.L[freq_idx] != 0 else None
                        
                        if None not in [rover_cp, base_cp, ref_rover_cp, ref_base_cp]:
                            # Form DD carrier phase
                            sd_rover_cp = rover_cp - ref_rover_cp
                            sd_base_cp = base_cp - ref_base_cp
                            dd_carrier = sd_rover_cp - sd_base_cp
                            
                            # Get wavelength based on frequency and system
                            from ..core.constants import FREQ_GPS_L1, FREQ_GPS_L2, FREQ_GPS_L5
                            from ..core.constants import FREQ_GAL_E1, FREQ_GAL_E5a, FREQ_GAL_E5b
                            from ..core.constants import FREQ_BDS_B1I, FREQ_BDS_B3I
                            
                            # Determine frequency based on system and freq_idx
                            sys_char = sys_map.get(sat2sys(sat), 'U')
                            freq = None
                            
                            if sys_char == 'G':  # GPS
                                freq = [FREQ_GPS_L1, FREQ_GPS_L2, FREQ_GPS_L5][freq_idx] if freq_idx < 3 else None
                            elif sys_char == 'E':  # Galileo
                                freq = [FREQ_GAL_E1, FREQ_GAL_E5a, FREQ_GAL_E5b][freq_idx] if freq_idx < 3 else None
                            elif sys_char == 'C':  # BeiDou
                                freq = [FREQ_BDS_B1I, FREQ_BDS_B3I][freq_idx] if freq_idx < 2 else None
                            elif sys_char == 'J':  # QZSS (same as GPS)
                                freq = [FREQ_GPS_L1, FREQ_GPS_L2, FREQ_GPS_L5][freq_idx] if freq_idx < 3 else None
                            
                            if freq:
                                wavelength = CLIGHT / freq
                
                dd_measurements.append({
                    'sat': sat,
                    'ref_sat': ref_sat,
                    'dd_obs': dd_obs,
                    'dd_carrier': dd_carrier,  # DD carrier phase in cycles
                    'wavelength': wavelength,   # Wavelength in meters
                    'sat_pos': data['pos'],
                    'ref_sat_pos': ref_data['pos'],
                    'sat_clk': data['clk'],
                    'ref_sat_clk': ref_data['clk'],
                    'elevation': data['elevation'],
                    'freq_idx': freq_idx
                })
    
    return dd_measurements


# Keep old synchronize_observations for compatibility  
def synchronize_observations(rover_obs, base_obs, gps_time, max_tdiff=30.0):
    """Legacy synchronization function - just returns base_obs if time close enough"""
    # This is a simplified version for backward compatibility
    # The real synchronization happens in combine_and_synchronize_observations
    return base_obs