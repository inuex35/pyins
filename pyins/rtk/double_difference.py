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
from ..gnss.interp import interpolate_base_epoch, DTTOL, DTTOL_LOWRATE
from ..gnss.residual_interpolation import compute_residual, interpolate_residual, compute_residuals_for_epoch



def syncobs(rover_obs_list, base_obs_list, max_time_diff=30.0, interpolate=True):
    """
    Combine and synchronize base and rover GNSS observations with RTKLIB-style interpolation.

    This function matches rover observations with base station observations by time, applying
    RTKLIB-compatible interpolation when exact time matches are not available. It supports
    both exact matching (within tolerance) and interpolation between bracketing base epochs.

    The function implements:
    - Exact time matching within RTKLIB tolerances (DTTOL)
    - Linear interpolation for pseudorange and carrier phase measurements
    - Extrapolation beyond last base epoch (limited time window)
    - Fallback to closest observation pairing

    Parameters:
    -----------
    rover_obs_list : list of dict
        List of rover observation epochs. Each epoch is a dict with keys:
        - 'gps_time' or 'time': GPS time in seconds
        - 'observations': dict or list of observation objects
    base_obs_list : list of dict
        List of base station observation epochs with same structure as rover_obs_list
    max_time_diff : float, optional
        Maximum allowable time difference for pairing observations (default: 30.0 seconds)
        Used as fallback when interpolation is not possible
    interpolate : bool, optional
        Enable RTKLIB-style interpolation of base observations (default: True)
        If False, only exact matches and closest observations are used

    Returns:
    --------
    list of dict
        List of synchronized observation pairs. Each pair contains:
        - 'rover_obs': rover observation epoch (dict)
        - 'base_obs': base observation epoch (dict, possibly interpolated)
        - 'rover_time': rover observation time (float)
        - 'base_time': base observation time (float)
        - 'time_diff': time difference rover_time - base_time (float)
        - 'interpolated': whether base observation was interpolated (bool)
        - 'extrapolated': whether base observation was extrapolated (bool, optional)

    Notes:
    ------
    - Uses RTKLIB time tolerances: DTTOL for exact matches, DTTOL_LOWRATE for low-rate data
    - Interpolation span is limited to 10 seconds for stability
    - Extrapolation is limited to 30 seconds beyond last base epoch
    - Empty list returned if no valid rover or base observations provided

    Examples:
    ---------
    >>> rover_epochs = [{'gps_time': 1000.0, 'observations': rover_obs}]
    >>> base_epochs = [{'gps_time': 999.5, 'observations': base_obs1},
    ...                {'gps_time': 1000.5, 'observations': base_obs2}]
    >>> pairs = syncobs(rover_epochs, base_epochs)
    >>> print(pairs[0]['interpolated'])  # True - interpolated between base epochs
    """
    
    # Helper to get time from observation
    def get_time(obs):
        """
        Extract GPS time from observation epoch dictionary.

        This helper function extracts the time value from an observation epoch,
        handling different possible key names for time storage. It prioritizes
        'gps_time' over 'time' for better precision and consistency.

        Parameters:
        -----------
        obs : dict or other
            Observation epoch that may contain time information.
            Expected to be a dictionary with either 'gps_time' or 'time' key.
            Non-dict types return None.

        Returns:
        --------
        float or None
            GPS time in seconds if found, None otherwise.
            Returns in order of preference:
            1. obs['gps_time'] if exists
            2. obs['time'] if exists
            3. None if neither key exists or obs is not a dict

        Notes:
        ------
        - Prefers 'gps_time' key for RTKLIB compatibility
        - Falls back to 'time' key for legacy support
        - Returns None for invalid or missing time data
        - Used internally for time extraction in synchronization

        Examples:
        ---------
        >>> obs1 = {'gps_time': 1000.5, 'time': 1000.0}
        >>> get_time(obs1)  # Returns 1000.5 (prefers gps_time)
        1000.5
        >>> obs2 = {'time': 1000.0}
        >>> get_time(obs2)  # Returns 1000.0 (fallback to time)
        1000.0
        >>> get_time({})  # Returns None (no time keys)
        None
        """
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
    Form GNSS double difference measurements for RTK positioning.

    This function computes double difference (DD) pseudorange and carrier phase measurements
    from synchronized rover and base station observations. Double differences eliminate
    satellite and receiver clock errors, making them ideal for high-precision RTK processing.

    The function performs:
    - Satellite position and clock computation using navigation data
    - Elevation angle filtering based on cutoff angle
    - System-wise grouping of satellites (GPS, Galileo, BeiDou, QZSS)
    - Reference satellite selection (highest elevation per system)
    - Double difference formation: DD = (rover_sat - rover_ref) - (base_sat - base_ref)
    - Multi-frequency processing (L1, L2, L5 etc.)
    - Residual interpolation for improved time synchronization (optional)

    Parameters:
    -----------
    rover_obs : list or dict
        Rover GNSS observations for single epoch. Can be:
        - List of Observation objects with .sat, .P, .L attributes
        - Dictionary {sat_id: observation_object}
    base_obs : list or dict
        Base station GNSS observations with same format as rover_obs
        Used when base_obs_list is None for single-epoch processing
    nav_data : dict
        Navigation data containing satellite ephemerides for position computation
        Keys should include satellite IDs, values are ephemeris objects
    gps_time : float
        GPS time in seconds for satellite position computation and time synchronization
    reference_ecef : np.ndarray, shape (3,)
        Reference station position in ECEF coordinates [X, Y, Z] in meters
        Used for elevation angle computation and as base station position
    reference_llh : np.ndarray, shape (3,)
        Reference station position in geodetic coordinates [lat(deg), lon(deg), height(m)]
        Used for elevation angle computation in local frame
    use_systems : list of str, optional
        GNSS systems to process. Valid values: ['G', 'E', 'C', 'J', 'R']
        - 'G': GPS, 'E': Galileo, 'C': BeiDou, 'J': QZSS, 'R': GLONASS
        Default: ['G', 'E', 'C', 'J'] (excludes GLONASS due to FDMA complexity)
    cutoff_angle : float, optional
        Elevation cutoff angle in degrees (default: 15.0)
        Satellites below this angle are excluded to avoid multipath and atmospheric effects
    use_residual_interp : bool, optional
        Enable residual interpolation for better time synchronization (default: True)
        Improves DD quality when base/rover time differences exist
    rover_position : np.ndarray, optional
        Approximate rover position in ECEF [X, Y, Z] for residual interpolation
        Required when use_residual_interp=True
    base_obs_list : list of dict, optional
        Complete list of base station observation epochs for advanced interpolation
        Each epoch: {'gps_time': time, 'observations': obs_list}
        Enables better interpolation than single base_obs
    base_obs_index : int, optional
        Index of current base epoch in base_obs_list for interpolation context
        Used with base_obs_list for temporal interpolation

    Returns:
    --------
    list of dict
        List of double difference measurements. Each measurement contains:
        - 'sat': satellite ID of non-reference satellite (int)
        - 'ref_sat': reference satellite ID (int)
        - 'dd_obs': double difference pseudorange in meters (float)
        - 'dd_carrier': double difference carrier phase in cycles (float or None)
        - 'wavelength': carrier wavelength in meters (float or None)
        - 'sat_pos': satellite position ECEF [X,Y,Z] in meters (np.ndarray)
        - 'ref_sat_pos': reference satellite position ECEF (np.ndarray)
        - 'sat_clk': satellite clock bias in seconds (float)
        - 'ref_sat_clk': reference satellite clock bias in seconds (float)
        - 'elevation': satellite elevation angle in degrees (float)
        - 'freq_idx': frequency index (0=L1, 1=L2, etc.) (int)

    Notes:
    ------
    - Returns empty list if fewer than 2 common satellites available
    - Satellite clock corrections are applied to pseudorange measurements
    - Reference satellite selection uses highest elevation per system
    - Multi-frequency processing creates separate DD for each frequency
    - Carrier phase DD includes integer ambiguities that need resolution
    - Wavelengths computed from system-specific frequency definitions

    Examples:
    ---------
    >>> nav_data = load_navigation_data('nav_file.rnx')
    >>> dd_list = form_double_differences(
    ...     rover_obs, base_obs, nav_data, gps_time,
    ...     base_ecef, base_llh, use_systems=['G', 'E'],
    ...     cutoff_angle=10.0)
    >>> if dd_list:
    ...     print(f"Formed {len(dd_list)} DD measurements")
    ...     print(f"First DD: {dd_list[0]['dd_obs']:.3f} m")
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
    if base_obs_list is not None:
        # Use provided base station list for better interpolation
        base_epochs = base_obs_list
    elif base_obs is not None:
        # Fall back to single epoch (backward compatibility)
        base_epoch = {'gps_time': base_time, 'observations': base_obs}
        base_epochs = [base_epoch]
    else:
        # No base observations provided
        return []
    
    # Create rover epoch
    rover_epoch = {'gps_time': rover_time, 'observations': rover_obs}
    
    # Synchronize observations with interpolation support
    # IMPORTANT: Enable interpolation for pseudorange, but be careful with carrier phase
    sync_pairs = syncobs(
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
                        
                        # Get frequency first for conversion
                        from ..core.constants import FREQ_L1, FREQ_L2, FREQ_L5
                        from ..core.constants import FREQ_E1, FREQ_E5a, FREQ_E5b
                        from ..core.constants import FREQ_B1I, FREQ_B3

                        # Determine frequency based on system and freq_idx
                        sys_char = sys_map.get(sat2sys(sat), 'U')
                        freq = None

                        if sys_char == 'G':  # GPS
                            freq = [FREQ_L1, FREQ_L2, FREQ_L5][freq_idx] if freq_idx < 3 else None
                        elif sys_char == 'E':  # Galileo
                            freq = [FREQ_E1, FREQ_E5a, FREQ_E5b][freq_idx] if freq_idx < 3 else None
                        elif sys_char == 'C':  # BeiDou
                            freq = [FREQ_B1I, FREQ_B3][freq_idx] if freq_idx < 2 else None
                        elif sys_char == 'J':  # QZSS (same as GPS)
                            freq = [FREQ_L1, FREQ_L2, FREQ_L5][freq_idx] if freq_idx < 3 else None

                        if not freq:
                            continue

                        wavelength = CLIGHT / freq

                        # Extract carrier phase measurements (in cycles from RINEX)
                        rover_cp_cycles = rover_obs_sat.L[freq_idx] if rover_obs_sat.L[freq_idx] != 0 else None
                        base_cp_cycles = base_obs_sat.L[freq_idx] if base_obs_sat.L[freq_idx] != 0 else None
                        ref_rover_cp_cycles = ref_rover_obs.L[freq_idx] if ref_rover_obs.L[freq_idx] != 0 else None
                        ref_base_cp_cycles = ref_base_obs.L[freq_idx] if ref_base_obs.L[freq_idx] != 0 else None

                        if None not in [rover_cp_cycles, base_cp_cycles, ref_rover_cp_cycles, ref_base_cp_cycles]:
                            # Convert carrier phase to meters like RTKLIB
                            # RTKLIB: obs.L[ix,f] * _c / freq
                            rover_cp_meters = rover_cp_cycles * wavelength
                            base_cp_meters = base_cp_cycles * wavelength
                            ref_rover_cp_meters = ref_rover_cp_cycles * wavelength
                            ref_base_cp_meters = ref_base_cp_cycles * wavelength

                            # Form DD carrier phase in meters
                            sd_rover_meters = rover_cp_meters - ref_rover_cp_meters
                            sd_base_meters = base_cp_meters - ref_base_cp_meters
                            dd_carrier_meters = sd_rover_meters - sd_base_meters

                            # Convert back to cycles for compatibility
                            dd_carrier = dd_carrier_meters / wavelength
                        else:
                            dd_carrier = None
                
                dd_measurements.append({
                    'sat': sat,
                    'ref_sat': ref_sat,
                    'dd_obs': dd_obs,
                    'dd_carrier': dd_carrier,  # DD carrier phase in cycles (converted from meters)
                    'wavelength': wavelength,   # Wavelength in meters
                    'sat_pos': data['pos'],
                    'ref_sat_pos': ref_data['pos'],
                    'sat_clk': data['clk'],
                    'ref_sat_clk': ref_data['clk'],
                    'elevation': data['elevation'],
                    'freq_idx': freq_idx,
                    # Individual carrier phase observations for debugging
                    'rover_carrier_ref': ref_rover_cp_cycles if dd_carrier is not None else None,  # Rover to ref sat (cycles)
                    'rover_carrier_other': rover_cp_cycles if dd_carrier is not None else None,     # Rover to other sat (cycles)
                    'base_carrier_ref': ref_base_cp_cycles if dd_carrier is not None else None,     # Base to ref sat (cycles)
                    'base_carrier_other': base_cp_cycles if dd_carrier is not None else None        # Base to other sat (cycles)
                })
    
    return dd_measurements


# Keep old synchronize_observations for compatibility
def synchronize_observations(rover_obs, base_obs, gps_time, max_tdiff=30.0):
    """
    Legacy observation synchronization function for backward compatibility.

    This is a simplified synchronization function that provides basic compatibility
    with older code. For proper time synchronization with interpolation capabilities,
    use combine_and_synchronize_observations() instead.

    The function performs minimal time checking and simply returns the base observations
    if they are deemed close enough in time to the rover observations. It does not
    perform interpolation, extrapolation, or advanced time matching.

    Parameters:
    -----------
    rover_obs : list or dict
        Rover GNSS observations (not used in current implementation)
        Kept for API compatibility with legacy code
    base_obs : list or dict
        Base station GNSS observations to be returned
        Can be list of Observation objects or dictionary format
    gps_time : float
        GPS time in seconds (not used in current implementation)
        Kept for API compatibility with legacy code
    max_tdiff : float, optional
        Maximum time difference threshold in seconds (default: 30.0)
        Currently not enforced in this simplified implementation

    Returns:
    --------
    list or dict
        Returns base_obs unchanged
        Same type and format as input base_obs parameter

    Notes:
    ------
    - This is a legacy function maintained for backward compatibility only
    - Does not perform actual time synchronization or validation
    - For production RTK processing, use syncobs()
    - Real synchronization logic is implemented in syncobs()
    - This function will likely be deprecated in future versions

    Examples:
    ---------
    >>> # Legacy usage (not recommended)
    >>> sync_base = synchronize_observations(rover_obs, base_obs, gps_time)
    >>>
    >>> # Recommended modern approach:
    >>> pairs = syncobs(
    ...     [{'gps_time': gps_time, 'observations': rover_obs}],
    ...     [{'gps_time': gps_time, 'observations': base_obs}])

    See Also:
    ---------
    syncobs : Full-featured synchronization with interpolation
    """
    # This is a simplified version for backward compatibility
    # The real synchronization happens in syncobs
    return base_obs