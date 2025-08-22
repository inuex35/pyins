#!/usr/bin/env python3
"""
Double Difference formation for GNSS observations
"""

import numpy as np
from typing import List, Dict, Optional
from ..core.constants import sat2sys, SYS_GPS, SYS_GLO, SYS_GAL, SYS_BDS, SYS_QZS, CLIGHT
from ..geometry.elevation import compute_elevation_angle
from ..gnss.ephemeris import satpos


def form_double_differences(rover_obs, base_obs, nav_data, gps_time, 
                           reference_ecef, reference_llh,
                           use_systems: List[str] = None,
                           cutoff_angle: float = 15.0) -> List[Dict]:
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
    
    # Default systems to use
    if use_systems is None:
        use_systems = ['G', 'E', 'C', 'J']  # Exclude GLONASS by default
    
    # Get common satellites
    rover_sats = {obs.sat for obs in rover_obs}
    base_sats = {obs.sat for obs in base_obs}
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
        base_obs_sat = next((o for o in base_obs if o.sat == obs.sat), None)
        if base_obs_sat is None:
            continue
        
        # Get pseudoranges
        rover_pr = obs.P[0] if obs.P[0] else obs.C[0]
        base_pr = base_obs_sat.P[0] if base_obs_sat.P[0] else base_obs_sat.C[0]
        
        if rover_pr is None or base_pr is None:
            continue
        
        # Apply satellite clock correction to pseudoranges
        # PR_corrected = PR_raw - c*sat_clk
        rover_pr_corrected = rover_pr - sat_clk * CLIGHT
        base_pr_corrected = base_pr - sat_clk * CLIGHT
        
        # Store satellite data
        sat_data[obs.sat] = {
            'sat_pos': sat_pos,
            'sat_clk': sat_clk,
            'elevation': elevation,
            'rover_pr': rover_pr_corrected,
            'base_pr': base_pr_corrected,
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
            
            # Single difference at rover
            sd_rover = data['rover_pr'] - ref_data['rover_pr']
            
            # Single difference at base
            sd_base = data['base_pr'] - ref_data['base_pr']
            
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
                'elevation': data['elevation']
            })
    
    return dd_measurements