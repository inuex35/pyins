#!/usr/bin/env python3
"""
RTKLIB-style residual interpolation for double difference processing
"""

import numpy as np
from ..core.constants import CLIGHT


def compute_residual(obs_pr, sat_pos, sat_clk, rcv_pos):
    """
    Compute observation residual = observation - computed range
    
    Parameters:
    -----------
    obs_pr : float
        Observed pseudorange (m)
    sat_pos : np.ndarray
        Satellite position ECEF (m)
    sat_clk : float
        Satellite clock bias (s)
    rcv_pos : np.ndarray
        Receiver position ECEF (m)
        
    Returns:
    --------
    float
        Residual including receiver clock bias (m)
    """
    # Geometric range
    geometric_range = np.linalg.norm(sat_pos[:3] - rcv_pos)
    
    # Computed range = geometric range - satellite clock
    computed_range = geometric_range - CLIGHT * sat_clk
    
    # Residual = observation - computed range
    # This residual includes receiver clock bias
    residual = obs_pr - computed_range
    
    return residual


def interpolate_residual(res_prev, res_curr, t_prev, t_curr, t_target):
    """
    Linearly interpolate residual to target time
    
    Parameters:
    -----------
    res_prev : float
        Previous residual
    res_curr : float
        Current residual
    t_prev : float
        Previous time
    t_curr : float
        Current time
    t_target : float
        Target time for interpolation
        
    Returns:
    --------
    float
        Interpolated residual
    """
    if res_prev is None or abs(t_curr - t_prev) < 1e-9:
        return res_curr
    
    # Linear interpolation
    dt = t_target - t_curr
    dt_total = t_curr - t_prev
    alpha = dt / dt_total
    
    return res_curr + alpha * (res_prev - res_curr)


def form_dd_from_residuals(rover_residuals, base_residuals, ref_sat, other_sat):
    """
    Form double difference from residuals
    
    Parameters:
    -----------
    rover_residuals : dict
        Rover residuals {sat_id: residual}
    base_residuals : dict
        Base residuals {sat_id: residual}
    ref_sat : int
        Reference satellite ID
    other_sat : int
        Other satellite ID
        
    Returns:
    --------
    float
        Double difference value
    """
    # Single differences
    sd_rover = rover_residuals[other_sat] - rover_residuals[ref_sat]
    sd_base = base_residuals[other_sat] - base_residuals[ref_sat]
    
    # Double difference
    dd = sd_rover - sd_base
    
    return dd


def compute_residuals_for_epoch(obs_list, sat_positions, sat_clocks, rcv_pos):
    """
    Compute residuals for all satellites in an epoch
    
    Parameters:
    -----------
    obs_list : list
        List of observations
    sat_positions : list
        List of satellite positions
    sat_clocks : list
        List of satellite clock biases
    rcv_pos : np.ndarray
        Receiver position ECEF
        
    Returns:
    --------
    dict
        Dictionary of residuals {sat_id: residual}
    """
    residuals = {}
    
    for i, obs in enumerate(obs_list):
        if sat_positions[i] is None or np.any(np.isnan(sat_positions[i])):
            continue
            
        sat_id = obs.sat
        
        # Get pseudorange (L1 C/A code)
        if hasattr(obs, 'P') and len(obs.P) > 0 and obs.P[0] != 0:
            obs_pr = obs.P[0]
        else:
            continue
        
        # Compute residual
        residual = compute_residual(
            obs_pr,
            sat_positions[i],
            sat_clocks[i],
            rcv_pos
        )
        
        residuals[sat_id] = residual
    
    return residuals