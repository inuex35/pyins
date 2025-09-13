#!/usr/bin/env python3
"""
RTKLIB-style observation interpolation
"""

import numpy as np

def interpolate_pseudorange(pr1, pr2, t1, t2, t_target):
    """
    Linearly interpolate pseudorange between two epochs
    
    Parameters:
    -----------
    pr1 : float
        Pseudorange at time t1
    pr2 : float
        Pseudorange at time t2
    t1 : float
        Time of first observation
    t2 : float
        Time of second observation  
    t_target : float
        Target time for interpolation
        
    Returns:
    --------
    float
        Interpolated pseudorange at t_target
    """
    if abs(t2 - t1) < 1e-9:
        return pr1
    
    # Linear interpolation factor
    alpha = (t_target - t1) / (t2 - t1)
    
    # Interpolate
    pr_interp = pr1 + alpha * (pr2 - pr1)
    
    return pr_interp


def interpolate_base_observations(base_obs_prev, base_obs_curr, base_obs_next, 
                                 rover_time, base_time):
    """
    RTKLIB-style interpolation of base observations to rover time
    
    Parameters:
    -----------
    base_obs_prev : dict
        Previous base observation (if available)
    base_obs_curr : dict
        Current base observation
    base_obs_next : dict
        Next base observation (if available)
    rover_time : float
        Rover observation time
    base_time : float
        Base observation time
        
    Returns:
    --------
    dict
        Interpolated base observations
    """
    time_diff = rover_time - base_time
    
    # If time difference is small, use current observation
    if abs(time_diff) < 0.001:  # Less than 1ms
        return base_obs_curr
    
    # Determine which observations to use for interpolation
    if time_diff > 0 and base_obs_next is not None:
        # Rover time is after base time, interpolate with next
        t1 = base_time
        t2 = base_obs_next.get('gps_time', base_time + 1.0)
        obs1 = base_obs_curr
        obs2 = base_obs_next
    elif time_diff < 0 and base_obs_prev is not None:
        # Rover time is before base time, interpolate with previous
        t1 = base_obs_prev.get('gps_time', base_time - 1.0)
        t2 = base_time
        obs1 = base_obs_prev
        obs2 = base_obs_curr
    else:
        # No adjacent observation available, return current
        return base_obs_curr
    
    # Check if interpolation span is reasonable
    if abs(t2 - t1) > 2.0:  # Don't interpolate over more than 2 seconds
        return base_obs_curr
    
    # Interpolate observations
    interp_obs = {}
    
    # Copy metadata
    interp_obs['gps_time'] = rover_time
    interp_obs['time'] = rover_time
    
    # Interpolate each satellite's observations
    obs_dict = {}
    
    # Get observations as dictionaries
    if isinstance(obs1.get('observations'), list):
        obs1_dict = {obs.sat: obs for obs in obs1['observations']}
    else:
        obs1_dict = obs1.get('observations', {})
        
    if isinstance(obs2.get('observations'), list):
        obs2_dict = {obs.sat: obs for obs in obs2['observations']}
    else:
        obs2_dict = obs2.get('observations', {})
    
    # Interpolate common satellites
    for sat in obs1_dict:
        if sat in obs2_dict:
            obs_interp = type(obs1_dict[sat])()  # Create new observation object
            
            # Copy satellite ID
            obs_interp.sat = sat
            
            # Interpolate pseudoranges
            if hasattr(obs1_dict[sat], 'P') and hasattr(obs2_dict[sat], 'P'):
                obs_interp.P = []
                for i in range(len(obs1_dict[sat].P)):
                    if obs1_dict[sat].P[i] != 0 and obs2_dict[sat].P[i] != 0:
                        pr_interp = interpolate_pseudorange(
                            obs1_dict[sat].P[i], obs2_dict[sat].P[i],
                            t1, t2, rover_time
                        )
                        obs_interp.P.append(pr_interp)
                    else:
                        obs_interp.P.append(0.0)
                obs_interp.P = np.array(obs_interp.P)
            
            # Copy other attributes
            if hasattr(obs1_dict[sat], 'L'):
                obs_interp.L = obs1_dict[sat].L  # Carrier phase needs more careful handling
            if hasattr(obs1_dict[sat], 'D'):
                obs_interp.D = obs1_dict[sat].D  # Doppler
            if hasattr(obs1_dict[sat], 'SNR'):
                obs_interp.SNR = obs1_dict[sat].SNR
            if hasattr(obs1_dict[sat], 'system'):
                obs_interp.system = obs1_dict[sat].system
                
            obs_dict[sat] = obs_interp
    
    interp_obs['observations'] = obs_dict
    
    return interp_obs