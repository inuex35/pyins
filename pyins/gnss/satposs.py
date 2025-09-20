"""
Satellite position computation with transmission time correction (RTKLIB-style)
"""

import numpy as np
from .ephemeris import compute_satellite_position

# Speed of light (m/s)
CLIGHT = 299792458.0

def satposs(obs_list, nav_data, obs_time=None):
    """
    Compute satellite positions and clocks at transmission time (RTKLIB-style)

    This function properly accounts for:
    1. Signal transmission time (pseudorange/c)
    2. Satellite clock bias correction
    3. Final position at corrected transmission time

    Parameters
    ----------
    obs_list : list of Observation
        List of observations with pseudoranges
    nav_data : NavigationData
        Navigation data (ephemeris)
    obs_time : float, optional
        Observation reception time (GPS seconds)
        If None, will try to get from first observation

    Returns
    -------
    rs : np.ndarray (n x 3)
        Satellite positions at transmission time (ECEF)
    dts : np.ndarray (n,)
        Satellite clock biases (seconds)
    var : np.ndarray (n,)
        Position variances (m^2)
    svh : np.ndarray (n,)
        Satellite health flags (0=healthy, -1=unhealthy)

    Notes
    -----
    This implementation follows RTKLIB's approach:
    1. t_tx = obs_time - pseudorange/c
    2. dt = ephclk(t_tx, eph)  # Get satellite clock bias
    3. t_tx_corrected = t_tx - dt
    4. Compute satellite position at t_tx_corrected
    """

    n = len(obs_list)
    rs = np.zeros((n, 3))
    dts = np.zeros(n)
    var = np.zeros(n)
    svh = np.zeros(n, dtype=int)

    # Get observation time
    if obs_time is None:
        if hasattr(obs_list[0], 'time'):
            obs_time = obs_list[0].time
        elif hasattr(obs_list[0], 'gps_time'):
            obs_time = obs_list[0].gps_time
        else:
            raise ValueError("Cannot determine observation time")

    for i, obs in enumerate(obs_list):
        # Get pseudorange (prefer C1/P1 on L1)
        pr = 0
        if hasattr(obs, 'observables') and 'C1' in obs.observables:
            pr = obs.observables['C1'].get('value', 0)
        elif hasattr(obs, 'P') and len(obs.P) > 0:
            pr = obs.P[0] if obs.P[0] > 0 else (obs.P[1] if len(obs.P) > 1 else 0)

        if pr <= 0:
            svh[i] = -1  # No valid pseudorange
            continue

        # Step 1: Calculate transmission time by signal travel time
        # t_tx = obs_time - pr/c
        t_tx = obs_time - pr / CLIGHT

        # Step 2: Get initial satellite position and clock at transmission time
        pos_init, clk_init, eph_var = compute_satellite_position(obs.sat, t_tx, nav_data)

        if pos_init is None:
            svh[i] = -1  # No ephemeris available
            continue

        # Step 3: Correct transmission time by satellite clock bias
        # This is what RTKLIB does: t = timeadd(t, -dt)
        t_tx_corrected = t_tx - clk_init

        # Step 4: Compute final satellite position at corrected transmission time
        pos_final, clk_final, eph_var_final = compute_satellite_position(obs.sat, t_tx_corrected, nav_data)

        if pos_final is None:
            svh[i] = -1
            continue

        # Store results
        rs[i, :] = pos_final[:3]  # Position (x, y, z)
        dts[i] = clk_final  # Clock bias
        var[i] = eph_var_final if eph_var_final is not None else 0.0
        svh[i] = 0  # Healthy

    return rs, dts, var, svh


def satposs_dict(obs_dict, nav_data, obs_time=None):
    """
    Compute satellite positions for observations stored in dictionary

    Parameters
    ----------
    obs_dict : dict
        Dictionary mapping satellite ID to observation
    nav_data : NavigationData
        Navigation data
    obs_time : float, optional
        Observation reception time

    Returns
    -------
    sat_positions : dict
        Dictionary mapping satellite ID to position
    sat_clocks : dict
        Dictionary mapping satellite ID to clock bias
    """

    # Convert dict to list
    obs_list = list(obs_dict.values())
    sat_ids = list(obs_dict.keys())

    # Compute positions
    rs, dts, var, svh = satposs(obs_list, nav_data, obs_time)

    # Convert back to dict
    sat_positions = {}
    sat_clocks = {}

    for i, sat_id in enumerate(sat_ids):
        if svh[i] == 0:  # Healthy
            sat_positions[sat_id] = rs[i, :]
            sat_clocks[sat_id] = dts[i]

    return sat_positions, sat_clocks