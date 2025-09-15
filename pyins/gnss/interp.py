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
RTKLIB-style observation interpolation implementation
Based on RTKLIB's interpobs() and syncobs() functions
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

from ..core.constants import CLIGHT, FREQ_L1, FREQ_L2, FREQ_L5
from ..core.stats import DTTOL, MAX_INTERP_TIME, MAXDTOE

# Derived constants
WAVELENGTH_L1 = CLIGHT / FREQ_L1  # L1 wavelength


def timediff(t1: float, t2: float) -> float:
    """Calculate time difference (considering GPS week rollover)"""
    dt = t1 - t2
    if dt > 302400.0:  # difference > 3.5 days
        dt -= 604800.0  # subtract 1 week
    elif dt < -302400.0:
        dt += 604800.0  # add 1 week
    return dt


def interp_pseudorange(pr1: float, pr2: float, t1: float, t2: float, t: float) -> float:
    """
    Linear interpolation of pseudorange (RTKLIB style)

    Parameters:
    -----------
    pr1, pr2 : float
        Pseudorange at previous and next epochs [m]
    t1, t2 : float
        Time of previous and next epochs [s]
    t : float
        Target interpolation time [s]

    Returns:
    --------
    float : Interpolated pseudorange [m]
    """
    if abs(t2 - t1) < 1e-9 or pr1 == 0 or pr2 == 0:
        return pr1
    
    # Linear interpolation coefficient
    alpha = (t - t1) / (t2 - t1)

    # Consider pseudorange rate of change
    # RTKLIB considers satellite motion
    pr_rate = (pr2 - pr1) / (t2 - t1)

    # Clip if rate is abnormally large (satellite velocity limit)
    MAX_RANGE_RATE = 1000.0  # m/s (conservative value)
    if abs(pr_rate) > MAX_RANGE_RATE:
        pr_rate = np.sign(pr_rate) * MAX_RANGE_RATE

    # Interpolate
    pr_interp = pr1 + pr_rate * (t - t1)
    
    return pr_interp


def interp_carrier_phase(L1: float, L2: float, D1: float, D2: float,
                        t1: float, t2: float, t: float, freq: float = FREQ_L1) -> float:
    """
    Carrier phase interpolation (using Doppler frequency)
    Mimics RTKLIB's interpobs function

    Parameters:
    -----------
    L1, L2 : float
        Carrier phase at previous and next epochs [cycle]
    D1, D2 : float
        Doppler frequency at previous and next epochs [Hz]
    t1, t2 : float
        Time of previous and next epochs [s]
    t : float
        Target interpolation time [s]
    freq : float
        Carrier frequency [Hz]

    Returns:
    --------
    float : Interpolated carrier phase [cycle]
    """
    if abs(t2 - t1) < 1e-9 or L1 == 0 or L2 == 0:
        return L1

    dt = t - t1
    dt_total = t2 - t1

    # RTKLIB approach: interpolation using Doppler
    # Estimate phase change from Doppler frequency
    if D1 != 0 and D2 != 0:
        # Use average Doppler (more stable)
        D_avg = (D1 + D2) / 2.0

        # wavelength = c / freq
        wavelength = CLIGHT / freq

        # Phase change from Doppler [cycles]
        # RTKLIB-py style: Doppler sign is opposite to phase rate
        # Negative Doppler means satellite approaching (phase increasing)
        phase_change = -D_avg * dt

        # Interpolated phase
        L_interp = L1 + phase_change
    else:
        # Linear interpolation when Doppler not available
        # For accumulated carrier phase, rate of change is relatively constant
        delta_L = L2 - L1

        # Correction for large time intervals
        # Properly interpolate even for 5-second interval data
        if abs(dt_total) > 1.0:
            # Calculate phase rate (cycles/second)
            phase_rate = delta_L / dt_total

            # Check for abnormal rate (satellite line-of-sight velocity usually within ±4000 cycles/s)
            if abs(phase_rate) > 10000:
                # Use nearest observation if abnormal
                if abs(dt) < abs(t2 - t):
                    return L1
                else:
                    return L2

            # Interpolate using rate
            L_interp = L1 + phase_rate * dt
        else:
            # Simple linear interpolation for short intervals
            L_interp = L1 + delta_L * dt / dt_total

    return L_interp


def interp_observation(obs1: dict, obs2: dict, t1: float, t2: float, t: float) -> dict:
    """
    Interpolation of entire observation (equivalent to RTKLIB's interpobs)

    Parameters:
    -----------
    obs1, obs2 : dict
        Observations at previous and next epochs
    t1, t2 : float
        Time of previous and next epochs
    t : float
        Target interpolation time

    Returns:
    --------
    dict : Interpolated observation
    """
    if abs(t - t1) < DTTOL:
        return obs1
    if abs(t - t2) < DTTOL:
        return obs2
    
    # Store interpolated observations
    obs_interp = {}

    # Interpolate pseudorange for each frequency
    if hasattr(obs1, 'P') and hasattr(obs2, 'P'):
        P_interp = []
        for i in range(min(len(obs1.P), len(obs2.P))):
            if obs1.P[i] != 0 and obs2.P[i] != 0:
                pr = interp_pseudorange(obs1.P[i], obs2.P[i], t1, t2, t)
                P_interp.append(pr)
            else:
                P_interp.append(0.0)
        obs_interp['P'] = np.array(P_interp)
    
    # Carrier phase interpolation (using Doppler, more carefully)
    if hasattr(obs1, 'L') and hasattr(obs2, 'L'):
        L_interp = []
        for i in range(min(len(obs1.L), len(obs2.L))):
            if obs1.L[i] != 0 and obs2.L[i] != 0:
                # If Doppler is available
                if hasattr(obs1, 'D') and hasattr(obs2, 'D') and \
                   i < len(obs1.D) and i < len(obs2.D) and \
                   obs1.D[i] != 0 and obs2.D[i] != 0:
                    # Determine frequency (L1, L2, L5, etc.)
                    if i == 0:
                        freq = FREQ_L1
                    elif i == 1:
                        freq = FREQ_L2
                    else:
                        freq = FREQ_L5

                    L = interp_carrier_phase(obs1.L[i], obs2.L[i],
                                           obs1.D[i], obs2.D[i],
                                           t1, t2, t, freq)
                else:
                    # RTKLIB style: perform linear interpolation even without Doppler
                    # This prevents carrier phase jumps due to time discontinuities
                    dt = t2 - t1
                    if dt != 0:
                        # Linear interpolation: L = L1 + (L2-L1) * (t-t1)/(t2-t1)
                        L = obs1.L[i] + (obs2.L[i] - obs1.L[i]) * (t - t1) / dt
                    else:
                        L = obs1.L[i]
                L_interp.append(L)
            else:
                L_interp.append(0.0)
        obs_interp['L'] = np.array(L_interp)
    
    # Copy other attributes
    obs_interp['sat'] = obs1.sat if hasattr(obs1, 'sat') else obs1.get('sat')
    obs_interp['system'] = obs1.system if hasattr(obs1, 'system') else obs1.get('system')
    
    return obs_interp


def syncobs_rtklib(rover_obs_list: List[dict], base_obs_list: List[dict]) -> List[Tuple[dict, dict, float]]:
    """
    Observation synchronization mimicking RTKLIB's syncobs function

    Parameters:
    -----------
    rover_obs_list : List[dict]
        Rover observation list
    base_obs_list : List[dict]
        Base station observation list

    Returns:
    --------
    List[Tuple[dict, dict, float]] : List of synchronized pairs
    """
    dttol = DTTOL
    
    synchronized = []
    i, j = 0, 0  # indices

    while i < len(rover_obs_list) and j < len(base_obs_list):
        # Get time
        t_rover = rover_obs_list[i].get('gps_time', rover_obs_list[i].get('time'))
        t_base = base_obs_list[j].get('gps_time', base_obs_list[j].get('time'))
        
        if t_rover is None or t_base is None:
            break
        
        dt = timediff(t_rover, t_base)
        
        if abs(dt) < dttol:
            # Time is close enough - add as pair
            synchronized.append((rover_obs_list[i], base_obs_list[j], dt))
            i += 1
            j += 1
        elif dt > 0:
            # Rover is ahead - try to interpolate base
            # Check if interpolation is possible (between current j and j+1)
            if j + 1 < len(base_obs_list):
                # Interpolate with current and next base observations
                t_prev = base_obs_list[j].get('gps_time', base_obs_list[j].get('time'))
                t_next = base_obs_list[j+1].get('gps_time', base_obs_list[j+1].get('time'))

                # RTKLIB style: allow interpolation up to MAX_INTERP_TIME
                max_interp_time = MAX_INTERP_TIME
                if t_prev <= t_rover <= t_next and (t_next - t_prev) <= max_interp_time:
                    # Perform interpolation
                    base_interp = interpolate_base_epoch(
                        base_obs_list[j], base_obs_list[j+1],
                        t_prev, t_next, t_rover
                    )
                    synchronized.append((rover_obs_list[i], base_interp, 0.0))
                    i += 1
                else:
                    # If cannot interpolate, advance base station
                    j += 1
            else:
                # No next base station
                break
        else:
            # Base is ahead - advance rover
            i += 1
            # RTKLIB usually doesn't interpolate rover side (for real-time processing)
    
    return synchronized


def interpolate_base_epoch(base1: dict, base2: dict, t1: float, t2: float, t: float) -> dict:
    """
    Interpolation of entire base station epoch

    Parameters:
    -----------
    base1, base2 : dict
        Previous and next base station epochs
    t1, t2 : float
        Time of previous and next epochs
    t : float
        Target interpolation time

    Returns:
    --------
    dict : Interpolated epoch
    """
    # Create interpolated epoch
    base_interp = {
        'gps_time': t,
        'time': t,
        'interpolated': True
    }

    # Get observations
    obs1 = base1.get('observations', {})
    obs2 = base2.get('observations', {})

    # Convert observations to dictionary format
    if isinstance(obs1, list):
        obs1_dict = {obs.sat: obs for obs in obs1}
    else:
        obs1_dict = obs1
    
    if isinstance(obs2, list):
        obs2_dict = {obs.sat: obs for obs in obs2}
    else:
        obs2_dict = obs2
    
    # Interpolate observations for common satellites
    interp_obs = {}
    for sat in obs1_dict:
        if sat in obs2_dict:
            # Interpolate observation for this satellite
            obs_interp = interp_observation(obs1_dict[sat], obs2_dict[sat], t1, t2, t)
            
            # Preserve original object type
            if hasattr(obs1_dict[sat], '__class__'):
                # Reconstruct as object
                # Observation class has required arguments, initialize with dummy values
                try:
                    new_obs = obs1_dict[sat].__class__(
                        time=t,
                        sat=sat,
                        system=obs1_dict[sat].system if hasattr(obs1_dict[sat], 'system') else 1
                    )
                except:
                    # Try to create without arguments
                    new_obs = type('Observation', (), {})()
                    new_obs.time = t
                    new_obs.sat = sat
                
                if isinstance(obs_interp, dict):
                    if 'P' in obs_interp:
                        new_obs.P = obs_interp['P']
                    if 'L' in obs_interp:
                        new_obs.L = obs_interp['L']
                else:
                    # obs_interp is already an Observation object
                    new_obs = obs_interp

                # Only process Doppler if we created new_obs from dict
                if isinstance(obs_interp, dict) and hasattr(obs1_dict[sat], 'D'):
                    # Linear interpolation for Doppler
                    alpha = (t - t1) / (t2 - t1)
                    if hasattr(obs2_dict[sat], 'D'):
                        new_obs.D = obs1_dict[sat].D + alpha * (obs2_dict[sat].D - obs1_dict[sat].D)
                    else:
                        new_obs.D = obs1_dict[sat].D
                if hasattr(obs1_dict[sat], 'system'):
                    new_obs.system = obs1_dict[sat].system
                if hasattr(obs1_dict[sat], 'SNR'):
                    new_obs.SNR = obs1_dict[sat].SNR
                if hasattr(obs1_dict[sat], 'LLI'):
                    new_obs.LLI = obs1_dict[sat].LLI
                if hasattr(obs1_dict[sat], 'code'):
                    new_obs.code = obs1_dict[sat].code
                    
                interp_obs[sat] = new_obs
            else:
                interp_obs[sat] = obs_interp
    
    base_interp['observations'] = interp_obs
    
    return base_interp