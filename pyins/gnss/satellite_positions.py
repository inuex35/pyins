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

#!/usr/bin/env python3
"""
Compute satellite positions and related information
"""


from ..core.constants import CLIGHT, SYS_BDS, SYS_GLO, sat2sys, sys2char
from ..core.data_structures import Observation
from ..core.unified_time import TimeCore, TimeSystem
from .ephemeris import eph2pos, seleph


def compute_satellite_info(observations: list[Observation],
                          nav_data,
                          used_satellites: list[int] = None) -> dict:
    """
    Compute satellite positions, clocks, and other information

    Parameters
    ----------
    observations : List[Observation]
        GNSS observations
    nav_data : NavigationData
        Navigation data (ephemeris)
    used_satellites : List[int], optional
        List of satellite numbers to compute. If None, compute for all valid observations

    Returns
    -------
    satellite_info : Dict
        Dictionary containing satellite positions, clocks, etc.
        Keys: 'positions', 'clocks', 'variances', 'systems', 'pseudoranges'
    """

    satellite_info = {
        'positions': {},  # satellite_number -> position [x, y, z]
        'clocks': {},     # satellite_number -> clock correction
        'variances': {},  # satellite_number -> position variance
        'systems': {},    # satellite_number -> satellite system
        'pseudoranges': {} # satellite_number -> pseudorange
    }

    # Create mapping from satellite number to observation
    obs_map = {obs.sat: obs for obs in observations}

    # Determine which satellites to process
    if used_satellites is None:
        # Process all satellites with valid pseudoranges (excluding GLONASS)
        satellites_to_process = []
        for obs in observations:
            if sat2sys(obs.sat) != SYS_GLO:  # Skip GLONASS
                pr = obs.P[0] if obs.P[0] > 0 else obs.P[1]
                if pr > 0:
                    satellites_to_process.append(obs.sat)
    else:
        satellites_to_process = used_satellites

    for sat_num in satellites_to_process:
        if sat_num not in obs_map:
            continue

        obs = obs_map[sat_num]

        # Get pseudorange
        pr = obs.P[0] if obs.P[0] > 0 else obs.P[1]
        if pr <= 0:
            continue

        # Get satellite system
        sys = sat2sys(obs.sat)
        sys_char = sys2char(sys)

        try:
            # Calculate transmission time
            tc_rx = TimeCore(obs.time)  # GPS time
            tc_tx = tc_rx - (pr / CLIGHT)

            # Get appropriate TOW for the satellite system
            if sys == SYS_BDS:
                tow = tc_tx.get_tow(TimeSystem.BDS)
            else:
                tow = tc_tx.get_tow(TimeSystem.GPS)

            # Get ephemeris
            eph = seleph(nav_data, tow, obs.sat)
            if eph is None:
                continue

            # Calculate satellite position
            sat_pos, sat_var, dts = eph2pos(tow, eph)

            # Store satellite information
            satellite_info['positions'][sat_num] = sat_pos
            satellite_info['clocks'][sat_num] = dts
            satellite_info['variances'][sat_num] = sat_var
            satellite_info['systems'][sat_num] = sys_char
            satellite_info['pseudoranges'][sat_num] = pr

        except Exception as e:
            print(f"Error computing satellite position for satellite {sat_num}: {e}")
            continue

    return satellite_info
