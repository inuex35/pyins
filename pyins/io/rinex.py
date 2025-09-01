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

"""RINEX file reading using gnsspy"""

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    import gnsspy
    import pandas as pd
    from gnsspy import read_navFile, read_obsFile
    GNSSPY_AVAILABLE = True
except ImportError:
    GNSSPY_AVAILABLE = False

from ..core import Ephemeris, GloEphemeris, GNSSTime, NavigationData, Observation
from ..core.constants import GPS_BDS_OFFSET
from ..core.time_conversions import (
    datetime_to_gps_seconds,
    datetime_to_tow,
    datetime_to_unix_seconds,  # For GLONASS time conversion
)

logger = logging.getLogger(__name__)

# RTKLIB-compatible signal code mapping
# Maps RINEX 3 observation codes to frequency indices with priorities
SIGNAL_PRIORITY_MAP = {
    'G': {  # GPS
        0: [  # L1 band (1575.42 MHz)
            'C1C', 'C1S', 'C1L', 'C1X', 'C1P', 'C1W', 'C1Y', 'C1M', 'C1N', 'P1'
        ],
        1: [  # L2 band (1227.60 MHz)
            'C2W', 'C2L', 'C2S', 'C2X', 'C2C', 'C2P', 'C2Y', 'C2M', 'C2N', 'C2D', 'P2'
        ],
        2: [  # L5 band (1176.45 MHz)
            'C5Q', 'C5I', 'C5X'
        ]
    },
    'E': {  # Galileo
        0: [  # E1 band (1575.42 MHz)
            'C1C', 'C1A', 'C1B', 'C1X', 'C1Z'
        ],
        1: [  # E5b band (1207.14 MHz)
            'C7Q', 'C7I', 'C7X'
        ],
        2: [  # E5a band (1176.45 MHz)
            'C5Q', 'C5I', 'C5X'
        ],
        3: [  # E5 wideband (1191.795 MHz)
            'C8Q', 'C8I', 'C8X'
        ],
        4: [  # E6 band (1278.75 MHz)
            'C6A', 'C6B', 'C6C', 'C6X', 'C6Z'
        ]
    },
    'C': {  # BeiDou
        0: [  # B1 band
            'C2I', 'C1I', 'C1Q', 'C1D', 'C1P', 'C1X', 'C1S', 'C1L'
        ],
        1: [  # B2I/B2b band
            'C7I', 'C7Q', 'C7X', 'C7D', 'C7P', 'C7Z'
        ],
        2: [  # B2a band (1176.45 MHz)
            'C5D', 'C5P', 'C5X'
        ],
        3: [  # B3 band (1268.52 MHz)
            'C6I', 'C6Q', 'C6A', 'C6X', 'C6D', 'C6P'
        ],
        4: [  # B2ab band
            'C8D', 'C8P', 'C8X'
        ]
    },
    'R': {  # GLONASS
        0: [  # G1 band (1602 + k*0.5625 MHz)
            'C1C', 'C1P', 'C4A', 'C4B', 'C4X', 'P1'
        ],
        1: [  # G2 band (1246 + k*0.4375 MHz)
            'C2C', 'C2P', 'P2'
        ],
        2: [  # G3 band
            'C3I', 'C3Q', 'C3X'
        ]
    },
    'J': {  # QZSS
        0: [  # L1 band (1575.42 MHz)
            'C1C', 'C1S', 'C1L', 'C1X', 'C1E', 'C1Z'
        ],
        1: [  # L2 band (1227.60 MHz)
            'C2S', 'C2L', 'C2X'
        ],
        2: [  # L5 band (1176.45 MHz)
            'C5Q', 'C5I', 'C5X', 'C5D', 'C5P', 'C5Z'
        ],
        3: [  # L6 band (1278.75 MHz)
            'C6S', 'C6L', 'C6X', 'C6E', 'C6Z'
        ]
    },
    'I': {  # IRNSS/NavIC
        0: [  # L5 band (1176.45 MHz)
            'C5A', 'C5B', 'C5C', 'C5X'
        ],
        1: [  # S band (2492.028 MHz)
            'C9A', 'C9B', 'C9C', 'C9X'
        ]
    },
    'S': {  # SBAS
        0: [  # L1 band
            'C1C'
        ],
        2: [  # L5 band
            'C5I', 'C5Q', 'C5X'
        ]
    }
}


class RinexObsReader:
    """RINEX observation file reader using gnsspy with RTKLIB signal code support"""

    def __init__(self, filename: str, max_frequencies: int = 5):
        self.filename = filename
        self.header = {}
        self.observations = []
        self.max_frequencies = max_frequencies  # Support up to 5 frequencies

    def read(self) -> list[dict]:
        """Read RINEX observation file"""
        logger.info(f"Reading RINEX observation file: {self.filename}")

        if GNSSPY_AVAILABLE:
            return self._read_with_gnsspy()
        else:
            raise ImportError("gnsspy not available for RINEX parsing")

    def _read_with_gnsspy(self) -> list[dict]:
        """Read using gnsspy library"""
        try:
            # Read observation file
            obs_data = read_obsFile(self.filename)

            # Extract header info
            self.header = {
                'approx_pos': obs_data.approx_position if hasattr(obs_data, 'approx_position') else None,
                'interval': obs_data.interval if hasattr(obs_data, 'interval') else None,
                'antenna_delta': obs_data.antenna_delta if hasattr(obs_data, 'antenna_delta') else None,
            }

            # Process epochs
            epochs = []

            # gnsspy returns observations as a DataFrame with MultiIndex (Epoch, SV)
            df = obs_data.observation

            # Group by epoch
            for epoch_time, epoch_df in df.groupby(level='Epoch'):
                # Convert epoch time to various time systems using new functions
                # Ensure epoch_time has timezone
                if epoch_time.tzinfo is None:
                    epoch_time = epoch_time.replace(tzinfo=timezone.utc)

                # Get GPS seconds and TOW
                gps_seconds = datetime_to_gps_seconds(epoch_time)
                gps_week, tow = datetime_to_tow(epoch_time)

                # Create GNSSTime object
                GNSSTime(week=int(gps_week), tow=float(tow))

                # RINEX times are in UTC
                if epoch_time.tzinfo is None:
                    utc_time = epoch_time.replace(tzinfo=timezone.utc)
                else:
                    utc_time = epoch_time.astimezone(timezone.utc)

                epoch_data = {
                    'time': datetime_to_unix_seconds(utc_time),  # Unix timestamp
                    'gps_time': gps_seconds,  # GPS seconds since GPS epoch
                    'gps_week': int(gps_week),  # GPS week number
                    'gps_tow': float(tow),  # Time of week in seconds
                    'n_sats': 0,
                    'observations': []
                }

                # Process each satellite in this epoch
                for sv_id in epoch_df.index.get_level_values('SV').unique():
                    # Extract PRN and system
                    sys_char = sv_id[0]
                    prn = int(sv_id[1:])

                    # Convert satellite ID to internal numbering
                    from ..core.satellite_numbering import prn_to_sat
                    sat_num = prn_to_sat(sys_char, prn)
                    if sat_num == 0:
                        continue  # Skip invalid satellites

                    # Get observation data for this satellite
                    try:
                        sat_data = epoch_df.xs(sv_id, level='SV')
                        # If multiple rows for same satellite, take the first one
                        if isinstance(sat_data, pd.DataFrame):
                            sat_data = sat_data.iloc[0]
                    except:
                        continue

                    # Extract observation values
                    obs = Observation(
                        time=epoch_data['gps_time'],  # Use GPS seconds for ephemeris matching
                        sat=sat_num,
                        system=0,  # Will be set by __post_init__
                    )

                    # Process each frequency using priority-based signal selection
                    for freq_idx in range(min(self.max_frequencies, len(obs.P))):
                        # Get priority list for this system and frequency
                        if sys_char in SIGNAL_PRIORITY_MAP and freq_idx in SIGNAL_PRIORITY_MAP[sys_char]:
                            signal_list = SIGNAL_PRIORITY_MAP[sys_char][freq_idx]
                            
                            # Try each signal in priority order for pseudorange
                            for signal_code in signal_list:
                                if signal_code in sat_data and not pd.isna(sat_data[signal_code]):
                                    obs.P[freq_idx] = float(sat_data[signal_code])
                                    break
                            
                            # Try corresponding carrier phase (replace C with L, P with L)
                            for signal_code in signal_list:
                                phase_code = signal_code.replace('C', 'L').replace('P', 'L')
                                if phase_code in sat_data and not pd.isna(sat_data[phase_code]):
                                    obs.L[freq_idx] = float(sat_data[phase_code])
                                    break
                            
                            # Try corresponding Doppler (replace C with D, P with D)
                            for signal_code in signal_list:
                                doppler_code = signal_code.replace('C', 'D').replace('P', 'D')
                                if doppler_code in sat_data and not pd.isna(sat_data[doppler_code]):
                                    obs.D[freq_idx] = float(sat_data[doppler_code])
                                    break
                            
                            # Try corresponding SNR (replace C with S, P with S)
                            for signal_code in signal_list:
                                snr_code = signal_code.replace('C', 'S').replace('P', 'S')
                                if snr_code in sat_data and not pd.isna(sat_data[snr_code]):
                                    obs.SNR[freq_idx] = float(sat_data[snr_code])
                                    break

                    # All signal processing is now handled by the priority-based system above
                    # The code below (lines 263-464) contains legacy hardcoded signal handling
                    # that can be safely removed as it's redundant with the priority-based system
                    
                    """
                    # Get L2 carrier phase (frequency index 1)
                    if sys_char == 'G':  # GPS
                        if 'L2W' in sat_data and not pd.isna(sat_data['L2W']):
                            obs.L[1] = float(sat_data['L2W'])
                        elif 'L2X' in sat_data and not pd.isna(sat_data['L2X']):
                            obs.L[1] = float(sat_data['L2X'])
                        elif 'L2L' in sat_data and not pd.isna(sat_data['L2L']):
                            obs.L[1] = float(sat_data['L2L'])
                        elif 'L2C' in sat_data and not pd.isna(sat_data['L2C']):
                            obs.L[1] = float(sat_data['L2C'])
                        elif 'L2S' in sat_data and not pd.isna(sat_data['L2S']):
                            obs.L[1] = float(sat_data['L2S'])
                        elif 'L2' in sat_data and not pd.isna(sat_data['L2']):
                            obs.L[1] = float(sat_data['L2'])
                    elif sys_char == 'R':  # GLONASS
                        if 'L2C' in sat_data and not pd.isna(sat_data['L2C']):
                            obs.L[1] = float(sat_data['L2C'])
                        elif 'L2P' in sat_data and not pd.isna(sat_data['L2P']):
                            obs.L[1] = float(sat_data['L2P'])
                    elif sys_char == 'E':  # Galileo E5b
                        if 'L7Q' in sat_data and not pd.isna(sat_data['L7Q']):
                            obs.L[1] = float(sat_data['L7Q'])
                        elif 'L7X' in sat_data and not pd.isna(sat_data['L7X']):
                            obs.L[1] = float(sat_data['L7X'])
                        elif 'L7I' in sat_data and not pd.isna(sat_data['L7I']):
                            obs.L[1] = float(sat_data['L7I'])
                    elif sys_char == 'C':  # BeiDou B2I
                        if 'L7I' in sat_data and not pd.isna(sat_data['L7I']):
                            obs.L[1] = float(sat_data['L7I'])
                        elif 'L7D' in sat_data and not pd.isna(sat_data['L7D']):
                            obs.L[1] = float(sat_data['L7D'])
                    elif sys_char == 'J':  # QZSS
                        if 'L2L' in sat_data and not pd.isna(sat_data['L2L']):
                            obs.L[1] = float(sat_data['L2L'])
                        elif 'L2X' in sat_data and not pd.isna(sat_data['L2X']):
                            obs.L[1] = float(sat_data['L2X'])

                    # Get Doppler
                    if sys_char == 'C':  # BeiDou
                        if 'D2I' in sat_data and not pd.isna(sat_data['D2I']):
                            obs.D[0] = float(sat_data['D2I'])
                        elif 'D1I' in sat_data and not pd.isna(sat_data['D1I']):
                            obs.D[0] = float(sat_data['D1I'])
                    else:
                        if 'D1C' in sat_data and not pd.isna(sat_data['D1C']):
                            obs.D[0] = float(sat_data['D1C'])
                        elif 'D1X' in sat_data and not pd.isna(sat_data['D1X']):  # Galileo E1 BC
                            obs.D[0] = float(sat_data['D1X'])
                        elif 'D1' in sat_data and not pd.isna(sat_data['D1']):
                            obs.D[0] = float(sat_data['D1'])

                    # Get SNR
                    if sys_char == 'C':  # BeiDou
                        if 'S2I' in sat_data and not pd.isna(sat_data['S2I']):
                            obs.SNR[0] = float(sat_data['S2I'])
                        elif 'S1I' in sat_data and not pd.isna(sat_data['S1I']):
                            obs.SNR[0] = float(sat_data['S1I'])
                    else:
                        if 'S1C' in sat_data and not pd.isna(sat_data['S1C']):
                            obs.SNR[0] = float(sat_data['S1C'])
                        elif 'S1' in sat_data and not pd.isna(sat_data['S1']):
                            obs.SNR[0] = float(sat_data['S1'])

                    # L2 observations
                    if 'C2W' in sat_data and not pd.isna(sat_data['C2W']):
                        obs.P[1] = float(sat_data['C2W'])
                    elif 'C2C' in sat_data and not pd.isna(sat_data['C2C']):
                        obs.P[1] = float(sat_data['C2C'])
                    elif 'P2' in sat_data and not pd.isna(sat_data['P2']):
                        obs.P[1] = float(sat_data['P2'])

                    if 'L2W' in sat_data and not pd.isna(sat_data['L2W']):
                        obs.L[1] = float(sat_data['L2W'])
                    elif 'L2C' in sat_data and not pd.isna(sat_data['L2C']):
                        obs.L[1] = float(sat_data['L2C'])
                    elif 'L2' in sat_data and not pd.isna(sat_data['L2']):
                        obs.L[1] = float(sat_data['L2'])

                    if 'D2W' in sat_data and not pd.isna(sat_data['D2W']):
                        obs.D[1] = float(sat_data['D2W'])
                    elif 'D2C' in sat_data and not pd.isna(sat_data['D2C']):
                        obs.D[1] = float(sat_data['D2C'])

                    if 'S2W' in sat_data and not pd.isna(sat_data['S2W']):
                        obs.SNR[1] = float(sat_data['S2W'])
                    elif 'S2C' in sat_data and not pd.isna(sat_data['S2C']):
                        obs.SNR[1] = float(sat_data['S2C'])

                    # L5/E5a/B2a observations
                    if 'C5I' in sat_data and not pd.isna(sat_data['C5I']):
                        obs.P[2] = float(sat_data['C5I'])
                    elif 'C5Q' in sat_data and not pd.isna(sat_data['C5Q']):
                        obs.P[2] = float(sat_data['C5Q'])
                    elif 'C5X' in sat_data and not pd.isna(sat_data['C5X']):
                        obs.P[2] = float(sat_data['C5X'])

                    if 'L5I' in sat_data and not pd.isna(sat_data['L5I']):
                        obs.L[2] = float(sat_data['L5I'])
                    elif 'L5Q' in sat_data and not pd.isna(sat_data['L5Q']):
                        obs.L[2] = float(sat_data['L5Q'])
                    elif 'L5X' in sat_data and not pd.isna(sat_data['L5X']):
                        obs.L[2] = float(sat_data['L5X'])

                    if 'D5I' in sat_data and not pd.isna(sat_data['D5I']):
                        obs.D[2] = float(sat_data['D5I'])
                    elif 'D5Q' in sat_data and not pd.isna(sat_data['D5Q']):
                        obs.D[2] = float(sat_data['D5Q'])
                    elif 'D5X' in sat_data and not pd.isna(sat_data['D5X']):
                        obs.D[2] = float(sat_data['D5X'])

                    if 'S5I' in sat_data and not pd.isna(sat_data['S5I']):
                        obs.SNR[2] = float(sat_data['S5I'])
                    elif 'S5Q' in sat_data and not pd.isna(sat_data['S5Q']):
                        obs.SNR[2] = float(sat_data['S5Q'])
                    elif 'S5X' in sat_data and not pd.isna(sat_data['S5X']):
                        obs.SNR[2] = float(sat_data['S5X'])

                    # L7/E5b/B2b observations (Galileo E5b, BeiDou B2b)
                    if 'C7I' in sat_data and not pd.isna(sat_data['C7I']):
                        obs.P[1] = float(sat_data['C7I'])
                    elif 'C7Q' in sat_data and not pd.isna(sat_data['C7Q']):
                        obs.P[1] = float(sat_data['C7Q'])
                    elif 'C7X' in sat_data and not pd.isna(sat_data['C7X']):
                        obs.P[1] = float(sat_data['C7X'])

                    if 'L7I' in sat_data and not pd.isna(sat_data['L7I']):
                        obs.L[1] = float(sat_data['L7I'])
                    elif 'L7Q' in sat_data and not pd.isna(sat_data['L7Q']):
                        obs.L[1] = float(sat_data['L7Q'])
                    elif 'L7X' in sat_data and not pd.isna(sat_data['L7X']):
                        obs.L[1] = float(sat_data['L7X'])

                    if 'D7I' in sat_data and not pd.isna(sat_data['D7I']):
                        obs.D[1] = float(sat_data['D7I'])
                    elif 'D7Q' in sat_data and not pd.isna(sat_data['D7Q']):
                        obs.D[1] = float(sat_data['D7Q'])
                    elif 'D7X' in sat_data and not pd.isna(sat_data['D7X']):
                        obs.D[1] = float(sat_data['D7X'])

                    if 'S7I' in sat_data and not pd.isna(sat_data['S7I']):
                        obs.SNR[1] = float(sat_data['S7I'])
                    elif 'S7Q' in sat_data and not pd.isna(sat_data['S7Q']):
                        obs.SNR[1] = float(sat_data['S7Q'])
                    elif 'S7X' in sat_data and not pd.isna(sat_data['S7X']):
                        obs.SNR[1] = float(sat_data['S7X'])

                    # Get L5/E5a/B2a observations (frequency index 2)
                    if sys_char == 'G':  # GPS L5
                        if 'C5Q' in sat_data and not pd.isna(sat_data['C5Q']):
                            obs.P[2] = float(sat_data['C5Q'])
                        elif 'C5X' in sat_data and not pd.isna(sat_data['C5X']):
                            obs.P[2] = float(sat_data['C5X'])
                        elif 'C5I' in sat_data and not pd.isna(sat_data['C5I']):
                            obs.P[2] = float(sat_data['C5I'])
                        
                        if 'L5Q' in sat_data and not pd.isna(sat_data['L5Q']):
                            obs.L[2] = float(sat_data['L5Q'])
                        elif 'L5X' in sat_data and not pd.isna(sat_data['L5X']):
                            obs.L[2] = float(sat_data['L5X'])
                        elif 'L5I' in sat_data and not pd.isna(sat_data['L5I']):
                            obs.L[2] = float(sat_data['L5I'])
                            
                    elif sys_char == 'E':  # Galileo E5a
                        if 'C5Q' in sat_data and not pd.isna(sat_data['C5Q']):
                            obs.P[2] = float(sat_data['C5Q'])
                        elif 'C5X' in sat_data and not pd.isna(sat_data['C5X']):
                            obs.P[2] = float(sat_data['C5X'])
                        elif 'C5I' in sat_data and not pd.isna(sat_data['C5I']):
                            obs.P[2] = float(sat_data['C5I'])
                        
                        if 'L5Q' in sat_data and not pd.isna(sat_data['L5Q']):
                            obs.L[2] = float(sat_data['L5Q'])
                        elif 'L5X' in sat_data and not pd.isna(sat_data['L5X']):
                            obs.L[2] = float(sat_data['L5X'])
                        elif 'L5I' in sat_data and not pd.isna(sat_data['L5I']):
                            obs.L[2] = float(sat_data['L5I'])
                            
                    elif sys_char == 'C':  # BeiDou B2a
                        if 'C5P' in sat_data and not pd.isna(sat_data['C5P']):
                            obs.P[2] = float(sat_data['C5P'])
                        elif 'C5D' in sat_data and not pd.isna(sat_data['C5D']):
                            obs.P[2] = float(sat_data['C5D'])
                        elif 'C5X' in sat_data and not pd.isna(sat_data['C5X']):
                            obs.P[2] = float(sat_data['C5X'])
                        
                        if 'L5P' in sat_data and not pd.isna(sat_data['L5P']):
                            obs.L[2] = float(sat_data['L5P'])
                        elif 'L5D' in sat_data and not pd.isna(sat_data['L5D']):
                            obs.L[2] = float(sat_data['L5D'])
                        elif 'L5X' in sat_data and not pd.isna(sat_data['L5X']):
                            obs.L[2] = float(sat_data['L5X'])
                            
                    elif sys_char == 'J':  # QZSS L5
                        if 'C5Q' in sat_data and not pd.isna(sat_data['C5Q']):
                            obs.P[2] = float(sat_data['C5Q'])
                        elif 'C5X' in sat_data and not pd.isna(sat_data['C5X']):
                            obs.P[2] = float(sat_data['C5X'])
                        
                        if 'L5Q' in sat_data and not pd.isna(sat_data['L5Q']):
                            obs.L[2] = float(sat_data['L5Q'])
                        elif 'L5X' in sat_data and not pd.isna(sat_data['L5X']):
                            obs.L[2] = float(sat_data['L5X'])

                    # Check other bands if needed, e.g. L6/L8 for Galileo
                    """

                    # Only add if we have valid data
                    if np.any(obs.P > 0) or np.any(obs.L > 0):
                        epoch_data['observations'].append(obs)
                        epoch_data['n_sats'] += 1

                if epoch_data['n_sats'] > 0:
                    epochs.append(epoch_data)

            logger.info(f"  Read {len(epochs)} epochs with gnsspy")
            return epochs

        except Exception as e:
            logger.error(f"Error reading with gnsspy: {e}")
            raise


class RinexNavReader:
    """RINEX navigation file reader"""

    def __init__(self, filename: str):
        self.filename = filename
        self.nav_data = NavigationData()

    def read(self) -> NavigationData:
        """Read RINEX navigation file"""
        if not self.filename:
            raise ValueError("No navigation filename provided")

        logger.info(f"Reading RINEX navigation file: {self.filename}")

        if GNSSPY_AVAILABLE:
            return self._read_with_gnsspy()
        else:
            raise ImportError("gnsspy not available for RINEX navigation parsing")

    def _read_with_gnsspy(self) -> NavigationData:
        """Read using gnsspy library"""
        try:
            # Read navigation file
            nav_obj = read_navFile(self.filename)

            # gnsspy returns a Navigation object with epoch and navigation lists
            if hasattr(nav_obj, 'navigation') and hasattr(nav_obj, 'epoch'):
                # Process ephemerides from the navigation DataFrame
                nav_df = nav_obj.navigation

                # The DataFrame has MultiIndex (Epoch, SV)
                # Process each row in the navigation DataFrame
                for idx in range(len(nav_df)):
                    try:
                        row = nav_df.iloc[idx]

                        # Extract satellite ID from MultiIndex
                        if hasattr(row, 'name') and isinstance(row.name, tuple) and len(row.name) > 1:
                            epoch_time = row.name[0]  # First element is Epoch
                            sat_id = row.name[1]      # Second element is SV
                        elif 'SV' in row and pd.notna(row['SV']):
                            sat_id = row['SV']
                            # Get epoch from epochs list if available
                            epoch_time = nav_obj.epoch[idx] if idx < len(nav_obj.epoch) else None
                        else:
                            logger.debug(f"No satellite ID found in row {idx}")
                            continue

                        sys_char = sat_id[0]
                        prn = int(sat_id[1:])

                        # Convert to internal numbering
                        from ..core.satellite_numbering import prn_to_sat
                        sat_num = prn_to_sat(sys_char, prn)
                        if sat_num == 0:
                            continue  # Skip invalid satellites

                        # Extract ephemeris data
                        # Convert epoch to GPS time
                        if epoch_time is not None:
                            gnss_time = GNSSTime.from_datetime(pd.Timestamp(epoch_time).to_pydatetime())
                        else:
                            # Use current time as fallback
                            gnss_time = GNSSTime.from_datetime(datetime.now())

                        # Handle GLONASS ephemeris differently
                        if sys_char == 'R':
                            # GLONASS uses different ephemeris format
                            # IMPORTANT: GLONASS times in RINEX are in UTC+3 hours (Moscow time)
                            # We need to convert to GPS time by adding leap seconds

                            # Check if data contains x, y, z (position in km)
                            if 'x' in row and pd.notna(row['x']) and 'y' in row and pd.notna(row['y']) and 'z' in row and pd.notna(row['z']):
                                # Get transmission time
                                # gnsspy provides tb (message frame time) in GPS week seconds
                                # For GLONASS, tb is rounded to 15-minute boundaries
                                trans_time = row['transmissionTime']
                                if isinstance(trans_time, str):
                                    trans_time = float(trans_time)
                                else:
                                    trans_time = float(trans_time) if pd.notna(trans_time) else gnss_time.tow

                                # GLONASS correction: tb is rounded to 15-minute boundaries
                                # We need to add the interval + leap seconds to get the correct epoch time
                                from ..core.constants import GPS_UTC_OFFSET, GLONASS_TB_INTERVAL
                                toc_gps = trans_time + GLONASS_TB_INTERVAL + GPS_UTC_OFFSET  # Add tb interval + leap seconds
                                tof_gps = toc_gps  # tof = toc for GLONASS
                                
                                # Get current GPS week and time of week from epoch
                                gps_tow = gnss_time.tow

                                # Adjust for day boundary (RTKLIB's adjday function)
                                # If the time difference is more than half a day, adjust by one day
                                dt = toc_gps - gps_tow
                                if dt < -43200:  # More than 12 hours behind
                                    toc_gps += 86400
                                    tof_gps += 86400
                                elif dt > 43200:  # More than 12 hours ahead
                                    toc_gps -= 86400
                                    tof_gps -= 86400

                                # Store as full GPS time for GLONASS
                                # This matches what RTKLIB does internally
                                gps_week = gnss_time.week
                                toe_gps_full = gps_week * 604800 + toc_gps  # Full GPS time
                                tof_gps_full = gps_week * 604800 + tof_gps  # Full GPS time

                                toe_gps = toe_gps_full  # Use full GPS time for GLONASS

                                # Clock bias - RTKLIB negates it
                                clock_bias = row['clockBias'] if 'clockBias' in row else 0.0
                                if isinstance(clock_bias, str):
                                    clock_bias = float(clock_bias)
                                else:
                                    clock_bias = float(clock_bias) if pd.notna(clock_bias) else 0.0

                                # Relative frequency bias
                                rel_freq = row['relFeqBias'] if 'relFeqBias' in row else 0.0
                                if isinstance(rel_freq, str):
                                    rel_freq = float(rel_freq)
                                else:
                                    rel_freq = float(rel_freq) if pd.notna(rel_freq) else 0.0

                                geph = GloEphemeris(
                                    sat=sat_num,
                                    iode=int(row['operationDay']) if 'operationDay' in row and pd.notna(row['operationDay']) else 0,
                                    frq=int(row['freqNumber']) if 'freqNumber' in row and pd.notna(row['freqNumber']) else 0,
                                    svh=int(row['health']) if 'health' in row and pd.notna(row['health']) else 0,
                                    flags=0,
                                    sva=0,
                                    age=0,
                                    toe=toe_gps_full,  # Full GPS time (week * 604800 + TOW)
                                    tof=tof_gps_full,  # Full GPS time (week * 604800 + TOW)
                                    pos=np.array([
                                        float(row['x']) * 1e3,  # gnsspy returns km, convert to m
                                        float(row['y']) * 1e3,
                                        float(row['z']) * 1e3
                                    ]),
                                    vel=np.array([
                                        float(row['vx']) * 1e3 if 'vx' in row and pd.notna(row['vx']) else 0.0,  # gnsspy returns km/s, convert to m/s
                                        float(row['vy']) * 1e3 if 'vy' in row and pd.notna(row['vy']) else 0.0,
                                        float(row['vz']) * 1e3 if 'vz' in row and pd.notna(row['vz']) else 0.0
                                    ]),
                                    acc=np.array([
                                        float(row['ax']) * 1e3 if 'ax' in row and pd.notna(row['ax']) else 0.0,  # gnsspy returns km/s^2, convert to m/s^2
                                        float(row['ay']) * 1e3 if 'ay' in row and pd.notna(row['ay']) else 0.0,
                                        float(row['az']) * 1e3 if 'az' in row and pd.notna(row['az']) else 0.0
                                    ]),
                                    taun=-clock_bias,  # GLONASS clock bias (negate as per RTKLIB)
                                    gamn=rel_freq,
                                    dtaun=0.0  # Not available in this format
                                )
                                self.nav_data.geph.append(geph)
                            else:
                                # Skip GLONASS ephemeris if position data is not available
                                logger.debug(f"Skipping GLONASS sat {sat_num} - no position data")
                                continue
                        else:
                            # Regular ephemeris for GPS, Galileo, BeiDou, QZSS
                            eph = Ephemeris(
                                sat=sat_num,
                                iode=int(row['iode']) if 'iode' in row and pd.notna(row['iode']) else 0,
                                iodc=int(row['iodc']) if 'iodc' in row and pd.notna(row['iodc']) else 0,
                                sva=int(row['sva']) if 'sva' in row and pd.notna(row['sva']) else 0,
                                svh=int(row['health']) if 'health' in row and pd.notna(row['health']) else 0,
                                week=int(row['week']) if 'week' in row and pd.notna(row['week']) else gnss_time.week,
                                code=int(row['code']) if 'code' in row and pd.notna(row['code']) else 0,
                                flag=0,
                                toe=float(row['toe']) if 'toe' in row and pd.notna(row['toe']) else gnss_time.tow,
                                toc=gnss_time.tow,  # toc is the time of clock (epoch time)
                                ttr=gnss_time.tow,  # ttr is the time of transmission
                                A=float(row['roota'])**2 if 'roota' in row and pd.notna(row['roota']) and row['roota'] > 0 else 0.0,  # roota is sqrt(A)
                                e=float(row['eccentricity']) if 'eccentricity' in row and pd.notna(row['eccentricity']) else 0.0,
                                i0=float(row['i0']) if 'i0' in row and pd.notna(row['i0']) else 0.0,
                                OMG0=float(row['bigomega0']) if 'bigomega0' in row and pd.notna(row['bigomega0']) else 0.0,
                                omg=float(row['smallomega']) if 'smallomega' in row and pd.notna(row['smallomega']) else 0.0,
                                M0=float(row['m0']) if 'm0' in row and pd.notna(row['m0']) else 0.0,
                                deln=float(row['delta_n']) if 'delta_n' in row and pd.notna(row['delta_n']) else 0.0,
                                OMGd=float(row['bigomegadot']) if 'bigomegadot' in row and pd.notna(row['bigomegadot']) else 0.0,
                                idot=float(row['idot']) if 'idot' in row and pd.notna(row['idot']) else 0.0,
                                crc=float(row['crc']) if 'crc' in row and pd.notna(row['crc']) else 0.0,
                                crs=float(row['crs']) if 'crs' in row and pd.notna(row['crs']) else 0.0,
                                cuc=float(row['cuc']) if 'cuc' in row and pd.notna(row['cuc']) else 0.0,
                                cus=float(row['cus']) if 'cus' in row and pd.notna(row['cus']) else 0.0,
                                cic=float(row['cic']) if 'cic' in row and pd.notna(row['cic']) else 0.0,
                                cis=float(row['cis']) if 'cis' in row and pd.notna(row['cis']) else 0.0,
                                toes=float(row['toe']) if 'toe' in row and pd.notna(row['toe']) else gnss_time.tow,
                                fit=float(row['fit']) if 'fit' in row and pd.notna(row['fit']) else 4.0,
                                f0=float(row['clockBias']) if 'clockBias' in row and pd.notna(row['clockBias']) else 0.0,
                                f1=float(row['relFeqBias']) if 'relFeqBias' in row and pd.notna(row['relFeqBias']) else 0.0,  # gnsspy uses relFeqBias
                                f2=float(row['clockDriftRate']) if 'clockDriftRate' in row and pd.notna(row['clockDriftRate']) else 0.0,
                                tgd=np.array([float(row['tgd']), 0.0]) if 'tgd' in row and pd.notna(row['tgd']) else np.zeros(2),
                            )

                            # Convert BeiDou time to GPS time
                            # Note: Modern RINEX files already store BeiDou ephemeris in GPS time
                            # Only convert if needed (check if toe is too small)
                            if sys_char == 'C' and False:  # Disabled - modern RINEX files use GPST
                                # BeiDou ephemeris times are in BDT, convert to GPST
                                # GPST = BDT + 14 seconds
                                eph.toe += GPS_BDS_OFFSET
                                eph.toc += GPS_BDS_OFFSET
                                eph.ttr += GPS_BDS_OFFSET

                            self.nav_data.eph.append(eph)

                    except Exception as e:
                        logger.debug(f"Error processing navigation row {idx}: {e}")
                        continue

            logger.info(f"  Read {len(self.nav_data.eph)} ephemerides and {len(self.nav_data.geph)} GLONASS ephemerides with gnsspy")
            # Sort ephemerides for efficient selection
            self.nav_data.sort_eph()
            return self.nav_data

        except Exception as e:
            logger.error(f"Error reading with gnsspy: {e}")
            raise
