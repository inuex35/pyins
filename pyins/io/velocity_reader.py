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

"""Velocity data reading utilities"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class VelocityReader:
    """Velocity data reader for various file formats"""

    def __init__(self, file_path: str, format: str = 'csv', time_format: str = 'gps'):
        """
        Initialize velocity reader

        Parameters:
        -----------
        file_path : str
            Path to velocity data file
        format : str
            File format ('csv', 'txt')
        time_format : str
            Time format ('gps', 'unix', 'utc')
        """
        self.file_path = Path(file_path)
        self.format = format.lower()
        self.time_format = time_format.lower()
        self.logger = logging.getLogger(__name__)
        self._data_cache = None  # Cache for loaded data

        if not self.file_path.exists():
            raise FileNotFoundError(f"Velocity file not found: {file_path}")

    def read(self, start_time: Optional[float] = None, duration: Optional[float] = None) -> pd.DataFrame:
        """
        Read velocity data from file

        Parameters:
        -----------
        start_time : float, optional
            Start time in GPS seconds
        duration : float, optional
            Duration in seconds

        Returns:
        --------
        pd.DataFrame
            DataFrame with columns: time, vx, vy, vz (velocities in m/s)
        """
        if self.format == 'csv':
            df = self._read_csv()
        else:
            raise ValueError(f"Unsupported format: {self.format}")

        # Filter by time if requested
        if start_time is not None:
            df = df[df['time'] >= start_time]
        if duration is not None and start_time is not None:
            df = df[df['time'] <= start_time + duration]

        return df

    def _read_csv(self) -> pd.DataFrame:
        """
        Read velocity data from CSV file with automatic coordinate conversion.

        This method reads velocity data from CSV files and handles various
        column naming conventions and coordinate systems. It automatically
        converts between different coordinate frames and standardizes the
        output format.

        Returns
        -------
        pd.DataFrame
            Velocity data with standardized columns:
            - time: timestamp in the specified time format
            - vx, vy, vz: velocity components in m/s (ENU frame)

        Raises
        ------
        ValueError
            If required velocity columns are missing after processing
        RuntimeError
            If there's an error reading or processing the CSV file

        Notes
        -----
        Supported CSV formats:
        1. Direct format: time, vx, vy, vz
        2. NED format: time, vel_n, vel_e, vel_d
        3. ENU format: time, vel_n, vel_e, vel_u

        Coordinate conversions:
        - NED to ENU: vx=vel_e, vy=vel_n, vz=-vel_d
        - ENU: vx=vel_e, vy=vel_n, vz=vel_u

        Supported alternative column names:
        - timestamp, unix_time, gps_time -> time
        - vel_x, velocity_x -> vx
        - vel_y, velocity_y -> vy
        - vel_z, velocity_z -> vz
        - vel_n, velocity_n -> vn (north)
        - vel_e, velocity_e -> ve (east)
        - vel_d, velocity_d -> vd (down)
        - vel_u, velocity_u -> vu (up)

        Time format conversion:
        - If time_format='unix': converts UNIX time to GPS time
        - If time_format='utc': raises NotImplementedError
        - If time_format='gps': uses time as-is
        """
        try:
            # Try reading with different column name variations
            df = pd.read_csv(self.file_path)

            # Standardize column names
            column_mapping = {
                'timestamp': 'time',
                'unix_time': 'time',
                'gps_time': 'time',
                'vel_x': 'vx',
                'vel_y': 'vy',
                'vel_z': 'vz',
                'vel_n': 'vn',
                'vel_e': 've',
                'vel_d': 'vd',
                'vel_u': 'vu',
                'velocity_x': 'vx',
                'velocity_y': 'vy',
                'velocity_z': 'vz',
                'velocity_n': 'vn',
                'velocity_e': 've',
                'velocity_d': 'vd',
                'velocity_u': 'vu'
            }

            # Apply column mapping
            df.columns = [column_mapping.get(col.lower().strip(), col.lower().strip())
                         for col in df.columns]

            # Check if we have NED or ENU coordinates and convert to ECEF/body frame
            if 'vn' in df.columns and 've' in df.columns:
                if 'vd' in df.columns:
                    # NED to ENU conversion
                    logger.info("Converting NED velocities to ENU")
                    df['vx'] = df['ve']
                    df['vy'] = df['vn']
                    df['vz'] = -df['vd']  # Up is negative of down
                elif 'vu' in df.columns:
                    # Already in ENU
                    logger.info("Velocity data in ENU coordinates")
                    df['vx'] = df['ve']
                    df['vy'] = df['vn']
                    df['vz'] = df['vu']

            # Ensure we have the required columns
            required_cols = ['time', 'vx', 'vy', 'vz']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Convert time if needed
            if self.time_format == 'unix':
                logger.info("Converting UNIX time to GPS time")
                # GPS time starts at January 6, 1980
                # UNIX time starts at January 1, 1970
                # Difference is 315964800 seconds (not accounting for leap seconds)
                GPS_UNIX_OFFSET = 315964800 - 18  # 18 leap seconds as of 2017
                df['time'] = df['time'] - GPS_UNIX_OFFSET
            elif self.time_format == 'utc':
                # Convert UTC datetime strings to GPS seconds
                logger.info("Converting UTC time to GPS time")
                # This would need more sophisticated handling
                raise NotImplementedError("UTC time conversion not yet implemented")

            # Sort by time
            df = df.sort_values('time').reset_index(drop=True)

            # Keep only required columns
            df = df[required_cols]

            logger.info(f"Loaded {len(df)} velocity samples")
            if len(df) > 0:
                logger.info(f"  Time range: {df['time'].min():.3f} - {df['time'].max():.3f}")
                logger.info("  Velocity stats (m/s):")
                logger.info(f"    vx: mean={df['vx'].mean():.3f}, std={df['vx'].std():.3f}")
                logger.info(f"    vy: mean={df['vy'].mean():.3f}, std={df['vy'].std():.3f}")
                logger.info(f"    vz: mean={df['vz'].mean():.3f}, std={df['vz'].std():.3f}")

            return df

        except Exception as e:
            raise RuntimeError(f"Error reading velocity CSV file: {e}")

    def get_velocity_at_time(self, gps_time: float, method: str = 'linear') -> Optional[np.ndarray]:
        """
        Get interpolated velocity at a specific GPS time.

        This method provides temporal interpolation of velocity data,
        allowing velocity estimation at any time within the data range.
        The velocity data is cached after the first read for efficiency.

        Parameters
        ----------
        gps_time : float
            GPS time in seconds at which to estimate velocity
        method : str, default 'linear'
            Interpolation method to use:
            - 'linear': Linear interpolation between adjacent points
            - 'nearest': Use velocity from nearest time point

        Returns
        -------
        np.ndarray or None
            Velocity vector [vx, vy, vz] in m/s in ENU coordinates.
            Returns None if the requested time is outside the data range.

        Notes
        -----
        - Data is cached internally for efficient repeated queries
        - Linear interpolation is recommended for smooth velocity estimates
        - Nearest neighbor can be used when discrete velocity values are preferred
        - The method automatically handles time bounds checking

        Examples
        --------
        >>> reader = VelocityReader('velocity_data.csv')
        >>> velocity = reader.get_velocity_at_time(123456.789)
        >>> if velocity is not None:
        ...     print(f\"Velocity: {velocity[0]:.2f}, {velocity[1]:.2f}, {velocity[2]:.2f} m/s\")
        """
        # Load data only once (cache it)
        if self._data_cache is None:
            self._data_cache = self.read()

        df = self._data_cache

        if gps_time < df['time'].min() or gps_time > df['time'].max():
            return None

        if method == 'nearest':
            idx = (df['time'] - gps_time).abs().argmin()
            return df.iloc[idx][['vx', 'vy', 'vz']].values
        elif method == 'linear':
            # Linear interpolation
            vx = np.interp(gps_time, df['time'], df['vx'])
            vy = np.interp(gps_time, df['time'], df['vy'])
            vz = np.interp(gps_time, df['time'], df['vz'])
            return np.array([vx, vy, vz])
        else:
            raise ValueError(f"Unknown interpolation method: {method}")
