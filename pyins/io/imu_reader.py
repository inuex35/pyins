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

"""IMU data reading utilities"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..sensors.imu import IMUData

logger = logging.getLogger(__name__)


class IMUReader:
    """IMU data reader for various file formats"""

    def __init__(self, file_path: str, format: str = 'csv', time_system: str = 'gps'):
        """
        Initialize IMU reader

        Parameters:
        -----------
        file_path : str
            Path to IMU data file
        format : str
            File format ('csv', 'txt', 'binary')
        time_system : str
            Time system of the input file ('unix', 'gps')
            - 'unix': Input is in UNIX time
            - 'gps': Input is in GPS time (default)
        """
        self.file_path = Path(file_path)
        self.format = format.lower()
        self.input_time_system = time_system
        self.logger = logging.getLogger(__name__)

        # GPS to UNIX time offset
        self.GPS_TO_UNIX_OFFSET = 315964800  # GPS epoch: January 6, 1980
        self.LEAP_SECONDS_2015 = 17  # Leap seconds as of 2015

        if not self.file_path.exists():
            raise FileNotFoundError(f"IMU file not found: {file_path}")

    def read(self, start_time: Optional[float] = None, duration: Optional[float] = None,
             output_time_system: str = 'same') -> pd.DataFrame:
        """
        Read IMU data from file with time system conversion

        Parameters:
        -----------
        start_time : float, optional
            Start time in the OUTPUT time system
        duration : float, optional
            Duration in seconds to load
        output_time_system : str
            Desired output time system ('unix', 'gps', 'same')
            - 'unix': Convert to UNIX time
            - 'gps': Convert to GPS time
            - 'same': Keep same as input (default)

        Returns:
        --------
        pd.DataFrame
            IMU data with columns: time, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
        """
        # First read the raw data
        if self.format == 'csv':
            df = self._read_csv(None, None)  # Read all data first
        elif self.format == 'txt':
            df = self._read_txt(None, None)
        else:
            raise ValueError(f"Unsupported format: {self.format}")

        # Convert time if needed
        if output_time_system != 'same' and output_time_system != self.input_time_system:
            df = self._convert_time_system(df, self.input_time_system, output_time_system)

        # Apply time filters after conversion
        if start_time is not None:
            df = df[df['time'] >= start_time]
        if duration is not None and start_time is not None:
            df = df[df['time'] <= start_time + duration]

        return df

    def _convert_time_system(self, df: pd.DataFrame, from_system: str, to_system: str) -> pd.DataFrame:
        """
        Convert time between different time systems.

        This method handles conversion between UNIX time and GPS time systems.
        GPS time starts at January 6, 1980, while UNIX time starts at
        January 1, 1970. The conversion also accounts for leap seconds.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing IMU data with time column
        from_system : str
            Source time system ('unix' or 'gps')
        to_system : str
            Target time system ('unix' or 'gps')

        Returns
        -------
        pd.DataFrame
            DataFrame with converted time column

        Notes
        -----
        - GPS to UNIX offset is 315964800 seconds
        - Leap seconds are accounted for (17 seconds as of 2015)
        - If from_system equals to_system, returns original DataFrame
        """
        if from_system == to_system:
            return df

        df = df.copy()

        if from_system == 'unix' and to_system == 'gps':
            # Convert UNIX to GPS
            df['time'] = df['time'] - self.GPS_TO_UNIX_OFFSET + self.LEAP_SECONDS_2015
            self.logger.info(f"Converted time from UNIX to GPS (offset: {-self.GPS_TO_UNIX_OFFSET + self.LEAP_SECONDS_2015})")
        elif from_system == 'gps' and to_system == 'unix':
            # Convert GPS to UNIX
            df['time'] = df['time'] + self.GPS_TO_UNIX_OFFSET - self.LEAP_SECONDS_2015
            self.logger.info(f"Converted time from GPS to UNIX (offset: {self.GPS_TO_UNIX_OFFSET - self.LEAP_SECONDS_2015})")

        return df

    def _read_csv(self, start_time: Optional[float] = None, duration: Optional[float] = None) -> pd.DataFrame:
        """
        Read IMU data from CSV file with automatic column mapping.

        This method reads IMU data from CSV files and handles various column
        naming conventions. It automatically maps common alternative names
        to the standard format expected by the pyins library.

        Parameters
        ----------
        start_time : float, optional
            Start time to filter data (not used in this implementation,
            filtering is done after reading)
        duration : float, optional
            Duration in seconds (not used in this implementation,
            filtering is done after reading)

        Returns
        -------
        pd.DataFrame
            IMU data with standardized columns:
            - time: timestamp
            - accel_x, accel_y, accel_z: acceleration in m/s²
            - gyro_x, gyro_y, gyro_z: angular velocity in rad/s

        Raises
        ------
        ValueError
            If required IMU columns are missing after column mapping

        Notes
        -----
        Supported alternative column names:
        - timestamp -> time
        - ax, ay, az -> accel_x, accel_y, accel_z
        - acc_x, acc_y, acc_z -> accel_x, accel_y, accel_z
        - gx, gy, gz -> gyro_x, gyro_y, gyro_z
        - wx, wy, wz -> gyro_x, gyro_y, gyro_z
        """
        self.logger.info(f"Reading IMU data from CSV: {self.file_path}")

        # Load IMU data
        imu_data = pd.read_csv(self.file_path)

        # Check for time column first
        if 'timestamp' in imu_data.columns and 'time' not in imu_data.columns:
            imu_data.rename(columns={'timestamp': 'time'}, inplace=True)

        # Check required columns
        required_cols = ['time', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
        if not all(col in imu_data.columns for col in required_cols):
            # Try alternative column names
            alt_mapping = {
                'ax': 'accel_x', 'ay': 'accel_y', 'az': 'accel_z',
                'acc_x': 'accel_x', 'acc_y': 'accel_y', 'acc_z': 'accel_z',
                'gx': 'gyro_x', 'gy': 'gyro_y', 'gz': 'gyro_z',
                'wx': 'gyro_x', 'wy': 'gyro_y', 'wz': 'gyro_z'
            }
            imu_data.rename(columns=alt_mapping, inplace=True)

            # Check again
            missing = [col for col in required_cols if col not in imu_data.columns]
            if missing:
                raise ValueError(f"Missing required IMU columns: {missing}")

        return self._apply_filters(imu_data, start_time, duration)

    def _read_txt(self, start_time: Optional[float] = None, duration: Optional[float] = None) -> pd.DataFrame:
        """
        Read IMU data from text file with space-separated values.

        This method reads IMU data from text files where values are
        separated by whitespace. The expected format is fixed-column
        with 7 values per line.

        Parameters
        ----------
        start_time : float, optional
            Start time to filter data (not used in this implementation,
            filtering is done after reading)
        duration : float, optional
            Duration in seconds (not used in this implementation,
            filtering is done after reading)

        Returns
        -------
        pd.DataFrame
            IMU data with columns:
            - time: timestamp
            - accel_x, accel_y, accel_z: acceleration in m/s²
            - gyro_x, gyro_y, gyro_z: angular velocity in rad/s

        Raises
        ------
        ValueError
            If the text file cannot be parsed or has invalid format

        Notes
        -----
        Expected text file format:
        time accel_x accel_y accel_z gyro_x gyro_y gyro_z

        Lines starting with '#' are treated as comments and skipped.
        Multiple whitespace characters are treated as single separator.
        """
        self.logger.info(f"Reading IMU data from TXT: {self.file_path}")

        # Read text file with space-separated values
        # Expected format: time accel_x accel_y accel_z gyro_x gyro_y gyro_z
        try:
            imu_data = pd.read_csv(
                self.file_path,
                sep=r'\s+',  # Multiple whitespace as separator
                names=['time', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z'],
                comment='#',  # Skip comment lines
                header=None
            )
        except Exception as e:
            raise ValueError(f"Failed to read IMU text file: {e}")

        return self._apply_filters(imu_data, start_time, duration)

    def _apply_filters(self, imu_data: pd.DataFrame, start_time: Optional[float], duration: Optional[float]) -> pd.DataFrame:
        """
        Apply time filters and sort IMU data chronologically.

        This method filters the IMU data by time range and sorts it
        chronologically. It also provides logging information about
        the loaded data including statistics.

        Parameters
        ----------
        imu_data : pd.DataFrame
            Raw IMU data to filter and sort
        start_time : float, optional
            Start time for filtering. If None, no start time filter is applied
        duration : float, optional
            Duration in seconds. If provided with start_time, data is filtered
            to start_time + duration. Ignored if start_time is None

        Returns
        -------
        pd.DataFrame
            Filtered and sorted IMU data

        Notes
        -----
        The method logs information about:
        - Number of loaded samples
        - Time range of the data
        - Estimated sampling rate based on median time difference
        """
        # Apply time filter if requested
        if start_time is not None:
            imu_data = imu_data[imu_data['time'] >= start_time]

        if duration is not None and len(imu_data) > 0 and start_time is not None:
            end_time = start_time + duration
            imu_data = imu_data[imu_data['time'] <= end_time]

        # Sort by time
        imu_data = imu_data.sort_values('time').reset_index(drop=True)

        self.logger.info(f"Loaded {len(imu_data)} IMU samples")
        if len(imu_data) > 0:
            dt = imu_data['time'].diff().median()
            freq = 1.0 / dt if dt > 0 else 0
            self.logger.info(f"  Time range: {imu_data['time'].iloc[0]:.3f} - {imu_data['time'].iloc[-1]:.3f}")
            self.logger.info(f"  Sampling rate: ~{freq:.1f} Hz")

        return imu_data

    def read_as_imu_data_list(self, start_time: Optional[float] = None, duration: Optional[float] = None) -> list[IMUData]:
        """
        Read IMU data and return as list of IMUData objects

        Returns:
        --------
        List[IMUData]
            List of IMU measurements as IMUData objects
        """
        df = self.read(start_time, duration)

        imu_data_list = []
        for _, row in df.iterrows():
            # Create 6D measurement vector [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
            measurement = np.array([
                row['accel_x'], row['accel_y'], row['accel_z'],
                row['gyro_x'], row['gyro_y'], row['gyro_z']
            ])

            imu_data = IMUData(
                timestamp=row['time'],
                sensor_id="imu_0",  # Default sensor ID
                data=measurement
            )
            imu_data_list.append(imu_data)

        return imu_data_list


class IMUPreprocessor:
    """IMU data preprocessing utilities"""

    @staticmethod
    def remove_bias(imu_data: pd.DataFrame, accel_bias: Optional[np.ndarray] = None,
                   gyro_bias: Optional[np.ndarray] = None, estimate_from_stationary: bool = True) -> pd.DataFrame:
        """
        Remove bias from IMU data

        Parameters:
        -----------
        imu_data : pd.DataFrame
            Raw IMU data
        accel_bias : np.ndarray, optional
            Known accelerometer bias [ax, ay, az]
        gyro_bias : np.ndarray, optional
            Known gyroscope bias [gx, gy, gz]
        estimate_from_stationary : bool
            Whether to estimate bias from stationary period if not provided

        Returns:
        --------
        pd.DataFrame
            Bias-corrected IMU data
        """
        logger = logging.getLogger(__name__)
        processed = imu_data.copy()

        if estimate_from_stationary:
            # Estimate bias from stationary period if not provided
            if accel_bias is None:
                # Use first 1 second as stationary
                stationary_mask = processed['time'] < processed['time'].iloc[0] + 1.0
                if stationary_mask.sum() > 10:
                    accel_bias = np.array([
                        processed.loc[stationary_mask, 'accel_x'].mean(),
                        processed.loc[stationary_mask, 'accel_y'].mean(),
                        processed.loc[stationary_mask, 'accel_z'].mean()
                    ])
                    # Subtract gravity from z-axis (assuming upright orientation)
                    accel_bias[2] -= 9.80665
                    logger.info(f"Estimated accel bias: {accel_bias}")

            if gyro_bias is None:
                stationary_mask = processed['time'] < processed['time'].iloc[0] + 1.0
                if stationary_mask.sum() > 10:
                    gyro_bias = np.array([
                        processed.loc[stationary_mask, 'gyro_x'].mean(),
                        processed.loc[stationary_mask, 'gyro_y'].mean(),
                        processed.loc[stationary_mask, 'gyro_z'].mean()
                    ])
                    logger.info(f"Estimated gyro bias: {gyro_bias}")

        # Remove bias
        if accel_bias is not None:
            processed['accel_x'] -= accel_bias[0]
            processed['accel_y'] -= accel_bias[1]
            processed['accel_z'] -= accel_bias[2]

        if gyro_bias is not None:
            processed['gyro_x'] -= gyro_bias[0]
            processed['gyro_y'] -= gyro_bias[1]
            processed['gyro_z'] -= gyro_bias[2]

        return processed

    @staticmethod
    def sync_with_gnss(gnss_epochs: list[dict], imu_data: pd.DataFrame, max_time_diff: float = 0.01) -> list[dict]:
        """
        Synchronize GNSS epochs with IMU data

        Parameters:
        -----------
        gnss_epochs : list
            List of GNSS observation epochs
        imu_data : pd.DataFrame
            IMU measurements
        max_time_diff : float
            Maximum time difference to consider synchronized (seconds)

        Returns:
        --------
        list
            List of synchronized epochs with structure:
            {
                'time': Unix timestamp,
                'gnss_obs': GNSS observations,
                'imu_data': List of IMU measurements between this and next epoch
            }
        """
        logger = logging.getLogger(__name__)

        synced_epochs = []

        # Process each GNSS epoch
        for i in range(len(gnss_epochs) - 1):
            gnss_time = gnss_epochs[i]['time']
            next_gnss_time = gnss_epochs[i + 1]['time']

            # Find IMU data between current and next GNSS epoch
            mask = (imu_data['time'] >= gnss_time) & (imu_data['time'] < next_gnss_time)
            imu_subset = imu_data[mask]

            # Skip if no IMU data
            if len(imu_subset) == 0:
                logger.warning(f"No IMU data for GNSS epoch at {gnss_time}")
                continue

            # Check time synchronization of first IMU sample
            time_diff = abs(imu_subset.iloc[0]['time'] - gnss_time)
            if time_diff > max_time_diff:
                logger.warning(f"Large time difference ({time_diff:.3f}s) at epoch {i}")

            # Create synchronized epoch
            # Convert IMU data to the format expected by INS processor
            imu_list = []
            for _, row in imu_subset.iterrows():
                imu_list.append({
                    'time': row['time'],
                    'accel': np.array([row['accel_x'], row['accel_y'], row['accel_z']]),
                    'gyro': np.array([row['gyro_x'], row['gyro_y'], row['gyro_z']])
                })

            synced_epoch = {
                'time': gnss_time,
                'gps_time': gnss_epochs[i].get('gps_time', gnss_time),  # Include GPS time if available
                'gnss_obs': gnss_epochs[i]['observations'],
                'imu_data': imu_list
            }

            synced_epochs.append(synced_epoch)

        logger.info(f"Synchronized {len(synced_epochs)} epochs from {len(gnss_epochs)} GNSS epochs")

        return synced_epochs


def load_imu_data(imu_file: str, start_time: Optional[float] = None,
                 duration: Optional[float] = None, format: str = 'csv') -> pd.DataFrame:
    """
    Convenience function to load IMU data from file

    Parameters:
    -----------
    imu_file : str
        Path to IMU data file
    start_time : float, optional
        Start time (Unix timestamp) to load from
    duration : float, optional
        Duration in seconds to load
    format : str
        File format ('csv', 'txt')

    Returns:
    --------
    pd.DataFrame
        IMU data with columns: time, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
    """
    reader = IMUReader(imu_file, format=format)
    return reader.read(start_time=start_time, duration=duration)


def preprocess_imu_data(imu_data: pd.DataFrame, remove_bias: bool = True,
                       accel_bias: Optional[np.ndarray] = None,
                       gyro_bias: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Convenience function to preprocess IMU data

    Parameters:
    -----------
    imu_data : pd.DataFrame
        Raw IMU data
    remove_bias : bool
        Whether to remove bias
    accel_bias : np.ndarray, optional
        Known accelerometer bias [ax, ay, az]
    gyro_bias : np.ndarray, optional
        Known gyroscope bias [gx, gy, gz]

    Returns:
    --------
    pd.DataFrame
        Preprocessed IMU data
    """
    if remove_bias:
        return IMUPreprocessor.remove_bias(imu_data, accel_bias, gyro_bias)
    return imu_data.copy()


def sync_gnss_imu(gnss_epochs: list[dict], imu_data: pd.DataFrame, max_time_diff: float = 0.01) -> list[dict]:
    """
    Convenience function to synchronize GNSS epochs with IMU data

    Parameters:
    -----------
    gnss_epochs : list
        List of GNSS observation epochs from load_gnss_data
    imu_data : pd.DataFrame
        IMU measurements from load_imu_data
    max_time_diff : float
        Maximum time difference to consider synchronized (seconds)

    Returns:
    --------
    list
        List of synchronized epochs
    """
    return IMUPreprocessor.sync_with_gnss(gnss_epochs, imu_data, max_time_diff)
