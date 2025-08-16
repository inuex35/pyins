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

"""Data synchronization for RTK processing"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import bisect
from ..core.data_structures import Observation
from ..io.rinex import RinexObsReader
from ..core.time_conversions import datetime_to_gps_seconds
import logging

try:
    from gnsspy import read_obsFile
    GNSSPY_AVAILABLE = True
except ImportError:
    GNSSPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class RTKDataSynchronizer:
    """Synchronize rover and base station GNSS observations"""
    
    def __init__(self, time_tolerance: float = 0.1):
        """
        Initialize data synchronizer
        
        Parameters:
        -----------
        time_tolerance : float
            Maximum time difference (seconds) for synchronization
        """
        self.time_tolerance = time_tolerance
        self.rover_data = defaultdict(list)
        self.base_data = defaultdict(list)
        self.rover_df = None
        self.base_df = None
        self.rover_obs_data = None
        self.base_obs_data = None
        self.use_pandas = GNSSPY_AVAILABLE
        
    def load_rover_data(self, rinex_file: str, max_duration: Optional[float] = None):
        """Load rover RINEX observation file
        
        Parameters:
        -----------
        rinex_file : str
            Path to rover RINEX file
        max_duration : Optional[float]
            Maximum duration in seconds to load
        """
        if self.use_pandas:
            logger.info(f"Loading rover data from {rinex_file} using pandas")
            
            # Read RINEX file
            self.rover_obs_data = read_obsFile(rinex_file)
            self.rover_df = self.rover_obs_data.observation
            
            # Get time range
            epochs = self.rover_df.index.get_level_values('Epoch').unique()
            start_time = epochs[0]
            
            if max_duration:
                # Filter by duration
                end_time = start_time + pd.Timedelta(seconds=max_duration)
                self.rover_df = self.rover_df.loc[self.rover_df.index.get_level_values('Epoch') <= end_time]
                logger.info(f"Limited rover data to {max_duration} seconds")
                
            # Also populate legacy format for compatibility
            for epoch in self.rover_df.index.get_level_values('Epoch').unique():
                gps_time = datetime_to_gps_seconds(epoch)
                observations = self._extract_observations(self.rover_df, epoch)
                self.rover_data[gps_time].extend(observations)
                
            logger.info(f"Loaded {len(self.rover_data)} rover epochs")
        else:
            # Fallback to old method
            reader = RinexObsReader(rinex_file)
            epochs = reader.read()
            
            count = 0
            start_time = None
            for epoch in epochs:
                if start_time is None:
                    start_time = epoch['gps_time']
                    
                if max_duration and (epoch['gps_time'] - start_time) > max_duration:
                    break
                    
                self.rover_data[epoch['gps_time']].extend(epoch['observations'])
                count += 1
                    
            logger.info(f"Loaded {len(self.rover_data)} rover epochs")
        
    def load_base_data(self, rinex_file: str):
        """Load base station RINEX observation file
        
        Only loads data within the rover time range
        """
        if self.use_pandas and self.rover_df is not None:
            logger.info(f"Loading base data from {rinex_file} using pandas")
            
            # Get rover time range
            rover_epochs = self.rover_df.index.get_level_values('Epoch').unique()
            rover_start = rover_epochs.min()
            rover_end = rover_epochs.max()
            
            # Read base RINEX
            self.base_obs_data = read_obsFile(rinex_file)
            base_df_full = self.base_obs_data.observation
            
            # Filter to rover time range (with tolerance)
            start_with_tol = rover_start - pd.Timedelta(seconds=self.time_tolerance)
            end_with_tol = rover_end + pd.Timedelta(seconds=self.time_tolerance)
            
            self.base_df = base_df_full.loc[
                (base_df_full.index.get_level_values('Epoch') >= start_with_tol) &
                (base_df_full.index.get_level_values('Epoch') <= end_with_tol)
            ]
            
            # Also populate legacy format for compatibility
            for epoch in self.base_df.index.get_level_values('Epoch').unique():
                gps_time = datetime_to_gps_seconds(epoch)
                observations = self._extract_observations(self.base_df, epoch)
                self.base_data[gps_time].extend(observations)
                
            logger.info(f"Loaded {len(self.base_data)} base epochs within rover time range")
        else:
            # Get rover time range
            if not self.rover_data:
                raise ValueError("Must load rover data first")
                
            rover_times = sorted(self.rover_data.keys())
            rover_start = rover_times[0]
            rover_end = rover_times[-1]
            
            reader = RinexObsReader(rinex_file)
            epochs = reader.read()
            
            count = 0
            skipped = 0
            for epoch in epochs:
                gps_time = epoch['gps_time']
                
                # Skip if outside rover time range
                if gps_time < rover_start - self.time_tolerance:
                    skipped += 1
                    continue
                if gps_time > rover_end + self.time_tolerance:
                    logger.info(f"Reached end of rover time range, stopping base data load")
                    break
                    
                self.base_data[gps_time].extend(epoch['observations'])
                count += 1
                
            logger.info(f"Loaded {len(self.base_data)} base epochs (skipped {skipped} before rover start)")
        
    def add_rover_observations(self, time: float, observations: List[Observation]):
        """Add rover observations for a specific epoch"""
        self.rover_data[time].extend(observations)
        
    def add_base_observations(self, time: float, observations: List[Observation]):
        """Add base observations for a specific epoch"""
        self.base_data[time].extend(observations)
        
    def get_synchronized_epochs(self) -> List[Tuple[float, List[Observation], List[Observation]]]:
        """
        Get synchronized observation epochs
        
        Returns:
        --------
        synchronized_epochs : List[Tuple[float, List[Observation], List[Observation]]]
            List of (time, rover_obs, base_obs) tuples
        """
        synchronized = []
        
        # Sort times
        rover_times = sorted(self.rover_data.keys())
        base_times = sorted(self.base_data.keys())
        
        if not rover_times or not base_times:
            return synchronized
        
        # Get rover time range
        rover_start = rover_times[0]
        rover_end = rover_times[-1]
        
        # Filter base times to rover range
        base_times_filtered = [t for t in base_times if rover_start <= t <= rover_end]
        logger.info(f"Filtered base epochs from {len(base_times)} to {len(base_times_filtered)} within rover time range")
        
        # Clear base data outside rover range to save memory
        for base_time in list(self.base_data.keys()):
            if base_time < rover_start or base_time > rover_end:
                del self.base_data[base_time]
            
        # Find matching epochs
        for rover_time in rover_times:
            # Find closest base time
            idx = bisect.bisect_left(base_times_filtered, rover_time)
            
            # Check both neighbors
            candidates = []
            if idx > 0:
                candidates.append((abs(rover_time - base_times_filtered[idx-1]), idx-1))
            if idx < len(base_times_filtered):
                candidates.append((abs(rover_time - base_times_filtered[idx]), idx))
                
            if not candidates:
                continue
                
            # Get closest match
            time_diff, best_idx = min(candidates)
            
            if time_diff <= self.time_tolerance:
                base_time = base_times_filtered[best_idx]
                synchronized.append((
                    rover_time,
                    self.rover_data[rover_time],
                    self.base_data[base_time]
                ))
                
        logger.info(f"Synchronized {len(synchronized)} epochs from {len(rover_times)} rover epochs")
        return synchronized
        
    def get_synchronized_epoch(self, time: float) -> Optional[Tuple[List[Observation], List[Observation]]]:
        """
        Get synchronized observations for a specific time
        
        Parameters:
        -----------
        time : float
            Requested epoch time
            
        Returns:
        --------
        synchronized : Optional[Tuple[List[Observation], List[Observation]]]
            (rover_obs, base_obs) if synchronized epoch exists, None otherwise
        """
        # Check if rover data exists
        if time not in self.rover_data:
            return None
            
        # Find closest base time
        base_times = sorted(self.base_data.keys())
        idx = bisect.bisect_left(base_times, time)
        
        candidates = []
        if idx > 0:
            candidates.append((abs(time - base_times[idx-1]), idx-1))
        if idx < len(base_times):
            candidates.append((abs(time - base_times[idx]), idx))
            
        if not candidates:
            return None
            
        time_diff, best_idx = min(candidates)
        
        if time_diff <= self.time_tolerance:
            base_time = base_times[best_idx]
            return self.rover_data[time], self.base_data[base_time]
            
        return None
        
    def clear(self):
        """Clear all stored data"""
        self.rover_data.clear()
        self.base_data.clear()
        self.rover_df = None
        self.base_df = None
        self.rover_obs_data = None
        self.base_obs_data = None
        
    def _extract_observations(self, df: pd.DataFrame, epoch) -> List[Observation]:
        """Extract observations for a specific epoch"""
        epoch_df = df.loc[epoch]
        observations = []
        
        # Convert epoch to GPS seconds
        gps_time = datetime_to_gps_seconds(epoch)
        
        for sv_id, row in epoch_df.iterrows():
            # Parse satellite ID
            sys_char = sv_id[0]
            prn = int(sv_id[1:])
            
            # Map system character to satellite number
            # Use pyins standard mapping
            if sys_char == 'G':  # GPS: 1-32
                sat = prn
            elif sys_char == 'R':  # GLONASS: 33-59
                sat = prn + 32
            elif sys_char == 'E':  # Galileo: 60-95
                sat = prn + 59
            elif sys_char == 'C':  # BeiDou: 96-163
                sat = prn + 95
            elif sys_char == 'J':  # QZSS: 164-172
                sat = prn + 163
            else:
                continue
                
            # Create observation
            obs = Observation(
                time=gps_time,  # Set correct GPS time
                sat=sat,
                system=0   # Will be set by __post_init__
            )
            
            # Set pseudorange values
            if not pd.isna(row.get('C1C', np.nan)):
                obs.P[0] = row['C1C']
            if not pd.isna(row.get('C2W', np.nan)):
                obs.P[1] = row['C2W']
                
            # Set carrier phase values
            if not pd.isna(row.get('L1C', np.nan)):
                obs.L[0] = row['L1C']
            if not pd.isna(row.get('L2W', np.nan)):
                obs.L[1] = row['L2W']
                
            # Set SNR values
            if not pd.isna(row.get('S1C', np.nan)):
                obs.SNR[0] = row['S1C']
            if not pd.isna(row.get('S2W', np.nan)):
                obs.SNR[1] = row['S2W']
            
            # Only add if we have valid pseudorange on L1
            if obs.P[0] > 0:
                observations.append(obs)
                
        return observations
        
    def get_approx_position(self, which='base') -> Optional[np.ndarray]:
        """Get approximate position from RINEX header
        
        Parameters:
        -----------
        which : str
            'base' or 'rover'
            
        Returns:
        --------
        position : Optional[np.ndarray]
            ECEF position or None if not available
        """
        if which == 'base' and self.base_obs_data:
            if hasattr(self.base_obs_data, 'approx_position'):
                return np.array(self.base_obs_data.approx_position)
        elif which == 'rover' and self.rover_obs_data:
            if hasattr(self.rover_obs_data, 'approx_position'):
                return np.array(self.rover_obs_data.approx_position)
        return None


class ObservationMatcher:
    """Match observations between rover and base for quality control"""
    
    def __init__(self):
        self.matched_stats = defaultdict(int)
        
    def match_observations(self, 
                         rover_obs: List[Observation], 
                         base_obs: List[Observation],
                         require_dual_freq: bool = False) -> Tuple[List[Observation], List[Observation]]:
        """
        Match observations between rover and base
        
        Parameters:
        -----------
        rover_obs : List[Observation]
            Rover observations
        base_obs : List[Observation]
            Base observations
        require_dual_freq : bool
            Whether to require dual frequency observations
            
        Returns:
        --------
        matched_rover : List[Observation]
            Matched rover observations
        matched_base : List[Observation]
            Matched base observations
        """
        # Create dictionaries by satellite
        rover_dict = {obs.sat: obs for obs in rover_obs}
        base_dict = {obs.sat: obs for obs in base_obs}
        
        matched_rover = []
        matched_base = []
        
        # Find common satellites
        common_sats = set(rover_dict.keys()) & set(base_dict.keys())
        
        for sat in common_sats:
            rover_sat = rover_dict[sat]
            base_sat = base_dict[sat]
            
            # Check if both have valid pseudorange
            if rover_sat.P1 <= 0 or base_sat.P1 <= 0:
                self.matched_stats['missing_P1'] += 1
                continue
                
            # Check if both have valid carrier phase
            if rover_sat.L1 <= 0 or base_sat.L1 <= 0:
                self.matched_stats['missing_L1'] += 1
                continue
                
            # Check dual frequency if required
            if require_dual_freq:
                if rover_sat.P2 <= 0 or base_sat.P2 <= 0:
                    self.matched_stats['missing_P2'] += 1
                    continue
                if rover_sat.L2 <= 0 or base_sat.L2 <= 0:
                    self.matched_stats['missing_L2'] += 1
                    continue
                    
            matched_rover.append(rover_sat)
            matched_base.append(base_sat)
            self.matched_stats['matched'] += 1
            
        self.matched_stats['total_rover'] += len(rover_obs)
        self.matched_stats['total_base'] += len(base_obs)
        
        return matched_rover, matched_base
        
    def get_statistics(self) -> Dict[str, int]:
        """Get matching statistics"""
        return dict(self.matched_stats)
        
    def reset_statistics(self):
        """Reset matching statistics"""
        self.matched_stats.clear()