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

"""SP3 precise ephemeris handling with download capability"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import interpolate
import warnings

from ..core.constants import CLIGHT, SYS_GPS, SYS_GLO, SYS_GAL, SYS_BDS, SYS_QZS
from ..core.unified_time import TimeCore, TimeSystem
from .sp3_downloader import download_sp3_cddis, download_clk_cddis, get_best_sp3_product
from .sp3_interpolation import interpolate_sp3_position, interpolate_sp3_clock


class SP3Ephemeris:
    """SP3 precise ephemeris reader and interpolator with download capability"""
    
    def __init__(self, cache_dir: str = "./sp3_cache"):
        """
        Initialize SP3 ephemeris handler
        
        Parameters
        ----------
        cache_dir : str
            Directory to cache downloaded SP3 files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sp3_data = {}
        self.clock_data = {}
        
    def download_sp3(self, date: datetime, product: str = "igs", 
                     overwrite: bool = False) -> Optional[str]:
        """
        Download SP3 file using the new HTTPS downloader
        
        Parameters
        ----------
        date : datetime
            Date for which to download SP3 file
        product : str
            Product type: 'igs' (final), 'igr' (rapid), 'igu' (ultra-rapid)
        overwrite : bool
            Whether to overwrite existing file
            
        Returns
        -------
        str or None
            Path to downloaded file or None if failed
        """
        return download_sp3_cddis(date, product, str(self.cache_dir), overwrite)
        
    def download_clk(self, date: datetime, product: str = "igs",
                     overwrite: bool = False) -> Optional[str]:
        """
        Download CLK file using the new HTTPS downloader
        
        Parameters
        ----------
        date : datetime
            Date for which to download CLK file
        product : str
            Product type: 'igs' (final), 'igr' (rapid)
        overwrite : bool
            Whether to overwrite existing file
            
        Returns
        -------
        str or None
            Path to downloaded file or None if failed
        """
        return download_clk_cddis(date, product, str(self.cache_dir), overwrite)
        
    def read_sp3(self, filepath: str) -> Dict:
        """
        Read SP3 file
        
        Parameters
        ----------
        filepath : str
            Path to SP3 file
            
        Returns
        -------
        dict
            Dictionary with satellite data
        """
        sp3_data = {}
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        current_epoch = None
        
        for line in lines:
            if line[0] == '*':
                # Epoch line
                parts = line.split()
                year = int(parts[1])
                month = int(parts[2])
                day = int(parts[3])
                hour = int(parts[4])
                minute = int(parts[5])
                second = float(parts[6])
                
                from datetime import timezone
                current_epoch = datetime(year, month, day, hour, minute, int(second), tzinfo=timezone.utc)
                
            elif line[0] == 'P':
                # Position line
                parts = line.split()
                sat_id = parts[0][1:]  # Remove 'P' prefix
                
                # Parse satellite system and PRN
                sys_char = sat_id[0]
                prn = int(sat_id[1:])
                
                # Convert to satellite number
                if sys_char == 'G':
                    sat_num = prn
                elif sys_char == 'R':
                    sat_num = prn + 100
                elif sys_char == 'E':
                    sat_num = prn + 200
                elif sys_char == 'C':
                    sat_num = prn + 300
                elif sys_char == 'J':
                    sat_num = prn + 400
                else:
                    continue
                    
                # Parse position (km) and clock (microseconds)
                x = float(parts[1]) * 1000  # Convert to meters
                y = float(parts[2]) * 1000
                z = float(parts[3]) * 1000
                clock = float(parts[4]) * 1e-6  # Convert to seconds
                
                if sat_num not in sp3_data:
                    sp3_data[sat_num] = {
                        'epochs': [],
                        'positions': [],
                        'clocks': []
                    }
                    
                sp3_data[sat_num]['epochs'].append(current_epoch)
                sp3_data[sat_num]['positions'].append([x, y, z])
                sp3_data[sat_num]['clocks'].append(clock)
                
        # Convert lists to numpy arrays
        for sat_num in sp3_data:
            sp3_data[sat_num]['epochs'] = np.array(sp3_data[sat_num]['epochs'])
            sp3_data[sat_num]['positions'] = np.array(sp3_data[sat_num]['positions'])
            sp3_data[sat_num]['clocks'] = np.array(sp3_data[sat_num]['clocks'])
            
        return sp3_data
        
    def interpolate_position(self, sat_num: int, time: TimeCore, 
                           poly_degree: int = 10, method: str = 'neville') -> Tuple[np.ndarray, float, float]:
        """
        Interpolate satellite position and clock at given time
        
        Uses RTKLIB/GNSSpy compatible interpolation methods.
        
        Parameters
        ----------
        sat_num : int
            Satellite number
        time : TimeCore
            Time for interpolation
        poly_degree : int
            Polynomial degree for interpolation (default 10 for RTKLIB compatibility)
        method : str
            Interpolation method: 'neville' (RTKLIB), 'polyfit' (GNSSpy), 'lagrange'
            
        Returns
        -------
        position : np.ndarray
            Satellite position [x, y, z] in meters
        clock : float
            Satellite clock correction in seconds
        variance : float
            Position variance
        """
        if sat_num not in self.sp3_data:
            return None, None, None
            
        sat_data = self.sp3_data[sat_num]
        
        # Convert time to datetime for comparison
        target_dt = time.get_datetime()
        
        # Find epochs for interpolation
        epochs = sat_data['epochs']
        positions = sat_data['positions']
        clocks = sat_data['clocks']
        
        # Convert epochs to seconds from first epoch
        time_seconds = np.array([(ep - epochs[0]).total_seconds() for ep in epochs])
        target_seconds = (target_dt - epochs[0]).total_seconds()
        
        # Check if target time is within range
        if target_seconds < time_seconds[0] or target_seconds > time_seconds[-1]:
            print(f"Warning: Time {target_dt} is outside SP3 data range for satellite {sat_num}")
            return None, None, None
        
        # Use the new interpolation methods
        position, pos_success = interpolate_sp3_position(
            time_seconds, positions, target_seconds, 
            method=method, degree=poly_degree
        )
        
        if not pos_success:
            return None, None, None
        
        # Interpolate clock (linear interpolation is standard)
        clock, clk_success = interpolate_sp3_clock(
            time_seconds, clocks, target_seconds, method='linear'
        )
        
        if not clk_success:
            clock = 0.0  # Default if clock interpolation fails
        
        # Estimate variance based on interpolation method and degree
        # For RTKLIB compatibility with degree 10: ~1cm accuracy
        # For GNSSpy with degree 16: ~2.5cm accuracy  
        if method == 'neville' and poly_degree == 10:
            variance = 0.01  # 1 cm (RTKLIB standard)
        elif method == 'polyfit' and poly_degree >= 16:
            variance = 0.025  # 2.5 cm (GNSSpy with high degree)
        else:
            variance = 0.05  # 5 cm (conservative estimate)
        
        return position, clock, variance
        
    def load_sp3_for_time(self, time: TimeCore, product: str = "igs",
                         download: bool = True) -> bool:
        """
        Load SP3 data for the given time
        
        Parameters
        ----------
        time : TimeCore
            Time for which to load SP3 data
        product : str
            Product type: 'igs', 'igr', 'igu'
        download : bool
            Whether to download if not available locally
            
        Returns
        -------
        bool
            True if successfully loaded
        """
        # Get date from time
        dt = time.get_datetime()
        
        # Check if we need files from adjacent days
        dates_needed = [dt.date()]
        if dt.hour < 3:
            dates_needed.append((dt - timedelta(days=1)).date())
        if dt.hour > 21:
            dates_needed.append((dt + timedelta(days=1)).date())
            
        sp3_files = []
        for date in dates_needed:
            # Check local cache first
            date_dt = datetime.combine(date, datetime.min.time())
            
            # Try to find existing file
            gps_epoch = datetime(1980, 1, 6)
            delta = date_dt - gps_epoch
            gps_week = delta.days // 7
            gps_dow = delta.days % 7
            
            if product == "igs":
                filename = f"igs{gps_week:04d}{gps_dow:01d}.sp3"
            elif product == "igr":
                filename = f"igr{gps_week:04d}{gps_dow:01d}.sp3"
            elif product == "igu":
                hour = (dt.hour // 6) * 6
                filename = f"igu{gps_week:04d}{gps_dow:01d}_{hour:02d}.sp3"
                
            filepath = self.cache_dir / filename
            
            if not filepath.exists() and download:
                # Download if not available - use best available product
                filepath = get_best_sp3_product(date_dt, str(self.cache_dir))
                if filepath is None:
                    print(f"Failed to obtain SP3 file for {date}")
                    continue
                else:
                    filepath = Path(filepath)
                    
            if filepath and Path(filepath).exists():
                sp3_files.append(str(filepath))
                
        if not sp3_files:
            print("No SP3 files available")
            return False
            
        # Read and merge SP3 data
        all_sp3_data = {}
        for filepath in sp3_files:
            sp3_data = self.read_sp3(filepath)
            
            # Merge data
            for sat_num, data in sp3_data.items():
                if sat_num not in all_sp3_data:
                    all_sp3_data[sat_num] = {
                        'epochs': [],
                        'positions': [],
                        'clocks': []
                    }
                all_sp3_data[sat_num]['epochs'].extend(data['epochs'])
                all_sp3_data[sat_num]['positions'].extend(data['positions'])
                all_sp3_data[sat_num]['clocks'].extend(data['clocks'])
                
        # Sort and convert to arrays
        for sat_num in all_sp3_data:
            # Sort by epoch
            epochs = np.array(all_sp3_data[sat_num]['epochs'])
            positions = np.array(all_sp3_data[sat_num]['positions'])
            clocks = np.array(all_sp3_data[sat_num]['clocks'])
            
            sort_idx = np.argsort(epochs)
            all_sp3_data[sat_num]['epochs'] = epochs[sort_idx]
            all_sp3_data[sat_num]['positions'] = positions[sort_idx]
            all_sp3_data[sat_num]['clocks'] = clocks[sort_idx]
            
        self.sp3_data = all_sp3_data
        return True
        
    def get_satellite_position(self, sat_num: int, time: TimeCore) -> Tuple[np.ndarray, float, float]:
        """
        Get satellite position and clock at given time
        
        Parameters
        ----------
        sat_num : int
            Satellite number
        time : TimeCore
            Time for position
            
        Returns
        -------
        position : np.ndarray
            Satellite position [x, y, z] in meters
        clock : float
            Satellite clock correction in seconds
        variance : float
            Position variance
        """
        # Ensure SP3 data is loaded for this time
        if not self.sp3_data:
            if not self.load_sp3_for_time(time):
                return None, None, None
                
        return self.interpolate_position(sat_num, time)