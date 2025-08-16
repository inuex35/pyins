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

"""GNSS Time Systems and Conversions"""

import numpy as np
from datetime import datetime, timedelta
from typing import Union, Optional
from .constants import GPST0, GST0, BDT0

class GNSSTime:
    """GNSS Time representation and conversion with type safety
    
    This class ensures that time systems are not accidentally mixed.
    All arithmetic operations check for compatible time systems.
    """
    
    def __init__(self, week: int = 0, tow: float = 0.0, time_sys: str = 'GPS'):
        """
        Initialize GNSS time
        
        Parameters:
        -----------
        week : int
            Week number
        tow : float
            Time of week in seconds
        time_sys : str
            Time system ('GPS', 'GAL', 'BDS', 'GLO', 'UTC')
        """
        self.week = int(week)
        self.tow = float(tow)
        self.time_sys = time_sys.upper()
        
        # Validate time system
        valid_systems = ['GPS', 'GAL', 'BDS', 'GLO', 'UTC']
        if self.time_sys not in valid_systems:
            raise ValueError(f"Invalid time system: {time_sys}. Must be one of {valid_systems}")
        
        # Normalize TOW to [0, 604800)
        while self.tow >= 604800:
            self.week += 1
            self.tow -= 604800
        while self.tow < 0:
            self.week -= 1
            self.tow += 604800
        
    @classmethod
    def from_datetime(cls, dt, time_sys='GPS'):
        """Create GNSSTime from datetime object"""
        if time_sys == 'GPS':
            ref_date = datetime(*GPST0)
        elif time_sys == 'GAL':
            ref_date = datetime(*GST0)
        elif time_sys == 'BDS':
            ref_date = datetime(*BDT0)
        else:
            raise ValueError(f"Unknown time system: {time_sys}")
            
        delta = dt - ref_date
        weeks = delta.days // 7
        tow = (delta.days % 7) * 86400 + delta.seconds + delta.microseconds * 1e-6
        
        return cls(weeks, tow, time_sys)
    
    @classmethod
    def from_gps_seconds(cls, gps_seconds, time_sys='GPS'):
        """Create GNSSTime from GPS seconds since GPS epoch"""
        week = int(gps_seconds // 604800)
        tow = gps_seconds % 604800
        return cls(week, tow, time_sys)
    
    @classmethod
    def from_mjd(cls, mjd, time_sys='GPS'):
        """Create GNSSTime from Modified Julian Day"""
        # MJD epoch: 1858-11-17 00:00:00
        mjd_epoch = datetime(1858, 11, 17)
        dt = mjd_epoch + timedelta(days=mjd)
        return cls.from_datetime(dt, time_sys)
    
    def to_datetime(self):
        """Convert to datetime object"""
        if self.time_sys == 'GPS':
            ref_date = datetime(*GPST0)
        elif self.time_sys == 'GAL':
            ref_date = datetime(*GST0)
        elif self.time_sys == 'BDS':
            ref_date = datetime(*BDT0)
        else:
            raise ValueError(f"Unknown time system: {self.time_sys}")
            
        return ref_date + timedelta(weeks=self.week, seconds=self.tow)
    
    def to_mjd(self):
        """Convert to Modified Julian Day"""
        dt = self.to_datetime()
        mjd_epoch = datetime(1858, 11, 17)
        delta = dt - mjd_epoch
        return delta.total_seconds() / 86400.0
    
    def to_gps_seconds(self):
        """Convert to GPS seconds since GPS epoch"""
        return self.week * 604800 + self.tow
    
    def add_seconds(self, seconds: float) -> 'GNSSTime':
        """Add seconds to time"""
        new_tow = self.tow + seconds
        week_delta = int(new_tow // 604800)
        new_tow = new_tow % 604800
        return GNSSTime(self.week + week_delta, new_tow, self.time_sys)
    
    def __add__(self, seconds: float) -> 'GNSSTime':
        """Add seconds using + operator"""
        if isinstance(seconds, (int, float)):
            return self.add_seconds(seconds)
        else:
            raise TypeError(f"Cannot add {type(seconds)} to GNSSTime")
    
    def __sub__(self, other: Union['GNSSTime', float]) -> Union[float, 'GNSSTime']:
        """Subtract time or seconds"""
        if isinstance(other, GNSSTime):
            if self.time_sys != other.time_sys:
                raise ValueError(f"Cannot subtract times with different systems: {self.time_sys} and {other.time_sys}")
            return (self.week - other.week) * 604800 + (self.tow - other.tow)
        elif isinstance(other, (int, float)):
            return self.add_seconds(-other)
        else:
            raise TypeError(f"Cannot subtract {type(other)} from GNSSTime")
    
    def __lt__(self, other: 'GNSSTime') -> bool:
        """Less than comparison"""
        if not isinstance(other, GNSSTime):
            return NotImplemented
        if self.time_sys != other.time_sys:
            raise ValueError(f"Cannot compare times with different systems: {self.time_sys} and {other.time_sys}")
        return (self.week, self.tow) < (other.week, other.tow)
    
    def __le__(self, other: 'GNSSTime') -> bool:
        """Less than or equal comparison"""
        if not isinstance(other, GNSSTime):
            return NotImplemented
        if self.time_sys != other.time_sys:
            raise ValueError(f"Cannot compare times with different systems: {self.time_sys} and {other.time_sys}")
        return (self.week, self.tow) <= (other.week, other.tow)
    
    def __gt__(self, other: 'GNSSTime') -> bool:
        """Greater than comparison"""
        if not isinstance(other, GNSSTime):
            return NotImplemented
        if self.time_sys != other.time_sys:
            raise ValueError(f"Cannot compare times with different systems: {self.time_sys} and {other.time_sys}")
        return (self.week, self.tow) > (other.week, other.tow)
    
    def __ge__(self, other: 'GNSSTime') -> bool:
        """Greater than or equal comparison"""
        if not isinstance(other, GNSSTime):
            return NotImplemented
        if self.time_sys != other.time_sys:
            raise ValueError(f"Cannot compare times with different systems: {self.time_sys} and {other.time_sys}")
        return (self.week, self.tow) >= (other.week, other.tow)
    
    def __eq__(self, other: 'GNSSTime') -> bool:
        """Equality comparison"""
        if not isinstance(other, GNSSTime):
            return NotImplemented
        return self.time_sys == other.time_sys and self.week == other.week and abs(self.tow - other.tow) < 1e-9
    
    def __str__(self):
        return f"{self.time_sys} Week: {self.week}, TOW: {self.tow:.3f}"
    
    def __repr__(self):
        return f"GNSSTime({self.week}, {self.tow}, '{self.time_sys}')"
    
    def convert_to(self, target_sys: str) -> 'GNSSTime':
        """Convert to a different time system
        
        Parameters:
        -----------
        target_sys : str
            Target time system ('GPS', 'GAL', 'BDS', 'GLO', 'UTC')
            
        Returns:
        --------
        GNSSTime
            Time in target system
        """
        if self.time_sys == target_sys:
            return GNSSTime(self.week, self.tow, self.time_sys)
        
        # Convert to GPS time as intermediate
        if self.time_sys == 'GPS':
            gps_time = self
        elif self.time_sys == 'GAL':
            # Galileo time is aligned with GPS time (same epoch)
            gps_time = GNSSTime(self.week, self.tow, 'GPS')
        elif self.time_sys == 'BDS':
            # BeiDou time offset: BDT = GPST - 14s (as of 2006)
            gps_seconds = self.to_gps_seconds() + 14.0
            gps_time = GNSSTime.from_gps_seconds(gps_seconds, 'GPS')
        elif self.time_sys == 'GLO':
            # GLONASS time is UTC + 3 hours (no leap seconds)
            # This is a simplified conversion - proper conversion needs leap seconds
            utc_seconds = self.to_gps_seconds() - 3 * 3600
            gps_seconds = utc_seconds + 18  # Approximate leap seconds
            gps_time = GNSSTime.from_gps_seconds(gps_seconds, 'GPS')
        else:
            raise ValueError(f"Conversion from {self.time_sys} not implemented")
        
        # Convert from GPS to target system
        if target_sys == 'GPS':
            return gps_time
        elif target_sys == 'GAL':
            return GNSSTime(gps_time.week, gps_time.tow, 'GAL')
        elif target_sys == 'BDS':
            gps_seconds = gps_time.to_gps_seconds()
            bdt_seconds = gps_seconds - 14.0
            return GNSSTime.from_gps_seconds(bdt_seconds, 'BDS')
        elif target_sys == 'GLO':
            gps_seconds = gps_time.to_gps_seconds()
            utc_seconds = gps_seconds - 18  # Approximate leap seconds
            glo_seconds = utc_seconds + 3 * 3600
            return GNSSTime.from_gps_seconds(glo_seconds, 'GLO')
        else:
            raise ValueError(f"Conversion to {target_sys} not implemented")
    
    def copy(self) -> 'GNSSTime':
        """Create a copy of this time instance"""
        return GNSSTime(self.week, self.tow, self.time_sys)


def gpst2utc(gpst, leap_seconds=18):
    """Convert GPS time to UTC (accounting for leap seconds)"""
    utc_dt = gpst.to_datetime() - timedelta(seconds=leap_seconds)
    return utc_dt


def utc2gpst(utc_dt, leap_seconds=18):
    """Convert UTC to GPS time (accounting for leap seconds)"""
    gps_dt = utc_dt + timedelta(seconds=leap_seconds)
    return GNSSTime.from_datetime(gps_dt, 'GPS')


def time_diff(t1, t2):
    """Compute time difference t1 - t2 in seconds"""
    if t1.time_sys != t2.time_sys:
        raise ValueError("Time systems must match for difference")
    
    dt = (t1.week - t2.week) * 604800 + (t1.tow - t2.tow)
    return dt


def timediff(time, tref):
    """Time difference accounting for week rollover
    
    Handles cases where one time is GPS time and the other is TOW
    """
    # If one is GPS time and the other is TOW, convert to same scale
    if time > 604800 and tref < 604800:
        time = time % 604800
    elif tref > 604800 and time < 604800:
        tref = tref % 604800
        
    dt = time - tref
    
    # Handle week rollover
    if abs(dt) > 302400:  # half week
        if dt > 0:
            dt -= 604800
        else:
            dt += 604800
    
    return dt


def gps_seconds_to_week_tow(gps_seconds: float) -> tuple:
    """
    Convert GPS seconds to GPS week number and time of week
    
    Parameters:
    -----------
    gps_seconds : float
        GPS seconds since GPS epoch (Jan 6, 1980 00:00:00 UTC)
        
    Returns:
    --------
    tuple : (week, tow)
        GPS week number and time of week in seconds
    """
    if gps_seconds < 0:
        raise ValueError(f"GPS seconds cannot be negative: {gps_seconds}")
    
    week = int(gps_seconds // 604800)
    tow = gps_seconds % 604800
    
    return week, tow


def week_tow_to_gps_seconds(week: int, tow: float) -> float:
    """
    Convert GPS week number and time of week to GPS seconds
    
    Parameters:
    -----------
    week : int
        GPS week number
    tow : float
        Time of week in seconds (0-604800)
        
    Returns:
    --------
    float
        GPS seconds since GPS epoch
    """
    if week < 0:
        raise ValueError(f"GPS week cannot be negative: {week}")
    
    if tow < 0 or tow >= 604800:
        raise ValueError(f"Time of week must be in range [0, 604800): {tow}")
    
    return week * 604800 + tow


def gps_seconds_to_tow(gps_seconds: float) -> float:
    """
    Convert GPS seconds to time of week (convenience function)
    
    Parameters:
    -----------
    gps_seconds : float
        GPS seconds since GPS epoch
        
    Returns:
    --------
    float
        Time of week in seconds
    """
    if gps_seconds < 0:
        raise ValueError(f"GPS seconds cannot be negative: {gps_seconds}")
    
    return gps_seconds % 604800


def is_gps_seconds(time_value: float) -> bool:
    """
    Heuristic to determine if a time value is GPS seconds or TOW
    
    Parameters:
    -----------
    time_value : float
        Time value to check
        
    Returns:
    --------
    bool
        True if likely GPS seconds, False if likely TOW
    """
    return time_value > 604800  # More than one week


def ensure_tow(time_value: float) -> float:
    """
    Ensure time value is in TOW format
    
    Parameters:
    -----------
    time_value : float
        Time value (either GPS seconds or TOW)
        
    Returns:
    --------
    float
        Time of week in seconds
    """
    if is_gps_seconds(time_value):
        return gps_seconds_to_tow(time_value)
    else:
        return time_value


def ensure_gps_seconds(time_value: float, week: int = None) -> float:
    """
    Ensure time value is in GPS seconds format
    
    Parameters:
    -----------
    time_value : float
        Time value (either GPS seconds or TOW)
    week : int, optional
        GPS week number (required if time_value is TOW)
        
    Returns:
    --------
    float
        GPS seconds since GPS epoch
    """
    if is_gps_seconds(time_value):
        return time_value
    else:
        if week is None:
            raise ValueError("GPS week number is required when converting from TOW")
        return week_tow_to_gps_seconds(week, time_value)


def gnss_time_diff(t1: Union[GNSSTime, float], t2: Union[GNSSTime, float], 
                   time_sys: str = 'GPS') -> float:
    """
    Compute time difference with automatic type handling
    
    Parameters:
    -----------
    t1, t2 : GNSSTime or float
        Times to compare (if float, assumed to be in time_sys)
    time_sys : str
        Time system for float values
        
    Returns:
    --------
    float
        Time difference t1 - t2 in seconds
    """
    # Convert floats to GNSSTime
    if isinstance(t1, (int, float)):
        if t1 > 604800:  # GPS seconds
            t1 = GNSSTime.from_gps_seconds(t1, time_sys)
        else:  # TOW
            t1 = GNSSTime(0, t1, time_sys)
    
    if isinstance(t2, (int, float)):
        if t2 > 604800:  # GPS seconds
            t2 = GNSSTime.from_gps_seconds(t2, time_sys)
        else:  # TOW
            t2 = GNSSTime(0, t2, time_sys)
    
    return t1 - t2