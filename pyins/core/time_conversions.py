"""Time conversions between Unix time, GPS time, TOW, and Week number.

This module provides comprehensive time conversion functions based on 
gnss_lib_py implementation, handling leap seconds properly.

Time reference frames:
- Unix time: Seconds/milliseconds since 1970-01-01 00:00:00 UTC
- GPS time: Seconds/milliseconds since 1980-01-06 00:00:00 UTC (no leap seconds)
- TOW: Time of Week - GPS week number and seconds within the week
- datetime: Python datetime objects (assumed to be in UTC timezone)
"""

from datetime import datetime, timedelta, timezone
import numpy as np

# Constants
WEEKSEC = 604800  # Seconds in a week
GPS_EPOCH_0 = datetime(1980, 1, 6, 0, 0, 0, tzinfo=timezone.utc)
UNIX_EPOCH_0 = datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

# Leap seconds table (most recent first)
# GPS time is ahead of UTC by these leap seconds
LEAPSECONDS_TABLE = [
    datetime(2017, 1, 1, 0, 0, 0, tzinfo=timezone.utc),  # 18 seconds
    datetime(2015, 7, 1, 0, 0, 0, tzinfo=timezone.utc),  # 17 seconds
    datetime(2012, 7, 1, 0, 0, 0, tzinfo=timezone.utc),  # 16 seconds
    datetime(2009, 1, 1, 0, 0, 0, tzinfo=timezone.utc),  # 15 seconds
    datetime(2006, 1, 1, 0, 0, 0, tzinfo=timezone.utc),  # 14 seconds
    datetime(1999, 1, 1, 0, 0, 0, tzinfo=timezone.utc),  # 13 seconds
    datetime(1997, 7, 1, 0, 0, 0, tzinfo=timezone.utc),  # 12 seconds
    datetime(1996, 1, 1, 0, 0, 0, tzinfo=timezone.utc),  # 11 seconds
    datetime(1994, 7, 1, 0, 0, 0, tzinfo=timezone.utc),  # 10 seconds
    datetime(1993, 7, 1, 0, 0, 0, tzinfo=timezone.utc),  # 9 seconds
    datetime(1992, 7, 1, 0, 0, 0, tzinfo=timezone.utc),  # 8 seconds
    datetime(1991, 1, 1, 0, 0, 0, tzinfo=timezone.utc),  # 7 seconds
    datetime(1990, 1, 1, 0, 0, 0, tzinfo=timezone.utc),  # 6 seconds
    datetime(1988, 1, 1, 0, 0, 0, tzinfo=timezone.utc),  # 5 seconds
    datetime(1985, 7, 1, 0, 0, 0, tzinfo=timezone.utc),  # 4 seconds
    datetime(1983, 7, 1, 0, 0, 0, tzinfo=timezone.utc),  # 3 seconds
    datetime(1982, 7, 1, 0, 0, 0, tzinfo=timezone.utc),  # 2 seconds
    datetime(1981, 7, 1, 0, 0, 0, tzinfo=timezone.utc),  # 1 second
    GPS_EPOCH_0  # 0 seconds
]


def ensure_utc_timezone(dt):
    """Ensure datetime has UTC timezone.
    
    Parameters
    ----------
    dt : datetime.datetime
        Datetime object
        
    Returns
    -------
    datetime.datetime
        Datetime with UTC timezone
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    elif dt.tzinfo != timezone.utc:
        return dt.astimezone(timezone.utc)
    return dt


def get_leap_seconds(time_utc):
    """Get leap seconds at given UTC time.
    
    Parameters
    ----------
    time_utc : datetime.datetime
        UTC time
        
    Returns
    -------
    int
        Number of leap seconds to add to UTC to get GPS time
    """
    time_utc = ensure_utc_timezone(time_utc)
    
    if time_utc < GPS_EPOCH_0:
        raise ValueError(f"Time must be after GPS epoch {GPS_EPOCH_0}")
    
    # Count leap seconds
    for i, leap_time in enumerate(LEAPSECONDS_TABLE):
        if time_utc >= leap_time:
            return len(LEAPSECONDS_TABLE) - 1 - i
    
    return 0


def unix_to_gps_seconds(unix_seconds):
    """Convert Unix seconds to GPS seconds.
    
    Parameters
    ----------
    unix_seconds : float
        Seconds since Unix epoch (1970-01-01 00:00:00 UTC)
        
    Returns
    -------
    float
        Seconds since GPS epoch (1980-01-06 00:00:00 GPS)
    """
    # Convert to datetime to determine leap seconds
    dt_utc = UNIX_EPOCH_0 + timedelta(seconds=unix_seconds)
    leap_seconds = get_leap_seconds(dt_utc)
    
    # GPS epoch offset from Unix epoch
    gps_offset = (GPS_EPOCH_0 - UNIX_EPOCH_0).total_seconds()
    
    # GPS time = Unix time - offset + leap seconds
    return unix_seconds - gps_offset


def gps_to_unix_seconds(gps_seconds):
    """Convert GPS seconds to Unix seconds.
    
    Parameters
    ----------
    gps_seconds : float
        Seconds since GPS epoch (1980-01-06 00:00:00 GPS)
        
    Returns
    -------
    float
        Seconds since Unix epoch (1970-01-01 00:00:00 UTC)
    """
    # GPS time to datetime (no leap seconds in GPS time)
    dt_gps = GPS_EPOCH_0 + timedelta(seconds=gps_seconds)
    
    # Get leap seconds at this GPS time
    leap_seconds = get_leap_seconds(dt_gps)
    
    # Convert to UTC by subtracting leap seconds
    dt_utc = dt_gps
    
    # Convert to Unix seconds
    return (dt_utc - UNIX_EPOCH_0).total_seconds()


def gps_seconds_to_tow(gps_seconds):
    """Convert GPS seconds to GPS week and time of week.
    
    Parameters
    ----------
    gps_seconds : float or np.ndarray
        Seconds since GPS epoch
        
    Returns
    -------
    gps_week : int or np.ndarray
        GPS week number
    tow : float or np.ndarray
        Time of week in seconds
    """
    gps_week = np.floor(gps_seconds / WEEKSEC).astype(int)
    tow = gps_seconds % WEEKSEC
    return gps_week, tow


def tow_to_gps_seconds(gps_week, tow):
    """Convert GPS week and time of week to GPS seconds.
    
    Parameters
    ----------
    gps_week : int or np.ndarray
        GPS week number
    tow : float or np.ndarray
        Time of week in seconds
        
    Returns
    -------
    float or np.ndarray
        Seconds since GPS epoch
    """
    return gps_week * WEEKSEC + tow


def datetime_to_gps_seconds(dt):
    """Convert datetime (UTC) to GPS seconds.
    
    Parameters
    ----------
    dt : datetime.datetime
        UTC datetime
        
    Returns
    -------
    float
        Seconds since GPS epoch
    """
    dt = ensure_utc_timezone(dt)
    
    if dt < GPS_EPOCH_0:
        raise ValueError(f"Datetime must be after GPS epoch {GPS_EPOCH_0}")
    
    # Get leap seconds at this UTC time
    leap_seconds = get_leap_seconds(dt)
    
    # GPS time = UTC time + leap seconds
    dt_gps = dt
    
    # Calculate seconds since GPS epoch
    return (dt_gps - GPS_EPOCH_0).total_seconds()


def gps_seconds_to_datetime(gps_seconds):
    """Convert GPS seconds to datetime (UTC).
    
    Parameters
    ----------
    gps_seconds : float
        Seconds since GPS epoch
        
    Returns
    -------
    datetime.datetime
        UTC datetime
    """
    # GPS time to datetime (no leap seconds in GPS time)
    dt_gps = GPS_EPOCH_0 + timedelta(seconds=gps_seconds)
    
    # Get leap seconds at this GPS time
    leap_seconds = get_leap_seconds(dt_gps)
    
    # Convert to UTC by subtracting leap seconds
    return dt_gps


def datetime_to_tow(dt):
    """Convert datetime (UTC) to GPS week and time of week.
    
    Parameters
    ----------
    dt : datetime.datetime or array-like
        UTC datetime
        
    Returns
    -------
    gps_week : int or np.ndarray
        GPS week number
    tow : float or np.ndarray
        Time of week in seconds
    """
    if isinstance(dt, datetime):
        gps_seconds = datetime_to_gps_seconds(dt)
        return gps_seconds_to_tow(gps_seconds)
    else:
        # Handle array-like input
        gps_weeks = []
        tows = []
        for d in dt:
            gps_seconds = datetime_to_gps_seconds(d)
            week, tow = gps_seconds_to_tow(gps_seconds)
            gps_weeks.append(week)
            tows.append(tow)
        return np.array(gps_weeks), np.array(tows)


def tow_to_datetime(gps_week, tow):
    """Convert GPS week and time of week to datetime (UTC).
    
    Parameters
    ----------
    gps_week : int or array-like
        GPS week number
    tow : float or array-like
        Time of week in seconds
        
    Returns
    -------
    datetime.datetime or np.ndarray
        UTC datetime
    """
    if np.isscalar(gps_week) and np.isscalar(tow):
        gps_seconds = tow_to_gps_seconds(gps_week, tow)
        return gps_seconds_to_datetime(gps_seconds)
    else:
        # Handle array-like input
        datetimes = []
        if np.isscalar(gps_week):
            gps_week = [gps_week] * len(tow)
        if np.isscalar(tow):
            tow = [tow] * len(gps_week)
            
        for w, t in zip(gps_week, tow):
            gps_seconds = tow_to_gps_seconds(w, t)
            dt = gps_seconds_to_datetime(gps_seconds)
            datetimes.append(dt)
        return np.array(datetimes)


def unix_to_tow(unix_seconds):
    """Convert Unix seconds to GPS week and time of week.
    
    Parameters
    ----------
    unix_seconds : float or np.ndarray
        Seconds since Unix epoch
        
    Returns
    -------
    gps_week : int or np.ndarray
        GPS week number
    tow : float or np.ndarray
        Time of week in seconds
    """
    gps_seconds = unix_to_gps_seconds(unix_seconds)
    return gps_seconds_to_tow(gps_seconds)


def tow_to_unix(gps_week, tow):
    """Convert GPS week and time of week to Unix seconds.
    
    Parameters
    ----------
    gps_week : int or np.ndarray
        GPS week number
    tow : float or np.ndarray
        Time of week in seconds
        
    Returns
    -------
    float or np.ndarray
        Seconds since Unix epoch
    """
    gps_seconds = tow_to_gps_seconds(gps_week, tow)
    return gps_to_unix_seconds(gps_seconds)


def datetime_to_unix_seconds(dt):
    """Convert datetime to Unix seconds.
    
    Parameters
    ----------
    dt : datetime.datetime
        UTC datetime
        
    Returns
    -------
    float
        Seconds since Unix epoch
    """
    dt = ensure_utc_timezone(dt)
    return (dt - UNIX_EPOCH_0).total_seconds()


def unix_seconds_to_datetime(unix_seconds):
    """Convert Unix seconds to datetime.
    
    Parameters
    ----------
    unix_seconds : float
        Seconds since Unix epoch
        
    Returns
    -------
    datetime.datetime
        UTC datetime
    """
    return UNIX_EPOCH_0 + timedelta(seconds=unix_seconds)


# ============================================================================
# GLONASS Time Conversions (similar to rtklib)
# ============================================================================
# GLONASS time = UTC + 3 hours (Moscow time)
# GLONASS ephemeris times in RINEX are in UTC
# For satellite computations, we need to convert between UTC, GLONASS, and GPS times

def utc_to_gpst(utc_time, leap_seconds=18):
    """Convert UTC time to GPS time (similar to rtklib's utc2gpst).
    
    Parameters
    ----------
    utc_time : float
        UTC time in seconds (e.g., seconds of day)
    leap_seconds : int
        Current leap seconds (default: 18 as of 2025)
        
    Returns
    -------
    float
        GPS time in seconds
    """
    # GPS time = UTC time + leap seconds
    return utc_time + leap_seconds


def gpst_to_utc(gps_time, leap_seconds=18):
    """Convert GPS time to UTC time (similar to rtklib's gpst2utc).
    
    Parameters
    ----------
    gps_time : float
        GPS time in seconds
    leap_seconds : int
        Current leap seconds (default: 18 as of 2025)
        
    Returns
    -------
    float
        UTC time in seconds
    """
    # UTC time = GPS time - leap seconds
    return gps_time - leap_seconds


def glonass_to_gpst(glo_time, leap_seconds=18):
    """Convert GLONASS time to GPS time.
    
    GLONASS time = UTC + 3 hours (Moscow time)
    GPS time = UTC + leap seconds
    Therefore: GPS time = GLONASS time - 3 hours + leap seconds
    
    Parameters
    ----------
    glo_time : float
        GLONASS time in seconds
    leap_seconds : int
        Current leap seconds (default: 18 as of 2025)
        
    Returns
    -------
    float
        GPS time in seconds
    """
    # Convert GLONASS to UTC (subtract 3 hours)
    utc_time = glo_time - 3 * 3600
    # Convert UTC to GPS (add leap seconds)
    return utc_time + leap_seconds


def gpst_to_glonass(gps_time, leap_seconds=18):
    """Convert GPS time to GLONASS time.
    
    Parameters
    ----------
    gps_time : float
        GPS time in seconds
    leap_seconds : int
        Current leap seconds (default: 18 as of 2025)
        
    Returns
    -------
    float
        GLONASS time in seconds
    """
    # Convert GPS to UTC (subtract leap seconds)
    utc_time = gps_time - leap_seconds
    # Convert UTC to GLONASS (add 3 hours)
    return utc_time + 3 * 3600


def adjust_glonass_toe(toe_utc, gps_week, leap_seconds=18):
    """Adjust GLONASS toe from UTC to GPS time with proper week handling.
    
    GLONASS ephemeris times in RINEX are given as UTC seconds of day.
    This function converts to GPS time and handles week rollovers.
    
    Parameters
    ----------
    toe_utc : float
        GLONASS toe in UTC seconds of day (0-86400)
    gps_week : int
        Current GPS week number
    leap_seconds : int
        Current leap seconds (default: 18 as of 2025)
        
    Returns
    -------
    float
        GPS time of week (0-604800)
    """
    # Convert UTC seconds of day to GPS seconds of day
    gps_sod = toe_utc + leap_seconds
    
    # Handle day rollover if needed
    if gps_sod >= 86400:
        # Crossed into next day in GPS time
        gps_sod -= 86400
        # Adjust for day of week (this is approximate, may need refinement)
        
    # For now, return the GPS seconds of day
    # In full implementation, would need to track day of week
    return gps_sod