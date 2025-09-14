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

"""Core Time Management System for GNSS

This module provides TimeCore - a central time class that can handle various time formats:
- GPS time (week + TOW)
- Unix timestamp
- UTC datetime
- Modified Julian Day (MJD)
- Other GNSS time systems (BDS, GAL, GLO)
"""

import warnings
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional, Union


class TimeFormat(Enum):
    """Enumeration of supported time formats"""
    GPS_WEEK_TOW = "gps_week_tow"
    GPS_SECONDS = "gps_seconds"
    UNIX = "unix"
    UTC = "utc"
    MJD = "mjd"
    BDS_WEEK_TOW = "bds_week_tow"
    BDS_SECONDS = "bds_seconds"
    GAL_WEEK_TOW = "gal_week_tow"
    GAL_SECONDS = "gal_seconds"
    GLO_WEEK_TOW = "glo_week_tow"
    UNKNOWN = "unknown"


class TimeSystem(Enum):
    """Time system enumeration"""
    GPS = "GPS"
    BDS = "BDS"
    GAL = "GAL"
    GLO = "GLO"
    UTC = "UTC"


# Time system constants
GPS_EPOCH = datetime(1980, 1, 6, 0, 0, 0, tzinfo=timezone.utc)
BDS_EPOCH = datetime(2006, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
GAL_EPOCH = datetime(1999, 8, 22, 0, 0, 0, tzinfo=timezone.utc)
MJD_EPOCH = datetime(1858, 11, 17, 0, 0, 0, tzinfo=timezone.utc)

# Leap seconds history (TAI - UTC)
# Source: https://www.ietf.org/timezones/data/leap-seconds.list
LEAP_SECONDS_HISTORY = [
    (datetime(1972, 1, 1, tzinfo=timezone.utc), 10),
    (datetime(1972, 7, 1, tzinfo=timezone.utc), 11),
    (datetime(1973, 1, 1, tzinfo=timezone.utc), 12),
    (datetime(1974, 1, 1, tzinfo=timezone.utc), 13),
    (datetime(1975, 1, 1, tzinfo=timezone.utc), 14),
    (datetime(1976, 1, 1, tzinfo=timezone.utc), 15),
    (datetime(1977, 1, 1, tzinfo=timezone.utc), 16),
    (datetime(1978, 1, 1, tzinfo=timezone.utc), 17),
    (datetime(1979, 1, 1, tzinfo=timezone.utc), 18),
    (datetime(1980, 1, 1, tzinfo=timezone.utc), 19),
    (datetime(1981, 7, 1, tzinfo=timezone.utc), 20),
    (datetime(1982, 7, 1, tzinfo=timezone.utc), 21),
    (datetime(1983, 7, 1, tzinfo=timezone.utc), 22),
    (datetime(1985, 7, 1, tzinfo=timezone.utc), 23),
    (datetime(1988, 1, 1, tzinfo=timezone.utc), 24),
    (datetime(1990, 1, 1, tzinfo=timezone.utc), 25),
    (datetime(1991, 1, 1, tzinfo=timezone.utc), 26),
    (datetime(1992, 7, 1, tzinfo=timezone.utc), 27),
    (datetime(1993, 7, 1, tzinfo=timezone.utc), 28),
    (datetime(1994, 7, 1, tzinfo=timezone.utc), 29),
    (datetime(1996, 1, 1, tzinfo=timezone.utc), 30),
    (datetime(1997, 7, 1, tzinfo=timezone.utc), 31),
    (datetime(1999, 1, 1, tzinfo=timezone.utc), 32),
    (datetime(2006, 1, 1, tzinfo=timezone.utc), 33),  # BDT epoch
    (datetime(2009, 1, 1, tzinfo=timezone.utc), 34),
    (datetime(2012, 7, 1, tzinfo=timezone.utc), 35),
    (datetime(2015, 7, 1, tzinfo=timezone.utc), 36),
    (datetime(2017, 1, 1, tzinfo=timezone.utc), 37),
]

# Current leap seconds (as of 2024)
LEAP_SECONDS = 18  # GPS-UTC offset (37 - 19 for GPS epoch)

# BeiDou Time offset calculation
# BDT started on 2006-01-01, when leap seconds = 33 (TAI-UTC)
# GPS started on 1980-01-06, when leap seconds = 19 (TAI-UTC)
# Therefore: GPS_BDS_OFFSET = leap_seconds(2006) - leap_seconds(1980) = 33 - 19 = 14
LEAP_SECONDS_AT_GPS_EPOCH = 19  # Leap seconds at GPS epoch (1980-01-06)
LEAP_SECONDS_AT_BDS_EPOCH = 33  # Leap seconds at BDS epoch (2006-01-01)
GPS_BDS_OFFSET = LEAP_SECONDS_AT_BDS_EPOCH - LEAP_SECONDS_AT_GPS_EPOCH  # = 14 seconds

# GLONASS Time System
# GLONASS Time = UTC + 3 hours (Moscow time)
# GLONASS includes leap seconds (follows UTC), unlike GPS/BDS
# GPS-GLONASS offset = current leap seconds (18 as of 2025)
GLO_UTC_OFFSET = 3 * 3600  # GLONASS = UTC + 3 hours
GPS_GLO_OFFSET = LEAP_SECONDS  # GPS-GLONASS offset = current leap seconds

# Constants
SECONDS_PER_WEEK = 604800
SECONDS_PER_DAY = 86400


def get_leap_seconds_at_date(dt: datetime) -> int:
    """Get leap seconds (TAI-UTC) at a given date

    Parameters
    ----------
    dt : datetime
        Date to get leap seconds for

    Returns
    -------
    int
        Number of leap seconds at that date
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    leap_seconds = 0
    for leap_date, seconds in LEAP_SECONDS_HISTORY:
        if dt >= leap_date:
            leap_seconds = seconds
        else:
            break

    return leap_seconds


class TimeCore:
    """Unified time representation for all GNSS and time systems.

    This class provides a central time management system that internally stores
    time as GPS seconds since the GPS epoch (1980-01-06) and offers conversions
    to and from all major time formats and GNSS time systems.

    The class handles:
    - GPS time (week + TOW)
    - BeiDou time (week + TOW)
    - Galileo time (week + TOW)
    - GLONASS time
    - Unix timestamp
    - UTC datetime
    - Modified Julian Day (MJD)

    All conversions properly account for leap seconds and system-specific
    epoch differences.

    Attributes
    ----------
    _gps_seconds : float
        Internal time storage as GPS seconds since GPS epoch

    Notes
    -----
    Time system relationships:
    - GPS time does not include leap seconds
    - BeiDou time is 14 seconds behind GPS time
    - GLONASS time includes leap seconds and is Moscow time (UTC+3)
    - Unix time includes leap seconds
    - UTC includes leap seconds

    Examples
    --------
    >>> # Create from GPS time
    >>> t = TimeCore.from_gps(2200, 432000.0)  # Week 2200, Sunday noon

    >>> # Convert to different formats
    >>> unix_time = t.to_unix()
    >>> utc_datetime = t.to_datetime()
    >>> mjd = t.to_mjd()

    >>> # Create from Unix timestamp
    >>> t2 = TimeCore.from_unix(1640995200.0)  # 2022-01-01 00:00:00 UTC
    """

    def __init__(self, gps_seconds: float):
        """Initialize with GPS seconds

        Parameters
        ----------
        gps_seconds : float
            Time in GPS seconds since GPS epoch (1980-01-06 00:00:00 UTC)
        """
        self._gps_seconds = float(gps_seconds)

    @classmethod
    def from_gps(cls, week: int, tow: float) -> 'TimeCore':
        """Create from GPS week and time of week

        Parameters
        ----------
        week : int
            GPS week number
        tow : float
            Time of week in seconds [0, 604800)

        Returns
        -------
        TimeCore
            Time instance
        """
        return cls(week * SECONDS_PER_WEEK + tow)

    @classmethod
    def from_unix(cls, unix_time: float) -> 'TimeCore':
        """Create from Unix timestamp

        Parameters
        ----------
        unix_time : float
            Unix timestamp (seconds since 1970-01-01 00:00:00 UTC)

        Returns
        -------
        TimeCore
            Time instance
        """
        # Convert Unix to GPS
        unix_dt = datetime.fromtimestamp(unix_time, tz=timezone.utc)
        gps_offset = (unix_dt - GPS_EPOCH).total_seconds()
        # Add leap seconds (Unix doesn't include them, GPS does)
        gps_seconds = gps_offset + LEAP_SECONDS
        return cls(gps_seconds)

    @classmethod
    def from_datetime(cls, dt: datetime) -> 'TimeCore':
        """Create from datetime object

        Parameters
        ----------
        dt : datetime
            Datetime object (should be UTC)

        Returns
        -------
        TimeCore
            Time instance
        """
        if dt.tzinfo is None:
            warnings.warn("Datetime has no timezone, assuming UTC", stacklevel=2)
            dt = dt.replace(tzinfo=timezone.utc)

        # Convert to Unix first, then to GPS
        unix_time = dt.timestamp()
        return cls.from_unix(unix_time)

    @classmethod
    def from_mjd(cls, mjd: float) -> 'TimeCore':
        """Create from Modified Julian Day

        Parameters
        ----------
        mjd : float
            Modified Julian Day

        Returns
        -------
        TimeCore
            Time instance
        """
        # Convert MJD to datetime
        dt = MJD_EPOCH + timedelta(days=mjd)
        return cls.from_datetime(dt)

    @classmethod
    def from_bds(cls, week: int, tow: float) -> 'TimeCore':
        """Create from BeiDou week and time of week

        BeiDou Time (BDT) started on 2006-01-01 00:00:00 UTC.
        At that time, there were 33 leap seconds (TAI-UTC).
        GPS Time started when there were 19 leap seconds.
        The offset between GPS and BDT is thus 33 - 19 = 14 seconds.

        Parameters
        ----------
        week : int
            BeiDou week number
        tow : float
            Time of week in seconds

        Returns
        -------
        TimeCore
            Time instance
        """
        # Convert BDS to GPS using the leap second difference
        bds_seconds = week * SECONDS_PER_WEEK + tow
        # BDT = GPS - GPS_BDS_OFFSET, so GPS = BDT + GPS_BDS_OFFSET
        bds_seconds + GPS_BDS_OFFSET

        # Calculate total GPS seconds
        bds_epoch_in_gps = (BDS_EPOCH - GPS_EPOCH).total_seconds()
        gps_seconds = bds_epoch_in_gps + bds_seconds + GPS_BDS_OFFSET

        return cls(gps_seconds)

    @classmethod
    def from_gal(cls, week: int, tow: float) -> 'TimeCore':
        """Create from Galileo week and time of week

        Parameters
        ----------
        week : int
            Galileo week number
        tow : float
            Time of week in seconds

        Returns
        -------
        TimeCore
            Time instance
        """
        # Galileo is aligned with GPS
        return cls.from_gps(week, tow)

    @classmethod
    def from_auto(cls, time_value: Union[float, tuple[int, float], datetime],
                  hint: Optional[TimeFormat] = None) -> 'TimeCore':
        """Create from automatic detection of time format

        Parameters
        ----------
        time_value : float, tuple, or datetime
            Time value in unknown format
        hint : TimeFormat, optional
            Hint about the time format

        Returns
        -------
        TimeCore
            Time instance
        """
        detected_format = hint if hint else cls.detect_time_format(time_value)

        if detected_format == TimeFormat.GPS_WEEK_TOW:
            return cls.from_gps(time_value[0], time_value[1])
        elif detected_format == TimeFormat.GPS_SECONDS:
            return cls(time_value)
        elif detected_format == TimeFormat.UNIX:
            return cls.from_unix(time_value)
        elif detected_format == TimeFormat.UTC:
            return cls.from_datetime(time_value)
        elif detected_format == TimeFormat.MJD:
            return cls.from_mjd(time_value)
        elif detected_format == TimeFormat.BDS_WEEK_TOW:
            return cls.from_bds(time_value[0], time_value[1])
        else:
            raise ValueError(f"Cannot create time from format: {detected_format}")

    @staticmethod
    def detect_time_format(time_value: Union[float, tuple[int, float], datetime]) -> TimeFormat:
        """Detect the format of a time value

        Parameters
        ----------
        time_value : float, tuple, or datetime
            Time value to analyze

        Returns
        -------
        TimeFormat
            Detected time format
        """
        if isinstance(time_value, datetime):
            return TimeFormat.UTC

        if isinstance(time_value, tuple) and len(time_value) == 2:
            week, tow = time_value
            if 0 <= tow < SECONDS_PER_WEEK:
                # Could be GPS, BDS, or GAL week+tow
                # Default to GPS without additional context
                return TimeFormat.GPS_WEEK_TOW

        if isinstance(time_value, (int, float)):
            time_value = float(time_value)

            # MJD range: ~40000 to ~70000 (1858 to 2050)
            if 40000 <= time_value <= 70000:
                return TimeFormat.MJD

            # TOW: 0 to 604800
            if 0 <= time_value < SECONDS_PER_WEEK:
                # Ambiguous - could be TOW or very early Unix time
                # Without context, assume TOW
                warnings.warn("Ambiguous time value - assuming TOW", stacklevel=2)
                return TimeFormat.UNKNOWN

            # Unix timestamp: ~0 (1970) to ~2e9 (2033)
            if 0 <= time_value <= 2e9:
                return TimeFormat.UNIX

            # GPS seconds: > 604800 (after first week)
            if time_value > SECONDS_PER_WEEK:
                # Could be GPS seconds or future Unix timestamp
                # Use year 2030 as threshold
                if time_value < 1.9e9:  # Before ~2030 in Unix time
                    # Likely GPS seconds (started 1980)
                    return TimeFormat.GPS_SECONDS
                else:
                    return TimeFormat.UNIX

        return TimeFormat.UNKNOWN

    def get_gps_week_tow(self) -> tuple[int, float]:
        """Get GPS week and time of week

        Returns
        -------
        tuple
            (week, tow) where week is GPS week number and tow is time of week in seconds
        """
        week = int(self._gps_seconds // SECONDS_PER_WEEK)
        tow = self._gps_seconds % SECONDS_PER_WEEK
        return week, tow

    def get_gps_seconds(self) -> float:
        """Get GPS seconds since GPS epoch

        Returns
        -------
        float
            GPS seconds
        """
        return self._gps_seconds

    def get_unix(self) -> float:
        """Get Unix timestamp

        Returns
        -------
        float
            Unix timestamp (seconds since 1970-01-01 00:00:00 UTC)
        """
        # GPS to UTC datetime
        gps_dt = GPS_EPOCH + timedelta(seconds=self._gps_seconds - LEAP_SECONDS)
        return gps_dt.timestamp()

    def get_datetime(self, tz=timezone.utc) -> datetime:
        """Get datetime object

        Parameters
        ----------
        tz : timezone, optional
            Timezone for the datetime (default: UTC)

        Returns
        -------
        datetime
            Datetime object
        """
        unix_time = self.get_unix()
        return datetime.fromtimestamp(unix_time, tz=tz)

    def get_mjd(self) -> float:
        """Get Modified Julian Day

        Returns
        -------
        float
            Modified Julian Day
        """
        dt = self.get_datetime()
        delta = dt - MJD_EPOCH
        return delta.total_seconds() / SECONDS_PER_DAY

    def get_bds_week_tow(self) -> tuple[int, float]:
        """Get BeiDou week and time of week

        BDT = GPS - GPS_BDS_OFFSET, where the offset comes from
        the leap second difference between the two epoch times.

        Returns
        -------
        tuple
            (week, tow) for BeiDou time system
        """
        # Convert GPS seconds to BDS seconds
        # GPS to BDS: subtract the leap second offset
        bds_seconds_from_gps = self._gps_seconds - GPS_BDS_OFFSET

        # Calculate offset from BDS epoch
        bds_epoch_in_gps = (BDS_EPOCH - GPS_EPOCH).total_seconds()
        bds_seconds = bds_seconds_from_gps - bds_epoch_in_gps

        # Handle negative values (before BDS epoch)
        if bds_seconds < 0:
            week = 0
            tow = 0
        else:
            week = int(bds_seconds // SECONDS_PER_WEEK)
            tow = bds_seconds % SECONDS_PER_WEEK

        return week, tow

    def get_bds_seconds(self) -> float:
        """Get BeiDou seconds since BDS epoch

        Returns
        -------
        float
            BeiDou seconds
        """
        week, tow = self.get_bds_week_tow()
        return week * SECONDS_PER_WEEK + tow

    def get_gal_week_tow(self) -> tuple[int, float]:
        """Get Galileo week and time of week

        Returns
        -------
        tuple
            (week, tow) for Galileo time system
        """
        # Galileo is aligned with GPS
        return self.get_gps_week_tow()

    def get_glo_time(self) -> tuple[int, float]:
        """Get GLONASS time (UTC + 3 hours)

        GLONASS time follows UTC with 3-hour offset (Moscow time).
        Unlike GPS/BDS, GLONASS includes leap seconds.

        Returns
        -------
        tuple
            (day_number, seconds_of_day) for GLONASS
        """
        # GPS to UTC (subtract leap seconds)
        utc_seconds = self._gps_seconds - LEAP_SECONDS
        utc_dt = GPS_EPOCH + timedelta(seconds=utc_seconds)

        # Add 3 hours for GLONASS (Moscow time)
        glo_dt = utc_dt + timedelta(hours=3)

        # GLONASS uses day number within 4-year period
        glo_epoch = datetime(1996, 1, 1, tzinfo=timezone.utc)
        days_since_epoch = (glo_dt - glo_epoch).days
        day_in_4year = days_since_epoch % (4 * 365 + 1)

        # Seconds of day in GLONASS time
        seconds_of_day = glo_dt.hour * 3600 + glo_dt.minute * 60 + glo_dt.second + glo_dt.microsecond / 1e6

        return day_in_4year, seconds_of_day

    def get_glo_seconds(self) -> float:
        """Get GLONASS seconds (GPS - leap seconds)

        GLONASS follows UTC, so we subtract leap seconds from GPS time.

        Returns
        -------
        float
            GLONASS time in seconds (for ephemeris calculations)
        """
        # For ephemeris calculations, use GPS - leap_seconds
        # The +3 hours is only for display/conversion purposes
        return self._gps_seconds - GPS_GLO_OFFSET

    def get_tow(self, system: TimeSystem = TimeSystem.GPS) -> float:
        """Get time of week for specified system

        Parameters
        ----------
        system : TimeSystem
            Time system (GPS, BDS, GAL, etc.)

        Returns
        -------
        float
            Time of week in seconds
        """
        if system == TimeSystem.GPS:
            _, tow = self.get_gps_week_tow()
        elif system == TimeSystem.BDS:
            _, tow = self.get_bds_week_tow()
        elif system == TimeSystem.GAL:
            _, tow = self.get_gal_week_tow()
        elif system == TimeSystem.GLO:
            # For GLONASS ephemeris, return GPS time minus leap seconds
            _, tow = self.get_gps_week_tow()
            return tow - GPS_GLO_OFFSET
        else:
            raise ValueError(f"TOW not available for system: {system}")

        return tow

    def add_seconds(self, seconds: float) -> 'TimeCore':
        """Add seconds to time

        Parameters
        ----------
        seconds : float
            Seconds to add

        Returns
        -------
        TimeCore
            New time instance
        """
        return TimeCore(self._gps_seconds + seconds)

    def __add__(self, seconds: float) -> 'TimeCore':
        """Add seconds using + operator"""
        return self.add_seconds(seconds)

    def __sub__(self, other: Union['TimeCore', float]) -> Union[float, 'TimeCore']:
        """Subtract time or seconds"""
        if isinstance(other, TimeCore):
            return self._gps_seconds - other._gps_seconds
        else:
            return self.add_seconds(-other)

    def __lt__(self, other: 'TimeCore') -> bool:
        """Less than comparison"""
        return self._gps_seconds < other._gps_seconds

    def __le__(self, other: 'TimeCore') -> bool:
        """Less than or equal comparison"""
        return self._gps_seconds <= other._gps_seconds

    def __gt__(self, other: 'TimeCore') -> bool:
        """Greater than comparison"""
        return self._gps_seconds > other._gps_seconds

    def __ge__(self, other: 'TimeCore') -> bool:
        """Greater than or equal comparison"""
        return self._gps_seconds >= other._gps_seconds

    def __eq__(self, other: 'TimeCore') -> bool:
        """Equality comparison"""
        if not isinstance(other, TimeCore):
            return False
        return abs(self._gps_seconds - other._gps_seconds) < 1e-9

    def __str__(self) -> str:
        """String representation"""
        week, tow = self.get_gps_week_tow()
        dt = self.get_datetime()
        return f"GPS({week}, {tow:.3f}) = {dt.strftime('%Y-%m-%d %H:%M:%S')} UTC"

    def __repr__(self) -> str:
        """Detailed representation"""
        return f"TimeCore(gps_seconds={self._gps_seconds})"

    def to_dict(self) -> dict[str, Union[float, tuple[int, float], str]]:
        """Convert to dictionary with all time formats

        Returns
        -------
        dict
            Dictionary containing all time representations
        """
        gps_week, gps_tow = self.get_gps_week_tow()
        bds_week, bds_tow = self.get_bds_week_tow()

        return {
            'gps_seconds': self._gps_seconds,
            'gps_week_tow': (gps_week, gps_tow),
            'unix': self.get_unix(),
            'utc': self.get_datetime().isoformat(),
            'mjd': self.get_mjd(),
            'bds_week_tow': (bds_week, bds_tow),
            'bds_seconds': self.get_bds_seconds(),
        }


# Convenience functions
def detect_time_format(time_value: Union[float, tuple[int, float], datetime]) -> TimeFormat:
    """Detect the format of a time value

    Parameters
    ----------
    time_value : float, tuple, or datetime
        Time value to analyze

    Returns
    -------
    TimeFormat
        Detected time format
    """
    return TimeCore.detect_time_format(time_value)


def current_time() -> TimeCore:
    """Get current time

    Returns
    -------
    TimeCore
        Current time
    """
    return TimeCore.from_datetime(datetime.now(timezone.utc))


def is_gps_time(time_value: float) -> bool:
    """Check if a float value is likely GPS time (not TOW)

    Parameters
    ----------
    time_value : float
        Time value to check

    Returns
    -------
    bool
        True if likely GPS seconds, False if likely TOW or other
    """
    fmt = detect_time_format(time_value)
    return fmt == TimeFormat.GPS_SECONDS


def is_unix_time(time_value: float) -> bool:
    """Check if a float value is likely Unix timestamp

    Parameters
    ----------
    time_value : float
        Time value to check

    Returns
    -------
    bool
        True if likely Unix timestamp
    """
    fmt = detect_time_format(time_value)
    return fmt == TimeFormat.UNIX


def convert_time(time_value: Union[float, tuple[int, float], datetime],
                 from_format: Optional[TimeFormat] = None,
                 to_format: TimeFormat = TimeFormat.GPS_SECONDS) -> Union[float, tuple[int, float]]:
    """Convert between time formats

    Parameters
    ----------
    time_value : float, tuple, or datetime
        Input time value
    from_format : TimeFormat, optional
        Source format (auto-detected if not provided)
    to_format : TimeFormat
        Target format

    Returns
    -------
    float or tuple
        Converted time value
    """
    # Create TimeCore
    unified = TimeCore.from_auto(time_value, hint=from_format)

    # Convert to target format
    if to_format == TimeFormat.GPS_SECONDS:
        return unified.get_gps_seconds()
    elif to_format == TimeFormat.GPS_WEEK_TOW:
        return unified.get_gps_week_tow()
    elif to_format == TimeFormat.UNIX:
        return unified.get_unix()
    elif to_format == TimeFormat.MJD:
        return unified.get_mjd()
    elif to_format == TimeFormat.BDS_SECONDS:
        return unified.get_bds_seconds()
    elif to_format == TimeFormat.BDS_WEEK_TOW:
        return unified.get_bds_week_tow()
    else:
        raise ValueError(f"Unsupported target format: {to_format}")
