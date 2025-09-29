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

"""Ephemeris selection and validation"""

from typing import Optional

import math

import numpy as np

import cssrlib.ephemeris
from cssrlib.gnss import gpst2time

from ..core.constants import *
from ..core.data_structures import Ephemeris, GloEphemeris, NavigationData

_WEEK_SECONDS = 604800.0


def _to_gtime(seconds: float, *, week_hint: int | None = None, reference_seconds: float | None = None):
    if week_hint is not None and 0.0 <= seconds < _WEEK_SECONDS:
        week = week_hint
        tow = seconds
    elif reference_seconds is not None and 0.0 <= seconds < _WEEK_SECONDS:
        ref_week = math.floor(reference_seconds / _WEEK_SECONDS)
        ref_tow = reference_seconds - ref_week * _WEEK_SECONDS
        week = int(ref_week)
        tow = seconds
        delta = tow - ref_tow
        half_week = _WEEK_SECONDS / 2.0
        if delta > half_week:
            week -= 1
        elif delta < -half_week:
            week += 1
    else:
        week = int(math.floor(seconds / _WEEK_SECONDS))
        tow = seconds - week * _WEEK_SECONDS
    return gpst2time(week, tow)


def geph2clk(geph: GloEphemeris, time: float) -> float:
    """Compute GLONASS satellite clock bias using cssrlib."""

    t_cssr = _to_gtime(time, reference_seconds=geph.toe)
    return float(cssrlib.ephemeris.geph2clk(t_cssr, geph))


def geph2pos(geph: GloEphemeris, time: float) -> tuple[np.ndarray, float, float]:
    """Compute GLONASS satellite position using cssrlib."""

    t_cssr = _to_gtime(time, reference_seconds=geph.toe)
    result = cssrlib.ephemeris.geph2pos(t_cssr, geph)

    if len(result) == 2:
        rs, dts = result
    else:
        rs, _, dts = result

    var = ERREPH_GLO**2
    return np.asarray(rs, dtype=float), var, float(dts)



def select_glonass_ephemeris(nav: NavigationData, sat: int, time: float) -> Optional[GloEphemeris]:
    """
    Select best GLONASS ephemeris for satellite at given time

    Parameters:
    -----------
    nav : NavigationData
        Navigation data containing GLONASS ephemerides
    sat : int
        Satellite number
    time : float
        Time of interest (GPST)

    Returns:
    --------
    geph : GloEphemeris or None
        Best GLONASS ephemeris or None if not found
    """
    if not hasattr(nav, 'geph') or not nav.geph:
        return None

    best_geph = None
    min_dt = float('inf')
    MAXDTOE_GLO = 1800.0  # 30 minutes for GLONASS

    for geph in nav.geph:
        if geph.sat != sat:
            continue

        # Check time validity
        dt = abs(time - geph.toe)
        if dt > MAXDTOE_GLO:
            continue

        if dt < min_dt:
            min_dt = dt
            best_geph = geph

    return best_geph


def select_ephemeris(nav: NavigationData, sat: int, time: float) -> Optional[Ephemeris]:
    """
    Select best ephemeris for satellite at given time

    Parameters:
    -----------
    nav : NavigationData
        Navigation data containing ephemerides
    sat : int
        Satellite number
    time : float
        Time of interest (TOW - Time of Week)

    Returns:
    --------
    eph : Ephemeris or None
        Best ephemeris or None if not found
    """
    best_eph = None
    min_dt = float('inf')

    for eph in nav.eph:
        if eph.sat != sat:
            continue

        # Check ephemeris validity using time of week
        if not is_ephemeris_valid_tow(eph, time):
            continue

        # Find closest ephemeris using time of week
        dt = abs(time - eph.toe)
        # Handle week wraparound
        if dt > 302400:  # Half week
            dt = 604800 - dt

        if dt < min_dt:
            min_dt = dt
            best_eph = eph

    return best_eph


def is_ephemeris_valid_tow(eph: Ephemeris, time_of_week: float) -> bool:
    """
    Check if ephemeris is valid at given time of week.

    Validates ephemeris data by checking satellite health and ensuring
    the time difference is within system-specific validity periods.
    Handles GPS week wraparound effects automatically.

    Parameters
    ----------
    eph : Ephemeris
        Ephemeris record to validate
    time_of_week : float
        Time of week in seconds (0-604800 range)

    Returns
    -------
    bool
        True if ephemeris is valid for the given time

    Notes
    -----
    System-specific validity periods:
    - GPS/QZSS: 2 hours (7200 seconds)
    - Galileo: 3 hours (10800 seconds)
    - BeiDou: 6 hours (21600 seconds)
    - GLONASS: 30 minutes (1800 seconds)
    - Other systems: 1 hour default

    The function automatically handles week wraparound by checking
    if dt > 302400 seconds (half week) and adjusting accordingly.

    Examples
    --------
    >>> is_valid = is_ephemeris_valid_tow(gps_eph, 345600.0)
    >>> print(f"Ephemeris valid: {is_valid}")
    """
    # Check SV health
    if eph.svh != 0:
        return False

    # Check time validity using time of week
    dt = abs(time_of_week - eph.toe)
    # Handle week wraparound
    if dt > 302400:  # Half week
        dt = 604800 - dt

    # System-specific validity periods
    sys = sat2sys(eph.sat)
    if sys == SYS_GPS or sys == SYS_QZS:
        max_dt = 7200.0  # 2 hours
    elif sys == SYS_GAL:
        max_dt = 10800.0  # 3 hours
    elif sys == SYS_BDS:
        max_dt = 21600.0  # 6 hours
    elif sys == SYS_GLO:
        max_dt = 1800.0   # 30 minutes
    else:
        max_dt = 3600.0   # 1 hour default

    return dt <= max_dt


def is_ephemeris_valid(eph: Ephemeris, time: float) -> bool:
    """
    Check if ephemeris is valid at given time with comprehensive validation.

    This function performs thorough validation of ephemeris data including
    satellite health checks, age validation, and fit interval considerations.
    It's the primary validation function used throughout the system.

    Parameters
    ----------
    eph : Ephemeris
        Ephemeris record to validate
    time : float
        Time of interest as Time of Week (seconds)

    Returns
    -------
    bool
        True if ephemeris is valid and healthy for the given time

    Notes
    -----
    Validation checks include:
    1. Satellite health (svh == 0 means healthy)
    2. Time difference within system-specific limits
    3. Week wraparound handling (604800 seconds)
    4. Custom fit interval if specified in ephemeris

    System validity periods:
    - GPS/QZSS: 2 hours
    - Galileo: 3 hours
    - BeiDou: 6 hours
    - GLONASS: 30 minutes
    - Default: 1 hour

    If ephemeris contains a fit interval (fit > 0), it overrides
    the default system validity period.

    Examples
    --------
    >>> if is_ephemeris_valid(eph, current_time):
    ...     pos, clk = compute_satellite_position(eph, current_time)
    """
    # Check SV health
    if eph.svh != 0:
        return False

    # Check time validity using time of week
    dt = abs(time - eph.toe)
    # Handle week wraparound
    if dt > 302400:  # Half week
        dt = 604800 - dt

    # System-specific validity periods
    sys = sat2sys(eph.sat)
    if sys == SYS_GPS or sys == SYS_QZS:
        max_dt = 7200.0  # 2 hours
    elif sys == SYS_GAL:
        max_dt = 10800.0  # 3 hours
    elif sys == SYS_BDS:
        max_dt = 21600.0  # 6 hours
    elif sys == SYS_GLO:
        max_dt = 1800.0   # 30 minutes (MAXDTOE_GLO)
    else:
        max_dt = 3600.0   # 1 hour default

    # Check fit interval if available
    if hasattr(eph, 'fit') and eph.fit > 0:
        max_dt = eph.fit * 3600.0 / 2.0

    return dt <= max_dt


def ephemeris_age(eph: Ephemeris, time: float) -> float:
    """
    Compute the age of ephemeris data relative to current time.

    Calculates the absolute time difference between the ephemeris
    reference time (toc - Time of Clock) and the given time,
    handling GPS week wraparound effects.

    Parameters
    ----------
    eph : Ephemeris
        Ephemeris record containing time reference (toc)
    time : float
        Current time as Time of Week (seconds)

    Returns
    -------
    float
        Age of ephemeris in seconds (always positive)

    Notes
    -----
    The function uses toc (Time of Clock) rather than toe (Time of
    Ephemeris) as the reference, which is more appropriate for
    age calculations related to clock accuracy.

    Week wraparound is handled by checking if the time difference
    exceeds half a week (302400 seconds) and adjusting accordingly.

    Examples
    --------
    >>> age = ephemeris_age(eph, current_time)
    >>> if age > 7200:  # Older than 2 hours
    ...     print("Warning: Old ephemeris data")
    """
    dt = abs(time - eph.toc)
    # Handle week wraparound
    if dt > 302400:  # Half week
        dt = 604800 - dt
    return dt


def select_eph_list(nav: NavigationData, time: float,
                    systems: int = SYS_ALL) -> list[Ephemeris]:
    """
    Select all valid ephemerides for specified satellite systems at given time.

    This function scans through all available ephemeris data and selects
    valid ephemerides for each satellite in the specified systems. It ensures
    only one ephemeris per satellite is returned (the first valid one found).

    Parameters
    ----------
    nav : NavigationData
        Navigation data container with ephemeris records
    time : float
        Time of interest as Time of Week (0-604800 seconds)
    systems : int, optional
        Bitmask of satellite systems to include (default: SYS_ALL)
        Valid values: SYS_GPS, SYS_GLO, SYS_GAL, SYS_BDS, SYS_QZS, etc.

    Returns
    -------
    list[Ephemeris]
        List of valid ephemeris records, one per satellite

    Notes
    -----
    The selection criteria:
    - Satellite system must match the systems bitmask
    - Ephemeris must pass validity checks (health, age, etc.)
    - Only the first valid ephemeris per satellite is included
    - System-specific validity periods are applied

    Examples
    --------
    >>> gps_ephs = select_eph_list(nav, time, SYS_GPS)
    >>> all_ephs = select_eph_list(nav, time)
    >>> print(f"Found {len(gps_ephs)} GPS ephemerides")
    """
    eph_list = []

    # Track which satellites we've already found
    found_sats = set()

    for eph in nav.eph:
        # Check system
        if not (sat2sys(eph.sat) & systems):
            continue

        # Skip if we already have ephemeris for this satellite
        if eph.sat in found_sats:
            continue

        # Check validity
        if is_ephemeris_valid(eph, time):
            eph_list.append(eph)
            found_sats.add(eph.sat)

    return eph_list


class EphemerisManager:
    """
    Manage ephemeris data storage, selection, and maintenance.

    This class provides a centralized manager for satellite ephemeris data,
    including storage, retrieval, validation, and cleanup of ephemeris records.
    It maintains ephemeris data for multiple satellites and automatically
    selects the best available ephemeris for position computations.

    Attributes
    ----------
    ephemerides : dict
        Dictionary mapping satellite numbers to lists of ephemeris records
    max_age : float
        Maximum age of ephemeris data to retain (seconds, default: 7200)

    Examples
    --------
    >>> manager = EphemerisManager()
    >>> manager.add_ephemeris(ephemeris_data)
    >>> eph = manager.get_ephemeris(sat=1, time=current_time)
    """

    def __init__(self):
        self.ephemerides = {}  # sat -> list of ephemeris
        self.max_age = 7200.0  # Maximum ephemeris age (2 hours)

    def add_ephemeris(self, eph: Ephemeris) -> None:
        """
        Add a new ephemeris record to the manager.

        The function checks for duplicates based on IODE, IODC, and toe
        to avoid storing redundant ephemeris data. Ephemerides are
        automatically sorted by time of ephemeris (toe).

        Parameters
        ----------
        eph : Ephemeris
            Ephemeris record to add

        Notes
        -----
        Duplicate detection uses:
        - IODE (Issue of Data Ephemeris)
        - IODC (Issue of Data Clock)
        - toe (Time of Ephemeris) with 1-second tolerance
        """
        if eph.sat not in self.ephemerides:
            self.ephemerides[eph.sat] = []

        # Check if this is a duplicate
        for existing in self.ephemerides[eph.sat]:
            if (existing.iode == eph.iode and
                existing.iodc == eph.iodc and
                abs(existing.toe - eph.toe) < 1.0):
                return  # Skip duplicate

        self.ephemerides[eph.sat].append(eph)

        # Sort by toe
        self.ephemerides[eph.sat].sort(key=lambda e: e.toe)

    def get_ephemeris(self, sat: int, time: float) -> Optional[Ephemeris]:
        """
        Get the best ephemeris for a satellite at a given time.

        Selects the ephemeris with the smallest time difference from
        the requested time, considering validity periods and week
        wraparound effects.

        Parameters
        ----------
        sat : int
            Satellite number
        time : float
            Time of interest (Time of Week, seconds)

        Returns
        -------
        Optional[Ephemeris]
            Best ephemeris if found and valid, None otherwise

        Notes
        -----
        The function:
        1. Checks all stored ephemerides for the satellite
        2. Validates each ephemeris using is_ephemeris_valid()
        3. Handles week wraparound (604800 seconds)
        4. Returns the ephemeris closest in time to the request
        """
        if sat not in self.ephemerides:
            return None

        best_eph = None
        min_dt = float('inf')

        for eph in self.ephemerides[sat]:
            if not is_ephemeris_valid(eph, time):
                continue

            dt = abs(time - eph.toe)
            # Handle week wraparound
            if dt > 302400:  # Half week
                dt = 604800 - dt

            if dt < min_dt:
                min_dt = dt
                best_eph = eph

        return best_eph

    def clean_old_ephemerides(self, current_time: float) -> None:
        """
        Remove ephemerides that exceed the maximum age limit.

        This function removes old ephemeris records to prevent
        memory growth and ensure only current data is used.
        Satellites with no remaining ephemerides are removed
        from the ephemerides dictionary.

        Parameters
        ----------
        current_time : float
            Current time reference for age calculation (seconds)

        Notes
        -----
        The cleanup process:
        1. Calculates age using ephemeris_age() function
        2. Removes ephemerides older than self.max_age
        3. Deletes satellite entries with no remaining ephemerides
        4. Helps prevent memory leaks in long-running applications
        """
        for sat in list(self.ephemerides.keys()):
            self.ephemerides[sat] = [
                eph for eph in self.ephemerides[sat]
                if ephemeris_age(eph, current_time) < self.max_age
            ]

            if not self.ephemerides[sat]:
                del self.ephemerides[sat]
