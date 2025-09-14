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

"""Satellite clock computation and correction"""


import numpy as np

from ..core.constants import *
from ..core.data_structures import Ephemeris
from ..core.time import timediff


def compute_satellite_clock(eph: Ephemeris, time: float) -> tuple[float, float]:
    """
    Compute satellite clock bias and drift

    Parameters:
    -----------
    eph : Ephemeris
        Satellite ephemeris
    time : float
        Time of interest (GPST)

    Returns:
    --------
    dts : float
        Satellite clock bias (s)
    ddts : float
        Satellite clock drift (s/s)
    """
    # Time from clock reference epoch
    dt = timediff(time, eph.toc)

    # Clock bias (polynomial model)
    dts = eph.f0 + eph.f1 * dt + eph.f2 * dt**2

    # Clock drift
    ddts = eph.f1 + 2.0 * eph.f2 * dt

    # Add relativistic correction for non-GEO satellites
    if eph.A > 0:  # Check if orbital parameters are available
        # Compute eccentric anomaly
        n = np.sqrt(GME / eph.A**3)
        M = eph.M0 + n * timediff(time, eph.toe)
        E = M
        for _ in range(10):
            E_old = E
            E = M + eph.e * np.sin(E)
            if abs(E - E_old) < 1e-12:
                break

        # Relativistic correction
        F = -2.0 * np.sqrt(GME) / CLIGHT**2
        dtr = F * eph.e * np.sqrt(eph.A) * np.sin(E)
        dts += dtr

        # Relativistic drift correction
        dE_dt = n / (1.0 - eph.e * np.cos(E))
        ddtr = F * eph.e * np.sqrt(eph.A) * np.cos(E) * dE_dt
        ddts += ddtr

    return dts, ddts


def apply_tgd_correction(eph: Ephemeris, freq_idx: int) -> float:
    """
    Apply Time Group Delay (TGD) correction for different satellite systems and frequencies.

    Time Group Delay represents the difference in signal propagation time between
    different frequencies within the satellite's electronics. This function computes
    the appropriate TGD correction based on the satellite system and frequency.

    Parameters
    ----------
    eph : Ephemeris
        Satellite ephemeris containing TGD parameters
    freq_idx : int
        Frequency index:
        - 0: L1/E1/B1 (primary frequency)
        - 1: L2/E5b/B2 (secondary frequency)
        - 2: L5/E5a/B3 (tertiary frequency)

    Returns
    -------
    float
        TGD correction in seconds

    Notes
    -----
    System-specific TGD handling:

    GPS:
    - L1: Direct TGD[0] application
    - L2: Scaled by frequency ratio to account for L1-L2 definition
    - L5: Uses TGD[1] if available

    Galileo:
    - E1: TGD[0] (BGD E1/E5a)
    - E5b: TGD[1] (BGD E5b/E1)
    - E5a: TGD[2] if available

    BeiDou:
    - Falls back to bias table if ephemeris TGD is missing or zero
    - Supports B1, B2, B3 frequencies

    Examples
    --------
    >>> tgd_l1 = apply_tgd_correction(gps_eph, 0)  # L1 correction
    >>> tgd_l2 = apply_tgd_correction(gps_eph, 1)  # L2 correction
    >>> print(f"L1 TGD: {tgd_l1*1e9:.1f} ns")
    """
    sys = sat2sys(eph.sat)

    if sys == SYS_GPS:
        if freq_idx == 0:  # L1
            return eph.tgd[0]
        elif freq_idx == 1:  # L2
            # TGD is defined for L1-L2, so L2 correction is different
            gamma = (FREQ_L1 / FREQ_L2)**2
            return eph.tgd[0] * gamma / (gamma - 1.0)
        elif freq_idx == 2:  # L5
            return eph.tgd[1] if len(eph.tgd) > 1 else 0.0

    elif sys == SYS_GAL:
        if freq_idx == 0:  # E1
            return eph.tgd[0]
        elif freq_idx == 1:  # E5b
            return eph.tgd[1] if len(eph.tgd) > 1 else 0.0
        elif freq_idx == 2:  # E5a
            return eph.tgd[2] if len(eph.tgd) > 2 else 0.0

    elif sys == SYS_BDS:
        # Check if TGD data is available in ephemeris
        if freq_idx == 0 and len(eph.tgd) > 0 and eph.tgd[0] != 0.0:  # B1
            return eph.tgd[0]
        elif freq_idx == 1 and len(eph.tgd) > 1 and eph.tgd[1] != 0.0:  # B2
            return eph.tgd[1]
        elif freq_idx == 2 and len(eph.tgd) > 2 and eph.tgd[2] != 0.0:  # B3
            return eph.tgd[2]
        else:
            # Use BeiDou bias table if ephemeris TGD is missing or zero
            from ..gnss.beidou_bias import get_beidou_tgd
            return get_beidou_tgd(eph.sat, freq_idx)

    return 0.0


def compute_clock_variance(eph: Ephemeris, time: float) -> float:
    """
    Compute satellite clock variance based on URA and ephemeris age.

    The clock variance is computed using the User Range Accuracy (URA)
    as a base measure and increased with ephemeris age to account for
    growing uncertainty over time.

    Parameters
    ----------
    eph : Ephemeris
        Satellite ephemeris containing URA (sva field)
    time : float
        Time of interest in GPS time (seconds)

    Returns
    -------
    float
        Clock variance in seconds squared (s²)

    Notes
    -----
    The variance computation includes:
    1. Base uncertainty from URA converted to time units
    2. Age-dependent factor that increases with time from toc
    3. Conversion from range accuracy to time accuracy using speed of light

    The age factor uses a linear model: variance increases by 100% per hour
    from the clock reference time (toc).

    Examples
    --------
    >>> var = compute_clock_variance(eph, current_time)
    >>> std_ns = np.sqrt(var) * 1e9  # Convert to nanoseconds
    >>> print(f"Clock uncertainty: {std_ns:.1f} ns")
    """
    # Base variance from SV accuracy
    from .satellite_position import ura_value
    std_base = ura_value(eph.sva) / CLIGHT  # Convert to time

    # Add age-dependent term
    age = abs(timediff(time, eph.toc))
    age_factor = 1.0 + age / 3600.0  # Increase uncertainty with age

    # Clock variance
    var = (std_base * age_factor)**2

    return var


class SatelliteClockManager:
    """
    Manage satellite clock corrections with caching and optimization.

    This class provides efficient computation and caching of satellite clock
    corrections to avoid redundant calculations. It maintains a cache of
    recently computed clock corrections and automatically manages memory
    by removing old entries.

    Attributes
    ----------
    clock_cache : dict
        Cache mapping (sat, time) tuples to (dts, ddts) corrections
    cache_tolerance : float
        Time tolerance for cache hits (seconds, default: 1.0)

    Notes
    -----
    The manager uses a simple caching strategy where corrections computed
    within the cache_tolerance time window are reused. This is particularly
    beneficial when processing multiple observations from the same epoch
    or when performing iterative computations.

    Cache cleanup is performed automatically when the cache size exceeds
    1000 entries, removing entries older than 5 minutes.

    Examples
    --------
    >>> manager = SatelliteClockManager()
    >>> dts, ddts = manager.get_clock_correction(eph, time)
    >>> print(f"Clock bias: {dts*1e9:.1f} ns")
    """

    def __init__(self):
        self.clock_cache = {}  # (sat, time) -> (dts, ddts)
        self.cache_tolerance = 1.0  # seconds

    def get_clock_correction(self, eph: Ephemeris, time: float,
                            use_cache: bool = True) -> tuple[float, float]:
        """
        Get satellite clock correction with optional caching.

        This method computes or retrieves cached satellite clock corrections,
        including both bias and drift components. It provides significant
        performance improvements when processing multiple observations from
        the same time epoch.

        Parameters
        ----------
        eph : Ephemeris
            Satellite ephemeris containing clock parameters (f0, f1, f2, toc)
        time : float
            Time of interest in GPS time (seconds)
        use_cache : bool, optional
            Whether to check and update the cache (default: True)

        Returns
        -------
        tuple[float, float]
            - dts : float
                Clock bias in seconds (positive = satellite clock fast)
            - ddts : float
                Clock drift in seconds per second

        Notes
        -----
        The method:
        1. Checks cache for recent computations within cache_tolerance
        2. Computes new corrections if not cached or cache disabled
        3. Updates cache with new results
        4. Performs automatic cache cleanup when size exceeds limits

        Clock corrections include:
        - Polynomial clock model (f0 + f1*dt + f2*dt²)
        - Relativistic correction for eccentric orbits
        - Proper time derivatives for drift computation

        Examples
        --------
        >>> manager = SatelliteClockManager()
        >>> dts, ddts = manager.get_clock_correction(eph, time)
        >>> # Apply correction to pseudorange
        >>> corrected_range = raw_range + dts * SPEED_OF_LIGHT
        """
        # Check cache
        if use_cache:
            for (sat, t), (dts, ddts) in self.clock_cache.items():
                if sat == eph.sat and abs(time - t) < self.cache_tolerance:
                    return dts, ddts

        # Compute clock correction
        dts, ddts = compute_satellite_clock(eph, time)

        # Update cache
        self.clock_cache[(eph.sat, time)] = (dts, ddts)

        # Clean old cache entries
        if len(self.clock_cache) > 1000:
            self._clean_cache(time)

        return dts, ddts

    def _clean_cache(self, current_time: float) -> None:
        """
        Remove old cache entries to prevent memory growth.

        This internal method removes cache entries older than the maximum
        age limit to prevent unbounded memory growth in long-running
        applications.

        Parameters
        ----------
        current_time : float
            Current time reference for age calculation (seconds)

        Notes
        -----
        The cleanup process:
        - Removes entries older than 5 minutes (300 seconds)
        - Preserves recent entries that may still be useful
        - Maintains cache performance by limiting size
        - Called automatically when cache exceeds 1000 entries
        """
        max_age = 300.0  # 5 minutes
        self.clock_cache = {
            (sat, t): (dts, ddts)
            for (sat, t), (dts, ddts) in self.clock_cache.items()
            if abs(current_time - t) < max_age
        }
