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

"""Core data structures for GNSS processing"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .constants import *


class SensorType(Enum):
    """Enumeration of supported sensor types.

    Attributes
    ----------
    GNSS : int
        Global Navigation Satellite System sensors
    IMU : int
        Inertial Measurement Unit sensors
    CAMERA : int
        Visual camera sensors
    LIDAR : int
        Light Detection and Ranging sensors
    WHEEL : int
        Wheel odometry sensors
    MAGNETOMETER : int
        Magnetic field sensors
    BAROMETER : int
        Atmospheric pressure sensors
    """
    GNSS = 1
    IMU = 2
    CAMERA = 3
    LIDAR = 4
    WHEEL = 5
    MAGNETOMETER = 6
    BAROMETER = 7

@dataclass
class Observation:
    """GNSS observation data for a single satellite at a specific epoch.

    This class contains all observables for a satellite including carrier phase,
    pseudorange, Doppler, and signal quality metrics across multiple frequency bands.

    Attributes
    ----------
    time : float
        Reception time in GPS time (GPST) seconds
    sat : int
        Satellite number using internal satellite numbering system
    system : int
        Satellite system ID (automatically determined from sat in __post_init__)
    L : np.ndarray
        Carrier phase measurements in cycles, shape (MAXBAND,)
    P : np.ndarray
        Pseudorange measurements in meters, shape (MAXBAND,)
    D : np.ndarray
        Doppler frequency measurements in Hz, shape (MAXBAND,)
    SNR : np.ndarray
        Signal-to-noise ratio in dB-Hz, shape (MAXBAND,)
    LLI : np.ndarray
        Loss of lock indicators, shape (MAXBAND,), dtype int
    code : np.ndarray
        Code type indicators, shape (MAXBAND,), dtype int

    Notes
    -----
    The arrays are indexed by frequency band (0=L1/E1/B1, 1=L2/E5b/B3, 2=L5/E5a/B2a).
    Zero values typically indicate no observation for that band.
    """
    time: float                    # reception time (GPST)
    sat: int                       # satellite number
    system: int                    # satellite system
    L: np.ndarray = field(default_factory=lambda: np.zeros(MAXBAND))  # carrier phase (cycles)
    P: np.ndarray = field(default_factory=lambda: np.zeros(MAXBAND))  # pseudorange (m)
    D: np.ndarray = field(default_factory=lambda: np.zeros(MAXBAND))  # doppler (Hz)
    SNR: np.ndarray = field(default_factory=lambda: np.zeros(MAXBAND)) # signal strength (dBHz)
    LLI: np.ndarray = field(default_factory=lambda: np.zeros(MAXBAND, dtype=int)) # loss of lock indicator
    code: np.ndarray = field(default_factory=lambda: np.zeros(MAXBAND, dtype=int)) # code indicator

    def __post_init__(self):
        self.system = sat2sys(self.sat)

    @property
    def prn(self):
        """Get PRN number from satellite number.

        Returns
        -------
        int
            PRN number within the satellite's constellation
        """
        from .constants import sat2prn
        return sat2prn(self.sat)

    def get_frequency(self, band, glo_fcn=0):
        """Get carrier frequency for given frequency band.

        Parameters
        ----------
        band : int
            Frequency band index (0=L1/E1/B1, 1=L2/E5b/B3, 2=L5/E5a/B2a)
        glo_fcn : int, optional
            GLONASS frequency channel number (-7 to +13), default 0

        Returns
        -------
        float
            Carrier frequency in Hz, or 0.0 if invalid band or system

        Notes
        -----
        For GLONASS satellites, the frequency depends on the frequency channel number (FCN).
        For other systems, glo_fcn parameter is ignored.
        """
        if self.system == SYS_GPS:
            freqs = [FREQ_L1, FREQ_L2, FREQ_L5]
        elif self.system == SYS_GLO:
            # GLONASS FDMA frequencies (need frequency channel number)
            freqs = [FREQ_G1 + glo_fcn * DFREQ_G1, FREQ_G2 + glo_fcn * DFREQ_G2, 0]
        elif self.system == SYS_GAL:
            freqs = [FREQ_E1, FREQ_E5b, FREQ_E5a]
        elif self.system == SYS_BDS:
            freqs = [FREQ_B1I, FREQ_B3, FREQ_B2a]
        elif self.system == SYS_QZS:
            # QZSS supports L1/L2/L5 (same as GPS)
            freqs = [FREQ_J1, FREQ_J2, FREQ_J5]
        else:
            return 0.0

        return freqs[band] if band < len(freqs) else 0.0

    def get_wavelength(self, band, glo_fcn=0):
        """Get carrier wavelength for given frequency band.

        Parameters
        ----------
        band : int
            Frequency band index (0=L1/E1/B1, 1=L2/E5b/B3, 2=L5/E5a/B2a)
        glo_fcn : int, optional
            GLONASS frequency channel number (-7 to +13), default 0

        Returns
        -------
        float
            Carrier wavelength in meters, or 0.0 if invalid band or system

        Notes
        -----
        Wavelength is calculated as speed of light divided by frequency.
        For GLONASS satellites, the wavelength depends on the frequency channel number.
        """
        freq = self.get_frequency(band, glo_fcn)
        return CLIGHT / freq if freq > 0 else 0.0


@dataclass
class Ephemeris:
    """Satellite ephemeris data for GPS/Galileo/BeiDou/QZSS navigation messages.

    This class contains broadcast ephemeris parameters needed for satellite
    position and clock computation using Keplerian orbital elements.

    Attributes
    ----------
    sat : int
        Satellite number (internal numbering system)
    iode : int
        Issue of Data Ephemeris
    iodc : int
        Issue of Data Clock
    sva : int
        SV accuracy (URA index for GPS/QZSS, SISA for Galileo, URAI for BeiDou)
    svh : int
        SV health flag
    week : int
        GPS week number (or system equivalent)
    code : int
        Code on L2 (GPS), data source (BeiDou), spare (Galileo)
    flag : int
        L2 P data flag (GPS), navigation type (BeiDou), spare (Galileo)
    toe : float
        Time of ephemeris reference epoch (seconds in week)
    toc : float
        Time of clock reference epoch (seconds in week)
    ttr : float
        Transmission time (seconds in week)
    A : float
        Semi-major axis (meters)
    e : float
        Eccentricity (dimensionless)
    i0 : float
        Inclination angle at reference epoch (radians)
    OMG0 : float
        Longitude of ascending node at weekly epoch (radians)
    omg : float
        Argument of perigee (radians)
    M0 : float
        Mean anomaly at reference epoch (radians)
    deln : float
        Mean motion difference from computed value (radians/second)
    OMGd : float
        Rate of right ascension (radians/second)
    idot : float
        Rate of inclination angle (radians/second)
    crc : float
        Amplitude of cosine harmonic correction term to orbit radius (meters)
    crs : float
        Amplitude of sine harmonic correction term to orbit radius (meters)
    cuc : float
        Amplitude of cosine harmonic correction term to argument of latitude (radians)
    cus : float
        Amplitude of sine harmonic correction term to argument of latitude (radians)
    cic : float
        Amplitude of cosine harmonic correction term to inclination (radians)
    cis : float
        Amplitude of sine harmonic correction term to inclination (radians)
    toes : float
        Time of ephemeris (seconds in week, for convenience)
    fit : float
        Fit interval (hours)
    f0 : float
        SV clock bias (seconds)
    f1 : float
        SV clock drift (seconds/second)
    f2 : float
        SV clock drift rate (seconds/second²)
    tgd : np.ndarray
        Group delay parameters, shape (4,)
    system : int
        Satellite system (automatically set in __post_init__)

    Notes
    -----
    The ephemeris data follows the standard broadcast navigation message format.
    System-specific variations are handled appropriately for each GNSS.
    """
    sat: int                # satellite number
    iode: int              # issue of data ephemeris
    iodc: int              # issue of data clock
    sva: int               # SV accuracy (URA index)
    svh: int               # SV health
    week: int              # GPS week
    code: int              # GPS/GAL: code on L2, BDS: data source
    flag: int              # GPS/GAL: L2 P data flag, BDS: nav type

    toe: float             # time of ephemeris (s)
    toc: float             # time of clock (s)
    ttr: float             # transmission time (s)

    A: float               # orbit semi-major axis (m)
    e: float               # eccentricity
    i0: float              # inclination angle at ref time (rad)
    OMG0: float            # longitude of ascending node (rad)
    omg: float             # argument of perigee (rad)
    M0: float              # mean anomaly at ref time (rad)
    deln: float            # mean motion difference (rad/s)
    OMGd: float            # rate of ascending node (rad/s)
    idot: float            # rate of inclination (rad/s)

    crc: float             # radius cosine correction (m)
    crs: float             # radius sine correction (m)
    cuc: float             # latitude cosine correction (rad)
    cus: float             # latitude sine correction (rad)
    cic: float             # inclination cosine correction (rad)
    cis: float             # inclination sine correction (rad)

    toes: float            # toe (s) in week
    fit: float             # fit interval (h)
    f0: float              # SV clock offset (s)
    f1: float              # SV clock drift (s/s)
    f2: float              # SV clock drift rate (s/s2)

    tgd: np.ndarray = field(default_factory=lambda: np.zeros(4))  # group delay

    def __post_init__(self):
        self.system = sat2sys(self.sat)


@dataclass
class GloEphemeris:
    """GLONASS broadcast ephemeris data.

    GLONASS uses a different ephemeris format compared to GPS/Galileo/BeiDou,
    providing satellite position, velocity, and acceleration directly rather
    than Keplerian orbital elements.

    Attributes
    ----------
    sat : int
        Satellite number (internal numbering system)
    iode : int
        Issue of Data Ephemeris (bits 0-6 of tb field)
    frq : int
        GLONASS frequency channel number (-7 to +13)
    svh : int
        Extended satellite health (bit 3: ln, bit 2: Cn_a, bit 1: Cn, bit 0: Bn)
    flags : int
        Status flags (bits 7-8: M, bit 6: P4, bit 5: P3, bit 4: P2, bits 2-3: P1, bits 0-1: P)
    sva : int
        Satellite accuracy index
    age : int
        Age of operation information (days)
    toe : float
        Epoch of ephemeris (GPS time seconds)
    tof : float
        Message frame time (GPS time seconds)
    pos : np.ndarray
        Satellite position in ECEF coordinates (meters), shape (3,)
    vel : np.ndarray
        Satellite velocity in ECEF coordinates (meters/second), shape (3,)
    acc : np.ndarray
        Satellite acceleration in ECEF coordinates (meters/second²), shape (3,)
    taun : float
        Satellite clock bias relative to GLONASS time (seconds)
    gamn : float
        Relative frequency bias (dimensionless)
    dtaun : float
        Time difference between L1 and L2 signals (seconds)
    system : int
        Satellite system (automatically set to SYS_GLO in __post_init__)

    Notes
    -----
    Based on RTKLIB geph_t structure. GLONASS ephemeris provides direct state vectors
    rather than orbital elements, requiring different algorithms for satellite
    position computation.
    """
    sat: int                # satellite number
    iode: int              # IODE (0-6 bit of tb field)
    frq: int               # Satellite frequency number (-7 to 13)
    svh: int               # Extended SVH (bit 3:ln, bit 2:Cn_a, bit 1:Cn, bit 0:Bn)
    flags: int             # Status flags (bits 7 8:M, bit 6:P4, bit 5:P3, bit 4:P2, bits 2 3:P1, bits 0 1:P)
    sva: int               # satellite accuracy
    age: int               # age of operation

    toe: float             # epoch of ephemeris (GPST)
    tof: float             # message frame time (GPST)

    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))     # satellite position (ECEF, m)
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))     # satellite velocity (ECEF, m/s)
    acc: np.ndarray = field(default_factory=lambda: np.zeros(3))     # satellite acceleration (ECEF, m/s^2)

    taun: float = 0.0      # SV clock bias (s)
    gamn: float = 0.0      # relative freq bias
    dtaun: float = 0.0     # delay between L1 and L2 (s)

    def __post_init__(self):
        self.system = SYS_GLO


@dataclass
class NavigationData:
    """Container for all navigation data including ephemerides and correction parameters.

    This class stores broadcast ephemerides, precise products, and system parameters
    needed for GNSS positioning and timing.

    Attributes
    ----------
    eph : list[Ephemeris]
        List of broadcast ephemerides for GPS/Galileo/BeiDou/QZSS satellites
    geph : list[GloEphemeris]
        List of GLONASS broadcast ephemerides
    peph : list
        List of precise ephemeris data (for high-precision applications)
    pclk : list
        List of precise clock data (for high-precision applications)
    ion_gps : np.ndarray
        GPS ionospheric correction parameters (α₀, α₁, α₂, α₃, β₀, β₁, β₂, β₃), shape (8,)
    ion_gal : np.ndarray
        Galileo ionospheric correction parameters (ai₀, ai₁, ai₂, flags), shape (4,)
    ion_bds : np.ndarray
        BeiDou ionospheric correction parameters (α₀, α₁, α₂, α₃, β₀, β₁, β₂, β₃), shape (8,)
    utc_gps : np.ndarray
        GPS UTC correction parameters (A₀, A₁, tot, WNt), shape (4,)
    utc_gal : np.ndarray
        Galileo UTC correction parameters (A₀G, A₁G, tot, WNot), shape (4,)
    utc_bds : np.ndarray
        BeiDou UTC correction parameters (A₀UTC, A₁UTC, tot, WNot), shape (4,)

    Notes
    -----
    The ionospheric and UTC parameters are broadcast in the navigation messages
    and used for single-frequency positioning and time conversion.
    """
    eph: list[Ephemeris] = field(default_factory=list)
    geph: list[GloEphemeris] = field(default_factory=list)  # GLONASS ephemeris
    peph: list = field(default_factory=list)  # Precise ephemeris
    pclk: list = field(default_factory=list)  # Precise clock

    ion_gps: np.ndarray = field(default_factory=lambda: np.zeros(8))   # GPS ionosphere parameters
    ion_gal: np.ndarray = field(default_factory=lambda: np.zeros(4))   # Galileo ionosphere parameters
    ion_bds: np.ndarray = field(default_factory=lambda: np.zeros(8))   # BeiDou ionosphere parameters

    utc_gps: np.ndarray = field(default_factory=lambda: np.zeros(4))   # GPS UTC parameters
    utc_gal: np.ndarray = field(default_factory=lambda: np.zeros(4))   # Galileo UTC parameters
    utc_bds: np.ndarray = field(default_factory=lambda: np.zeros(4))   # BeiDou UTC parameters

    def find_eph(self, sat, time):
        """Find broadcast ephemeris for satellite at given time.

        Searches for the ephemeris with the closest time of ephemeris (toe)
        to the requested time for the specified satellite.

        Parameters
        ----------
        sat : int
            Satellite number (internal numbering system)
        time : float
            Requested time in GPS time (seconds)

        Returns
        -------
        Ephemeris or None
            Best matching ephemeris for the satellite, or None if not found

        Notes
        -----
        Returns the ephemeris with minimum |time - toe| for the given satellite.
        Does not check ephemeris validity interval.
        """
        best_eph = None
        min_dt = float('inf')

        for eph in self.eph:
            if eph.sat != sat:
                continue
            dt = abs(time - eph.toe)
            if dt < min_dt:
                min_dt = dt
                best_eph = eph

        return best_eph

    def find_geph(self, sat, time):
        """Find GLONASS ephemeris for satellite at given time.

        Searches for the GLONASS ephemeris with the closest time of ephemeris (toe)
        to the requested time for the specified satellite.

        Parameters
        ----------
        sat : int
            Satellite number (internal numbering system for GLONASS satellite)
        time : float
            Requested time in GPS time (seconds)

        Returns
        -------
        GloEphemeris or None
            Best matching GLONASS ephemeris for the satellite, or None if not found

        Notes
        -----
        Returns the GLONASS ephemeris with minimum |time - toe| for the given satellite.
        GLONASS ephemerides have shorter validity periods than GPS ephemerides.
        """
        best_geph = None
        min_dt = float('inf')

        for geph in self.geph:
            if geph.sat != sat:
                continue
            dt = abs(time - geph.toe)
            if dt < min_dt:
                min_dt = dt
                best_geph = geph

        return best_geph

    def sort_eph(self):
        """Sort ephemerides by satellite number, then by time.

        Sorts both broadcast ephemerides (eph) and GLONASS ephemerides (geph)
        in ascending order first by satellite number, then by time of ephemeris.

        Notes
        -----
        This method modifies the ephemeris lists in place. Sorting improves
        efficiency of ephemeris lookup operations.
        """
        self.eph.sort(key=lambda x: (x.sat, x.toe))
        if self.geph:
            self.geph.sort(key=lambda x: (x.sat, x.toe))


@dataclass
class Solution:
    """GNSS positioning solution with position, velocity, and quality metrics.

    This class contains the results of GNSS positioning computation including
    position, velocity, receiver clock biases, and associated uncertainties.

    Attributes
    ----------
    time : float
        Solution epoch time in GPS time (seconds)
    type : int
        Solution type (SOLQ_NONE, SOLQ_FIX, SOLQ_FLOAT, etc.)
    rr : np.ndarray
        Position in ECEF coordinates (X, Y, Z in meters), shape (3,)
    vv : np.ndarray
        Velocity in ECEF coordinates (Vx, Vy, Vz in m/s), shape (3,)
    dtr : np.ndarray
        Receiver clock biases for different systems (seconds), shape (6,)
    qr : np.ndarray
        Position and velocity covariance matrix (m² and m²/s²), shape (6, 6)
    ns : int
        Number of satellites used in solution
    age : float
        Age of differential corrections (seconds)
    ratio : float
        Ambiguity ratio factor for RTK solutions

    Notes
    -----
    The covariance matrix qr is ordered as [X, Y, Z, Vx, Vy, Vz].
    Clock biases in dtr are ordered by system: [GPS, GLO, GAL, BDS, QZS, SBS].
    """
    time: float                    # solution time (GPST)
    type: int = SOLQ_NONE         # solution type

    rr: np.ndarray = field(default_factory=lambda: np.zeros(3))      # position (ECEF, m)
    vv: np.ndarray = field(default_factory=lambda: np.zeros(3))      # velocity (ECEF, m/s)
    dtr: np.ndarray = field(default_factory=lambda: np.zeros(6))     # receiver clock bias (s)

    qr: np.ndarray = field(default_factory=lambda: np.zeros((6, 6))) # position/velocity covariance

    ns: int = 0                    # number of satellites
    age: float = 0.0              # age of differential (s)
    ratio: float = 0.0            # ambiguity ratio factor

    def get_llh(self):
        """Get geodetic position in latitude, longitude, height.

        Converts ECEF coordinates to geodetic coordinates using WGS84 ellipsoid.

        Returns
        -------
        np.ndarray
            Geodetic position [latitude, longitude, height] where:
            - latitude in radians (positive north)
            - longitude in radians (positive east)
            - height in meters above WGS84 ellipsoid

        Examples
        --------
        >>> sol = Solution(time=0.0)
        >>> sol.rr = np.array([4194304.0, 171887.0, 4780316.0])  # Example ECEF
        >>> llh = sol.get_llh()
        >>> lat_deg = llh[0] * 180/np.pi  # Convert to degrees
        """
        from ..coordinate import ecef2llh
        return ecef2llh(self.rr)

    def get_enu_cov(self):
        """Get position covariance matrix in local ENU coordinates.

        Transforms the ECEF position covariance to local East-North-Up coordinates
        at the solution position.

        Returns
        -------
        np.ndarray
            3x3 covariance matrix in ENU coordinates (m²)
            Order: [East, North, Up]

        Notes
        -----
        The transformation is performed at the current solution position.
        Only the position part of the covariance matrix is transformed.

        Examples
        --------
        >>> sol = Solution(time=0.0)
        >>> sol.rr = np.array([4194304.0, 171887.0, 4780316.0])
        >>> sol.qr[:3, :3] = np.eye(3) * 0.01  # 1cm std in each ECEF axis
        >>> enu_cov = sol.get_enu_cov()
        >>> enu_std = np.sqrt(np.diag(enu_cov))  # [east_std, north_std, up_std]
        """
        from ..coordinate import covecef2enu
        llh = self.get_llh()
        return covecef2enu(llh, self.qr[:3, :3])
