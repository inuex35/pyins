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
    """Sensor type enumeration"""
    GNSS = 1
    IMU = 2
    CAMERA = 3
    LIDAR = 4
    WHEEL = 5
    MAGNETOMETER = 6
    BAROMETER = 7

@dataclass
class Observation:
    """GNSS observation data"""
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
        """Get PRN number from satellite number"""
        from .constants import sat2prn
        return sat2prn(self.sat)

    def get_frequency(self, band, glo_fcn=0):
        """Get carrier frequency for given band"""
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
        """Get carrier wavelength for given band"""
        freq = self.get_frequency(band, glo_fcn)
        return CLIGHT / freq if freq > 0 else 0.0


@dataclass
class Ephemeris:
    """Satellite ephemeris data"""
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
    """GLONASS broadcast ephemeris (based on RTKLIB geph_t)"""
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
    """Navigation data container"""
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
        """Find ephemeris for satellite at given time"""
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
        """Find GLONASS ephemeris for satellite at given time"""
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
        """Sort ephemerides by satellite number, then by time"""
        self.eph.sort(key=lambda x: (x.sat, x.toe))
        if self.geph:
            self.geph.sort(key=lambda x: (x.sat, x.toe))


@dataclass
class Solution:
    """Position solution"""
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
        """Get geodetic position (lat, lon, height)"""
        from ..coordinate import ecef2llh
        return ecef2llh(self.rr)

    def get_enu_cov(self):
        """Get ENU covariance matrix"""
        from ..coordinate import covecef2enu
        llh = self.get_llh()
        return covecef2enu(llh, self.qr[:3, :3])
