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

"""GNSS Constants and System Parameters"""

import numpy as np

# Physical Constants
CLIGHT = 299792458.0  # speed of light (m/s)

# GPS frequencies
FREQ_L1 = 1.57542E9   # L1 frequency (Hz)
FREQ_L2 = 1.22760E9   # L2 frequency (Hz)
FREQ_L5 = 1.17645E9   # L5 frequency (Hz)

# GLONASS frequencies
FREQ_G1 = 1.60200E9   # GLONASS G1 base frequency (Hz)
FREQ_G2 = 1.24600E9   # GLONASS G2 base frequency (Hz)
DFREQ_G1 = 0.56250E6  # GLONASS G1 channel spacing (Hz)
DFREQ_G2 = 0.43750E6  # GLONASS G2 channel spacing (Hz)

# Galileo frequencies
FREQ_E1 = 1.57542E9   # E1 frequency (Hz) - same as GPS L1
FREQ_E5a = 1.17645E9  # E5a frequency (Hz) - same as GPS L5
FREQ_E5b = 1.20714E9  # E5b frequency (Hz)
FREQ_E5 = 1.191795E9  # E5 (E5a+E5b) frequency (Hz)
FREQ_E6 = 1.27875E9   # E6 frequency (Hz)

# BeiDou frequencies
FREQ_B1I = 1.561098E9  # BeiDou B1I frequency (Hz)
FREQ_B1C = 1.57542E9   # BeiDou B1C frequency (Hz) - same as GPS L1
FREQ_B2a = 1.17645E9   # BeiDou B2a frequency (Hz) - same as GPS L5
FREQ_B2b = 1.20714E9   # BeiDou B2b frequency (Hz) - same as Galileo E5b
FREQ_B2 = 1.191795E9   # BeiDou B2 (B2a+B2b) frequency (Hz)
FREQ_B3 = 1.26852E9    # BeiDou B3 frequency (Hz)

# QZSS frequencies (same as GPS)
FREQ_J1 = FREQ_L1      # QZSS L1 frequency (Hz)
FREQ_J2 = FREQ_L2      # QZSS L2 frequency (Hz)
FREQ_J5 = FREQ_L5      # QZSS L5 frequency (Hz)

# SBAS frequencies (same as GPS L1/L5)
FREQ_S1 = FREQ_L1      # SBAS L1 frequency (Hz)
FREQ_S5 = FREQ_L5      # SBAS L5 frequency (Hz)

# IRNSS frequencies
FREQ_I5 = FREQ_L5      # IRNSS L5 frequency (Hz)
FREQ_IS = 2.492028E9   # IRNSS S frequency (Hz)

# GNSS System IDs
SYS_GPS = 0x01    # GPS
SYS_GLO = 0x02    # GLONASS
SYS_GAL = 0x04    # Galileo
SYS_BDS = 0x08    # BeiDou
SYS_QZS = 0x10    # QZSS
SYS_SBS = 0x20    # SBAS
SYS_IRN = 0x40    # IRNSS
SYS_ALL = 0xFF    # All systems

# Time System Parameters
GPST0 = [1980, 1, 6, 0, 0, 0]  # GPS time reference epoch
GST0 = [1999, 8, 22, 0, 0, 0]  # Galileo time reference epoch
BDT0 = [2006, 1, 1, 0, 0, 0]   # BeiDou time reference epoch

# Time system offsets (as of 2025)
GPS_GLONASS_OFFSET = 18.0      # GPS-GLONASS time offset (seconds)
GPS_UTC_OFFSET = 18.0          # GPS-UTC leap seconds (as of 2025)
GPS_BDS_OFFSET = 14.0          # GPS-BeiDou time offset (seconds)

# Earth Parameters (WGS84)
RE_WGS84 = 6378137.0           # earth semimajor axis (m)
FE_WGS84 = 1.0 / 298.257223563 # earth flattening
OMGE = 7.2921151467E-5         # earth angular velocity (rad/s)
GME = 3.986004418E14           # earth gravitational constant
J2 = 1.082627E-3               # J2 coefficient

# System-specific gravitational constants
MU_GPS = 3.9860050E14          # GPS gravitational constant
MU_GAL = 3.986004418E14        # Galileo gravitational constant
MU_GLO = 3.9860044E14          # GLONASS gravitational constant
MU_BDS = 3.986004418E14        # BeiDou gravitational constant

# System-specific earth angular velocities
OMGE_GAL = 7.2921151467E-5     # Galileo earth angular velocity
OMGE_GLO = 7.292115E-5         # GLONASS earth angular velocity
OMGE_BDS = 7.292115E-5         # BeiDou earth angular velocity

# GLONASS-specific parameters
J2_GLO = 1.082627E-3           # GLONASS J2 coefficient
RE_GLO = 6378136.0             # GLONASS earth radius

# Additional WGS84 constants from navlib
RP_WGS84 = 6356752.31425       # polar radius (semi-minor axis) (m)
E_WGS84 = 0.0818191908425      # WGS84 eccentricity
E2_WGS84 = E_WGS84 * E_WGS84   # WGS84 eccentricity squared
ERREPH_GLO = 5.0               # GLONASS ephemeris error (m)
TSTEP = 120.0                 # GLONASS orbit integration step (s)

# Unit conversions
R2D = 180.0 / np.pi            # radians to degrees
D2R = np.pi / 180.0            # degrees to radians

# Gravity constant
G_GRAVITY = 9.80665            # Earth's gravity constant (m/s^2)

# Error/Threshold Constants
ERR_SAAS = 0.3      # Saastamoinen model error std (m)
ERR_BRDCI = 0.5     # broadcast ionosphere model error factor
ERR_CBIAS = 0.3     # code bias error std (m)
REL_HUMI = 0.7      # relative humidity
MAXBAND = 10        # max frequency bands
MAXANT = 64         # max antenna number

# Solution Status
SOLQ_NONE = 0       # no solution
SOLQ_FIX = 1        # fixed solution
SOLQ_FLOAT = 2      # float solution
SOLQ_SBAS = 3       # SBAS solution
SOLQ_DGPS = 4       # DGPS solution
SOLQ_SINGLE = 5     # single point positioning
SOLQ_PPP = 6        # PPP solution
SOLQ_DR = 7         # dead reckoning

# Observation Types
CODE_NONE = 0       # none or unknown
CODE_L1C = 1        # L1C/A,G1C/A,E1C (GPS,GLO,GAL,QZS,SBS)
CODE_L1P = 2        # L1P,G1P (GPS,GLO)
CODE_L1W = 3        # L1 Z-tracking (GPS)
CODE_L1Y = 4        # L1Y (GPS)
CODE_L1M = 5        # L1M (GPS)
CODE_L1N = 6        # L1N (GPS)
CODE_L1S = 7        # L1S (GPS,SBS)
CODE_L1L = 8        # L1L (GPS,QZS)
CODE_L1X = 9        # L1X (GPS,QZS)

# Wavelengths
def lam_carr(sys, freq):
    """Get carrier wavelength"""
    return CLIGHT / freq if freq > 0 else 0.0

# Add SYS_NONE for invalid satellites
SYS_NONE = 0x00

# Satellite system functions
def sat2sys(sat):
    """Get satellite system from satellite number

    Uses unified satellite numbering from satellite_numbering.py
    """
    from .satellite_numbering import SATELLITE_RANGES

    if sat <= 0 or sat > 255:
        return SYS_NONE

    for sys_id, ranges in SATELLITE_RANGES.items():
        for start, end in ranges:
            if start <= sat <= end:
                return sys_id

    return SYS_NONE

def sat2prn(sat):
    """Get PRN number from satellite number

    Uses unified satellite numbering from satellite_numbering.py
    """
    from .satellite_numbering import sat_to_prn
    return sat_to_prn(sat)

def prn2sat(prn, sys):
    """Get satellite number from PRN and system

    Uses unified satellite numbering from satellite_numbering.py

    Parameters:
    -----------
    prn : int
        PRN number
    sys : int
        Satellite system (SYS_GPS, SYS_GLO, etc.)

    Returns:
    --------
    int
        Satellite number
    """
    from .satellite_numbering import SYS_TO_CHAR, prn_to_sat

    # Get system character
    sys_char = SYS_TO_CHAR.get(sys, None)
    if sys_char is None:
        return 0

    return prn_to_sat(sys_char, prn)

def sys2char(sys):
    """Convert system ID to character"""
    syschar = {
        SYS_GPS: 'G',
        SYS_GLO: 'R',
        SYS_GAL: 'E',
        SYS_BDS: 'C',
        SYS_QZS: 'J',
        SYS_SBS: 'S',
        SYS_IRN: 'I'
    }
    return syschar.get(sys, ' ')

def char2sys(c):
    """Convert character to system ID"""
    charmap = {
        'G': SYS_GPS,
        'R': SYS_GLO,
        'E': SYS_GAL,
        'C': SYS_BDS,
        'J': SYS_QZS,
        'S': SYS_SBS,
        'I': SYS_IRN
    }
    return charmap.get(c.upper(), 0)
