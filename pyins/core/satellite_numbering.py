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

"""Unified satellite numbering system for pyins.

This module provides consistent satellite numbering across all GNSS constellations,
converting between PRN numbers and internal satellite numbers. The unified numbering
system simplifies satellite identification and avoids conflicts between different
constellation numbering schemes.

The satellite number ranges are:
- GPS (G): 1-32
- SBAS (S): 33-64, 133-140
- GLONASS (R): 65-88
- Galileo (E): 97-132
- BeiDou (C): 141-203
- QZSS (J): 210-216
- IRNSS (I): 230-243
"""

# Define system IDs (duplicated from constants.py to avoid circular import)
SYS_NONE = 0x00
SYS_GPS = 0x01
SYS_GLO = 0x02
SYS_GAL = 0x04
SYS_BDS = 0x08
SYS_QZS = 0x10
SYS_SBS = 0x20
SYS_IRN = 0x40

# Define satellite number ranges for each system
SATELLITE_RANGES = {
    SYS_GPS: [(1, 32)],                    # GPS: 1-32
    SYS_SBS: [(33, 64), (133, 140)],      # SBAS: 33-64, 133-140
    SYS_GLO: [(65, 88)],                   # GLONASS: 65-88
    SYS_GAL: [(97, 132)],                  # Galileo: 97-132
    SYS_BDS: [(141, 203)],                 # BeiDou: 141-203 (BDS-2: 141-177, BDS-3: 178-203)
    SYS_QZS: [(210, 216)],                 # QZSS: 210-216
    SYS_IRN: [(230, 243)],                 # IRNSS: 230-243
}

# System ID to character mapping
SYS_TO_CHAR = {
    SYS_GPS: 'G',
    SYS_GLO: 'R',
    SYS_GAL: 'E',
    SYS_BDS: 'C',
    SYS_QZS: 'J',
    SYS_SBS: 'S',
    SYS_IRN: 'I',
}

# Character to system ID mapping
CHAR_TO_SYS = {v: k for k, v in SYS_TO_CHAR.items()}

def prn_to_sat(system_char, prn):
    """Convert system character and PRN to internal satellite number.

    Transforms constellation-specific PRN numbers into a unified internal satellite
    numbering scheme that avoids conflicts between different GNSS systems.

    Parameters
    ----------
    system_char : str
        Single character system identifier:
        - 'G': GPS
        - 'R': GLONASS
        - 'E': Galileo
        - 'C': BeiDou
        - 'J': QZSS
        - 'S': SBAS
        - 'I': IRNSS/NavIC
    prn : int
        PRN number within the constellation (constellation-specific range)

    Returns
    -------
    int
        Internal satellite number (1-255), or 0 if invalid PRN or system

    Notes
    -----
    The internal satellite number ranges are:
    - GPS (G): PRN 1-32 → sat 1-32
    - SBAS (S): PRN 120-151 → sat 33-64, PRN 152-159 → sat 133-140
    - GLONASS (R): PRN 1-24 → sat 65-88
    - Galileo (E): PRN 1-36 → sat 97-132
    - BeiDou (C): PRN 1-63 → sat 141-203 (BDS-2: 1-37, BDS-3: 38-63)
    - QZSS (J): PRN 1-7 → sat 210-216
    - IRNSS (I): PRN 1-14 → sat 230-243

    Examples
    --------
    >>> prn_to_sat('G', 1)  # GPS PRN 1
    1
    >>> prn_to_sat('R', 1)  # GLONASS PRN 1
    65
    >>> prn_to_sat('E', 1)  # Galileo PRN 1
    97
    >>> prn_to_sat('X', 1)  # Invalid system
    0
    """
    if system_char == 'G':  # GPS
        if 1 <= prn <= 32:
            return prn
    elif system_char == 'R':  # GLONASS
        if 1 <= prn <= 24:
            return prn + 64
    elif system_char == 'E':  # Galileo
        if 1 <= prn <= 36:
            return prn + 96
    elif system_char == 'C':  # BeiDou
        if 1 <= prn <= 37:
            return prn + 140  # BeiDou-2
        elif 38 <= prn <= 63:
            return prn + 140  # BeiDou-3 (178-203)
    elif system_char == 'J':  # QZSS
        if 1 <= prn <= 7:
            return prn + 209  # J01-J07 -> 210-216
    elif system_char == 'S':  # SBAS
        if 120 <= prn <= 151:
            return prn - 87  # SBAS 120-151 -> 33-64
        elif 152 <= prn <= 159:
            return prn - 19  # SBAS 152-159 -> 133-140
    elif system_char == 'I':  # IRNSS
        if 1 <= prn <= 14:
            return prn + 200

    return 0  # Invalid


def sat_to_prn(sat):
    """Convert internal satellite number to constellation-specific PRN.

    Transforms unified internal satellite numbers back to the original PRN
    numbers used within each GNSS constellation.

    Parameters
    ----------
    sat : int
        Internal satellite number (1-255)

    Returns
    -------
    int
        PRN number within the constellation, or 0 if invalid satellite number

    Notes
    -----
    The conversion follows these mappings:
    - GPS: sat 1-32 → PRN 1-32
    - SBAS: sat 33-64 → PRN 120-151, sat 133-140 → PRN 152-159
    - GLONASS: sat 65-88 → PRN 1-24
    - Galileo: sat 97-132 → PRN 1-36
    - BeiDou: sat 141-203 → PRN 1-63
    - QZSS: sat 210-216 → PRN 1-7
    - IRNSS: sat 230-243 → PRN 1-14

    Examples
    --------
    >>> sat_to_prn(1)    # GPS satellite 1
    1
    >>> sat_to_prn(65)   # GLONASS satellite 65
    1
    >>> sat_to_prn(97)   # Galileo satellite 97
    1
    >>> sat_to_prn(999)  # Invalid satellite
    0
    """
    if sat <= 0 or sat > 255:
        return 0
    elif 1 <= sat <= 32:  # GPS
        return sat
    elif 33 <= sat <= 64:  # SBAS
        return sat - 33 + 120
    elif 65 <= sat <= 88:  # GLONASS
        return sat - 64
    elif 97 <= sat <= 132:  # Galileo
        return sat - 96
    elif 133 <= sat <= 140:  # SBAS
        return sat - 133 + 152
    elif 141 <= sat <= 203:  # BeiDou
        return sat - 140
    elif 210 <= sat <= 216:  # QZSS
        return sat - 209
    elif 230 <= sat <= 243:  # IRNSS
        return sat - 229
    else:
        return 0
