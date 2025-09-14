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

"""Satellite utilities for GNSS processing"""

from enum import Enum
from typing import Optional

import numpy as np


class SatelliteSystem(Enum):
    """Satellite system enumeration"""
    GPS = "GPS"
    GLONASS = "GLONASS"
    GALILEO = "Galileo"
    BEIDOU = "BeiDou"
    QZSS = "QZSS"
    SBAS = "SBAS"
    UNKNOWN = "Unknown"


def get_satellite_system(sat_num: int) -> SatelliteSystem:
    """
    Get satellite system from satellite number (PRN)

    Parameters
    ----------
    sat_num : int
        Satellite PRN number

    Returns
    -------
    SatelliteSystem
        Satellite system

    Notes
    -----
    PRN ranges based on RINEX standards:
    - GPS: 1-32
    - GLONASS: 38-61, 65-96
    - Galileo: 71-106, 201-236
    - BeiDou: 87-172, 401-437
    - QZSS: 193-202
    - SBAS: 120-158
    """
    if 1 <= sat_num <= 32:
        return SatelliteSystem.GPS
    elif 38 <= sat_num <= 61 or 65 <= sat_num <= 96:
        return SatelliteSystem.GLONASS
    elif 71 <= sat_num <= 106 or 201 <= sat_num <= 236:
        return SatelliteSystem.GALILEO
    elif 87 <= sat_num <= 172 or 401 <= sat_num <= 437:
        return SatelliteSystem.BEIDOU
    elif 193 <= sat_num <= 202:
        return SatelliteSystem.QZSS
    elif 120 <= sat_num <= 158:
        return SatelliteSystem.SBAS
    else:
        return SatelliteSystem.UNKNOWN


def get_satellite_system_str(sat_num: int) -> str:
    """
    Get satellite system as string from satellite number (PRN)

    Parameters
    ----------
    sat_num : int
        Satellite PRN number

    Returns
    -------
    str
        Satellite system name or "Unknown(N)" where N is the satellite number
    """
    system = get_satellite_system(sat_num)
    if system == SatelliteSystem.UNKNOWN:
        return f"Unknown({sat_num})"
    return system.value


def count_satellites_by_system(sat_list: list[int]) -> dict[str, int]:
    """
    Count satellites by system

    Parameters
    ----------
    sat_list : List[int]
        List of satellite PRN numbers

    Returns
    -------
    Dict[str, int]
        Dictionary with system names as keys and counts as values
    """
    counts = {}
    for sat_num in sat_list:
        system = get_satellite_system_str(sat_num)
        counts[system] = counts.get(system, 0) + 1
    return counts


def get_system_frequency_info(system: SatelliteSystem) -> dict[str, float]:
    """
    Get frequency information for a satellite system

    Parameters
    ----------
    system : SatelliteSystem
        Satellite system

    Returns
    -------
    Dict[str, float]
        Dictionary with frequency band names as keys and frequencies in Hz as values
    """
    frequency_info = {
        SatelliteSystem.GPS: {
            "L1": 1575.42e6,
            "L2": 1227.60e6,
            "L5": 1176.45e6
        },
        SatelliteSystem.GLONASS: {
            "L1": 1602.0e6,  # Base frequency, actual varies by satellite
            "L2": 1246.0e6   # Base frequency, actual varies by satellite
        },
        SatelliteSystem.GALILEO: {
            "E1": 1575.42e6,
            "E5a": 1176.45e6,
            "E5b": 1207.14e6,
            "E6": 1278.75e6
        },
        SatelliteSystem.BEIDOU: {
            "B1": 1561.098e6,
            "B2": 1207.14e6,
            "B3": 1268.52e6
        },
        SatelliteSystem.QZSS: {
            "L1": 1575.42e6,
            "L2": 1227.60e6,
            "L5": 1176.45e6
        }
    }

    return frequency_info.get(system, {})


def get_wavelength(sat_num: int, frequency_band: str = "L1") -> float:
    """
    Get signal wavelength for a satellite

    Parameters
    ----------
    sat_num : int
        Satellite PRN number
    frequency_band : str, optional
        Frequency band (default: "L1")

    Returns
    -------
    float
        Wavelength in meters, or 0.0 if unknown
    """
    system = get_satellite_system(sat_num)
    freq_info = get_system_frequency_info(system)

    # Map common frequency band names
    band_mapping = {
        "L1": ["L1", "E1", "B1"],
        "L2": ["L2", "E5a", "B2"],
        "L5": ["L5", "E5b", "B3"]
    }

    # Try to find frequency
    frequency = None
    if frequency_band in freq_info:
        frequency = freq_info[frequency_band]
    else:
        # Try mapped bands
        for mapped_bands in band_mapping.get(frequency_band, []):
            if mapped_bands in freq_info:
                frequency = freq_info[mapped_bands]
                break

    if frequency is None:
        return 0.0

    # Speed of light in m/s
    c = 299792458.0
    return c / frequency


def validate_satellite_number(sat_num: int) -> bool:
    """
    Validate if satellite number is in valid range

    Parameters
    ----------
    sat_num : int
        Satellite PRN number

    Returns
    -------
    bool
        True if valid, False otherwise
    """
    return 1 <= sat_num <= 500  # Conservative upper bound


def get_system_constellation_size(system: SatelliteSystem) -> int:
    """
    Get typical constellation size for a satellite system

    Parameters
    ----------
    system : SatelliteSystem
        Satellite system

    Returns
    -------
    int
        Typical number of satellites in constellation
    """
    constellation_sizes = {
        SatelliteSystem.GPS: 32,
        SatelliteSystem.GLONASS: 24,
        SatelliteSystem.GALILEO: 30,
        SatelliteSystem.BEIDOU: 35,
        SatelliteSystem.QZSS: 7,
        SatelliteSystem.SBAS: 10
    }

    return constellation_sizes.get(system, 0)


class ProcessingOptions:
    """
    Processing options for GNSS data processing.

    This class manages configuration settings for GNSS processing, including
    satellite exclusion/inclusion and navigation system selection.

    Attributes
    ----------
    exsats : np.ndarray
        Excluded satellites array of length 500. Values:
        - 0: Auto selection (default)
        - 1: Force exclude satellite
        - 2: Force include satellite
    navsys : int
        Navigation systems bitmask. Default 0x1F enables GPS|GLO|GAL|BDS|QZS.
        Bitmask values:
        - 0x01: GPS
        - 0x02: GLONASS
        - 0x04: Galileo
        - 0x08: BeiDou
        - 0x10: QZSS
        - 0x20: SBAS

    Examples
    --------
    >>> opt = ProcessingOptions()
    >>> opt.set_excluded_satellite(15, exclude=True)  # Exclude GPS satellite 15
    >>> opt.set_navigation_systems(gps=True, glonass=False, galileo=True)
    """
    def __init__(self):
        """
        Initialize processing options with default settings.

        Notes
        -----
        Default settings enable all major GNSS systems (GPS, GLONASS, Galileo,
        BeiDou, QZSS) with automatic satellite selection for all satellites.
        """
        self.exsats = np.zeros(500, dtype=int)  # Excluded satellites array (0: auto, 1: exclude, 2: include)
        self.navsys = 0x1F  # Navigation systems (bit mask: GPS|GLO|GAL|BDS|QZS)

    def set_excluded_satellite(self, sat_num: int, exclude: bool = True):
        """
        Set satellite exclusion status.

        Parameters
        ----------
        sat_num : int
            Satellite PRN number (1-500)
        exclude : bool, optional
            True to exclude satellite, False to force include (default: True)

        Notes
        -----
        This method overrides automatic satellite selection:
        - exclude=True: Satellite will be excluded from processing
        - exclude=False: Satellite will be forced to be included
        - Only satellites with valid PRN numbers (1-500) are affected

        Examples
        --------
        >>> opt = ProcessingOptions()
        >>> opt.set_excluded_satellite(15, exclude=True)   # Exclude GPS satellite 15
        >>> opt.set_excluded_satellite(22, exclude=False)  # Force include GPS satellite 22
        """
        if 1 <= sat_num <= 500:
            self.exsats[sat_num - 1] = 1 if exclude else 2

    def set_navigation_systems(self, gps=True, glonass=True, galileo=True, beidou=True, qzss=True, sbas=False):
        """
        Set navigation systems to use in processing.

        Parameters
        ----------
        gps : bool, optional
            Enable GPS system (default: True)
        glonass : bool, optional
            Enable GLONASS system (default: True)
        galileo : bool, optional
            Enable Galileo system (default: True)
        beidou : bool, optional
            Enable BeiDou system (default: True)
        qzss : bool, optional
            Enable QZSS system (default: True)
        sbas : bool, optional
            Enable SBAS system (default: False)

        Notes
        -----
        This method sets the navigation systems bitmask (navsys) by combining
        individual system flags. Disabled systems will have their satellites
        automatically excluded from processing.

        The bitmask values are:
        - GPS: 0x01
        - GLONASS: 0x02
        - Galileo: 0x04
        - BeiDou: 0x08
        - QZSS: 0x10
        - SBAS: 0x20

        Examples
        --------
        >>> opt = ProcessingOptions()
        >>> opt.set_navigation_systems(gps=True, glonass=False, galileo=True)
        >>> # Only GPS and Galileo will be used

        >>> opt.set_navigation_systems(sbas=True)  # Enable SBAS in addition to defaults
        """
        self.navsys = 0
        if gps: self.navsys |= 0x01       # SYS_GPS
        if glonass: self.navsys |= 0x02   # SYS_GLO
        if galileo: self.navsys |= 0x04   # SYS_GAL
        if beidou: self.navsys |= 0x08    # SYS_BDS
        if qzss: self.navsys |= 0x10      # SYS_QZS
        if sbas: self.navsys |= 0x20      # SYS_SBS


def get_system_bitmask(system: SatelliteSystem) -> int:
    """
    Get system bitmask for navigation system selection.

    Parameters
    ----------
    system : SatelliteSystem
        Satellite system enum value

    Returns
    -------
    int
        Bitmask value for the satellite system, 0 if unknown

    Notes
    -----
    Bitmask values correspond to RTKLIB system definitions:
    - GPS: 0x01 (SYS_GPS)
    - GLONASS: 0x02 (SYS_GLO)
    - Galileo: 0x04 (SYS_GAL)
    - BeiDou: 0x08 (SYS_BDS)
    - QZSS: 0x10 (SYS_QZS)
    - SBAS: 0x20 (SYS_SBS)

    Examples
    --------
    >>> mask = get_system_bitmask(SatelliteSystem.GPS)
    >>> print(f"GPS bitmask: 0x{mask:02X}")
    GPS bitmask: 0x01

    >>> combined = get_system_bitmask(SatelliteSystem.GPS) | get_system_bitmask(SatelliteSystem.GALILEO)
    >>> print(f"GPS+Galileo: 0x{combined:02X}")
    GPS+Galileo: 0x05
    """
    system_masks = {
        SatelliteSystem.GPS: 0x01,
        SatelliteSystem.GLONASS: 0x02,
        SatelliteSystem.GALILEO: 0x04,
        SatelliteSystem.BEIDOU: 0x08,
        SatelliteSystem.QZSS: 0x10,
        SatelliteSystem.SBAS: 0x20
    }
    return system_masks.get(system, 0)


def satexclude(sat: int, svh: int, opt: Optional[ProcessingOptions] = None) -> bool:
    """
    Test if satellite should be excluded

    Parameters
    ----------
    sat : int
        Satellite number (PRN)
    svh : int
        SV health flag (0: healthy, >0: unhealthy, <0: no ephemeris)
    opt : ProcessingOptions, optional
        Processing options containing exclusion settings

    Returns
    -------
    bool
        True if satellite should be excluded, False otherwise

    Notes
    -----
    This function implements the same logic as RTKLIB's satexclude:
    - Excludes satellites with no ephemeris (svh < 0)
    - Respects manual exclusion/inclusion settings in opt.exsats
    - Excludes satellites from unselected navigation systems
    - Checks satellite health (unhealthy if svh > 0)
    - Special handling for QZSS LEX health flag
    """
    # Check if ephemeris is available
    if svh < 0:
        return True  # Ephemeris unavailable

    # Get satellite system
    sys = get_satellite_system(sat)
    sys_mask = get_system_bitmask(sys)

    if opt is not None:
        # Check manual exclusion settings
        if 1 <= sat <= 500:
            if opt.exsats[sat - 1] == 1:
                return True  # Manually excluded
            if opt.exsats[sat - 1] == 2:
                return False  # Manually included

        # Check if satellite system is selected
        if not (sys_mask & opt.navsys):
            return True  # System not selected

    # Special handling for QZSS LEX health
    if sys == SatelliteSystem.QZSS:
        svh &= 0xFE  # Mask QZSS LEX health bit

    # Check satellite health
    if svh:
        # Satellite is unhealthy
        return True

    return False  # Satellite is not excluded
