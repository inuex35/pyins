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

"""BeiDou satellite utilities"""

from ..core.constants import SYS_BDS, sat2prn, sat2sys


def get_beidou_orbit_type(sat):
    """
    Determine BeiDou satellite orbit type from satellite number

    Parameters
    ----------
    sat : int
        Satellite number

    Returns
    -------
    str
        'GEO', 'IGSO', 'MEO', or None if not BeiDou
    """
    if sat2sys(sat) != SYS_BDS:
        return None

    prn = sat2prn(sat)

    # BeiDou-2 and BeiDou-3 classification
    # GEO: C01-C05, C59-C63 (Geostationary Earth Orbit)
    # IGSO: C06-C18, C31-C40, C56-C58 (Inclined Geosynchronous Orbit)
    # MEO: C19-C30, C41-C46 (Medium Earth Orbit)

    if prn <= 5 or (59 <= prn <= 63):
        return 'GEO'
    elif (6 <= prn <= 18) or (31 <= prn <= 40) or (56 <= prn <= 58):
        return 'IGSO'
    elif (19 <= prn <= 30) or (41 <= prn <= 46):
        return 'MEO'
    else:
        # Future satellites
        return 'MEO'  # Default to MEO for unknown PRNs


def get_expected_range(orbit_type):
    """
    Get expected range for different orbit types

    Parameters
    ----------
    orbit_type : str
        'GEO', 'IGSO', or 'MEO'

    Returns
    -------
    tuple
        (min_range, max_range) in meters
    """
    if orbit_type == 'GEO':
        # GEO satellites at ~35,786 km altitude
        # Range typically 36,000-42,000 km depending on elevation
        return (35000e3, 43000e3)
    elif orbit_type == 'IGSO':
        # IGSO satellites at ~35,786 km altitude
        # Range typically 36,000-42,000 km
        return (35000e3, 43000e3)
    elif orbit_type == 'MEO':
        # MEO satellites at ~21,500 km altitude
        # Range typically 20,000-27,000 km
        return (19000e3, 28000e3)
    else:
        # Default range for all satellites
        return (19000e3, 43000e3)


def is_beidou_geo(sat):
    """Check if satellite is BeiDou GEO"""
    return get_beidou_orbit_type(sat) == 'GEO'


def is_beidou_igso(sat):
    """Check if satellite is BeiDou IGSO"""
    return get_beidou_orbit_type(sat) == 'IGSO'


def is_beidou_meo(sat):
    """Check if satellite is BeiDou MEO"""
    return get_beidou_orbit_type(sat) == 'MEO'


def needs_special_transformation(sat):
    """
    Check if BeiDou satellite needs special coordinate transformation

    GEO and IGSO satellites need special handling due to their
    orbit characteristics and the BeiDou coordinate system

    Parameters
    ----------
    sat : int
        Satellite number

    Returns
    -------
    bool
        True if special transformation is needed
    """
    orbit_type = get_beidou_orbit_type(sat)
    return orbit_type in ['GEO', 'IGSO']
