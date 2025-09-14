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

"""GNSS frequency management functions.

This module provides functions for managing GNSS carrier frequencies and wavelengths
across different satellite constellations. It supports all major GNSS systems and
handles constellation-specific frequency plans, including GLONASS FDMA channels.

The module supports the following GNSS constellations:
- GPS (Global Positioning System): L1, L2, L5 frequencies
- GLONASS: G1, G2 with Frequency Division Multiple Access (FDMA)
- Galileo: E1, E5a, E5b frequencies
- BeiDou: B1I, B2a, B3 frequencies
- QZSS (Quasi-Zenith Satellite System): J1, J2, J5 frequencies
- SBAS (Satellite-Based Augmentation Systems): S1, S5 frequencies
- IRNSS (Indian Regional Navigation Satellite System): I5, IS frequencies

Functions:
    sat2freq: Get carrier frequency for a specific satellite and frequency index
    sat2wavelength: Get carrier wavelength for a specific satellite and frequency index

Notes:
    - Frequency indices: 0=L1/B1I/E1, 1=L2/B2I/E5b, 2=L5/B2a/E5a
    - GLONASS uses FDMA with frequency channel numbers from -7 to +6
    - All frequencies are in Hz, wavelengths in meters
    - Uses speed of light constant CLIGHT for wavelength calculations
"""

from ..core.constants import *
from ..core.constants import sat2sys


def sat2freq(sat, frq_idx=0, glo_fcn=0):
    """Get carrier frequency for a specific satellite and frequency index.

    Determines the carrier frequency based on the satellite constellation and
    frequency index. Handles constellation-specific frequency plans including
    GLONASS FDMA channels which require a frequency channel number.

    Parameters
    ----------
    sat : int
        Satellite number (PRN) identifying the satellite. The constellation is
        determined from the satellite number:
        - GPS: 1-32
        - GLONASS: 65-96 (internal numbering)
        - Galileo: 301-336 (internal numbering)
        - BeiDou: 401-463 (internal numbering)
        - QZSS: 193-202 (internal numbering)
        - SBAS: 120-158 (internal numbering)
        - IRNSS: 401-414 (internal numbering, overlaps with BeiDou)
    frq_idx : int, optional
        Frequency index specifying which frequency band to use:
        - 0: Primary frequency (L1/G1/E1/B1I/J1/S1/I5)
        - 1: Secondary frequency (L2/G2/E5b/B3/J2/IS)
        - 2: Tertiary frequency (L5/E5a/B2a/J5/S5)
        Default is 0.
    glo_fcn : int, optional
        GLONASS frequency channel number for FDMA channels, ranging from -7 to +6.
        Only used for GLONASS satellites. Default is 0.

    Returns
    -------
    float
        Carrier frequency in Hz. Returns 0.0 if the constellation or frequency
        index is not supported or invalid.

    Notes
    -----
    Frequency mappings by constellation:

    GPS:
        - L1: 1575.42 MHz
        - L2: 1227.60 MHz
        - L5: 1176.45 MHz

    GLONASS (FDMA):
        - G1: 1602 MHz + fcn * 0.5625 MHz
        - G2: 1246 MHz + fcn * 0.4375 MHz

    Galileo:
        - E1: 1575.42 MHz
        - E5a: 1176.45 MHz
        - E5b: 1207.14 MHz

    BeiDou:
        - B1I: 1561.098 MHz
        - B2a: 1176.45 MHz
        - B3: 1268.52 MHz

    Examples
    --------
    >>> # GPS L1 frequency
    >>> freq = sat2freq(sat=1, frq_idx=0)
    >>> print(f"GPS L1: {freq/1e6:.2f} MHz")

    >>> # GLONASS G1 with frequency channel +1
    >>> freq = sat2freq(sat=65, frq_idx=0, glo_fcn=1)
    >>> print(f"GLONASS G1+1: {freq/1e6:.4f} MHz")

    >>> # Galileo E5a frequency
    >>> freq = sat2freq(sat=301, frq_idx=2)
    >>> print(f"Galileo E5a: {freq/1e6:.2f} MHz")
    """
    sys = sat2sys(sat)

    if sys == SYS_GPS:
        if frq_idx == 0:
            return FREQ_L1
        elif frq_idx == 1:
            return FREQ_L2
        elif frq_idx == 2:
            return FREQ_L5

    elif sys == SYS_GLO:
        if frq_idx == 0:
            return FREQ_G1 + glo_fcn * DFREQ_G1
        elif frq_idx == 1:
            return FREQ_G2 + glo_fcn * DFREQ_G2

    elif sys == SYS_GAL:
        if frq_idx == 0:
            return FREQ_E1
        elif frq_idx == 1:
            return FREQ_E5b
        elif frq_idx == 2:
            return FREQ_E5a

    elif sys == SYS_BDS:
        if frq_idx == 0:
            return FREQ_B1I  # B1I for BeiDou MEO/IGSO
        elif frq_idx == 1:
            return FREQ_B3   # B3 for BeiDou
        elif frq_idx == 2:
            return FREQ_B2a  # B2a

    elif sys == SYS_QZS:
        if frq_idx == 0:
            return FREQ_J1
        elif frq_idx == 1:
            return FREQ_J2
        elif frq_idx == 2:
            return FREQ_J5

    elif sys == SYS_SBS:
        if frq_idx == 0:
            return FREQ_S1
        elif frq_idx == 2:
            return FREQ_S5

    elif sys == SYS_IRN:
        if frq_idx == 0:
            return FREQ_I5
        elif frq_idx == 1:
            return FREQ_IS

    return 0.0


def sat2wavelength(sat, frq_idx=0, glo_fcn=0):
    """Get carrier wavelength for a specific satellite and frequency index.

    Computes the carrier wavelength by first determining the carrier frequency
    and then applying the relationship λ = c/f, where c is the speed of light
    and f is the frequency. The wavelength is fundamental for carrier phase
    measurements and ambiguity resolution in GNSS processing.

    Parameters
    ----------
    sat : int
        Satellite number (PRN) identifying the satellite. The constellation is
        determined from the satellite number using the same mapping as sat2freq.
    frq_idx : int, optional
        Frequency index specifying which frequency band to use:
        - 0: Primary frequency (L1/G1/E1/B1I/J1/S1/I5)
        - 1: Secondary frequency (L2/G2/E5b/B3/J2/IS)
        - 2: Tertiary frequency (L5/E5a/B2a/J5/S5)
        Default is 0.
    glo_fcn : int, optional
        GLONASS frequency channel number for FDMA channels, ranging from -7 to +6.
        Only used for GLONASS satellites. Default is 0.

    Returns
    -------
    float
        Carrier wavelength in meters. Returns 0.0 if the frequency is invalid
        or the constellation/frequency index is not supported.

    Notes
    -----
    The wavelength calculation uses the relationship:
        λ = c/f
    where:
    - λ is the wavelength in meters
    - c is the speed of light (299,792,458 m/s)
    - f is the frequency in Hz

    Typical wavelengths:
    - GPS L1: ~0.19 m
    - GPS L2: ~0.244 m
    - GPS L5: ~0.255 m
    - GLONASS G1: ~0.187 m (varies with frequency channel)
    - GLONASS G2: ~0.241 m (varies with frequency channel)

    Examples
    --------
    >>> # GPS L1 wavelength
    >>> wavelength = sat2wavelength(sat=1, frq_idx=0)
    >>> print(f"GPS L1 wavelength: {wavelength:.4f} m")

    >>> # GLONASS G1 wavelength with frequency channel +1
    >>> wavelength = sat2wavelength(sat=65, frq_idx=0, glo_fcn=1)
    >>> print(f"GLONASS G1+1 wavelength: {wavelength:.4f} m")

    >>> # Compare wavelengths for dual-frequency processing
    >>> wl_l1 = sat2wavelength(sat=1, frq_idx=0)
    >>> wl_l2 = sat2wavelength(sat=1, frq_idx=1)
    >>> print(f"GPS L1/L2 wavelength ratio: {wl_l2/wl_l1:.4f}")

    See Also
    --------
    sat2freq : Get carrier frequency for a satellite
    """
    freq = sat2freq(sat, frq_idx, glo_fcn)
    if freq > 0:
        return CLIGHT / freq
    return 0.0

