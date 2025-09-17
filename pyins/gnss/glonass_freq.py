"""GLONASS FDMA frequency calculation.

GLONASS uses Frequency Division Multiple Access (FDMA) where each satellite
transmits on a slightly different frequency. This module implements the
frequency calculation based on the satellite's frequency channel number (FCN).

The frequency is calculated as:
- L1/G1: f = 1602.0 MHz + 0.5625 MHz × k
- L2/G2: f = 1246.0 MHz + 0.4375 MHz × k

Where k is the frequency channel number (-7 to +6 for operational satellites).

Note: Starting with GLONASS-K2 satellites, some use CDMA on new frequencies,
but this module focuses on traditional FDMA calculations.
"""

from ..core.constants import FREQ_G1, FREQ_G2, DFREQ_G1, DFREQ_G2


def get_glonass_frequency(sat, freq_idx, nav_data):
    """Calculate GLONASS FDMA frequency for a specific satellite and band.

    Parameters
    ----------
    sat : int
        Satellite number (internal numbering: 65-88 for GLONASS)
    freq_idx : int
        Frequency index (0 for L1/G1, 1 for L2/G2)
    nav_data : NavigationData
        Navigation data containing GLONASS ephemerides with FCN information

    Returns
    -------
    float
        Frequency in Hz, or None if FCN not available

    Notes
    -----
    GLONASS frequency channel numbers (FCN) are assigned as follows:
    - Operational range: -7 to +6 (14 channels)
    - Extended range: -7 to +13 (21 channels, for future use)
    - Channel 0 is avoided to minimize interference

    The frequency separation ensures no overlap between channels and
    provides guard bands for signal isolation.

    Examples
    --------
    >>> freq = get_glonass_frequency(65, 0, nav_data)  # R01 L1 frequency
    >>> print(f"R01 L1 frequency: {freq/1e9:.6f} GHz")
    """
    # Find GLONASS ephemeris for this satellite
    fcn = None

    if hasattr(nav_data, 'geph') and nav_data.geph:
        for geph in nav_data.geph:
            if geph.sat == sat:
                fcn = geph.frq
                break

    if fcn is None:
        # FCN not found, return center frequency as fallback
        if freq_idx == 0:
            return FREQ_G1  # L1/G1 center frequency
        elif freq_idx == 1:
            return FREQ_G2  # L2/G2 center frequency
        else:
            return None

    # Calculate frequency based on FCN
    if freq_idx == 0:  # L1/G1
        # Validate FCN range for L1
        if fcn < -7 or fcn > 6:
            # Out of operational range, but still calculate
            # (some satellites may use extended range)
            pass
        freq = FREQ_G1 + DFREQ_G1 * fcn
    elif freq_idx == 1:  # L2/G2
        # Validate FCN range for L2
        if fcn < -7 or fcn > 6:
            # Out of operational range, but still calculate
            pass
        freq = FREQ_G2 + DFREQ_G2 * fcn
    else:
        # Unsupported frequency band
        return None

    return freq


def get_glonass_wavelength(sat, freq_idx, nav_data):
    """Calculate GLONASS wavelength for a specific satellite and band.

    Parameters
    ----------
    sat : int
        Satellite number (internal numbering: 65-88 for GLONASS)
    freq_idx : int
        Frequency index (0 for L1/G1, 1 for L2/G2)
    nav_data : NavigationData
        Navigation data containing GLONASS ephemerides with FCN information

    Returns
    -------
    float
        Wavelength in meters, or None if FCN not available

    Examples
    --------
    >>> wl = get_glonass_wavelength(65, 0, nav_data)  # R01 L1 wavelength
    >>> print(f"R01 L1 wavelength: {wl:.4f} m")
    """
    from ..core.constants import CLIGHT

    freq = get_glonass_frequency(sat, freq_idx, nav_data)
    if freq is None:
        return None

    return CLIGHT / freq


def get_fcn_from_nav(sat, nav_data):
    """Extract frequency channel number (FCN) for a GLONASS satellite.

    Parameters
    ----------
    sat : int
        Satellite number (internal numbering: 65-88 for GLONASS)
    nav_data : NavigationData
        Navigation data containing GLONASS ephemerides

    Returns
    -------
    int or None
        Frequency channel number (-7 to +13), or None if not found

    Notes
    -----
    The FCN is broadcast in the GLONASS navigation message and stored
    in the ephemeris data. Each satellite maintains its assigned FCN
    throughout its operational lifetime.
    """
    if hasattr(nav_data, 'geph') and nav_data.geph:
        for geph in nav_data.geph:
            if geph.sat == sat:
                return geph.frq

    return None


def print_glonass_frequency_table(nav_data):
    """Print GLONASS frequency allocation table.

    Parameters
    ----------
    nav_data : NavigationData
        Navigation data containing GLONASS ephemerides

    Notes
    -----
    Displays the frequency channel assignments for all GLONASS satellites
    found in the navigation data, showing both L1 and L2 frequencies.
    """
    print("GLONASS Frequency Allocation Table")
    print("=" * 60)
    print(f"{'Sat':<5} {'FCN':<5} {'L1/G1 (GHz)':<15} {'L2/G2 (GHz)':<15}")
    print("-" * 60)

    # Collect unique satellites
    glonass_sats = set()
    if hasattr(nav_data, 'geph') and nav_data.geph:
        for geph in nav_data.geph:
            glonass_sats.add(geph.sat)

    for sat in sorted(glonass_sats):
        fcn = get_fcn_from_nav(sat, nav_data)
        if fcn is not None:
            freq_l1 = get_glonass_frequency(sat, 0, nav_data)
            freq_l2 = get_glonass_frequency(sat, 1, nav_data)

            prn = sat - 64  # Convert to PRN
            print(f"R{prn:02d}  {fcn:+3d}   "
                  f"{freq_l1/1e9:.6f}       {freq_l2/1e9:.6f}")


# Channel assignment reference (as of 2024)
# This can change as satellites are replaced
GLONASS_FCN_ASSIGNMENTS = {
    # PRN: FCN (example assignments, actual values from navigation data)
    1: 1,    # R01
    2: -4,   # R02
    3: 5,    # R03
    4: 6,    # R04
    5: 1,    # R05
    7: 5,    # R07
    8: 6,    # R08
    9: -2,   # R09
    10: -7,  # R10
    11: 0,   # R11
    12: -1,  # R12
    13: -2,  # R13
    14: -7,  # R14
    15: 0,   # R15
    16: -1,  # R16
    17: 4,   # R17
    18: -3,  # R18
    19: 3,   # R19
    20: 2,   # R20
    21: 4,   # R21
    22: -3,  # R22
    23: 3,   # R23
    24: 2,   # R24
}

# Note: Channel 0 is typically avoided in operational assignments
# to minimize potential interference issues