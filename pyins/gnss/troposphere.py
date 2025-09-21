"""Tropospheric delay models for GNSS.

This module implements tropospheric delay models to correct for the
signal delay caused by the neutral atmosphere (troposphere).
"""

import numpy as np


def saastamoinen_model(elevation_deg, altitude_m=0, latitude_deg=45.0):
    """Saastamoinen tropospheric delay model.

    Computes the tropospheric delay using the Saastamoinen model,
    which is widely used in GNSS processing for its simplicity and accuracy.

    Parameters
    ----------
    elevation_deg : float or array_like
        Satellite elevation angle in degrees
    altitude_m : float, optional
        Receiver altitude in meters (default: 0)
    latitude_deg : float, optional
        Receiver latitude in degrees (default: 45)

    Returns
    -------
    float or array_like
        Tropospheric delay in meters

    Notes
    -----
    The model computes both hydrostatic (dry) and wet components of the
    tropospheric delay. For elevations below 5 degrees, the delay is
    set to 0 to avoid numerical issues.

    References
    ----------
    Saastamoinen, J. (1972), "Atmospheric correction for the troposphere
    and stratosphere in radio ranging of satellites"
    """
    # Convert to radians
    el_rad = np.radians(elevation_deg)
    lat_rad = np.radians(latitude_deg)

    # Skip very low elevations
    if np.any(elevation_deg < 5.0):
        if np.isscalar(elevation_deg):
            return 0.0
        else:
            delays = np.zeros_like(elevation_deg)
            mask = elevation_deg >= 5.0
            if np.any(mask):
                delays[mask] = saastamoinen_model(
                    elevation_deg[mask], altitude_m, latitude_deg
                )
            return delays

    # Standard atmosphere parameters at sea level
    P0 = 1013.25  # Pressure in mbar
    T0 = 288.15   # Temperature in Kelvin
    e0 = 11.691   # Water vapor pressure in mbar (50% humidity)

    # Adjust for altitude using standard atmosphere model
    P = P0 * (1 - 0.0000226 * altitude_m) ** 5.225
    T = T0 - 0.0065 * altitude_m
    e = e0 * (1 - 0.0000226 * altitude_m) ** 5.225

    # Zenith hydrostatic delay
    zhd = 0.0022768 * P / (1 - 0.00266 * np.cos(2 * lat_rad) - 0.00028 * altitude_m/1000)

    # Zenith wet delay
    zwd = 0.0022768 * (1255/T + 0.05) * e / (1 - 0.00266 * np.cos(2 * lat_rad) - 0.00028 * altitude_m/1000)

    # Simple mapping function (1/sin(el))
    # For more accuracy, could use Niell or VMF mapping functions
    mapping = 1.0 / np.sin(el_rad)

    # Total delay
    delay = (zhd + zwd) * mapping

    return delay


def troposphere_correction(elevation_deg, pos_llh, model='saastamoinen'):
    """Compute tropospheric delay correction.

    Parameters
    ----------
    elevation_deg : float or array_like
        Satellite elevation angle in degrees
    pos_llh : array_like
        Position in lat/lon/height (radians, radians, meters)
    model : str, optional
        Troposphere model to use. Currently only 'saastamoinen' is supported.

    Returns
    -------
    float or array_like
        Tropospheric delay in meters
    """
    if model == 'saastamoinen':
        lat_deg = np.degrees(pos_llh[0])
        alt_m = pos_llh[2]
        return saastamoinen_model(elevation_deg, alt_m, lat_deg)
    else:
        raise ValueError(f"Unknown troposphere model: {model}")


def compute_dd_troposphere_correction(elevation_ref, elevation_other,
                                     rover_llh, base_llh, model='saastamoinen'):
    """Compute double-differenced tropospheric correction.

    For double differences, the tropospheric delays partially cancel out,
    especially for short baselines. This function computes the residual
    tropospheric delay after double differencing.

    Parameters
    ----------
    elevation_ref : float
        Reference satellite elevation angle at rover (degrees)
    elevation_other : float
        Other satellite elevation angle at rover (degrees)
    rover_llh : array_like
        Rover position in lat/lon/height (radians, radians, meters)
    base_llh : array_like
        Base position in lat/lon/height (radians, radians, meters)
    model : str, optional
        Troposphere model to use

    Returns
    -------
    float
        Double-differenced tropospheric correction in meters

    Notes
    -----
    The DD tropospheric correction is:
    DD_trop = (T_rover_other - T_rover_ref) - (T_base_other - T_base_ref)

    For short baselines (<10km), this is typically small (<0.1m).
    For longer baselines or large altitude differences, it can be significant.
    """
    # Compute tropospheric delays at rover
    trop_rover_ref = troposphere_correction(elevation_ref, rover_llh, model)
    trop_rover_other = troposphere_correction(elevation_other, rover_llh, model)

    # Compute tropospheric delays at base
    # Note: elevation angles at base are slightly different than at rover
    # For simplicity, we use the same elevation angles (valid for short baselines)
    trop_base_ref = troposphere_correction(elevation_ref, base_llh, model)
    trop_base_other = troposphere_correction(elevation_other, base_llh, model)

    # Form single differences
    sd_rover = trop_rover_other - trop_rover_ref
    sd_base = trop_base_other - trop_base_ref

    # Form double difference
    dd_trop = sd_rover - sd_base

    return dd_trop