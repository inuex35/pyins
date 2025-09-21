"""Ionospheric delay models for GNSS.

This module implements ionospheric delay models to correct for the
signal delay caused by the ionosphere, particularly important for
single-frequency GNSS processing.
"""

import numpy as np
from ..core.constants import CLIGHT, FREQ_L1


def klobuchar_model(lat, lon, az, el, tow, alpha, beta):
    """Klobuchar ionospheric delay model.

    Standard GPS ionospheric correction model using broadcast parameters.

    Parameters
    ----------
    lat : float
        Receiver latitude in radians
    lon : float
        Receiver longitude in radians
    az : float
        Satellite azimuth in radians
    el : float
        Satellite elevation in radians
    tow : float
        GPS time of week in seconds
    alpha : array_like
        Alpha coefficients [alpha0, alpha1, alpha2, alpha3]
    beta : array_like
        Beta coefficients [beta0, beta1, beta2, beta3]

    Returns
    -------
    float
        Ionospheric delay in meters at L1 frequency
    """
    # Earth-centered angle (semi-circle)
    psi = 0.0137 / (el/np.pi + 0.11) - 0.022

    # Subionospheric latitude
    phi_i = lat/np.pi + psi * np.cos(az)
    phi_i = np.clip(phi_i, -0.416, 0.416)

    # Subionospheric longitude
    lambda_i = lon/np.pi + psi * np.sin(az) / np.cos(phi_i * np.pi)

    # Geomagnetic latitude
    phi_m = phi_i + 0.064 * np.cos((lambda_i - 1.617) * np.pi)

    # Local time
    t = 4.32e4 * lambda_i + tow
    t = t % 86400
    if t < 0:
        t += 86400

    # Amplitude of ionospheric delay
    amp = alpha[0] + phi_m * (alpha[1] + phi_m * (alpha[2] + phi_m * alpha[3]))
    amp = max(0, amp)

    # Period of ionospheric delay
    per = beta[0] + phi_m * (beta[1] + phi_m * (beta[2] + phi_m * beta[3]))
    per = max(72000, per)

    # Phase of ionospheric delay
    x = 2 * np.pi * (t - 50400) / per

    # Slant factor
    f = 1.0 + 16.0 * (0.53 - el/np.pi) ** 3

    # Ionospheric delay
    if abs(x) < 1.57:
        iono = CLIGHT * f * (5e-9 + amp * (1 - x*x/2 + x**4/24))
    else:
        iono = CLIGHT * f * 5e-9

    return iono


def simple_ionosphere_model(elevation_deg, tec_zenith=10.0):
    """Simple ionospheric delay model.

    Simplified model using a constant zenith TEC value.

    Parameters
    ----------
    elevation_deg : float or array_like
        Satellite elevation angle in degrees
    tec_zenith : float
        Zenith total electron content in TECU (1 TECU = 1e16 electrons/m^2)
        Default is 10 TECU (moderate ionosphere)

    Returns
    -------
    float or array_like
        Ionospheric delay in meters at L1 frequency
    """
    # Convert to radians
    el_rad = np.radians(elevation_deg)

    # Skip very low elevations
    if np.any(elevation_deg < 5.0):
        if np.isscalar(elevation_deg):
            return 0.0
        else:
            delays = np.zeros_like(elevation_deg)
            mask = elevation_deg >= 5.0
            if np.any(mask):
                delays[mask] = simple_ionosphere_model(elevation_deg[mask], tec_zenith)
            return delays

    # Mapping function (simple cosecant)
    mapping = 1.0 / np.sin(el_rad)

    # Ionospheric delay at L1
    # 1 TECU causes ~0.162m delay at L1
    K = 40.3e16 / (FREQ_L1 ** 2)  # ~0.162 m/TECU
    delay = K * tec_zenith * mapping

    return delay


def ionosphere_correction(elevation_deg, pos_llh, az_deg=0, tow=0,
                        alpha=None, beta=None, model='simple'):
    """Compute ionospheric delay correction.

    Parameters
    ----------
    elevation_deg : float or array_like
        Satellite elevation angle in degrees
    pos_llh : array_like
        Position in lat/lon/height (radians, radians, meters)
    az_deg : float
        Satellite azimuth in degrees (for Klobuchar model)
    tow : float
        GPS time of week in seconds (for Klobuchar model)
    alpha : array_like, optional
        Alpha coefficients for Klobuchar model
    beta : array_like, optional
        Beta coefficients for Klobuchar model
    model : str
        Model to use: 'simple', 'klobuchar', or 'none'

    Returns
    -------
    float or array_like
        Ionospheric delay in meters at L1 frequency
    """
    if model == 'none':
        return 0.0
    elif model == 'simple':
        return simple_ionosphere_model(elevation_deg)
    elif model == 'klobuchar':
        if alpha is None or beta is None:
            # Use default values if not provided
            alpha = [0.1118e-7, -0.7451e-8, -0.5961e-7, 0.1192e-6]
            beta = [0.1270e6, -0.1966e6, 0.6554e5, 0.2621e6]
        return klobuchar_model(
            pos_llh[0], pos_llh[1],
            np.radians(az_deg), np.radians(elevation_deg),
            tow, alpha, beta
        )
    else:
        raise ValueError(f"Unknown ionosphere model: {model}")


def compute_dd_ionosphere_correction(elevation_ref, elevation_other,
                                    rover_llh, base_llh,
                                    az_ref=0, az_other=0, tow=0,
                                    model='simple', baseline_km=0):
    """Compute double-differenced ionospheric correction.

    For double differences, ionospheric delays partially cancel,
    especially for short baselines. For longer baselines, spatial
    gradients in the ionosphere cause residual delays.

    Parameters
    ----------
    elevation_ref : float
        Reference satellite elevation at rover (degrees)
    elevation_other : float
        Other satellite elevation at rover (degrees)
    rover_llh : array_like
        Rover position in lat/lon/height (radians, radians, meters)
    base_llh : array_like
        Base position in lat/lon/height (radians, radians, meters)
    az_ref : float
        Reference satellite azimuth (degrees)
    az_other : float
        Other satellite azimuth (degrees)
    tow : float
        GPS time of week in seconds
    model : str
        Ionosphere model to use

    Returns
    -------
    float
        Double-differenced ionospheric correction in meters
    """
    # For longer baselines, add spatial gradient
    # Approximate: 1-2 ppm/km difference in ionosphere
    if baseline_km == 0:
        # Compute baseline if not provided
        import numpy as np
        from ..coordinate.transforms import llh2ecef
        rover_ecef = llh2ecef(rover_llh[0], rover_llh[1], rover_llh[2])
        base_ecef = llh2ecef(base_llh[0], base_llh[1], base_llh[2])
        baseline_km = np.linalg.norm(rover_ecef - base_ecef) / 1000

    # Scale factor for spatial ionosphere variation
    # About 0.1-0.2 TECU per 10km baseline
    tec_gradient = 0.15 * (baseline_km / 10.0)  # TECU difference

    # Compute ionospheric delays at rover
    iono_rover_ref = ionosphere_correction(
        elevation_ref, rover_llh, az_ref, tow, model=model
    )
    iono_rover_other = ionosphere_correction(
        elevation_other, rover_llh, az_other, tow, model=model
    )

    # Compute ionospheric delays at base with gradient
    # Base has slightly different TEC
    if model == 'simple':
        # Add gradient effect
        gradient_factor = 1.0 - tec_gradient / 10.0  # Relative TEC change
        iono_base_ref = simple_ionosphere_model(elevation_ref, 10.0 * gradient_factor)
        iono_base_other = simple_ionosphere_model(elevation_other, 10.0 * gradient_factor)
    else:
        iono_base_ref = ionosphere_correction(
            elevation_ref, base_llh, az_ref, tow, model=model
        )
        iono_base_other = ionosphere_correction(
            elevation_other, base_llh, az_other, tow, model=model
        )

    # Form single differences
    sd_rover = iono_rover_other - iono_rover_ref
    sd_base = iono_base_other - iono_base_ref

    # Form double difference
    dd_iono = sd_rover - sd_base

    return dd_iono