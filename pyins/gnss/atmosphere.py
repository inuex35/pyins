"""
Atmospheric corrections for GNSS signals
Implements ionospheric and tropospheric delay models
"""

import numpy as np
from ..core.constants import CLIGHT, FREQ_L1, FREQ_L2, FREQ_L5

# Earth parameters
RE_WGS84 = 6378137.0  # Earth radius (m)
FE_WGS84 = 1.0 / 298.257223563  # Earth flattening


def ionosphere_klobuchar(lat, lon, azimuth, elevation, tow, alpha, beta):
    """
    Klobuchar ionospheric delay model (broadcast model)
    
    Args:
        lat: Receiver latitude (rad)
        lon: Receiver longitude (rad)
        azimuth: Satellite azimuth (rad)
        elevation: Satellite elevation (rad)
        tow: GPS time of week (s)
        alpha: Alpha coefficients [α0, α1, α2, α3]
        beta: Beta coefficients [β0, β1, β2, β3]
    
    Returns:
        iono_delay: Ionospheric delay on L1 (m)
    """
    if elevation <= 0:
        return 0.0
    
    # Earth centered angle (semi-circles)
    psi = 0.0137 / (elevation / np.pi + 0.11) - 0.022
    
    # Subionospheric latitude
    phi_i = lat / np.pi + psi * np.cos(azimuth)
    phi_i = np.clip(phi_i, -0.416, 0.416)
    
    # Subionospheric longitude
    lambda_i = lon / np.pi + psi * np.sin(azimuth) / np.cos(phi_i * np.pi)
    
    # Geomagnetic latitude
    phi_m = phi_i + 0.064 * np.cos((lambda_i - 1.617) * np.pi)
    
    # Local time
    t = 43200.0 * lambda_i + tow
    t = t % 86400.0
    if t < 0:
        t += 86400.0
    
    # Amplitude of ionospheric delay
    amp = alpha[0] + alpha[1] * phi_m + alpha[2] * phi_m**2 + alpha[3] * phi_m**3
    amp = max(0.0, amp)
    
    # Period of ionospheric delay
    per = beta[0] + beta[1] * phi_m + beta[2] * phi_m**2 + beta[3] * phi_m**3
    per = max(72000.0, per)
    
    # Phase of ionospheric delay
    x = 2.0 * np.pi * (t - 50400.0) / per
    
    # Slant factor
    f = 1.0 + 16.0 * (0.53 - elevation / np.pi)**3
    
    # Ionospheric delay
    if abs(x) < 1.57:
        iono_delay = CLIGHT * f * (5e-9 + amp * (1.0 - x**2 / 2.0 + x**4 / 24.0))
    else:
        iono_delay = CLIGHT * f * 5e-9
    
    return iono_delay


def ionosphere_free_combination(P1, P2, f1=FREQ_L1, f2=FREQ_L2):
    """
    Ionosphere-free linear combination of pseudoranges
    
    Args:
        P1: Pseudorange on frequency 1 (m)
        P2: Pseudorange on frequency 2 (m)
        f1: Frequency 1 (Hz)
        f2: Frequency 2 (Hz)
    
    Returns:
        P_IF: Ionosphere-free pseudorange (m)
    """
    gamma = (f1 / f2) ** 2
    P_IF = (gamma * P1 - P2) / (gamma - 1.0)
    return P_IF


def ionosphere_tec_from_dual_freq(P1, P2, f1=FREQ_L1, f2=FREQ_L2):
    """
    Estimate TEC (Total Electron Content) from dual-frequency measurements
    
    Args:
        P1: Pseudorange on frequency 1 (m)
        P2: Pseudorange on frequency 2 (m)
        f1: Frequency 1 (Hz)
        f2: Frequency 2 (Hz)
    
    Returns:
        tec: Total Electron Content (TECU = 10^16 electrons/m^2)
    """
    K = 40.3e16  # Ionospheric constant
    tec = (P2 - P1) * (f1**2 * f2**2) / (K * (f1**2 - f2**2)) / 1e16
    return tec


def troposphere_saastamoinen(lat, height, elevation, pressure=1013.25, temperature=15.0, humidity=50.0):
    """
    Saastamoinen tropospheric delay model
    
    Args:
        lat: Receiver latitude (rad)
        height: Receiver height above sea level (m)
        elevation: Satellite elevation (rad)
        pressure: Surface pressure (hPa)
        temperature: Surface temperature (°C)
        humidity: Relative humidity (%)
    
    Returns:
        tropo_delay: Tropospheric delay (m)
    """
    if elevation <= 0:
        return 0.0
    
    # Standard atmosphere
    if pressure == 1013.25 and temperature == 15.0:
        # Use standard atmosphere model
        pressure = 1013.25 * (1.0 - 2.2557e-5 * height) ** 5.2568
        temperature = 15.0 - 6.5e-3 * height
        e = 6.108 * humidity * np.exp((17.15 * temperature) / (234.7 + temperature)) / 100.0
    else:
        # Partial pressure of water vapor
        es = 6.108 * np.exp((17.15 * temperature) / (234.7 + temperature))
        e = humidity / 100.0 * es
    
    # Convert temperature to Kelvin
    T = temperature + 273.15
    
    # Zenith delays
    z_hyd = 0.002277 * pressure / (1.0 - 0.00266 * np.cos(2.0 * lat) - 0.00028 * height / 1000.0)
    z_wet = 0.002277 * (1255.0 / T + 0.05) * e
    
    # Mapping function (simple 1/sin(el))
    map_hyd = 1.0 / np.sin(elevation)
    map_wet = 1.0 / np.sin(elevation)
    
    # Total delay
    tropo_delay = z_hyd * map_hyd + z_wet * map_wet
    
    return tropo_delay


def troposphere_hopfield(height, elevation, pressure=1013.25, temperature=15.0, humidity=50.0):
    """
    Hopfield tropospheric delay model
    
    Args:
        height: Receiver height above sea level (m)
        elevation: Satellite elevation (rad)
        pressure: Surface pressure (hPa)
        temperature: Surface temperature (°C)
        humidity: Relative humidity (%)
    
    Returns:
        tropo_delay: Tropospheric delay (m)
    """
    if elevation <= 0:
        return 0.0
    
    # Convert to Kelvin
    T = temperature + 273.15
    
    # Partial pressure of water vapor
    es = 6.108 * np.exp((17.15 * temperature) / (234.7 + temperature))
    e = humidity / 100.0 * es
    
    # Heights of troposphere layers
    h_dry = 40136.0 + 148.72 * (T - 273.15)  # Dry component height
    h_wet = 11000.0  # Wet component height
    
    # Refractivity
    N_dry = 77.64 * pressure / T
    N_wet = -12.96 * e / T + 371900.0 * e / T**2
    
    # Zenith delays
    z_dry = 1e-6 * N_dry * (h_dry - height) if height < h_dry else 0.0
    z_wet = 1e-6 * N_wet * (h_wet - height) if height < h_wet else 0.0
    
    # Mapping function
    sin_el = np.sin(elevation)
    map_factor = 1.0 / np.sqrt(1.0 - (RE_WGS84 / (RE_WGS84 + h_dry))**2 * (1.0 - sin_el**2))
    
    # Total delay
    tropo_delay = (z_dry + z_wet) * map_factor
    
    return tropo_delay


def troposphere_niell_mapping(elevation, lat, height, doy):
    """
    Niell mapping function for troposphere
    
    Args:
        elevation: Satellite elevation (rad)
        lat: Receiver latitude (rad)
        height: Receiver height (m)
        doy: Day of year
    
    Returns:
        map_dry: Dry mapping function
        map_wet: Wet mapping function
    """
    # Coefficients for dry mapping (latitude dependent)
    lat_deg = np.degrees(abs(lat))
    
    if lat_deg <= 15:
        a = 1.2769934e-3
        b = 2.9153695e-3
        c = 62.610505e-3
    elif lat_deg <= 30:
        a = 1.2683230e-3
        b = 2.9152299e-3
        c = 62.837393e-3
    elif lat_deg <= 45:
        a = 1.2465397e-3
        b = 2.9288445e-3
        c = 63.721774e-3
    elif lat_deg <= 60:
        a = 1.2196049e-3
        b = 2.9022565e-3
        c = 63.824265e-3
    else:
        a = 1.2045996e-3
        b = 2.9024912e-3
        c = 64.258455e-3
    
    # Height correction
    a_ht = 2.53e-5
    b_ht = 5.49e-3
    c_ht = 1.14e-3
    
    # Apply height correction
    ht_corr = 1.0 / np.sin(elevation) - mapping_function_form(elevation, a, b, c)
    ht_corr_km = ht_corr * height / 1000.0
    
    map_dry = mapping_function_form(elevation, a, b, c) + ht_corr_km
    
    # Wet mapping (simpler, height independent)
    a_wet = 5.8021897e-4
    b_wet = 1.4275268e-3
    c_wet = 4.3472961e-2
    
    map_wet = mapping_function_form(elevation, a_wet, b_wet, c_wet)
    
    return map_dry, map_wet


def mapping_function_form(elevation, a, b, c):
    """
    Common form for mapping functions
    
    m(e) = (1 + a/(1 + b/(1 + c))) / (sin(e) + a/(sin(e) + b/(sin(e) + c)))
    """
    sin_el = np.sin(elevation)
    
    numerator = 1.0 + a / (1.0 + b / (1.0 + c))
    denominator = sin_el + a / (sin_el + b / (sin_el + c))
    
    return numerator / denominator


def apply_atmospheric_corrections(pseudorange, satellite_pos, receiver_pos, 
                                 elevation, azimuth, time, frequency,
                                 iono_model='klobuchar', tropo_model='saastamoinen',
                                 iono_params=None, weather_params=None):
    """
    Apply ionospheric and tropospheric corrections to pseudorange
    
    Args:
        pseudorange: Raw pseudorange (m)
        satellite_pos: Satellite ECEF position (m)
        receiver_pos: Receiver ECEF position (m)
        elevation: Satellite elevation (rad)
        azimuth: Satellite azimuth (rad)
        time: GPS time
        frequency: Signal frequency (Hz)
        iono_model: Ionosphere model ('klobuchar', 'iono_free', None)
        tropo_model: Troposphere model ('saastamoinen', 'hopfield', None)
        iono_params: Ionosphere model parameters
        weather_params: Weather parameters for troposphere
    
    Returns:
        corrected_pr: Corrected pseudorange (m)
        iono_delay: Ionospheric delay (m)
        tropo_delay: Tropospheric delay (m)
    """
    from ..coordinate.transforms import ecef2llh
    
    # Get receiver position in geodetic coordinates
    receiver_llh = ecef2llh(receiver_pos)
    lat, lon, height = receiver_llh
    
    # Ionospheric correction
    iono_delay = 0.0
    if iono_model == 'klobuchar' and iono_params is not None:
        alpha = iono_params.get('alpha', [0, 0, 0, 0])
        beta = iono_params.get('beta', [0, 0, 0, 0])
        tow = time % 604800.0  # GPS time of week
        
        # Klobuchar gives L1 delay
        iono_delay_L1 = ionosphere_klobuchar(lat, lon, azimuth, elevation, tow, alpha, beta)
        
        # Scale to actual frequency
        iono_delay = iono_delay_L1 * (FREQ_L1 / frequency) ** 2
    
    # Tropospheric correction
    tropo_delay = 0.0
    if tropo_model == 'saastamoinen':
        if weather_params is None:
            weather_params = {'pressure': 1013.25, 'temperature': 15.0, 'humidity': 50.0}
        
        tropo_delay = troposphere_saastamoinen(
            lat, height, elevation,
            weather_params['pressure'],
            weather_params['temperature'],
            weather_params['humidity']
        )
    elif tropo_model == 'hopfield':
        if weather_params is None:
            weather_params = {'pressure': 1013.25, 'temperature': 15.0, 'humidity': 50.0}
        
        tropo_delay = troposphere_hopfield(
            height, elevation,
            weather_params['pressure'],
            weather_params['temperature'],
            weather_params['humidity']
        )
    
    # Apply corrections
    corrected_pr = pseudorange - iono_delay - tropo_delay
    
    return corrected_pr, iono_delay, tropo_delay


def estimate_ztd(observations, satellite_positions, receiver_pos, mapping_func='simple'):
    """
    Estimate Zenith Total Delay (ZTD) from multiple satellite observations
    
    Args:
        observations: List of pseudorange observations
        satellite_positions: List of satellite positions
        receiver_pos: Receiver position
        mapping_func: Mapping function type
    
    Returns:
        ztd: Estimated zenith total delay (m)
        residuals: Observation residuals
    """
    # Implementation for ZTD estimation
    # This would typically be part of the position solution
    pass