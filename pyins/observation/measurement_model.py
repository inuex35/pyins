"""GNSS measurement models including atmospheric corrections"""

import numpy as np
from typing import Tuple, Optional
from ..core.constants import *
from ..coordinate import ecef2llh

def troposphere_model(el: float, lat: float, h: float, 
                     model: str = 'saastamoinen') -> float:
    """
    Compute tropospheric delay
    
    Parameters:
    -----------
    el : float
        Elevation angle (rad)
    lat : float
        Latitude (rad)
    h : float
        Height (m)
    model : str
        Troposphere model ('saastamoinen', 'hopfield')
        
    Returns:
    --------
    delay : float
        Tropospheric delay (m)
    """
    if model == 'saastamoinen':
        return saastamoinen_model(el, lat, h)
    elif model == 'hopfield':
        return hopfield_model(el, lat, h)
    else:
        raise ValueError(f"Unknown troposphere model: {model}")


def saastamoinen_model(el: float, lat: float, h: float) -> float:
    """
    Saastamoinen troposphere model
    
    Parameters:
    -----------
    el : float
        Elevation angle (rad)
    lat : float
        Latitude (rad) 
    h : float
        Height (m)
        
    Returns:
    --------
    delay : float
        Tropospheric delay (m)
    """
    if el <= 0:
        return 0.0
        
    # Standard atmosphere parameters
    temp0 = 288.15  # Temperature at sea level (K)
    pres0 = 1013.25  # Pressure at sea level (mbar)
    humi = REL_HUMI  # Relative humidity
    
    # Height correction
    temp = temp0 - 0.0065 * h
    pres = pres0 * (1.0 - 0.0065 * h / temp0) ** 5.225
    e = 6.108 * humi * np.exp((17.15 * temp - 4684.0) / (temp - 38.45))
    
    # Zenith delays
    z = np.pi / 2.0 - el
    zhd = 0.0022768 * pres / (1.0 - 0.00266 * np.cos(2.0 * lat) - 0.00028 * h / 1000.0)
    zwd = 0.002277 * (1255.0 / temp + 0.05) * e
    
    # Mapping function (simple)
    mapfh = 1.0 / (np.sin(el) + 0.00143 / (np.tan(el) + 0.0445))
    mapfw = 1.0 / (np.sin(el) + 0.00035 / (np.tan(el) + 0.017))
    
    return zhd * mapfh + zwd * mapfw


def hopfield_model(el: float, lat: float, h: float) -> float:
    """
    Hopfield troposphere model
    
    Parameters:
    -----------
    el : float
        Elevation angle (rad)
    lat : float
        Latitude (rad)
    h : float  
        Height (m)
        
    Returns:
    --------
    delay : float
        Tropospheric delay (m)
    """
    if el <= 0:
        return 0.0
        
    # Standard atmosphere
    temp0 = 288.15
    pres0 = 1013.25
    humi = REL_HUMI
    
    # Height correction
    temp = temp0 - 0.0065 * h
    pres = pres0 * (temp / temp0) ** 5.225
    e = 6.108 * humi * np.exp((17.15 * temp - 4684.0) / (temp - 38.45))
    
    # Scale heights
    hw = 11000.0  # Wet component scale height
    hd = 40136.0 + 148.72 * (temp - 273.16)  # Dry component scale height
    
    # Refractivities
    Nd = 77.64 * pres / temp
    Nw = -12.92 * e / temp + 371900.0 * e / (temp ** 2)
    
    # Delays
    dry_delay = Nd * 1e-6 * hd * 5 / np.sin(np.sqrt(el ** 2 + 2.25 * (np.pi / 180) ** 2))
    wet_delay = Nw * 1e-6 * hw * 5 / np.sin(np.sqrt(el ** 2 + 2.25 * (np.pi / 180) ** 2))
    
    return dry_delay + wet_delay


def ionosphere_model(el: float, az: float, lat: float, lon: float,
                    time: float, ion_params: np.ndarray,
                    freq: float = FREQ_L1) -> float:
    """
    Klobuchar ionosphere model
    
    Parameters:
    -----------
    el : float
        Elevation angle (rad)
    az : float
        Azimuth angle (rad)
    lat : float
        Latitude (rad)
    lon : float
        Longitude (rad)
    time : float
        Time (GPST)
    ion_params : np.ndarray
        Ionosphere parameters (8 values: alpha0-3, beta0-3)
    freq : float
        Signal frequency (Hz)
        
    Returns:
    --------
    delay : float
        Ionospheric delay (m)
    """
    if el <= 0 or len(ion_params) < 8:
        return 0.0
        
    # Earth-centered angle
    psi = 0.0137 / (el / np.pi + 0.11) - 0.022
    
    # Subionospheric latitude
    phi = lat / np.pi + psi * np.cos(az)
    phi = np.clip(phi, -0.416, 0.416)
    
    # Subionospheric longitude
    lam = lon / np.pi + psi * np.sin(az) / np.cos(phi * np.pi)
    
    # Geomagnetic latitude
    phi_m = phi + 0.064 * np.cos((lam - 1.617) * np.pi)
    
    # Local time
    t = 43200.0 * lam + time
    t = t % 86400.0
    if t < 0:
        t += 86400.0
        
    # Amplitude and period
    amp = ion_params[0] + phi_m * (ion_params[1] + phi_m * (ion_params[2] + phi_m * ion_params[3]))
    per = ion_params[4] + phi_m * (ion_params[5] + phi_m * (ion_params[6] + phi_m * ion_params[7]))
    
    amp = max(0.0, amp)
    per = max(72000.0, per)
    
    # Phase
    x = 2.0 * np.pi * (t - 50400.0) / per
    
    # Ionospheric delay
    if abs(x) < 1.57:
        delay = CLIGHT * (5e-9 + amp * (1.0 - x**2 / 2.0 + x**4 / 24.0))
    else:
        delay = CLIGHT * 5e-9
        
    # Obliquity factor
    f = 1.0 + 16.0 * (0.53 - el / np.pi) ** 3
    
    # Frequency scaling
    delay *= f * (FREQ_L1 / freq) ** 2
    
    return delay


def dual_frequency_iono_correction(P1: float, P2: float, 
                                  f1: float = FREQ_L1, 
                                  f2: float = FREQ_L2) -> float:
    """
    Dual-frequency ionosphere correction
    
    Parameters:
    -----------
    P1 : float
        Pseudorange on frequency 1 (m)
    P2 : float
        Pseudorange on frequency 2 (m)
    f1 : float
        Frequency 1 (Hz)
    f2 : float
        Frequency 2 (Hz)
        
    Returns:
    --------
    iono_free : float
        Ionosphere-free pseudorange (m)
    """
    if P1 == 0.0 or P2 == 0.0:
        return 0.0
        
    # Ionosphere-free combination
    gamma = (f1 / f2) ** 2
    iono_free = (gamma * P1 - P2) / (gamma - 1.0)
    
    return iono_free


def windup_correction(sat_pos: np.ndarray, sat_vel: np.ndarray,
                     rcv_pos: np.ndarray, prev_windup: float = 0.0) -> float:
    """
    Carrier phase windup correction
    
    Parameters:
    -----------
    sat_pos : np.ndarray
        Satellite position (ECEF)
    sat_vel : np.ndarray
        Satellite velocity (ECEF)
    rcv_pos : np.ndarray
        Receiver position (ECEF)
    prev_windup : float
        Previous windup value (cycles)
        
    Returns:
    --------
    windup : float
        Phase windup correction (cycles)
    """
    # Unit vectors
    k = np.array([0, 0, 1])  # Up direction
    
    # Satellite antenna orientation
    es = -sat_pos / np.linalg.norm(sat_pos)  # From satellite to Earth center
    
    # Satellite y-axis (cross-track)
    ys = np.cross(es, sat_vel)
    ys = ys / np.linalg.norm(ys)
    
    # Satellite x-axis (along-track)
    xs = np.cross(ys, es)
    
    # Receiver antenna orientation (simplified - assuming vertical)
    llh = ecef2llh(rcv_pos)
    lat, lon = llh[0], llh[1]
    
    # ENU to ECEF rotation
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)
    
    # Receiver axes in ECEF
    er = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat])
    xr = np.array([-sin_lon, cos_lon, 0])
    yr = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])
    
    # Effective dipole vectors
    ds = xs - k * np.dot(k, xs) + np.cross(k, ys)
    dr = xr - k * np.dot(k, xr) + np.cross(k, yr)
    
    # Phase windup
    los = sat_pos - rcv_pos
    los = los / np.linalg.norm(los)
    
    # Compute windup angle
    dp = np.dot(ds, dr)
    dq = np.dot(los, np.cross(ds, dr))
    windup = np.arctan2(dq, dp) / (2.0 * np.pi)
    
    # Unwrap phase
    dwindup = windup - prev_windup
    windup = prev_windup + dwindup - np.round(dwindup)
    
    return windup


class MeasurementModel:
    """Complete GNSS measurement model"""
    
    def __init__(self):
        self.tropo_model = 'saastamoinen'
        self.iono_model = 'klobuchar'
        self.enable_windup = True
        self.windup_state = {}  # sat -> windup value
        
    def compute_modeled_range(self,
                            sat_pos: np.ndarray,
                            sat_vel: np.ndarray,
                            sat_clk: float,
                            rcv_pos: np.ndarray,
                            rcv_clk: float,
                            sat: int,
                            freq: float,
                            ion_params: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """
        Compute complete modeled range including all corrections
        
        Parameters:
        -----------
        sat_pos : np.ndarray
            Satellite position (ECEF)
        sat_vel : np.ndarray
            Satellite velocity (ECEF)
        sat_clk : float
            Satellite clock bias (s)
        rcv_pos : np.ndarray
            Receiver position (ECEF)
        rcv_clk : float
            Receiver clock bias (s)
        sat : int
            Satellite number
        freq : float
            Signal frequency (Hz)
        ion_params : np.ndarray, optional
            Ionosphere parameters
            
        Returns:
        --------
        rho_code : float
            Modeled pseudorange (m)
        rho_phase : float
            Modeled carrier phase (m)
        """
        # Geometric range
        from .pseudorange import compute_range, sagnac_correction, elevation_angle
        rho, e = compute_range(sat_pos, rcv_pos)
        
        # Sagnac effect
        sagnac = sagnac_correction(sat_pos, rcv_pos)
        
        # Clock correction
        clock_corr = CLIGHT * (rcv_clk - sat_clk)
        
        # Receiver position in geodetic coordinates
        llh = ecef2llh(rcv_pos)
        lat, lon, h = llh[0], llh[1], llh[2]
        
        # Elevation and azimuth
        el = elevation_angle(sat_pos, rcv_pos)
        az = self._compute_azimuth(sat_pos, rcv_pos)
        
        # Tropospheric delay
        tropo = troposphere_model(el, lat, h, self.tropo_model)
        
        # Ionospheric delay
        if ion_params is not None and self.iono_model == 'klobuchar':
            iono = ionosphere_model(el, az, lat, lon, 0.0, ion_params, freq)
        else:
            iono = 0.0
            
        # Pseudorange model
        rho_code = rho + sagnac + clock_corr + tropo + iono
        
        # Carrier phase model (ionosphere has opposite sign)
        rho_phase = rho + sagnac + clock_corr + tropo - iono
        
        # Phase windup correction
        if self.enable_windup:
            if sat not in self.windup_state:
                self.windup_state[sat] = 0.0
                
            windup = windup_correction(sat_pos, sat_vel, rcv_pos, 
                                     self.windup_state[sat])
            self.windup_state[sat] = windup
            
            # Convert to meters
            wavelength = CLIGHT / freq
            rho_phase += windup * wavelength
            
        return rho_code, rho_phase
    
    def _compute_azimuth(self, sat_pos: np.ndarray, rcv_pos: np.ndarray) -> float:
        """Compute satellite azimuth angle"""
        # Convert to ENU
        llh = ecef2llh(rcv_pos)
        lat, lon = llh[0], llh[1]
        
        # ENU transformation
        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)
        sin_lon = np.sin(lon)
        cos_lon = np.cos(lon)
        
        R = np.array([
            [-sin_lon, cos_lon, 0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]
        ])
        
        los = sat_pos - rcv_pos
        enu = R @ los
        
        # Azimuth
        az = np.arctan2(enu[0], enu[1])
        
        return az