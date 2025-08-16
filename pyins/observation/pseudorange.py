"""Pseudorange measurement processing"""

import numpy as np
from typing import Tuple, Optional, List
from ..core.constants import *
from ..core.data_structures import Observation, Ephemeris
from ..satellite.satellite_position import compute_satellite_position
from ..satellite.clock import compute_satellite_clock, apply_tgd_correction

def compute_range(sat_pos: np.ndarray, rcv_pos: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute geometric range and unit vector
    
    Parameters:
    -----------
    sat_pos : np.ndarray
        Satellite position (3x1)
    rcv_pos : np.ndarray
        Receiver position (3x1)
        
    Returns:
    --------
    rho : float
        Geometric range (m)
    e : np.ndarray
        Unit vector from receiver to satellite
    """
    dr = sat_pos - rcv_pos
    rho = np.linalg.norm(dr)
    e = dr / rho if rho > 0 else np.zeros(3)
    
    return rho, e


def sagnac_correction(sat_pos: np.ndarray, rcv_pos: np.ndarray) -> float:
    """
    Compute Sagnac effect correction
    
    Parameters:
    -----------
    sat_pos : np.ndarray
        Satellite position in ECEF
    rcv_pos : np.ndarray
        Receiver position in ECEF
        
    Returns:
    --------
    dt_sagnac : float
        Sagnac correction (m)
    """
    return OMGE * (sat_pos[0] * rcv_pos[1] - sat_pos[1] * rcv_pos[0]) / CLIGHT


def compute_pseudorange_residual(obs: Observation, 
                               eph: Ephemeris,
                               rcv_pos: np.ndarray,
                               rcv_clk: float,
                               freq_idx: int = 0) -> Tuple[float, np.ndarray, float]:
    """
    Compute pseudorange residual
    
    Parameters:
    -----------
    obs : Observation
        GNSS observation
    eph : Ephemeris
        Satellite ephemeris
    rcv_pos : np.ndarray
        Receiver position in ECEF
    rcv_clk : float
        Receiver clock bias (s)
    freq_idx : int
        Frequency index
        
    Returns:
    --------
    residual : float
        Pseudorange residual (m)
    H : np.ndarray
        Measurement Jacobian [dx, dy, dz, dt]
    var : float
        Measurement variance
    """
    # Check if pseudorange is available
    if obs.P[freq_idx] == 0.0:
        return 0.0, np.zeros(4), 0.0
    
    # Signal transmission time
    tau = obs.P[freq_idx] / CLIGHT
    t_tx = obs.time - tau
    
    # Satellite position at transmission time
    sat_pos, sat_clk, sat_var = compute_satellite_position(eph, t_tx)
    
    # Geometric range
    rho, e = compute_range(sat_pos, rcv_pos)
    
    # Sagnac correction
    sagnac = sagnac_correction(sat_pos, rcv_pos)
    
    # TGD correction
    tgd = apply_tgd_correction(eph, freq_idx)
    
    # BeiDou bias correction (ISB)
    beidou_isb_correction = 0.0
    from ..core.constants import sat2sys, SYS_BDS
    if sat2sys(eph.sat) == SYS_BDS:
        from ..gnss.beidou_bias import get_beidou_isb
        freq_codes = ['B1I', 'B2I', 'B3I']
        freq_code = freq_codes[min(freq_idx, 2)]
        beidou_isb_correction = get_beidou_isb(freq_code)  # meters
    
    # Modeled pseudorange
    pr_model = rho + sagnac + CLIGHT * (rcv_clk - sat_clk + tgd) + beidou_isb_correction
    
    # Residual
    residual = obs.P[freq_idx] - pr_model
    
    # Jacobian
    H = np.zeros(4)
    H[:3] = -e  # Partial derivatives w.r.t position
    H[3] = CLIGHT  # Partial derivative w.r.t clock
    
    # Measurement variance
    el = elevation_angle(sat_pos, rcv_pos)
    var = pseudorange_variance(obs.SNR[freq_idx], el)
    
    return residual, H, var


def pseudorange_variance(snr: float, elevation: float) -> float:
    """
    Compute pseudorange measurement variance
    
    Parameters:
    -----------
    snr : float
        Signal-to-noise ratio (dBHz)
    elevation : float
        Satellite elevation angle (rad)
        
    Returns:
    --------
    var : float
        Measurement variance (m^2)
    """
    # Base standard deviation
    sigma_base = 0.3  # meters
    
    # SNR-dependent factor
    if snr > 0:
        snr_factor = 10.0 ** (-snr / 20.0)
    else:
        snr_factor = 1.0
        
    # Elevation-dependent factor
    el_deg = np.rad2deg(elevation)
    if el_deg > 5.0:
        el_factor = 1.0 / np.sin(elevation)
    else:
        el_factor = 1.0 / np.sin(np.deg2rad(5.0))
        
    # Total variance
    sigma = sigma_base * snr_factor * el_factor
    return sigma ** 2


def elevation_angle(sat_pos: np.ndarray, rcv_pos: np.ndarray) -> float:
    """
    Compute satellite elevation angle
    
    Parameters:
    -----------
    sat_pos : np.ndarray
        Satellite position in ECEF
    rcv_pos : np.ndarray
        Receiver position in ECEF
        
    Returns:
    --------
    elevation : float
        Elevation angle (rad)
    """
    # Vector from receiver to satellite
    los = sat_pos - rcv_pos
    
    # Convert receiver position to geodetic
    from ..coordinate import ecef2llh
    llh = ecef2llh(rcv_pos)
    lat, lon = llh[0], llh[1]
    
    # ENU transformation matrix
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)
    
    R = np.array([
        [-sin_lon, cos_lon, 0],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]
    ])
    
    # Transform to ENU
    enu = R @ los
    
    # Elevation angle
    horizontal_dist = np.sqrt(enu[0]**2 + enu[1]**2)
    elevation = np.arctan2(enu[2], horizontal_dist)
    
    return elevation


class PseudorangeProcessor:
    """Process pseudorange measurements"""
    
    def __init__(self):
        self.min_elevation = np.deg2rad(10.0)  # Minimum elevation angle
        self.max_residual = 30.0  # Maximum residual for outlier detection
        
    def process_observations(self, 
                           observations: List[Observation],
                           ephemerides: List[Ephemeris],
                           rcv_pos: np.ndarray,
                           rcv_clk: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process multiple pseudorange observations
        
        Parameters:
        -----------
        observations : List[Observation]
            List of GNSS observations
        ephemerides : List[Ephemeris]
            List of ephemerides
        rcv_pos : np.ndarray
            Receiver position
        rcv_clk : float
            Receiver clock bias
            
        Returns:
        --------
        residuals : np.ndarray
            Measurement residuals
        H : np.ndarray
            Measurement Jacobian matrix
        R : np.ndarray
            Measurement covariance matrix
        """
        valid_obs = []
        
        # Create ephemeris lookup
        eph_dict = {eph.sat: eph for eph in ephemerides}
        
        # Process each observation
        for obs in observations:
            if obs.sat not in eph_dict:
                continue
                
            eph = eph_dict[obs.sat]
            
            # Check elevation
            sat_pos, _, _ = compute_satellite_position(eph, obs.time)
            el = elevation_angle(sat_pos, rcv_pos)
            
            if el < self.min_elevation:
                continue
                
            # Compute residual
            res, h, var = compute_pseudorange_residual(obs, eph, rcv_pos, rcv_clk)
            
            # Outlier check
            if abs(res) > self.max_residual:
                continue
                
            valid_obs.append((res, h, var))
            
        if not valid_obs:
            return np.array([]), np.array([]), np.array([])
            
        # Stack results
        n = len(valid_obs)
        residuals = np.array([res for res, _, _ in valid_obs])
        H = np.vstack([h for _, h, _ in valid_obs])
        R = np.diag([var for _, _, var in valid_obs])
        
        return residuals, H, R