"""Earth-Centered Inertial (ECI) coordinate transformations"""

import numpy as np
from typing import Union
from ..core.constants import OMGE, RE_WGS84, FE_WGS84
from .transforms import llh2ecef, ecef2llh, ecef2ned, ned2ecef, ecef2enu, enu2ecef
from .dcm import eci2ecef_dcm, ecef2eci_dcm


def eci2ecef(xyz_eci: np.ndarray, t: float) -> np.ndarray:
    """
    Convert ECI to ECEF coordinates
    
    Parameters:
    -----------
    xyz_eci : np.ndarray
        ECI coordinates [x, y, z] (m)
    t : float
        Time since reference epoch (s)
        
    Returns:
    --------
    xyz_ecef : np.ndarray
        ECEF coordinates [x, y, z] (m)
    """
    C_i_e = eci2ecef_dcm(t)
    return C_i_e @ xyz_eci


def ecef2eci(xyz_ecef: np.ndarray, t: float) -> np.ndarray:
    """
    Convert ECEF to ECI coordinates
    
    Parameters:
    -----------
    xyz_ecef : np.ndarray
        ECEF coordinates [x, y, z] (m)
    t : float
        Time since reference epoch (s)
        
    Returns:
    --------
    xyz_eci : np.ndarray
        ECI coordinates [x, y, z] (m)
    """
    C_e_i = ecef2eci_dcm(t)
    return C_e_i @ xyz_ecef


def lla2eci(lla: np.ndarray, t: float) -> np.ndarray:
    """
    Convert geodetic coordinates to ECI
    
    Parameters:
    -----------
    lla : np.ndarray
        Geodetic coordinates [lat, lon, height] (rad, rad, m)
    t : float
        Time since reference epoch (s)
        
    Returns:
    --------
    xyz_eci : np.ndarray
        ECI coordinates [x, y, z] (m)
    """
    xyz_ecef = llh2ecef(lla)
    return ecef2eci(xyz_ecef, t)


def eci2lla(xyz_eci: np.ndarray, t: float) -> np.ndarray:
    """
    Convert ECI to geodetic coordinates
    
    Parameters:
    -----------
    xyz_eci : np.ndarray
        ECI coordinates [x, y, z] (m)
    t : float
        Time since reference epoch (s)
        
    Returns:
    --------
    lla : np.ndarray
        Geodetic coordinates [lat, lon, height] (rad, rad, m)
    """
    xyz_ecef = eci2ecef(xyz_eci, t)
    return ecef2llh(xyz_ecef)


def eci2ned(xyz_eci: np.ndarray, lla0: np.ndarray, t: float) -> np.ndarray:
    """
    Convert ECI to local NED coordinates
    
    Parameters:
    -----------
    xyz_eci : np.ndarray
        ECI coordinates [x, y, z] (m)
    lla0 : np.ndarray
        Origin geodetic coordinates [lat, lon, height] (rad, rad, m)
    t : float
        Time since reference epoch (s)
        
    Returns:
    --------
    ned : np.ndarray
        Local NED coordinates [n, e, d] (m)
    """
    xyz_ecef = eci2ecef(xyz_eci, t)
    return ecef2ned(xyz_ecef, lla0)


def ned2eci(ned: np.ndarray, lla0: np.ndarray, t: float) -> np.ndarray:
    """
    Convert local NED to ECI coordinates
    
    Parameters:
    -----------
    ned : np.ndarray
        Local NED coordinates [n, e, d] (m)
    lla0 : np.ndarray
        Origin geodetic coordinates [lat, lon, height] (rad, rad, m)
    t : float
        Time since reference epoch (s)
        
    Returns:
    --------
    xyz_eci : np.ndarray
        ECI coordinates [x, y, z] (m)
    """
    xyz_ecef = ned2ecef(ned, lla0)
    return ecef2eci(xyz_ecef, t)


def eci2enu(xyz_eci: np.ndarray, lla0: np.ndarray, t: float) -> np.ndarray:
    """
    Convert ECI to local ENU coordinates
    
    Parameters:
    -----------
    xyz_eci : np.ndarray
        ECI coordinates [x, y, z] (m)
    lla0 : np.ndarray
        Origin geodetic coordinates [lat, lon, height] (rad, rad, m)
    t : float
        Time since reference epoch (s)
        
    Returns:
    --------
    enu : np.ndarray
        Local ENU coordinates [e, n, u] (m)
    """
    xyz_ecef = eci2ecef(xyz_eci, t)
    return ecef2enu(xyz_ecef, lla0)


def enu2eci(enu: np.ndarray, lla0: np.ndarray, t: float) -> np.ndarray:
    """
    Convert local ENU to ECI coordinates
    
    Parameters:
    -----------
    enu : np.ndarray
        Local ENU coordinates [e, n, u] (m)
    lla0 : np.ndarray
        Origin geodetic coordinates [lat, lon, height] (rad, rad, m)
    t : float
        Time since reference epoch (s)
        
    Returns:
    --------
    xyz_eci : np.ndarray
        ECI coordinates [x, y, z] (m)
    """
    xyz_ecef = enu2ecef(enu, lla0)
    return ecef2eci(xyz_ecef, t)