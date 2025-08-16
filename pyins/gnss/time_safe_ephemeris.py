"""Time-safe ephemeris functions using GNSSTime class"""

import numpy as np
from typing import Union, Tuple, Optional, List
from ..core.constants import *
from ..core.data_structures import Ephemeris, GloEphemeris, Observation, NavigationData
from ..core.time import GNSSTime, gnss_time_diff
from .ephemeris import seleph as _seleph, eph2clk as _eph2clk, eph2pos as _eph2pos
from ..satellite.satellite_position import (
    compute_satellite_position as _compute_satellite_position,
    compute_glonass_position as _compute_glonass_position
)


def timediff_safe(t1: Union[GNSSTime, float], t2: Union[GNSSTime, float], 
                  time_sys: str = 'GPS') -> float:
    """Time difference with type safety
    
    Parameters
    ----------
    t1, t2 : GNSSTime or float
        Times to compare
    time_sys : str
        Time system for float values
        
    Returns
    -------
    float
        Time difference t1 - t2 in seconds
    """
    return gnss_time_diff(t1, t2, time_sys)


def seleph_safe(nav: NavigationData, t: GNSSTime, sat: int) -> Optional[Union[Ephemeris, GloEphemeris]]:
    """Select ephemeris with GNSSTime
    
    Parameters
    ----------
    nav : NavigationData
        Navigation data
    t : GNSSTime
        Time of interest
    sat : int
        Satellite number
        
    Returns
    -------
    Ephemeris or GloEphemeris or None
        Best ephemeris for the satellite
    """
    # Convert to GPS seconds for legacy function
    gps_seconds = t.to_gps_seconds()
    return _seleph(nav, gps_seconds, sat)


def eph2clk_safe(t: GNSSTime, eph: Union[Ephemeris, GloEphemeris]) -> float:
    """Compute satellite clock with GNSSTime
    
    Parameters
    ----------
    t : GNSSTime
        Time of interest
    eph : Ephemeris or GloEphemeris
        Ephemeris data
        
    Returns
    -------
    float
        Satellite clock bias in seconds
    """
    # Get system
    sys = sat2sys(eph.sat)
    
    # Convert time to appropriate system
    if sys == SYS_BDS:
        # BeiDou uses BDT
        t_sys = t.convert_to('BDS')
    elif sys == SYS_GLO:
        # GLONASS uses GLO time
        t_sys = t.convert_to('GLO')
    elif sys == SYS_GAL:
        # Galileo uses GAL time
        t_sys = t.convert_to('GAL')
    else:
        # GPS/QZSS use GPS time
        t_sys = t.convert_to('GPS')
    
    # Use TOW for computation
    return _eph2clk(t_sys.tow, eph)


def eph2pos_safe(t: GNSSTime, eph: Union[Ephemeris, GloEphemeris]) -> Tuple[np.ndarray, float, float]:
    """Compute satellite position with GNSSTime
    
    Parameters
    ----------
    t : GNSSTime
        Time of interest
    eph : Ephemeris or GloEphemeris
        Ephemeris data
        
    Returns
    -------
    rs : np.ndarray
        Satellite position (3,)
    var : float
        Position variance
    dts : float
        Satellite clock bias
    """
    # Get system
    sys = sat2sys(eph.sat)
    
    # Convert time to appropriate system
    if sys == SYS_BDS:
        t_sys = t.convert_to('BDS')
    elif sys == SYS_GLO:
        t_sys = t.convert_to('GLO')
    elif sys == SYS_GAL:
        t_sys = t.convert_to('GAL')
    else:
        t_sys = t.convert_to('GPS')
    
    # Use TOW for computation
    return _eph2pos(t_sys.tow, eph)


def compute_satellite_position_safe(eph: Ephemeris, t: GNSSTime) -> Tuple[np.ndarray, float, np.ndarray]:
    """Compute satellite position with GNSSTime
    
    Parameters
    ----------
    eph : Ephemeris
        Broadcast ephemeris
    t : GNSSTime
        Signal transmission time
        
    Returns
    -------
    rs : np.ndarray
        Satellite position in ECEF (m)
    dts : float
        Satellite clock bias (s)
    var : np.ndarray
        Satellite position and clock variance
    """
    # Get system and convert time
    sys = sat2sys(eph.sat)
    
    if sys == SYS_BDS:
        t_sys = t.convert_to('BDS')
    elif sys == SYS_GAL:
        t_sys = t.convert_to('GAL')
    else:
        t_sys = t.convert_to('GPS')
    
    # Use float for legacy function
    return _compute_satellite_position(eph, t_sys.tow)


def compute_glonass_position_safe(geph: GloEphemeris, t: GNSSTime) -> Tuple[np.ndarray, float, float]:
    """Compute GLONASS satellite position with GNSSTime
    
    Parameters
    ----------
    geph : GloEphemeris
        GLONASS broadcast ephemeris
    t : GNSSTime
        Signal transmission time
        
    Returns
    -------
    rs : np.ndarray
        Satellite position in ECEF (m)
    var : float
        Position variance
    dts : float
        Satellite clock bias (s)
    """
    # Convert to GLONASS time
    t_glo = t.convert_to('GLO')
    
    # Use TOW for legacy function
    return _compute_glonass_position(geph, t_glo.tow)


def satpos_safe(obs: List[Observation], nav: NavigationData) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute satellite positions with GNSSTime
    
    Parameters
    ----------
    obs : list of Observation
        Observation data
    nav : NavigationData
        Navigation data
        
    Returns
    -------
    rs : np.ndarray
        Satellite positions (n x 3)
    dts : np.ndarray
        Satellite clock biases (n)
    var : np.ndarray
        Position variances (n)
    svh : np.ndarray
        Satellite health (n)
    """
    n = len(obs)
    rs = np.zeros((n, 3))
    dts = np.zeros(n)
    var = np.zeros(n)
    svh = np.zeros(n, dtype=int)
    
    for i, ob in enumerate(obs):
        # Get pseudorange (prefer L1)
        pr = ob.P[0] if ob.P[0] > 0 else ob.P[1]
        if pr == 0:
            svh[i] = -1
            continue
        
        # Convert observation time to GNSSTime if needed
        if isinstance(ob.time, (int, float)):
            # Assume GPS time
            if ob.time > 604800:
                t_rx = GNSSTime.from_gps_seconds(ob.time, 'GPS')
            else:
                t_rx = GNSSTime(0, ob.time, 'GPS')
        else:
            t_rx = ob.time
            
        # Signal transmission time
        t_tx = t_rx - (pr / CLIGHT)
        
        # Select ephemeris
        eph = seleph_safe(nav, t_tx, ob.sat)
        if eph is None:
            svh[i] = -1
            continue
            
        # Satellite clock correction
        dt_sat = eph2clk_safe(t_tx, eph)
        
        # Corrected transmission time
        t_tx_corrected = t_tx - dt_sat
        
        # Satellite position at transmission time
        rs[i], var[i], dts[i] = eph2pos_safe(t_tx_corrected, eph)
        svh[i] = getattr(eph, 'svh', 0)
        
    return rs, dts, var, svh