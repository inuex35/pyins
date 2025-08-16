"""Ephemeris selection and validation"""

import numpy as np
from typing import Optional, List
from ..core.data_structures import Ephemeris, NavigationData, GloEphemeris
from ..core.constants import *

def geph2clk(geph: GloEphemeris, time: float) -> float:
    """
    GLONASS ephemeris to satellite clock bias
    
    Parameters:
    -----------
    geph : GloEphemeris
        GLONASS ephemeris
    time : float
        Time of interest (GPST)
        
    Returns:
    --------
    dts : float
        Satellite clock bias (s)
    """
    t = time - geph.toe
    
    # Iterative solution for clock bias
    ts = t
    for i in range(2):
        t = ts - (-geph.taun + geph.gamn * t)
    
    return -geph.taun + geph.gamn * t


def geph2pos(geph: GloEphemeris, time: float) -> tuple:
    """
    GLONASS ephemeris to satellite position and clock bias
    
    Parameters:
    -----------
    geph : GloEphemeris
        GLONASS ephemeris
    time : float
        Time of interest (GPST)
        
    Returns:
    --------
    rs : np.ndarray
        Satellite position (ECEF, m)
    dts : float
        Satellite clock bias (s)
    var : float
        Position and clock variance (m^2)
    """
    t = time - geph.toe
    dts = -geph.taun + geph.gamn * t
    x = np.array((*geph.pos, *geph.vel))
    
    # Debug trace (commented out)
    # trace(4, 'geph2pos: sat=%d\n' % geph.sat)
    tt = -TSTEP if t < 0 else TSTEP
    while abs(t) > 1E-5:  #1E-9
        if abs(t) < TSTEP:
            tt = t
        x = glorbit(tt, x, geph.acc)
        t -= tt

    var = ERREPH_GLO**2
    return x[0:3], var, dts


def glorbit(t, x, acc):
    """ glonass position and velocity by numerical integration """
    k1 = deq(x, acc)
    w =x + k1 * t / 2
    k2 = deq(w, acc)
    w = x + k2 * t / 2
    k3 = deq(w, acc)
    w = x + k3 * t
    k4 = deq(w, acc)
    x += (k1 + 2 * k2 + 2 * k3 + k4) * t / 6
    return x


def deq(x, acc):
    """glonass orbit differential equations """
    xdot = np.zeros(6)
    r2 = np.dot(x[0:3], x[0:3])
    if r2 <= 0.0:
        return xdot
    r3 = r2 * np.sqrt(r2)
    omg2 = OMGE_GLO**2

    a = 1.5 * J2_GLO * MU_GLO * RE_GLO**2 / r2 / r3 
    b = 5.0 * x[2]**2 / r2 
    c = -MU_GLO / r3 - a * (1.0 - b)
    xdot[0:3] = x[3:6]
    xdot[3] = (c + omg2) * x[0] + 2.0 * OMGE_GLO * x[4] + acc[0]
    xdot[4] = (c + omg2) * x[1] - 2.0 * OMGE_GLO * x[3] + acc[1]
    xdot[5] = (c - 2.0 * a) * x[2] + acc[2]
    return xdot

def select_glonass_ephemeris(nav: NavigationData, sat: int, time: float) -> Optional[GloEphemeris]:
    """
    Select best GLONASS ephemeris for satellite at given time
    
    Parameters:
    -----------
    nav : NavigationData
        Navigation data containing GLONASS ephemerides
    sat : int
        Satellite number
    time : float
        Time of interest (GPST)
        
    Returns:
    --------
    geph : GloEphemeris or None
        Best GLONASS ephemeris or None if not found
    """
    if not hasattr(nav, 'geph') or not nav.geph:
        return None
        
    best_geph = None
    min_dt = float('inf')
    MAXDTOE_GLO = 1800.0  # 30 minutes for GLONASS
    
    for geph in nav.geph:
        if geph.sat != sat:
            continue
            
        # Check time validity
        dt = abs(time - geph.toe)
        if dt > MAXDTOE_GLO:
            continue
            
        if dt < min_dt:
            min_dt = dt
            best_geph = geph
            
    return best_geph


def select_ephemeris(nav: NavigationData, sat: int, time: float) -> Optional[Ephemeris]:
    """
    Select best ephemeris for satellite at given time
    
    Parameters:
    -----------
    nav : NavigationData
        Navigation data containing ephemerides
    sat : int
        Satellite number
    time : float
        Time of interest (TOW - Time of Week)
        
    Returns:
    --------
    eph : Ephemeris or None
        Best ephemeris or None if not found
    """
    best_eph = None
    min_dt = float('inf')
    
    for eph in nav.eph:
        if eph.sat != sat:
            continue
            
        # Check ephemeris validity using time of week
        if not is_ephemeris_valid_tow(eph, time):
            continue
            
        # Find closest ephemeris using time of week
        dt = abs(time - eph.toe)
        # Handle week wraparound
        if dt > 302400:  # Half week
            dt = 604800 - dt
            
        if dt < min_dt:
            min_dt = dt
            best_eph = eph
            
    return best_eph


def is_ephemeris_valid_tow(eph: Ephemeris, time_of_week: float) -> bool:
    """
    Check if ephemeris is valid at given time of week
    
    Parameters:
    -----------
    eph : Ephemeris
        Ephemeris to check
    time_of_week : float
        Time of week (0-604800 seconds)
        
    Returns:
    --------
    valid : bool
        True if ephemeris is valid
    """
    # Check SV health
    if eph.svh != 0:
        return False
        
    # Check time validity using time of week
    dt = abs(time_of_week - eph.toe)
    # Handle week wraparound
    if dt > 302400:  # Half week
        dt = 604800 - dt
    
    # System-specific validity periods
    sys = sat2sys(eph.sat)
    if sys == SYS_GPS or sys == SYS_QZS:
        max_dt = 7200.0  # 2 hours
    elif sys == SYS_GAL:
        max_dt = 10800.0  # 3 hours  
    elif sys == SYS_BDS:
        max_dt = 21600.0  # 6 hours
    elif sys == SYS_GLO:
        max_dt = 1800.0   # 30 minutes
    else:
        max_dt = 3600.0   # 1 hour default
        
    return dt <= max_dt


def is_ephemeris_valid(eph: Ephemeris, time: float) -> bool:
    """
    Check if ephemeris is valid at given time
    
    Parameters:
    -----------
    eph : Ephemeris
        Ephemeris to check
    time : float
        Time of interest (TOW - Time of Week)
        
    Returns:
    --------
    valid : bool
        True if ephemeris is valid
    """
    # Check SV health
    if eph.svh != 0:
        return False
        
    # Check time validity using time of week
    dt = abs(time - eph.toe)
    # Handle week wraparound
    if dt > 302400:  # Half week
        dt = 604800 - dt
    
    # System-specific validity periods
    sys = sat2sys(eph.sat)
    if sys == SYS_GPS or sys == SYS_QZS:
        max_dt = 7200.0  # 2 hours
    elif sys == SYS_GAL:
        max_dt = 10800.0  # 3 hours  
    elif sys == SYS_BDS:
        max_dt = 21600.0  # 6 hours
    elif sys == SYS_GLO:
        max_dt = 1800.0   # 30 minutes (MAXDTOE_GLO)
    else:
        max_dt = 3600.0   # 1 hour default
        
    # Check fit interval if available
    if hasattr(eph, 'fit') and eph.fit > 0:
        max_dt = eph.fit * 3600.0 / 2.0
        
    return dt <= max_dt


def ephemeris_age(eph: Ephemeris, time: float) -> float:
    """
    Compute age of ephemeris
    
    Parameters:
    -----------
    eph : Ephemeris
        Ephemeris
    time : float
        Current time (TOW - Time of Week)
        
    Returns:
    --------
    age : float
        Age in seconds
    """
    dt = abs(time - eph.toc)
    # Handle week wraparound
    if dt > 302400:  # Half week
        dt = 604800 - dt
    return dt


def select_eph_list(nav: NavigationData, time: float, 
                    systems: int = SYS_ALL) -> List[Ephemeris]:
    """
    Select valid ephemerides for all satellites at given time
    
    Parameters:
    -----------
    nav : NavigationData
        Navigation data
    time : float
        Time of interest (TOW - Time of Week)
    systems : int
        Satellite systems to include (bitmask)
        
    Returns:
    --------
    eph_list : List[Ephemeris]
        List of valid ephemerides
    """
    eph_list = []
    
    # Track which satellites we've already found
    found_sats = set()
    
    for eph in nav.eph:
        # Check system
        if not (sat2sys(eph.sat) & systems):
            continue
            
        # Skip if we already have ephemeris for this satellite
        if eph.sat in found_sats:
            continue
            
        # Check validity
        if is_ephemeris_valid(eph, time):
            eph_list.append(eph)
            found_sats.add(eph.sat)
            
    return eph_list


class EphemerisManager:
    """Manage ephemeris data and updates"""
    
    def __init__(self):
        self.ephemerides = {}  # sat -> list of ephemeris
        self.max_age = 7200.0  # Maximum ephemeris age (2 hours)
        
    def add_ephemeris(self, eph: Ephemeris):
        """Add new ephemeris"""
        if eph.sat not in self.ephemerides:
            self.ephemerides[eph.sat] = []
            
        # Check if this is a duplicate
        for existing in self.ephemerides[eph.sat]:
            if (existing.iode == eph.iode and 
                existing.iodc == eph.iodc and
                abs(existing.toe - eph.toe) < 1.0):
                return  # Skip duplicate
                
        self.ephemerides[eph.sat].append(eph)
        
        # Sort by toe
        self.ephemerides[eph.sat].sort(key=lambda e: e.toe)
        
    def get_ephemeris(self, sat: int, time: float) -> Optional[Ephemeris]:
        """Get best ephemeris for satellite at time"""
        if sat not in self.ephemerides:
            return None
            
        best_eph = None
        min_dt = float('inf')
        
        for eph in self.ephemerides[sat]:
            if not is_ephemeris_valid(eph, time):
                continue
                
            dt = abs(time - eph.toe)
            # Handle week wraparound
            if dt > 302400:  # Half week
                dt = 604800 - dt
            
            if dt < min_dt:
                min_dt = dt
                best_eph = eph
                
        return best_eph
    
    def clean_old_ephemerides(self, current_time: float):
        """Remove old ephemerides"""
        for sat in list(self.ephemerides.keys()):
            self.ephemerides[sat] = [
                eph for eph in self.ephemerides[sat]
                if ephemeris_age(eph, current_time) < self.max_age
            ]
            
            if not self.ephemerides[sat]:
                del self.ephemerides[sat]