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

"""Ephemeris selection and satellite position computation

Supports both RINEX broadcast ephemeris and SP3 precise ephemeris.
"""

import numpy as np
from typing import Optional, Tuple, Union
from ..core.constants import *
from ..core.data_structures import Ephemeris
from .beidou_geo_transform import is_beidou_geo_igso, beidou_geo_transform
from ..core.unified_time import TimeCore, TimeSystem


def timediff(t1, t2):
    """Time difference in seconds handling week rollover
    
    Properly handles GPS time and time of week differences
    """
    # Convert both times to TOW if one is GPS time
    if t1 > 604800:
        t1 = t1 % 604800
    if t2 > 604800:
        t2 = t2 % 604800
        
    # Calculate difference
    dt = t1 - t2
    
    # Handle week rollover
    if dt > 302400:  # More than half week
        dt -= 604800
    elif dt < -302400:
        dt += 604800
        
    return dt


def dtadjust(t1, t2, tw=604800):
    """Calculate delta time considering week-rollover
    
    Parameters
    ----------
    t1 : float
        Time 1 (can be GPS seconds or time of week)
    t2 : float  
        Time 2 (can be GPS seconds or time of week)
    tw : float
        Week duration in seconds (default: 604800)
        
    Returns
    -------
    float
        Adjusted time difference
    """
    dt = timediff(t1, t2)
    if dt > tw / 2:
        dt -= tw
    elif dt < -tw / 2:
        dt += tw
    return dt


def seleph(nav, t, sat):
    """
    Select ephemeris for satellite at given time
    
    Based on rtklib-py implementation
    
    Parameters
    ----------
    nav : NavigationData
        Navigation data containing ephemerides
    t : float
        Time of interest (GPS seconds)
    sat : int
        Satellite number
        
    Returns
    -------
    Ephemeris or GloEphemeris or None
        Best ephemeris for the satellite at given time
    """
    dt_best = 1e10
    eph_best = None
    
    # Get satellite system
    sys = sat2sys(sat)
    
    if sys != SYS_GLO:
        # GPS, Galileo, BeiDou, QZSS, etc.
        for eph in nav.eph:
            if eph.sat != sat:
                continue
                
            # For Galileo, check signal availability
            if sys == SYS_GAL:
                # TODO: Check Galileo signal codes when needed
                pass
                
            # Calculate time difference from time of ephemeris
            # Convert to time of week if needed
            t_week = t % 604800 if t > 604800 else t
            dt = abs(dtadjust(t_week, eph.toe))
            
            # Select ephemeris with minimum time difference
            if dt <= dt_best:
                dt_best = dt
                eph_best = eph
            else:
                # Ephemerides are sorted by time, so we can break
                break
                
    else:
        # GLONASS uses geph
        # For GLONASS, prefer past ephemeris over future ones
        # This matches RTKLIB behavior
        for geph in nav.geph:
            if geph.sat != sat:
                continue
                
            # Convert to time of week if needed
            t_week = t % 604800 if t > 604800 else t
            
            # Calculate time difference with week rollover handling
            dt_raw = dtadjust(t_week, geph.toe)
            
            # For GLONASS, prefer past ephemeris (dt_raw >= 0)
            # Add small penalty for future ephemeris
            if dt_raw < 0:
                # Future ephemeris - add penalty
                dt = abs(dt_raw) + 3600.0  # Add 1 hour penalty
            else:
                # Past ephemeris - use as is
                dt = abs(dt_raw)
            
            if dt <= dt_best:
                dt_best = dt
                eph_best = geph
            else:
                # Since ephemerides are sorted by time, we can break
                # only if we're past the best match
                if dt_best < 7200.0:  # If we have a good match (< 2 hours)
                    break
    
    return eph_best


def eph2clk(t, eph):
    """
    Calculate satellite clock bias from ephemeris
    
    Parameters
    ----------
    t : float
        Time of week in seconds (GPS TOW)
        - For BeiDou, will be converted to BDT internally
    eph : Ephemeris or GloEphemeris
        Broadcast ephemeris
        
    Returns
    -------
    float
        Satellite clock bias (seconds) in GPS time system
    """
    # Check if this is GLONASS ephemeris
    if hasattr(eph, 'taun'):  # GLONASS ephemeris
        # GLONASS clock model: dts = -taun + gamn * (t - toe)
        # Following RTKLIB's geph2clk implementation
        ts = timediff(t, eph.toe)
        
        # Limit time difference
        if abs(ts) > 3600.0:  # 1 hour for GLONASS
            return 0.0
            
        # Iterative calculation (RTKLIB approach)
        t = ts
        for _ in range(2):
            t = ts - (-eph.taun + eph.gamn * t)
            
        # Return clock bias in seconds
        return -eph.taun + eph.gamn * t
    
    # Regular ephemeris (GPS, Galileo, BeiDou, QZSS)
    # Time difference from time of clock
    # Convert to time of week if needed
    t_week = t % 604800 if t > 604800 else t
    
    # Get satellite system
    sys = sat2sys(eph.sat)
    
    # For BeiDou, convert GPS TOW to BDT TOW
    if sys == SYS_BDS:
        # BeiDou ephemeris toc is in BDT
        t_bdt = t_week - GPS_BDS_OFFSET
        if t_bdt < 0:
            t_bdt += 604800
        dt = dtadjust(t_bdt, eph.toc)
    else:
        dt = dtadjust(t_week, eph.toc)
    
    # Limit time difference for extrapolation
    if abs(dt) > 7200.0:  # 2 hours
        return 0.0
    
    # Satellite clock polynomial
    # dts = a0 + a1*dt + a2*dt^2
    dts = eph.f0 + eph.f1 * dt + eph.f2 * dt * dt
    
    return dts


def eph2pos(t, eph):
    """
    Compute satellite position from broadcast ephemeris
    
    Parameters
    ----------
    t : float
        Time of week in seconds:
        - GPS TOW for GPS/QZSS/Galileo/GLONASS
        - GPS TOW for BeiDou (will be converted to BDT internally)
    eph : Ephemeris or GloEphemeris
        Broadcast ephemeris
        
    Returns
    -------
    rs : np.ndarray
        Satellite position in ECEF (m)
    var : float
        Position variance (m^2)
    dts : float
        Satellite clock bias (s) in GPS time system
    """
    # Check if this is GLONASS ephemeris
    if hasattr(eph, 'taun'):  # GLONASS ephemeris
        # Import and use the geph2pos function from satellite.ephemeris module
        from ..satellite.ephemeris import geph2pos
        rs, var, dts = geph2pos(eph, t)
        return rs, var, dts
    
    # Get system-specific constants
    sys = sat2sys(eph.sat)
    
    if sys == SYS_GAL:
        mu = MU_GAL
        omge = OMGE_GAL
    elif sys == SYS_BDS:
        mu = MU_BDS
        omge = OMGE_BDS
    else:  # GPS, QZSS
        mu = MU_GPS
        omge = OMGE
    
    # Time from ephemeris reference epoch
    # Convert to time of week if needed
    t_week = t % 604800 if t > 604800 else t
    
    # For BeiDou, convert GPS TOW to BDT TOW
    if sys == SYS_BDS:
        # BeiDou ephemeris is stored in BDT
        # BDT = GPS - 14 seconds
        t_bdt = t_week - GPS_BDS_OFFSET
        if t_bdt < 0:
            t_bdt += 604800
        tk = dtadjust(t_bdt, eph.toe)
    else:
        tk = dtadjust(t_week, eph.toe)
    
    # Limit time difference
    if tk > 302400.0:
        tk -= 604800.0
    elif tk < -302400.0:
        tk += 604800.0
    
    # Mean anomaly
    n = np.sqrt(mu / eph.A**3)  # Mean motion
    M = eph.M0 + (n + eph.deln) * tk
    
    # Kepler's equation for eccentric anomaly
    E = M
    for _ in range(30):  # Max iterations
        E_old = E
        E = M + eph.e * np.sin(E)
        if abs(E - E_old) < 1e-13:
            break
    
    # True anomaly
    sinE = np.sin(E)
    cosE = np.cos(E)
    v = np.arctan2(np.sqrt(1.0 - eph.e**2) * sinE, cosE - eph.e)
    
    # Argument of latitude
    u = v + eph.omg
    
    # Radius
    r = eph.A * (1.0 - eph.e * cosE)
    
    # Inclination
    i = eph.i0 + eph.idot * tk
    
    # Corrections
    sin2u = np.sin(2.0 * u)
    cos2u = np.cos(2.0 * u)
    
    u += eph.cus * sin2u + eph.cuc * cos2u
    r += eph.crs * sin2u + eph.crc * cos2u
    i += eph.cis * sin2u + eph.cic * cos2u
    
    # Positions in orbital plane
    x = r * np.cos(u)
    y = r * np.sin(u)
    
    # Check for BeiDou GEO satellites (PRN <= 5 or PRN >= 59)
    prn = sat2prn(eph.sat)
    
    # Constants for -5 degree rotation (ref [9] table 4-1)
    SIN_5 = -0.08715574274765817  # sin(-5 deg)
    COS_5 = 0.99619469809174553   # cos(-5 deg)
    
    # Earth-fixed coordinates
    sini = np.sin(i)
    cosi = np.cos(i)
    
    if sys == SYS_BDS and (prn <= 5 or prn >= 59):
        # BeiDou GEO satellites (C01-C05, C59-C63) need special transformation
        # Following RTKLIB ephpos.c exactly (ref [9] table 4-1)
        
        # Use O = OMG0 + OMGd * tk - omge * toes for BeiDou GEO
        O = eph.OMG0 + eph.OMGd * tk - omge * eph.toes
        sinO, cosO = np.sin(O), np.cos(O)
        
        # Compute position without earth rotation first
        xg = x * cosO - y * cosi * sinO
        yg = x * sinO + y * cosi * cosO
        zg = y * sini
        
        # Earth rotation
        ome = omge * tk
        sino = np.sin(ome)
        coso = np.cos(ome)
        
        # Apply -5 degree rotation for BeiDou GEO satellites
        rs = np.array([
            xg * coso + yg * sino * COS_5 + zg * sino * SIN_5,
            -xg * sino + yg * coso * COS_5 + zg * coso * SIN_5,
            -yg * SIN_5 + zg * COS_5
        ])
    else:
        # Standard transformation for other satellites (including BeiDou MEO and IGSO)
        # Note: BeiDou MEO and IGSO satellites use standard transformation without -5 deg rotation
        # Use O = OMG0 + (OMGd - omge) * tk - omge * toes for standard satellites
        O = eph.OMG0 + (eph.OMGd - omge) * tk - omge * eph.toes
        sinO, cosO = np.sin(O), np.cos(O)
        
        rs = np.array([
            x * cosO - y * cosi * sinO,
            x * sinO + y * cosi * cosO,
            y * sini
        ])
    
    # Satellite clock bias
    dts = eph2clk(t, eph)
    
    # Relativistic correction
    dts -= 2.0 * np.sqrt(mu * eph.A) * eph.e * sinE / CLIGHT**2
    
    # Variance (simplified - should use URA)
    var = 3.0**2  # 3m standard deviation
    
    return rs, var, dts


def satpos(obs, nav):
    """
    Compute satellite positions and clocks for all satellites in observation
    (Based on SPP implementation)
    
    Parameters
    ----------
    obs : list of Observation
        Observation data
    nav : NavigationData or SP3Ephemeris
        Navigation data (broadcast) or SP3 ephemeris handler
        
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
    # Check if SP3 or broadcast ephemeris
    from .sp3_ephemeris import SP3Ephemeris
    if isinstance(nav, SP3Ephemeris):
        return satpos_sp3(obs, nav)
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
            
        # Get time as float (GPS seconds)
        if hasattr(ob, 'time'):
            gps_time = ob.time if isinstance(ob.time, (int, float)) else ob.time
        else:
            svh[i] = -1
            continue
            
        # Calculate transmission time
        t_tx = gps_time - pr / CLIGHT
        
        # Convert to week and TOW for ephemeris selection
        from ..core.time import gps_seconds_to_week_tow
        week, tow = gps_seconds_to_week_tow(t_tx)
        
        # Select ephemeris
        eph = seleph(nav, tow, ob.sat)
        if eph is None:
            svh[i] = -1
            continue
            
        try:
            # Get satellite system
            sys = sat2sys(ob.sat)
            
            # For GLONASS, use GPS time directly (no conversion)
            # The GLONASS ephemeris computation should handle time internally
            t_tx_tow = t_tx % 604800  # Convert to TOW (GPS time)
            
            # Satellite clock correction
            dt_sat = eph2clk(t_tx_tow, eph)
            dts[i] = dt_sat
            
            # Satellite position at corrected transmission time
            result = eph2pos(t_tx_tow - dt_sat, eph)
            
            if result is not None:
                # Handle different return types
                if isinstance(result, tuple) and len(result) >= 2:
                    rs[i] = result[0]
                    if len(result) >= 2 and not np.isnan(result[1]).any():
                        var[i] = result[1] if np.isscalar(result[1]) else 0.0
                else:
                    svh[i] = -1
                    continue
                    
                # Sanity check on position
                radius = np.linalg.norm(rs[i])
                if radius < 20000e3 or radius > 50000e3:
                    svh[i] = -1
                    rs[i] = np.zeros(3)
                    dts[i] = 0.0
                    continue
                    
            else:
                svh[i] = -1
                
        except Exception as e:
            svh[i] = -1
            continue
        # Check satellite health
        svh[i] = getattr(eph, 'svh', 0)
        
    return rs, dts, var, svh


def satpos_sp3(obs, sp3_eph, poly_degree: int = 10, method: str = 'neville'):
    """
    Compute satellite positions using SP3 precise ephemeris
    
    Parameters
    ----------
    obs : list of Observation
        Observation data
    sp3_eph : SP3Ephemeris
        SP3 ephemeris handler with loaded data
    poly_degree : int
        Polynomial degree for interpolation (default 10 for RTKLIB)
    method : str
        Interpolation method ('neville', 'polyfit', 'lagrange')
        
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
            
        # Get time
        if hasattr(ob, 'time'):
            if isinstance(ob.time, TimeCore):
                time = ob.time
            else:
                # Assume GPS seconds
                time = TimeCore(ob.time)
        else:
            svh[i] = -1
            continue
            
        # Calculate transmission time
        tau = pr / CLIGHT
        transmit_time = time - tau
        
        # Iterate for better accuracy
        for _ in range(2):
            # Get satellite position from SP3
            pos, clk, position_var = sp3_eph.interpolate_position(
                ob.sat, transmit_time,
                poly_degree=poly_degree,
                method=method
            )
            
            if pos is None:
                svh[i] = -1
                break
                
            # Update transmission time
            tau = np.linalg.norm(pos) / CLIGHT
            transmit_time = time - tau
        
        if pos is not None:
            # Earth rotation correction
            omega_e = 7.2921151467e-5  # Earth rotation rate (rad/s)
            theta = omega_e * tau
            
            # Rotation matrix
            R = np.array([
                [np.cos(theta), np.sin(theta), 0],
                [-np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
            
            # Apply Earth rotation correction
            rs[i] = R @ pos
            dts[i] = clk
            var[i] = position_var
            svh[i] = 0  # SP3 doesn't have health flags, assume healthy
            
            # Sanity check on position
            radius = np.linalg.norm(rs[i])
            if radius < 20000e3 or radius > 50000e3:
                svh[i] = -1
                rs[i] = np.zeros(3)
                dts[i] = 0.0
        else:
            svh[i] = -1
    
    return rs, dts, var, svh


def compute_satellite_position(sat_num: int, time: Union[TimeCore, float], 
                              nav_or_sp3, pseudorange: Optional[float] = None,
                              prefer_mgex: bool = True,
                              **kwargs) -> Tuple[np.ndarray, float, float]:
    """
    Unified function to compute satellite position from either broadcast or SP3 ephemeris
    
    Parameters
    ----------
    sat_num : int
        Satellite number
    time : TimeCore or float
        Time (GPS seconds or TimeCore object)
    nav_or_sp3 : dict or SP3Ephemeris
        Navigation data or SP3 ephemeris object
    pseudorange : float, optional
        Pseudorange for signal transmission time calculation
    prefer_mgex : bool
        If True and using SP3, prefer MGEX products for multi-GNSS
    **kwargs
        Additional arguments for SP3 interpolation
    
    Parameters
    ----------
    sat_num : int
        Satellite number
    time : TimeCore or float
        Time (TimeCore object or GPS seconds)
    nav_or_sp3 : NavigationData or SP3Ephemeris
        Navigation data (broadcast) or SP3 ephemeris handler
    pseudorange : float, optional
        Pseudorange for transmission time calculation
    **kwargs : dict
        Additional arguments for SP3 interpolation (poly_degree, method)
        
    Returns
    -------
    position : np.ndarray
        Satellite position [x, y, z] in ECEF meters
    clock : float
        Satellite clock correction in seconds
    variance : float
        Position variance
    """
    from .sp3_ephemeris import SP3Ephemeris
    
    # Convert time to TimeCore if needed
    if not isinstance(time, TimeCore):
        time = TimeCore(time)
    
    # Handle SP3 ephemeris
    if isinstance(nav_or_sp3, SP3Ephemeris):
        # Calculate transmission time if pseudorange provided
        if pseudorange and pseudorange > 0:
            tau = pseudorange / CLIGHT
            transmit_time = time - tau
            
            # Iterate for better accuracy
            for _ in range(2):
                pos, clk, var = nav_or_sp3.interpolate_position(
                    sat_num, transmit_time,
                    poly_degree=kwargs.get('poly_degree', 10),
                    method=kwargs.get('method', 'neville')
                )
                if pos is None:
                    return None, None, None
                tau = np.linalg.norm(pos) / CLIGHT
                transmit_time = time - tau
        else:
            transmit_time = time
        
        # Get position
        pos, clk, var = nav_or_sp3.interpolate_position(
            sat_num, transmit_time,
            poly_degree=kwargs.get('poly_degree', 10),
            method=kwargs.get('method', 'neville')
        )
        
        # Earth rotation correction if pseudorange provided
        if pos is not None and pseudorange and pseudorange > 0:
            omega_e = 7.2921151467e-5
            theta = omega_e * tau
            R = np.array([
                [np.cos(theta), np.sin(theta), 0],
                [-np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
            pos = R @ pos
        
        return pos, clk, var
    
    # Handle broadcast ephemeris
    else:
        gps_seconds = time.get_gps_seconds()
        
        # Select ephemeris
        eph = seleph(nav_or_sp3, gps_seconds, sat_num)
        if eph is None:
            return None, None, None
        
        # Calculate transmission time if pseudorange provided
        if pseudorange and pseudorange > 0:
            tau = pseudorange / CLIGHT
            transmit_time = gps_seconds - tau
            
            # Iterate for better accuracy
            for _ in range(2):
                pos, vel, clk = eph2pos(transmit_time, eph)
                if pos is None:
                    return None, None, None
                tau = np.linalg.norm(pos) / CLIGHT
                transmit_time = gps_seconds - tau
        else:
            transmit_time = gps_seconds
        
        # Compute position
        pos, vel, clk = eph2pos(transmit_time, eph)
        
        if pos is None:
            return None, None, None
        
        # Earth rotation correction if pseudorange provided
        if pseudorange and pseudorange > 0:
            omega_e = 7.2921151467e-5
            theta = omega_e * tau
            R = np.array([
                [np.cos(theta), np.sin(theta), 0],
                [-np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
            pos = R @ pos
        
        # Estimate variance from SV accuracy
        if hasattr(eph, 'sva'):
            if eph.sva <= 6:
                var = (2**(1 + eph.sva/2))**2
            else:
                var = (2**eph.sva)**2
        else:
            var = 4.0**2  # Default 4m
        
        return pos, clk, var