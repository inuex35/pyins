"""Single Point Positioning (SPP) implementation based on rtklib-py"""

import numpy as np
from numpy.linalg import norm
from datetime import datetime
from pyins.core.data_structures import Observation, NavigationData, Solution
from pyins.core.constants import (
    CLIGHT, OMGE, RE_WGS84, FE_WGS84,
    SYS_GPS, SYS_GLO, SYS_GAL, SYS_BDS, SYS_QZS, 
    sat2sys, sys2char
)
from pyins.gnss.ephemeris import seleph, eph2pos
from pyins.coordinate import ecef2llh
from pyins.core.unified_time import TimeCore, TimeSystem

# Constants
MAXITR = 10          # max iterations
MIN_EL = 5.0         # min elevation in degrees
ERR_SAAS = 0.3       # Saastamoinen model error
ERR_ION = 5.0        # Ionosphere error
ERR_CBIAS = 0.3      # Code bias error


def tropmodel_simple(pos, el):
    """Simple tropospheric model (Saastamoinen-like)"""
    if el < np.deg2rad(MIN_EL):
        return 0.0
    
    # Standard atmosphere at sea level
    P0 = 1013.25  # hPa
    T0 = 288.15   # K
    e0 = 11.75    # hPa (water vapor pressure)
    
    # Height correction
    h = pos[2] if len(pos) > 2 else 0.0
    if h < 0:
        h = 0.0
    
    # Pressure and temperature at height
    P = P0 * (1 - 2.26e-5 * h) ** 5.225
    T = T0 - 6.5e-3 * h
    e = e0 * (T / T0) ** 4.0
    
    # Zenith delays
    zhd = 0.0022768 * P / (1 - 0.00266 * np.cos(2 * pos[0]) - 0.00028e-3 * h)
    zwd = 0.0022768 * (1255 / T + 0.05) * e
    
    # Mapping function (simple)
    mapf = 1.0 / np.sin(el)
    
    return (zhd + zwd) * mapf


def sagnac_correction(sat_pos, rec_pos):
    """Sagnac effect correction"""
    return (OMGE / CLIGHT) * (sat_pos[0] * rec_pos[1] - sat_pos[1] * rec_pos[0])


def geodist(sat_pos, rec_pos):
    """Geometric distance and unit vector"""
    diff = sat_pos - rec_pos
    r = norm(diff)
    if r > 0:
        e = diff / r
    else:
        e = np.zeros(3)
    return r, e


def satazel(pos, e):
    """Satellite azimuth/elevation from receiver position and line-of-sight vector"""
    lat, lon, h = pos[0], pos[1], pos[2]
    
    # ENU transformation matrix
    R = np.array([
        [-np.sin(lon), np.cos(lon), 0],
        [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)],
        [np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)]
    ])
    
    # Transform to ENU
    enu = R @ e
    
    # Azimuth and elevation
    az = np.arctan2(enu[0], enu[1])
    if az < 0:
        az += 2 * np.pi
    el = np.arcsin(enu[2])
    
    return az, el


def varerr(sys, el):
    """Variance of pseudorange error"""
    a = 0.003  # Base error (m)
    b = 0.003  # Elevation-dependent error (m)
    
    s_el = np.sin(el)
    if s_el <= 0:
        return 100.0  # Large error for negative elevation
    
    # Basic elevation-dependent model
    var = (a ** 2) + (b / s_el) ** 2
    
    # Add system-specific errors
    if sys == SYS_GLO:
        var *= 1.5  # GLONASS typically has larger errors
    elif sys == SYS_BDS:
        var *= 1.2  # BeiDou slightly larger errors
        
    return var


def single_point_positioning(observations, nav_data, initial_pos=None, 
                           systems_to_use=None, frequencies_to_use=None):
    """
    Perform single point positioning using iterative least squares
    
    Parameters:
    -----------
    observations : list
        List of GNSS observations
    nav_data : NavigationData
        Navigation data with ephemerides
    initial_pos : np.ndarray, optional
        Initial position estimate (ECEF)
    systems_to_use : list, optional
        List of satellite system characters to use (e.g., ['G', 'E'])
    frequencies_to_use : list, optional
        List of frequency bands to use (e.g., ['L1', 'L5'])
        
    Returns:
    --------
    solution : Solution
        Position solution
    used_sats : list
        List of satellites used in the solution
    """
    if not observations:
        return None, []
    
    # Default systems to use (exclude GLONASS)
    if systems_to_use is None:
        systems_to_use = ['G', 'E', 'C', 'J', 'R']  # GPS, Galileo, BeiDou, QZSS (no GLONASS)
    print(f"Systems to use: {systems_to_use}")
    # Filter observations by system
    filtered_obs = []
    for obs in observations:
        sys = sat2sys(obs.sat)
        if sys == 0:  # Skip invalid satellites
            continue
        sys_char = sys2char(sys)
        if sys_char in systems_to_use:
            filtered_obs.append(obs)
    
    print(f"Total observations: {len(observations)}, Filtered: {len(filtered_obs)}")
    if len(filtered_obs) < 4:
        print(f"Insufficient observations: {len(filtered_obs)}")
        # Debug: show satellite systems
        for obs in observations[:5]:
            sys_char = sys2char(sat2sys(obs.sat))
            print(f"  Sat {obs.sat} -> System: {sys_char}")
        return None, []
    
    # Initial state [x, y, z, dtr_gps, dtr_glo, dtr_gal, dtr_bds]
    # Note: Following rtklib-py approach, we use separate clock for each system
    if initial_pos is None:
        x = np.zeros(7)
    else:
        x = np.zeros(7)
        x[:3] = initial_pos.copy()
    
    # Iterative least squares
    print(f"Initial state: pos={x[:3]/1e3} km")
    for iteration in range(MAXITR):
        H = []
        v = []
        var = []
        used_sats = []
        
        pos = x[:3]
        dtr = x[3:]  # Clock biases in seconds
        
        # Convert to LLH for elevation calculation
        if norm(pos) > RE_WGS84:
            llh = ecef2llh(pos)
        else:
            llh = np.array([0, 0, 0])  # Use equator for initial guess
        
        for obs in filtered_obs:
            # Get pseudorange
            pr = obs.P[0] if obs.P[0] > 0 else obs.P[1]
            if pr <= 0:
                if iteration == 0:
                    print(f"  Sat {obs.sat}: No valid pseudorange")
                continue
            
            # Calculate transmission time
            # Convert observation time to TimeCore if needed
            if isinstance(obs.time, (int, float)):
                tc_rx = TimeCore.from_auto(obs.time)
            else:
                tc_rx = obs.time  # Already TimeCore
                
            # Signal transmission time
            tc_tx = tc_rx - (pr / CLIGHT)
            
            # Get appropriate TOW for ephemeris selection
            sat_sys = sat2sys(obs.sat)
            if sat_sys == SYS_BDS:
                tow = tc_tx.get_tow(TimeSystem.BDS)
            else:
                tow = tc_tx.get_tow(TimeSystem.GPS)
            
            # Get ephemeris (seleph expects TOW)
            eph = seleph(nav_data, tow, obs.sat)
            if eph is None:
                if iteration == 0:
                    print(f"  No ephemeris for sat {obs.sat} at tow {tow}")
                continue
            
            # Get satellite position and clock
            # Note: Modern RINEX files typically store BeiDou ephemeris in GPS time
            # So we don't need to convert to BDT for ephemeris calculations
            sat_pos, sat_var, dts = eph2pos(tow, eph)
            if np.any(np.isnan(sat_pos)) or norm(sat_pos) < RE_WGS84:
                if iteration == 0:
                    print(f"  Invalid sat position for sat {obs.sat}")
                continue
            
            # Geometric range and unit vector
            r, e = geodist(sat_pos, pos)
            if r <= 0:
                continue
            
            # Elevation check
            if norm(pos) > RE_WGS84:
                az, el = satazel(llh, e)
                if el < np.deg2rad(MIN_EL):
                    continue
            else:
                el = np.pi/4  # Assume 45 degrees for initial iteration
            
            # System-specific clock bias
            sys = sat2sys(obs.sat)
            if sys == SYS_GPS or sys == SYS_QZS:
                clk_bias = dtr[0]
                clk_idx = 0
            elif sys == SYS_GLO:
                clk_bias = dtr[0] + dtr[1]
                clk_idx = 1
            elif sys == SYS_GAL:
                clk_bias = dtr[0] + dtr[2]
                clk_idx = 2
            elif sys == SYS_BDS:
                clk_bias = dtr[0] + dtr[3]
                clk_idx = 3
            else:
                continue
            
            # Corrections
            dtrp = tropmodel_simple(llh, el) if iteration > 0 else 0.0
            dion = 0.0  # No ionosphere correction for single frequency
            sagnac = sagnac_correction(sat_pos, pos)
            
            # Pseudorange residual
            # pr = r + c*dtr - c*dts + dion + dtrp + sagnac
            # Note: clk_bias is in seconds, need to multiply by c
            res = pr - (r + sagnac + clk_bias * CLIGHT - dts * CLIGHT + dion + dtrp)
            
            # Design matrix row
            H_row = np.zeros(7)
            H_row[:3] = -e  # Position partials
            H_row[3] = CLIGHT  # GPS clock (multiply by c since state is in seconds)
            if clk_idx > 0 and clk_idx < 4:
                H_row[3 + clk_idx] = CLIGHT  # ISB term
            
            H.append(H_row)
            v.append(res)
            var.append(varerr(sys, el))
            used_sats.append(obs.sat)
            
            if iteration == 0 and len(v) <= 5:
                print(f"  Sat {obs.sat}: r={r/1e3:.1f}km, res={res/1e3:.1f}km, el={np.rad2deg(el):.1f}°, pos_norm={norm(pos)/1e3:.1f}km")
        
        print(f"Iteration {iteration}: {len(v)} valid measurements")
        if len(v) < 4:
            print(f"Insufficient valid measurements: {len(v)}")
            if iteration == 0:
                print(f"  Checked {len(filtered_obs)} satellites")
                valid_pr = sum(1 for obs in filtered_obs if obs.P[0] > 0 or obs.P[1] > 0)
                print(f"  Valid pseudoranges: {valid_pr}")
                # Count failures at each step
                no_eph = 0
                bad_pos = 0
                low_el = 0
                for obs in filtered_obs[:10]:  # Check first 10
                    pr = obs.P[0] if obs.P[0] > 0 else obs.P[1]
                    if pr <= 0:
                        continue
                    t_tx = obs.time - pr / CLIGHT
                    from pyins.core.time import gps_seconds_to_week_tow
                    week, tow = gps_seconds_to_week_tow(t_tx)
                    eph = seleph(nav_data, tow, obs.sat)
                    if eph is None:
                        no_eph += 1
                    else:
                        sat_pos, _, _ = eph2pos(tow, eph)
                        if np.any(np.isnan(sat_pos)) or norm(sat_pos) < RE_WGS84:
                            bad_pos += 1
                print(f"  No ephemeris: {no_eph}, Bad position: {bad_pos}")
            return None, []
        
        # Convert to arrays
        H = np.array(H)
        v = np.array(v)
        var = np.array(var)
        
        # Remove unused clock parameters
        active_params = [True, True, True, True]  # x, y, z, dtr_gps always active
        for i in range(1, 4):
            col_sum = np.sum(np.abs(H[:, 3+i]))
            if col_sum == 0:
                active_params.append(False)
            else:
                active_params.append(True)
        
        # Keep only active parameters
        active_idx = np.where(active_params[:len(H[0])])[0]
        H_reduced = H[:, active_idx]
        
        # Weighted least squares
        W = np.diag(1.0 / var)
        
        try:
            # Normal equation: (H^T W H) dx = H^T W v
            N = H_reduced.T @ W @ H_reduced
            b = H_reduced.T @ W @ v
            dx_reduced = np.linalg.solve(N, b)
            
            # Map back to full state update
            dx = np.zeros(7)
            dx[active_idx] = dx_reduced
            
        except np.linalg.LinAlgError:
            print("Failed to solve normal equations")
            return None, []
        
        # Update state
        x += dx
        
        print(f"  dx: pos=[{dx[0]/1e3:.3f}, {dx[1]/1e3:.3f}, {dx[2]/1e3:.3f}] km, clock={dx[3]*1e3:.3f} ms")
        print(f"  Updated pos: [{x[0]/1e3:.1f}, {x[1]/1e3:.1f}, {x[2]/1e3:.1f}] km, norm={norm(x[:3])/1e3:.1f} km")
        
        # Check convergence
        if norm(dx[:3]) < 1e-4:
            break
    
    # Create solution
    pos = x[:3]
    dtr = x[3:]
    
    # Covariance matrix (simplified)
    try:
        Q = np.linalg.inv(H.T @ W @ H)
    except:
        Q = np.eye(7) * 100.0
    
    solution = Solution(
        time=observations[0].time,
        type=5,  # Single point
        rr=pos,
        dtr=dtr,
        qr=Q[:3, :3],  # Position covariance
        ns=len(used_sats)
    )
    
    return solution, used_sats


def main():
    """Example usage"""
    # TimeCore is now imported at the top
    
    print("Single Point Positioning Example")
    print("=" * 40)
    
    # Create sample data
    current_tc = TimeCore.from_datetime(datetime.now())
    
    # Sample observations
    observations = []
    for prn in [1, 3, 7, 11, 15, 20, 25, 30]:
        obs = Observation(
            time=current_tc.get_gps_seconds(),
            sat=prn,
            system=SYS_GPS,  # GPS system
            P=np.array([20e6 + np.random.uniform(0, 5e6), 0, 0]),
            SNR=np.array([40 + np.random.uniform(0, 10), 0, 0])
        )
        observations.append(obs)
    
    # Empty navigation data for example
    nav_data = NavigationData()
    
    # Run SPP
    solution, used_sats = single_point_positioning(observations, nav_data)
    
    if solution:
        llh = ecef2llh(solution.rr)
        print(f"\nSolution found:")
        print(f"  Position: {solution.rr}")
        print(f"  LLH: {np.rad2deg(llh[0]):.6f}°, {np.rad2deg(llh[1]):.6f}°, {llh[2]:.1f}m")
        print(f"  Clock bias: {solution.dtr[0]*1e9:.1f} ns")
        print(f"  Satellites used: {len(used_sats)}")
    else:
        print("No solution found")


if __name__ == "__main__":
    main()