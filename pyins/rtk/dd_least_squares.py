"""Double Difference Least Squares Solution for RTK"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from ..core.data_structures import Observation
from ..core.constants import CLIGHT, sat2sys, SYS_GLO
from ..gnss.ephemeris import seleph, eph2pos
from ..core.unified_time import TimeCore, TimeSystem
from .double_difference import DoubleDifferenceProcessor
import copy


def interpolate_observations(obs1, obs2, t1, t2, t_target):
    """
    Linear interpolation of observations to target time
    
    Parameters:
    -----------
    obs1, obs2 : Observation
        Observations at time t1 and t2
    t1, t2 : float
        Times of observations (GPS seconds)
    t_target : float
        Target time for interpolation
        
    Returns:
    --------
    obs_interp : Observation
        Interpolated observation
    """
    # Calculate interpolation weight
    alpha = (t_target - t1) / (t2 - t1)
    
    # Create interpolated observation
    obs_interp = copy.deepcopy(obs1)
    
    # Interpolate pseudorange
    for i in range(3):
        if obs1.P[i] > 0 and obs2.P[i] > 0:
            obs_interp.P[i] = (1 - alpha) * obs1.P[i] + alpha * obs2.P[i]
    
    # Interpolate carrier phase
    for i in range(3):
        if obs1.L[i] != 0 and obs2.L[i] != 0:
            obs_interp.L[i] = (1 - alpha) * obs1.L[i] + alpha * obs2.L[i]
    
    # Update time
    obs_interp.time = t_target
    
    return obs_interp


def interpolate_epoch(epoch1, epoch2, t_target):
    """
    Interpolate entire epoch to target time
    
    Returns interpolated observations for all common satellites
    """
    t1 = epoch1['gps_time']
    t2 = epoch2['gps_time']
    
    # Get common satellites
    sats1 = {obs.sat: obs for obs in epoch1['observations']}
    sats2 = {obs.sat: obs for obs in epoch2['observations']}
    common_sats = set(sats1.keys()) & set(sats2.keys())
    
    # Interpolate each satellite
    interp_obs = []
    for sat in common_sats:
        obs_interp = interpolate_observations(
            sats1[sat], sats2[sat], t1, t2, t_target
        )
        interp_obs.append(obs_interp)
    
    # Create interpolated epoch
    interp_epoch = {
        'time': t_target,
        'gps_time': t_target,
        'gps_week': epoch1['gps_week'],
        'gps_tow': t_target - epoch1['gps_week'] * 604800,
        'n_sats': len(interp_obs),
        'observations': interp_obs
    }
    
    return interp_epoch


class DDLeastSquares:
    """Double Difference Least Squares solver for RTK positioning"""
    
    def __init__(self, dd_processor: Optional[DoubleDifferenceProcessor] = None):
        """
        Initialize DD least squares solver
        
        Parameters:
        -----------
        dd_processor : DoubleDifferenceProcessor, optional
            DD processor instance
        """
        self.dd_processor = dd_processor or DoubleDifferenceProcessor()
        
    def solve_baseline(self, 
                      rover_obs: List[Observation],
                      base_obs: List[Observation],
                      nav_data: dict,
                      base_pos: np.ndarray,
                      initial_baseline: Optional[np.ndarray] = None,
                      max_iter: int = 5,
                      convergence_threshold: float = 1e-4) -> Tuple[np.ndarray, float, int]:
        """
        Solve for baseline vector using DD least squares
        
        Parameters:
        -----------
        rover_obs : List[Observation]
            Rover observations
        base_obs : List[Observation]
            Base observations
        nav_data : dict
            Navigation data with ephemerides
        base_pos : np.ndarray
            Base station position in ECEF
        initial_baseline : np.ndarray, optional
            Initial baseline estimate (default: zero)
        max_iter : int
            Maximum iterations
        convergence_threshold : float
            Convergence threshold in meters
            
        Returns:
        --------
        baseline : np.ndarray
            Estimated baseline vector
        residual_rms : float
            RMS of residuals
        n_iter : int
            Number of iterations
        """
        # Filter valid observations (no GLONASS)
        rover_obs_valid = [obs for obs in rover_obs 
                          if obs.P[0] > 0 and obs.L[0] > 0 and sat2sys(obs.sat) != SYS_GLO]
        base_obs_valid = [obs for obs in base_obs 
                         if obs.P[0] > 0 and obs.L[0] > 0 and sat2sys(obs.sat) != SYS_GLO]
        
        # Form double differences
        dd_pr, dd_cp, sat_pairs, ref_sats = self.dd_processor.form_double_differences(
            rover_obs_valid, base_obs_valid
        )
        
        if len(dd_pr) < 3:
            raise ValueError(f"Not enough DD observations: {len(dd_pr)}")
        
        # Calculate satellite positions
        tc_rx = TimeCore.from_auto(rover_obs[0].time)
        sat_positions = self._calculate_satellite_positions(
            rover_obs_valid, nav_data, tc_rx
        )
        
        # Filter valid DD pairs
        valid_pairs = [(ref, other) for ref, other in sat_pairs 
                      if ref in sat_positions and other in sat_positions]
        valid_dd_pr = dd_pr[:len(valid_pairs)]
        
        # Initial baseline
        if initial_baseline is None:
            baseline = np.zeros(3)
        else:
            baseline = initial_baseline.copy()
        
        # Iterative least squares
        for iteration in range(max_iter):
            # Current rover position
            rover_pos = base_pos + baseline
            
            # Compute geometry matrix
            H = self.dd_processor.compute_dd_geometry_matrix(
                sat_positions, rover_pos, valid_pairs
            )
            
            # Compute DD residuals
            residuals = self.dd_processor.compute_dd_residuals(
                valid_dd_pr, sat_positions, rover_pos, base_pos, 
                valid_pairs, use_carrier=False
            )
            
            # Least squares solution
            # Normal equation: (H^T * H) * dx = H^T * residuals
            try:
                HTH = H.T @ H
                HTr = H.T @ residuals
                dx = np.linalg.solve(HTH, HTr)
            except np.linalg.LinAlgError:
                raise ValueError("Matrix is singular, cannot solve")
            
            # Update baseline
            baseline += dx
            
            # Check convergence
            if np.linalg.norm(dx) < convergence_threshold:
                break
        
        # Final residuals
        rover_pos = base_pos + baseline
        final_residuals = self.dd_processor.compute_dd_residuals(
            valid_dd_pr, sat_positions, rover_pos, base_pos, 
            valid_pairs, use_carrier=False
        )
        residual_rms = np.sqrt(np.mean(final_residuals**2))
        
        return baseline, residual_rms, iteration + 1
        
    def _calculate_satellite_positions(self, 
                                     observations: List[Observation],
                                     nav_data: dict,
                                     tc_rx: TimeCore) -> Dict[int, np.ndarray]:
        """Calculate satellite positions for all observed satellites"""
        sat_positions = {}
        
        for obs in observations:
            if obs.P[0] <= 0:
                continue
                
            # Calculate transmission time
            tc_tx = tc_rx - (obs.P[0] / CLIGHT)
            
            # Get appropriate TOW
            sys = sat2sys(obs.sat)
            if sys == 4:  # SYS_BDS
                tow = tc_tx.get_tow(TimeSystem.BDS)
            else:
                tow = tc_tx.get_tow(TimeSystem.GPS)
            
            # Get ephemeris and position
            eph = seleph(nav_data, tow, obs.sat)
            if eph is not None:
                sat_pos, _, _ = eph2pos(tow, eph)
                if not np.any(np.isnan(sat_pos)):
                    sat_positions[obs.sat] = sat_pos
        
        return sat_positions