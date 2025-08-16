"""Main RTK processor integrating all RTK components"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from ..core.data_structures import Observation, NavigationData
from ..satellite.ephemeris import select_ephemeris
from ..satellite.satellite_position import compute_satellite_position
from .double_difference import DoubleDifferenceProcessor
from .ambiguity_resolution import RTKAmbiguityManager
from .cycle_slip import CycleSlipDetector
from ..coordinate.transforms import ecef2llh


class RTKProcessor:
    """Complete RTK processing pipeline"""
    
    def __init__(self, base_position: np.ndarray):
        """
        Initialize RTK processor
        
        Parameters:
        -----------
        base_position : np.ndarray
            Base station position in ECEF (m)
        """
        self.base_position = base_position
        
        # Processing components
        self.dd_processor = DoubleDifferenceProcessor()
        self.ambiguity_manager = RTKAmbiguityManager()
        self.cycle_slip_detector = CycleSlipDetector()
        
        # Processing state
        self.last_fix_time = 0.0
        self.continuous_fix_epochs = 0
        self.baseline_length = 0.0
        
        # Quality metrics
        self.fix_ratio = 0.0
        self.position_accuracy = 0.0
        
    def process_epoch(self, 
                     rover_observations: List[Observation],
                     base_observations: List[Observation],
                     nav_data: NavigationData,
                     time: float,
                     rover_position_approx: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process one epoch of RTK observations
        
        Parameters:
        -----------
        rover_observations : List[Observation]
            Rover observations
        base_observations : List[Observation]
            Base station observations
        nav_data : NavigationData
            Navigation data with ephemerides
        time : float
            Current time (GPS time)
        rover_position_approx : np.ndarray
            Approximate rover position (ECEF)
            
        Returns:
        --------
        position : np.ndarray
            Rover position (ECEF)
        info : Dict
            Processing information and quality metrics
        """
        info = {
            'fix_type': 'float',
            'num_satellites': 0,
            'baseline_length': 0.0,
            'ratio': 0.0,
            'position_accuracy': 0.0,
            'cycle_slips': {},
            'fixed_ambiguities': {}
        }
        
        try:
            # Detect cycle slips
            rover_slips = self.cycle_slip_detector.detect_cycle_slips(rover_observations, time)
            base_slips = self.cycle_slip_detector.detect_cycle_slips(base_observations, time)
            info['cycle_slips'] = {**rover_slips, **base_slips}
            
            # Reset ambiguities for satellites with cycle slips
            self._handle_cycle_slips(rover_slips, base_slips)
            
            # Form double differences
            dd_pseudorange, dd_carrier, sat_pairs = self.dd_processor.form_double_differences(
                rover_observations, base_observations)
                
            if len(sat_pairs) < 1:
                return rover_position_approx, info
                
            info['num_satellites'] = len(sat_pairs) + 1  # +1 for reference satellite
            
            # Get satellite positions and clocks
            sat_positions, sat_clocks = self._compute_satellite_states(
                sat_pairs, nav_data, time)
                
            # Compute geometry matrix
            H = self.dd_processor.compute_dd_geometry_matrix(
                sat_positions, rover_position_approx, sat_pairs)
                
            # Process measurements
            if len(dd_carrier) > 0:
                # RTK with carrier phase
                position, fix_info = self._process_rtk_carrier(
                    dd_pseudorange, dd_carrier, H, sat_pairs, 
                    sat_positions, sat_clocks, rover_position_approx, time)
                info.update(fix_info)
            else:
                # Code-only differential positioning
                position = self._process_code_differential(
                    dd_pseudorange, H, rover_position_approx)
                info['fix_type'] = 'dgps'
                
            # Update quality metrics
            self.baseline_length = np.linalg.norm(position - self.base_position)
            info['baseline_length'] = self.baseline_length
            
            # Position accuracy estimation
            if info['fix_type'] == 'fixed':
                info['position_accuracy'] = 0.01 + 0.001 * self.baseline_length  # cm + ppm
            elif info['fix_type'] == 'float':
                info['position_accuracy'] = 0.1 + 0.01 * self.baseline_length   # dm + cm/km
            else:
                info['position_accuracy'] = 1.0 + 0.1 * self.baseline_length    # m + dm/km
                
            return position, info
            
        except Exception as e:
            print(f"RTK processing error: {e}")
            return rover_position_approx, info
            
    def _compute_satellite_states(self, 
                                sat_pairs: List[Tuple[int, int]],
                                nav_data: NavigationData,
                                time: float) -> Tuple[Dict[int, np.ndarray], Dict[int, float]]:
        """Compute satellite positions and clocks"""
        sat_positions = {}
        sat_clocks = {}
        
        # Get all unique satellites
        all_sats = set()
        for ref_sat, sat in sat_pairs:
            all_sats.add(ref_sat)
            all_sats.add(sat)
            
        for sat in all_sats:
            eph = select_ephemeris(nav_data, sat, time)
            if eph is not None:
                pos, clk, var = compute_satellite_position(eph, time)
                sat_positions[sat] = pos
                sat_clocks[sat] = clk
                
        return sat_positions, sat_clocks
        
    def _process_rtk_carrier(self, 
                           dd_pseudorange: np.ndarray,
                           dd_carrier: np.ndarray,
                           H: np.ndarray,
                           sat_pairs: List[Tuple[int, int]],
                           sat_positions: Dict[int, np.ndarray],
                           sat_clocks: Dict[int, float],
                           rover_pos_approx: np.ndarray,
                           time: float) -> Tuple[np.ndarray, Dict]:
        """Process RTK with carrier phase measurements"""
        n_dd = len(sat_pairs)
        wavelength = 299792458.0 / 1575.42e6  # L1 wavelength
        
        # Setup observation equation: y = H*dx + ambiguities
        # where y = [dd_pseudorange; dd_carrier_range]
        y = np.concatenate([dd_pseudorange, dd_carrier * wavelength])
        
        # Design matrix
        H_full = np.vstack([H, H])  # Same geometry for both pseudorange and carrier
        
        # Add ambiguity columns for carrier phase
        H_amb = np.zeros((2 * n_dd, n_dd))
        H_amb[n_dd:, :] = wavelength * np.eye(n_dd)  # Only carrier phase affected by ambiguities
        
        H_combined = np.hstack([H_full, H_amb])
        
        # Weight matrix
        sigma_pr = 3.0  # m
        sigma_cp = 0.01 * wavelength  # 1cm converted to meters
        W = np.diag(np.concatenate([
            np.full(n_dd, 1/sigma_pr**2),
            np.full(n_dd, 1/sigma_cp**2)
        ]))
        
        # Float solution
        N = H_combined.T @ W @ H_combined
        b = H_combined.T @ W @ y
        
        try:
            x_float = np.linalg.solve(N, b)
            P_float = np.linalg.inv(N)
        except np.linalg.LinAlgError:
            return rover_pos_approx, {'fix_type': 'float', 'ratio': 0.0}
            
        # Extract position and ambiguity estimates
        dx_float = x_float[:3]
        amb_float = x_float[3:]
        P_amb = P_float[3:, 3:]
        
        # Try integer ambiguity resolution
        satellites = [pair[1] for pair in sat_pairs]  # Non-reference satellites
        fixed_ambs, fix_status = self.ambiguity_manager.update_ambiguities(
            satellites, amb_float, P_amb)
            
        info = {}
        
        # Check if any ambiguities are fixed
        num_fixed = sum(fix_status.values())
        if num_fixed > 0:
            # Fixed solution
            amb_fixed = np.array([fixed_ambs.get(sat, amb_float[i]) 
                                for i, sat in enumerate(satellites)])
                                
            # Recompute position with fixed ambiguities
            y_corrected = y.copy()
            y_corrected[n_dd:] -= wavelength * amb_fixed
            
            # Position-only least squares
            dx_fixed = np.linalg.lstsq(H, y_corrected[:n_dd], rcond=None)[0]
            
            info['fix_type'] = 'fixed' if num_fixed == len(satellites) else 'partial'
            info['fixed_ambiguities'] = fixed_ambs
            info['ratio'] = self.ambiguity_manager.resolver.ratio_threshold  # Placeholder
            
            self.continuous_fix_epochs += 1
            self.last_fix_time = time
            
            return rover_pos_approx + dx_fixed, info
        else:
            # Float solution
            info['fix_type'] = 'float'
            info['ratio'] = 0.0
            self.continuous_fix_epochs = 0
            
            return rover_pos_approx + dx_float, info
            
    def _process_code_differential(self, 
                                 dd_pseudorange: np.ndarray,
                                 H: np.ndarray,
                                 rover_pos_approx: np.ndarray) -> np.ndarray:
        """Process code-only differential positioning"""
        # Simple least squares
        try:
            dx = np.linalg.lstsq(H, dd_pseudorange, rcond=None)[0]
            return rover_pos_approx + dx
        except np.linalg.LinAlgError:
            return rover_pos_approx
            
    def _handle_cycle_slips(self, 
                          rover_slips: Dict[int, bool],
                          base_slips: Dict[int, bool]):
        """Handle detected cycle slips by resetting ambiguities"""
        all_slips = set()
        
        for sat, slip in rover_slips.items():
            if slip:
                all_slips.add(sat)
                
        for sat, slip in base_slips.items():
            if slip:
                all_slips.add(sat)
                
        # Reset ambiguities for affected satellites
        for sat in all_slips:
            if sat in self.ambiguity_manager.fixed_ambiguities:
                del self.ambiguity_manager.fixed_ambiguities[sat]
            if sat in self.ambiguity_manager.fix_status:
                self.ambiguity_manager.fix_status[sat] = False
                
    def get_quality_metrics(self) -> Dict:
        """Get current processing quality metrics"""
        return {
            'baseline_length': self.baseline_length,
            'continuous_fix_epochs': self.continuous_fix_epochs,
            'last_fix_time': self.last_fix_time,
            'position_accuracy': self.position_accuracy
        }
        
    def reset_ambiguities(self):
        """Reset all ambiguity tracking"""
        self.ambiguity_manager.fixed_ambiguities.clear()
        self.ambiguity_manager.fix_status.clear()
        self.continuous_fix_epochs = 0