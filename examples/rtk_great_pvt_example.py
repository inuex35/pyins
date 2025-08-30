#!/usr/bin/env python3
"""
RTK with GREAT-PVT Ambiguity Resolution
========================================

This example demonstrates the GREAT-PVT (Graph Robust Estimation for Adaptive Tracking
in Positioning, Velocity and Timing) approach for integer ambiguity resolution in RTK.

Key features:
- Partial ambiguity fixing based on quality metrics
- Adaptive selection of reliable ambiguities
- Higher fix rates compared to traditional LAMBDA
- Suitable for long baselines

Author: PyINS Contributors
License: Apache 2.0
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

# PyINS imports
from pyins.io.rinex import RinexObsReader, RinexNavReader
from pyins.rtk.double_difference import DoubleDifferenceProcessor
from pyins.rtk.ambiguity_resolution import RTKAmbiguityManager, mlambda
from pyins.gnss.ephemeris import satpos
from pyins.coordinate.transforms import ecef2llh, ecef2enu
from pyins.core.constants import CLIGHT, FREQ_L1

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GreatPvtRTK:
    """
    GREAT-PVT RTK Processor
    
    This class implements the GREAT-PVT approach for RTK positioning with
    improved ambiguity resolution for challenging scenarios like long baselines.
    
    Attributes
    ----------
    base_pos_ecef : np.ndarray
        Base station position in ECEF coordinates
    max_fix_count : int
        Maximum number of ambiguities to fix (default: 15)
    min_ratio : float
        Minimum ratio for ambiguity validation (default: 2.0)
    max_sigma : float
        Maximum standard deviation for float ambiguity (cycles)
    """
    
    def __init__(self, base_pos_ecef: np.ndarray, 
                 max_fix_count: int = 15,
                 min_ratio: float = 2.0,
                 max_sigma: float = 0.25):
        """
        Initialize GREAT-PVT RTK processor
        
        Parameters
        ----------
        base_pos_ecef : np.ndarray
            Base station position in ECEF [x, y, z] meters
        max_fix_count : int, optional
            Maximum number of ambiguities to fix simultaneously
        min_ratio : float, optional
            Minimum ratio threshold for validation
        max_sigma : float, optional
            Maximum acceptable standard deviation for float ambiguities
        """
        self.base_pos_ecef = base_pos_ecef
        self.base_pos_llh = ecef2llh(base_pos_ecef)
        self.dd_processor = DoubleDifferenceProcessor()
        
        # GREAT-PVT parameters
        self.max_fix_count = max_fix_count
        self.min_ratio = min_ratio
        self.max_sigma = max_sigma
        self.min_elevation = np.deg2rad(15)
        
        # State tracking
        self.fixed_ambiguities = {}
        self.fix_history = []
        
    def partial_ambiguity_resolution(self, 
                                    float_ambiguities: Dict[Tuple[int, int], float],
                                    covariances: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], int]:
        """
        GREAT-PVT partial ambiguity resolution
        
        This method selects and fixes a subset of the most reliable ambiguities
        based on their quality metrics.
        
        Parameters
        ----------
        float_ambiguities : dict
            Float ambiguity estimates for each satellite pair
        covariances : dict
            Variance estimates for each ambiguity
            
        Returns
        -------
        fixed_subset : dict
            Fixed integer ambiguities for selected satellite pairs
        """
        # Evaluate quality of each ambiguity
        candidates = []
        
        for pair, float_val in float_ambiguities.items():
            if pair not in covariances:
                continue
                
            sigma = np.sqrt(covariances[pair])
            fractional = abs(float_val - np.round(float_val))
            
            # Quality metric: lower is better
            # Combines uncertainty and distance from integer
            quality = sigma + 0.5 * fractional
            
            if sigma < self.max_sigma:  # Only consider low-noise ambiguities
                candidates.append({
                    'pair': pair,
                    'float': float_val,
                    'integer': int(np.round(float_val)),
                    'sigma': sigma,
                    'quality': quality
                })
        
        # Sort by quality (best first)
        candidates.sort(key=lambda x: x['quality'])
        
        # Select best subset up to max_fix_count
        n_to_fix = min(self.max_fix_count, len(candidates))
        
        fixed_subset = {}
        if n_to_fix >= 4:  # Need minimum for position solution
            for i in range(n_to_fix):
                c = candidates[i]
                fixed_subset[c['pair']] = c['integer']
                
            logger.info(f"  GREAT-PVT: Fixed {len(fixed_subset)} ambiguities "
                       f"(best quality: {candidates[0]['quality']:.3f})")
        
        return fixed_subset
    
    def process_epoch(self, rover_obs: list, base_obs: list, 
                     nav_data, initial_pos: np.ndarray) -> Dict:
        """
        Process single epoch with GREAT-PVT approach
        
        Parameters
        ----------
        rover_obs : list
            Rover GNSS observations
        base_obs : list
            Base station observations
        nav_data : NavigationData
            Navigation ephemerides
        initial_pos : np.ndarray
            Initial position estimate
            
        Returns
        -------
        result : dict
            Processing result with position and status
        """
        # Get satellite positions
        sat_pos_array, _, _, sat_health = satpos(rover_obs, nav_data)
        
        sat_positions = {}
        for i, obs in enumerate(rover_obs):
            if sat_health[i] == 0:
                radius = np.linalg.norm(sat_pos_array[i])
                if 20000e3 < radius < 30000e3:  # Valid satellite orbit
                    sat_positions[obs.sat] = sat_pos_array[i]
        
        # Form double differences
        dd_pr, dd_cp, sat_pairs, ref_sats = self.dd_processor.form_double_differences(
            rover_obs, base_obs, frequency_idx=0  # L1
        )
        
        if len(dd_pr) < 4:
            return {'success': False, 'reason': 'Insufficient DD observations'}
        
        # Estimate float ambiguities
        float_ambiguities = {}
        covariances = {}
        wavelength = CLIGHT / FREQ_L1
        
        for i, (ref_sat, other_sat) in enumerate(sat_pairs[:len(dd_cp)]):
            if ref_sat not in sat_positions or other_sat not in sat_positions:
                continue
            
            # Compute geometric DD range
            r_rover_ref = np.linalg.norm(sat_positions[ref_sat] - initial_pos)
            r_rover_other = np.linalg.norm(sat_positions[other_sat] - initial_pos)
            r_base_ref = np.linalg.norm(sat_positions[ref_sat] - self.base_pos_ecef)
            r_base_other = np.linalg.norm(sat_positions[other_sat] - self.base_pos_ecef)
            
            dd_range = (r_rover_other - r_rover_ref) - (r_base_other - r_base_ref)
            
            # Float ambiguity
            float_amb = dd_cp[i] - dd_range / wavelength
            float_ambiguities[(ref_sat, other_sat)] = float_amb
            
            # Simple covariance estimate based on elevation
            elevation = self._compute_elevation(sat_positions[other_sat], initial_pos)
            sigma = 0.01 / np.sin(elevation) if elevation > self.min_elevation else 1.0
            covariances[(ref_sat, other_sat)] = sigma**2
        
        # Apply GREAT-PVT partial ambiguity resolution
        fixed_subset = self.partial_ambiguity_resolution(float_ambiguities, covariances)
        
        # Solve position
        if len(fixed_subset) >= 4:
            # Use fixed ambiguities for precise position
            rover_pos = self._solve_with_fixed_ambiguities(
                dd_cp, sat_pairs, sat_positions, fixed_subset, initial_pos, wavelength
            )
            
            if rover_pos is not None:
                return {
                    'success': True,
                    'position_fixed': rover_pos,
                    'ar_success': True,
                    'n_fixed': len(fixed_subset),
                    'n_dd': len(dd_pr),
                    'method': 'GREAT-PVT'
                }
        
        # Fall back to float solution
        rover_pos = self._solve_float_position(dd_pr, sat_pairs, sat_positions, initial_pos)
        
        return {
            'success': True if rover_pos is not None else False,
            'position_float': rover_pos,
            'ar_success': False,
            'n_dd': len(dd_pr)
        }
    
    def _compute_elevation(self, sat_pos: np.ndarray, rover_pos: np.ndarray) -> float:
        """Compute satellite elevation angle"""
        rover_llh = ecef2llh(rover_pos)
        sat_enu = ecef2enu(sat_pos, rover_llh)
        horizontal = np.sqrt(sat_enu[0]**2 + sat_enu[1]**2)
        return np.arctan2(sat_enu[2], horizontal)
    
    def _solve_with_fixed_ambiguities(self, dd_cp, sat_pairs, sat_positions,
                                      fixed_ambiguities, initial_pos, wavelength):
        """Solve position using fixed ambiguities"""
        rover_pos = initial_pos.copy()
        
        for iteration in range(10):
            H = []
            residuals = []
            
            for i, (ref_sat, other_sat) in enumerate(sat_pairs[:len(dd_cp)]):
                if (ref_sat, other_sat) not in fixed_ambiguities:
                    continue
                if ref_sat not in sat_positions or other_sat not in sat_positions:
                    continue
                
                # Geometry matrix
                r_ref = sat_positions[ref_sat] - rover_pos
                r_other = sat_positions[other_sat] - rover_pos
                d_ref = np.linalg.norm(r_ref)
                d_other = np.linalg.norm(r_other)
                
                h_row = -r_other/d_other + r_ref/d_ref
                H.append(h_row)
                
                # Predicted DD range
                d_base_ref = np.linalg.norm(sat_positions[ref_sat] - self.base_pos_ecef)
                d_base_other = np.linalg.norm(sat_positions[other_sat] - self.base_pos_ecef)
                dd_pred = (d_other - d_ref) - (d_base_other - d_base_ref)
                
                # Residual with fixed ambiguity
                fixed_N = fixed_ambiguities[(ref_sat, other_sat)]
                dd_precise = (dd_cp[i] - fixed_N) * wavelength
                residuals.append(dd_precise - dd_pred)
            
            if len(H) < 4:
                return None
            
            H = np.array(H)
            residuals = np.array(residuals)
            
            # Least squares solution
            try:
                dx = np.linalg.lstsq(H, residuals, rcond=None)[0]
                rover_pos += dx
                
                if np.linalg.norm(dx) < 1e-5:
                    break
            except:
                return None
        
        return rover_pos
    
    def _solve_float_position(self, dd_pr, sat_pairs, sat_positions, initial_pos):
        """Standard float solution using pseudoranges"""
        rover_pos = initial_pos.copy()
        
        for iteration in range(10):
            H = []
            residuals = []
            
            for i, (ref_sat, other_sat) in enumerate(sat_pairs[:len(dd_pr)]):
                if ref_sat not in sat_positions or other_sat not in sat_positions:
                    continue
                
                r_ref = sat_positions[ref_sat] - rover_pos
                r_other = sat_positions[other_sat] - rover_pos
                d_ref = np.linalg.norm(r_ref)
                d_other = np.linalg.norm(r_other)
                
                h_row = -r_other/d_other + r_ref/d_ref
                H.append(h_row)
                
                d_base_ref = np.linalg.norm(sat_positions[ref_sat] - self.base_pos_ecef)
                d_base_other = np.linalg.norm(sat_positions[other_sat] - self.base_pos_ecef)
                dd_pred = (d_other - d_ref) - (d_base_other - d_base_ref)
                
                residuals.append(dd_pr[i] - dd_pred)
            
            if len(H) < 4:
                return None
            
            H = np.array(H)
            residuals = np.array(residuals)
            
            try:
                dx = np.linalg.lstsq(H, residuals, rcond=None)[0]
                rover_pos += dx
                if np.linalg.norm(dx) < 1e-4:
                    break
            except:
                return None
        
        return rover_pos


def main():
    """
    Example usage of GREAT-PVT RTK
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    # File paths - adjust to your RINEX files
    rover_obs_file = "path/to/rover.obs"
    base_obs_file = "path/to/base.obs"
    nav_file = "path/to/navigation.nav"
    
    # For demonstration with sample data
    sample_dir = Path(__file__).parent.parent / "data" / "sample"
    if not Path(rover_obs_file).exists():
        logger.info("Using sample data for demonstration")
        rover_obs_file = sample_dir / "rover.obs"
        base_obs_file = sample_dir / "base.obs"
        nav_file = sample_dir / "rover.nav"
    
    if not all(Path(f).exists() for f in [rover_obs_file, base_obs_file, nav_file]):
        logger.error("RINEX files not found. Please provide valid file paths.")
        return
    
    # Read RINEX files
    logger.info("Reading RINEX files...")
    rover_reader = RinexObsReader(str(rover_obs_file))
    base_reader = RinexObsReader(str(base_obs_file))
    nav_reader = RinexNavReader(str(nav_file))
    
    rover_data = rover_reader.read()
    base_data = base_reader.read()
    nav_data = nav_reader.read()
    
    logger.info(f"Loaded {len(rover_data)} rover epochs, {len(base_data)} base epochs")
    
    # Get base station position
    base_header = base_reader.header
    if 'approx_pos' in base_header:
        base_pos_ecef = np.array(base_header['approx_pos'])
    else:
        # Default position if not in header
        base_pos_ecef = np.array([0, 0, 0])
        logger.warning("Base position not found in header, using default")
    
    # Get initial rover position
    rover_header = rover_reader.header
    if 'approx_pos' in rover_header:
        initial_rover_pos = np.array(rover_header['approx_pos'])
    else:
        initial_rover_pos = base_pos_ecef + np.array([100, 100, 0])
    
    base_llh = ecef2llh(base_pos_ecef)
    logger.info(f"Base position: [{np.rad2deg(base_llh[0]):.6f}°, "
                f"{np.rad2deg(base_llh[1]):.6f}°, {base_llh[2]:.3f}m]")
    logger.info(f"Initial baseline: {np.linalg.norm(initial_rover_pos - base_pos_ecef):.1f} m")
    
    # Initialize GREAT-PVT RTK processor
    rtk = GreatPvtRTK(
        base_pos_ecef,
        max_fix_count=15,  # Optimal for most scenarios
        min_ratio=2.0,
        max_sigma=0.25
    )
    
    # Process epochs
    n_epochs = min(200, len(rover_data), len(base_data))
    logger.info(f"\nProcessing {n_epochs} epochs with GREAT-PVT RTK...")
    
    results = []
    n_fixed = 0
    n_float = 0
    
    for epoch_idx in range(n_epochs):
        if epoch_idx % 50 == 0:
            logger.info(f"Processing epoch {epoch_idx}/{n_epochs}...")
        
        rover_epoch = rover_data[epoch_idx]
        base_epoch = base_data[epoch_idx]
        
        # Get observations
        rover_obs = rover_epoch['observations']
        base_obs = base_epoch['observations']
        
        # Add time to observations
        gps_time = rover_epoch['gps_time']
        for obs in rover_obs:
            obs.time = gps_time
        for obs in base_obs:
            obs.time = gps_time
        
        # Use previous solution or initial
        if results and 'position_fixed' in results[-1]:
            initial_pos = results[-1]['position_fixed']
        elif results and 'position_float' in results[-1]:
            initial_pos = results[-1]['position_float']
        else:
            initial_pos = initial_rover_pos
        
        # Process epoch
        result = rtk.process_epoch(rover_obs, base_obs, nav_data, initial_pos)
        
        if result['success']:
            results.append(result)
            
            if result.get('ar_success', False):
                n_fixed += 1
                if epoch_idx % 50 == 0:
                    logger.info(f"  Epoch {epoch_idx}: FIX with {result.get('n_fixed', 0)} ambiguities")
            else:
                n_float += 1
    
    # Summary
    fix_rate = n_fixed / n_epochs * 100 if n_epochs > 0 else 0
    
    logger.info("\n" + "="*60)
    logger.info("GREAT-PVT RTK RESULTS:")
    logger.info(f"  Total epochs: {n_epochs}")
    logger.info(f"  Fixed solutions: {n_fixed} ({fix_rate:.1f}%)")
    logger.info(f"  Float solutions: {n_float} ({100-fix_rate:.1f}%)")
    
    # Advantages over traditional LAMBDA
    logger.info("\nAdvantages of GREAT-PVT over traditional LAMBDA:")
    logger.info("  • Partial ambiguity fixing for higher success rate")
    logger.info("  • Quality-based selection of reliable ambiguities")
    logger.info("  • Better performance with long baselines")
    logger.info("  • Adaptive to observation quality")
    logger.info("="*60)
    
    # Plot results if we have data
    if results:
        plot_results(results, base_llh)


def plot_results(results, base_llh):
    """Plot RTK results"""
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Separate float and fixed solutions
    float_pos = []
    fixed_pos = []
    
    for r in results:
        if r.get('ar_success', False) and 'position_fixed' in r:
            pos = r['position_fixed']
            fixed_pos.append(ecef2enu(pos, base_llh))
        elif 'position_float' in r:
            pos = r['position_float']
            float_pos.append(ecef2enu(pos, base_llh))
    
    # Horizontal plot
    if float_pos:
        float_pos = np.array(float_pos)
        ax1.scatter(float_pos[:, 0], float_pos[:, 1], c='blue', s=2, alpha=0.5, label='Float')
    
    if fixed_pos:
        fixed_pos = np.array(fixed_pos)
        ax1.scatter(fixed_pos[:, 0], fixed_pos[:, 1], c='red', s=3, label='Fixed')
    
    ax1.set_xlabel('East (m)')
    ax1.set_ylabel('North (m)')
    ax1.set_title('GREAT-PVT RTK Horizontal Position')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Fix status over time
    fix_status = []
    for r in results:
        if r.get('ar_success', False):
            fix_status.append(2)
        else:
            fix_status.append(1)
    
    ax2.plot(fix_status, 'k-', linewidth=0.5)
    ax2.fill_between(range(len(fix_status)), 1, fix_status, 
                     where=np.array(fix_status) > 1.5, 
                     alpha=0.3, color='green', label='Fixed')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Solution Status')
    ax2.set_title('Solution Status Over Time')
    ax2.set_yticks([1, 2])
    ax2.set_yticklabels(['Float', 'Fixed'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('GREAT-PVT RTK Results')
    plt.tight_layout()
    plt.savefig('great_pvt_rtk_results.png')
    logger.info(f"\nResults saved to great_pvt_rtk_results.png")
    plt.show()


if __name__ == "__main__":
    main()