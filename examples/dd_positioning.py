#!/usr/bin/env python3
"""
Accurate Double Difference Positioning Example
===============================================
This example demonstrates accurate DD (Double Difference) positioning 
with proper geometry computation and iterative least squares solving.

Key improvements:
- Correct DD geometry matrix computation
- Proper iterative least squares with convergence checking
- Exclusion of problematic satellites (e.g., GLONASS R18)
- System-specific weighting
- Baseline sanity checking
"""

import sys
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt

# Import modules
from pyins.io.rinex import RinexObsReader, RinexNavReader
from pyins.rtk.double_difference import DoubleDifferenceProcessor
from pyins.gnss.ephemeris import satpos, seleph, eph2pos
from pyins.gnss.spp import single_point_positioning
from pyins.coordinate.transforms import ecef2llh, llh2ecef, ecef2enu
from pyins.core.constants import (
    CLIGHT, sat2sys, sat2prn, 
    SYS_GPS, SYS_GAL, SYS_BDS, SYS_GLO, SYS_QZS
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class AccurateDDSolver:
    """Accurate Double Difference Solver"""
    
    def __init__(self, base_pos_ecef: np.ndarray):
        """
        Initialize DD solver
        
        Parameters
        ----------
        base_pos_ecef : np.ndarray
            Base station position in ECEF
        """
        self.base_pos_ecef = base_pos_ecef
        self.base_pos_llh = ecef2llh(base_pos_ecef)
        self.dd_processor = DoubleDifferenceProcessor()
        
    def compute_dd_geometry_matrix(self, sat_positions: dict, rover_pos: np.ndarray,
                                   sat_pairs: list) -> np.ndarray:
        """
        Compute DD geometry matrix correctly
        
        The DD observation equation is:
        DD = (ρ_r^j - ρ_r^i) - (ρ_b^j - ρ_b^i)
        
        Where:
        - i is reference satellite
        - j is other satellite
        - r is rover
        - b is base
        """
        H = []
        
        for ref_sat, other_sat in sat_pairs:
            if ref_sat not in sat_positions or other_sat not in sat_positions:
                continue
                
            # Satellite positions
            pos_ref = sat_positions[ref_sat]
            pos_other = sat_positions[other_sat]
            
            # Ranges from rover to satellites
            r_rover_ref = pos_ref - rover_pos
            r_rover_other = pos_other - rover_pos
            
            # Unit vectors
            d_rover_ref = np.linalg.norm(r_rover_ref)
            d_rover_other = np.linalg.norm(r_rover_other)
            
            u_rover_ref = r_rover_ref / d_rover_ref
            u_rover_other = r_rover_other / d_rover_other
            
            # DD geometry row: derivative of DD with respect to rover position
            # ∂DD/∂x = -u_j + u_i (other - reference)
            h_row = -u_rover_other + u_rover_ref
            H.append(h_row)
            
        return np.array(H)
    
    def compute_dd_predicted(self, sat_positions: dict, rover_pos: np.ndarray,
                            sat_pairs: list) -> np.ndarray:
        """
        Compute predicted DD measurements based on geometry
        """
        dd_pred = []
        
        for ref_sat, other_sat in sat_pairs:
            if ref_sat not in sat_positions or other_sat not in sat_positions:
                continue
                
            # Satellite positions
            pos_ref = sat_positions[ref_sat]
            pos_other = sat_positions[other_sat]
            
            # Compute ranges
            # Rover to satellites
            d_rover_ref = np.linalg.norm(pos_ref - rover_pos)
            d_rover_other = np.linalg.norm(pos_other - rover_pos)
            
            # Base to satellites
            d_base_ref = np.linalg.norm(pos_ref - self.base_pos_ecef)
            d_base_other = np.linalg.norm(pos_other - self.base_pos_ecef)
            
            # DD predicted
            dd = (d_rover_other - d_rover_ref) - (d_base_other - d_base_ref)
            dd_pred.append(dd)
            
        return np.array(dd_pred)
    
    def solve_position(self, dd_measurements: np.ndarray, sat_positions: dict,
                      sat_pairs: list, initial_pos: np.ndarray,
                      max_iter: int = 10, threshold: float = 1e-4) -> tuple:
        """
        Solve for rover position using iterative least squares
        
        Returns
        -------
        rover_pos : np.ndarray
            Rover position in ECEF
        residuals : np.ndarray
            Final residuals
        n_iter : int
            Number of iterations
        """
        rover_pos = initial_pos.copy()
        
        for iteration in range(max_iter):
            # Filter valid satellite pairs
            valid_indices = []
            for i, (ref_sat, other_sat) in enumerate(sat_pairs[:len(dd_measurements)]):
                if ref_sat in sat_positions and other_sat in sat_positions:
                    # Check same system (important for DD)
                    if sat2sys(ref_sat) == sat2sys(other_sat):
                        valid_indices.append(i)
            
            if len(valid_indices) < 4:
                raise ValueError(f"Not enough valid DD pairs: {len(valid_indices)}")
            
            # Select valid measurements and pairs
            valid_dd = dd_measurements[valid_indices]
            valid_pairs = [sat_pairs[i] for i in valid_indices]
            
            # Compute geometry matrix
            H = self.compute_dd_geometry_matrix(sat_positions, rover_pos, valid_pairs)
            
            # Compute predicted DD
            dd_pred = self.compute_dd_predicted(sat_positions, rover_pos, valid_pairs)
            
            # Residuals (observed - predicted)
            residuals = valid_dd - dd_pred
            
            # Weight matrix (can be adjusted based on satellite system)
            W = np.eye(len(residuals))
            
            # Downweight GLONASS if present
            for i, (ref_sat, other_sat) in enumerate(valid_pairs):
                if sat2sys(ref_sat) == SYS_GLO:
                    W[i, i] = 0.5
            
            # Weighted least squares solution
            try:
                HTW = H.T @ W
                HTWH = HTW @ H
                HTWr = HTW @ residuals
                dx = np.linalg.solve(HTWH, HTWr)
            except np.linalg.LinAlgError:
                logger.warning(f"Singular matrix at iteration {iteration}")
                break
            
            # Update position
            rover_pos += dx
            
            # Check convergence
            if np.linalg.norm(dx) < threshold:
                logger.debug(f"Converged after {iteration + 1} iterations")
                break
        
        # Final residuals
        dd_pred_final = self.compute_dd_predicted(sat_positions, rover_pos, valid_pairs)
        residuals_final = valid_dd - dd_pred_final
        
        return rover_pos, residuals_final, iteration + 1


def process_accurate_dd():
    """Process DD with accurate solver"""
    
    # File paths
    rover_obs_file = "okujo_test/main.obs"
    base_obs_file = "okujo_test/base.obs"
    nav_file = "okujo_test/rover.nav"
    
    # Check files
    for file_path in [rover_obs_file, base_obs_file, nav_file]:
        if not Path(file_path).exists():
            logger.error(f"File not found: {file_path}")
            return None
    
    logger.info("Reading RINEX files...")
    
    # Read files
    rover_reader = RinexObsReader(rover_obs_file)
    base_reader = RinexObsReader(base_obs_file)
    nav_reader = RinexNavReader(nav_file)
    
    rover_obs_all = rover_reader.read()
    base_obs_all = base_reader.read()
    nav_data = nav_reader.read()
    
    logger.info(f"Read {len(rover_obs_all)} rover epochs, {len(base_obs_all)} base epochs")
    
    # Base station position (from RINEX header)
    base_pos_ecef = np.array([-3961905.0321, 3348994.2448, 3698212.3456])
    base_pos_llh = ecef2llh(base_pos_ecef)
    
    # Rover approximate position (from RINEX header)
    rover_approx_ecef = np.array([-3963730.2015, 3355699.6439, 3690265.5217])
    rover_approx_llh = ecef2llh(rover_approx_ecef)
    
    logger.info(f"Base position (LLH): [{base_pos_llh[0]:.8f}, {base_pos_llh[1]:.8f}, {base_pos_llh[2]:.3f}]")
    logger.info(f"Rover approx (LLH): [{rover_approx_llh[0]:.8f}, {rover_approx_llh[1]:.8f}, {rover_approx_llh[2]:.3f}]")
    
    # Expected baseline (approximate)
    expected_baseline = rover_approx_ecef - base_pos_ecef
    expected_baseline_enu = ecef2enu(rover_approx_ecef, base_pos_llh)
    logger.info(f"Expected baseline (ENU): [{expected_baseline_enu[0]:.3f}, "
                f"{expected_baseline_enu[1]:.3f}, {expected_baseline_enu[2]:.3f}] m")
    logger.info(f"Expected baseline length: {np.linalg.norm(expected_baseline_enu):.3f} m")
    
    # Initialize solver
    dd_solver = AccurateDDSolver(base_pos_ecef)
    dd_processor = DoubleDifferenceProcessor()
    
    # Process epochs
    n_epochs = min(len(rover_obs_all), len(base_obs_all))  # Process all epochs
    logger.info(f"Processing {n_epochs} epochs...")
    
    results = []
    last_good_pos = rover_approx_ecef
    
    for epoch_idx in range(n_epochs):
        if epoch_idx % 50 == 0:
            logger.info(f"Processing epoch {epoch_idx}/{n_epochs}...")
        
        rover_epoch = rover_obs_all[epoch_idx]
        base_epoch = base_obs_all[epoch_idx]
        
        # Get observations
        rover_obs = rover_epoch['observations']
        base_obs = base_epoch['observations']
        gps_time = rover_epoch['gps_time']
        
        # Add time to observations
        for obs in rover_obs:
            obs.time = gps_time
        for obs in base_obs:
            obs.time = gps_time
        
        # Compute satellite positions
        sat_pos_array, sat_clk_array, _, sat_health = satpos(rover_obs, nav_data)
        
        # Build satellite position dictionary
        sat_positions = {}
        for i, obs in enumerate(rover_obs):
            if sat_health[i] == 0 and np.linalg.norm(sat_pos_array[i]) > 0:
                # Basic sanity check
                radius = np.linalg.norm(sat_pos_array[i])
                if 20000e3 < radius < 30000e3:
                    # Exclude GLONASS R18 if problematic
                    if sat2sys(obs.sat) == SYS_GLO and sat2prn(obs.sat) == 18:
                        continue
                    sat_positions[obs.sat] = sat_pos_array[i]
        
        # Form double differences
        dd_pr, dd_cp, sat_pairs, ref_sats = dd_processor.form_double_differences(
            rover_obs, base_obs, frequency_idx=0
        )
        
        if len(dd_pr) < 4:
            logger.debug(f"Epoch {epoch_idx}: Not enough DD ({len(dd_pr)})")
            continue
        
        # Get initial position
        if epoch_idx == 0 or len(results) == 0:
            # Try SPP for first epoch
            try:
                spp_solution, _ = single_point_positioning(rover_obs, nav_data)
                if spp_solution and spp_solution.rr is not None:
                    initial_pos = spp_solution.rr
                    logger.debug(f"Using SPP initial: {ecef2llh(initial_pos)}")
                else:
                    initial_pos = last_good_pos
            except:
                initial_pos = last_good_pos
        else:
            # Use previous solution
            initial_pos = last_good_pos
        
        # Solve for position
        try:
            rover_pos, residuals, n_iter = dd_solver.solve_position(
                dd_pr, sat_positions, sat_pairs, initial_pos,
                max_iter=10, threshold=0.001
            )
            
            # Convert to LLH and ENU
            rover_llh = ecef2llh(rover_pos)
            baseline_enu = ecef2enu(rover_pos, base_pos_llh)
            baseline_length = np.linalg.norm(baseline_enu)
            rms = np.sqrt(np.mean(residuals**2))
            
            # Sanity check - baseline shouldn't be too large
            if baseline_length > 50000:  # 50 km is unreasonable
                logger.debug(f"Epoch {epoch_idx}: Unreasonable baseline {baseline_length:.1f}m")
                continue
            
            results.append({
                'epoch': epoch_idx,
                'time': gps_time,
                'position_ecef': rover_pos,
                'position_llh': rover_llh,
                'baseline_enu': baseline_enu,
                'baseline_length': baseline_length,
                'n_dd': len(residuals),
                'rms': rms,
                'n_iter': n_iter
            })
            
            last_good_pos = rover_pos
            
        except Exception as e:
            logger.debug(f"Epoch {epoch_idx} failed: {e}")
            continue
    
    logger.info(f"Successfully processed {len(results)}/{n_epochs} epochs")
    
    return results, base_pos_llh


def plot_accurate_results(results, base_pos_llh):
    """Plot accurate DD results"""
    
    if not results:
        logger.error("No results to plot")
        return
    
    # Extract data
    times = np.array([r['time'] for r in results])
    times = times - times[0]  # Relative time
    
    baselines_enu = np.array([r['baseline_enu'] for r in results])
    baseline_lengths = np.array([r['baseline_length'] for r in results])
    n_dd = np.array([r['n_dd'] for r in results])
    rms = np.array([r['rms'] for r in results])
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Horizontal trajectory
    ax1 = plt.subplot(2, 3, 1)
    scatter = ax1.scatter(baselines_enu[:, 0], baselines_enu[:, 1], 
                         c=times, cmap='viridis', s=10, alpha=0.7)
    ax1.set_xlabel('East (m)')
    ax1.set_ylabel('North (m)')
    ax1.set_title('Horizontal Trajectory (Accurate DD)')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    plt.colorbar(scatter, ax=ax1, label='Time (s)')
    
    # Add mean position
    mean_e = np.mean(baselines_enu[:, 0])
    mean_n = np.mean(baselines_enu[:, 1])
    ax1.plot(mean_e, mean_n, 'r*', markersize=10, label='Mean')
    ax1.legend()
    
    # 2. Position time series
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(times, baselines_enu[:, 0], 'r-', label='East', linewidth=0.8)
    ax2.plot(times, baselines_enu[:, 1], 'g-', label='North', linewidth=0.8)
    ax2.plot(times, baselines_enu[:, 2], 'b-', label='Up', linewidth=0.8)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (m)')
    ax2.set_title('Position Time Series')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Height variation
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(times, baselines_enu[:, 2], 'b-', linewidth=1)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Height (m)')
    ax3.set_title('Height Variation')
    ax3.grid(True, alpha=0.3)
    mean_h = np.mean(baselines_enu[:, 2])
    ax3.axhline(y=mean_h, color='r', linestyle='--', alpha=0.5, label=f'Mean: {mean_h:.2f}m')
    ax3.legend()
    
    # 4. Baseline length
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(times, baseline_lengths, 'purple', linewidth=1)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Baseline Length (m)')
    ax4.set_title('Baseline Length')
    ax4.grid(True, alpha=0.3)
    mean_len = np.mean(baseline_lengths)
    ax4.axhline(y=mean_len, color='r', linestyle='--', alpha=0.5, label=f'Mean: {mean_len:.1f}m')
    ax4.legend()
    
    # 5. RMS residuals
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(times, rms, 'm-', linewidth=1)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('RMS (m)')
    ax5.set_title('RMS Residuals')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0, max(rms) * 1.1])
    
    # 6. Number of DD measurements
    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(times, n_dd, 'k-', linewidth=1)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('# DD Measurements')
    ax6.set_title('DD Measurements Used')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Accurate Double Difference Positioning', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    output_file = 'pyins_dd_accurate.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"Plot saved to {output_file}")
    
    # Print statistics
    logger.info("\n=== Accurate DD Statistics ===")
    logger.info(f"Processed epochs: {len(results)}")
    logger.info(f"Mean baseline (ENU): [{mean_e:.3f}, {mean_n:.3f}, {mean_h:.3f}] m")
    logger.info(f"STD baseline (ENU): [{np.std(baselines_enu[:, 0]):.3f}, "
                f"{np.std(baselines_enu[:, 1]):.3f}, {np.std(baselines_enu[:, 2]):.3f}] m")
    logger.info(f"Mean baseline length: {mean_len:.3f} m")
    logger.info(f"Average DD measurements: {np.mean(n_dd):.1f}")
    logger.info(f"Average RMS: {np.mean(rms):.3f} m")
    
    # Save results to file
    output_pos = 'pyins_dd_accurate.pos'
    with open(output_pos, 'w') as f:
        f.write("% Accurate DD Positioning Results\n")
        f.write("% epoch, time, lat, lon, height, east, north, up, baseline_len, n_dd, rms\n")
        for r in results:
            f.write(f"{r['epoch']:6d}, {r['time']:15.3f}, ")
            f.write(f"{r['position_llh'][0]:12.8f}, {r['position_llh'][1]:13.8f}, {r['position_llh'][2]:10.4f}, ")
            f.write(f"{r['baseline_enu'][0]:10.3f}, {r['baseline_enu'][1]:10.3f}, {r['baseline_enu'][2]:10.3f}, ")
            f.write(f"{r['baseline_length']:10.3f}, {r['n_dd']:3d}, {r['rms']:8.3f}\n")
    logger.info(f"Results saved to {output_pos}")
    
    plt.show()


def main():
    """Main function"""
    logger.info("=" * 60)
    logger.info("Accurate Double Difference Positioning")
    logger.info("=" * 60)
    
    # Process DD
    results, base_pos_llh = process_accurate_dd()
    
    if results:
        # Plot results
        plot_accurate_results(results, base_pos_llh)
    else:
        logger.error("No results obtained")


if __name__ == "__main__":
    main()