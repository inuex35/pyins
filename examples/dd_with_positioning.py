#!/usr/bin/env python3
"""
Double Difference with Position Calculation and Plotting
========================================================
This example performs DD processing, calculates positions, and plots results.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Add pyins to path
sys.path.insert(0, './external/pyins')
sys.path.append('.')

# Import pyins modules
from pyins.io.rinex import RinexObsReader, RinexNavReader
from pyins.rtk.double_difference import DoubleDifferenceProcessor
from pyins.rtk.dd_least_squares import DDLeastSquares
from pyins.gnss.ephemeris import satpos
from pyins.gnss.spp import single_point_positioning
from pyins.coordinate.transforms import ecef2llh, llh2ecef, ecef2enu
from pyins.core.constants import sat2sys, SYS_GPS, SYS_GAL, SYS_BDS, SYS_GLO

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_dd_with_positions():
    """Process DD and calculate positions"""
    
    # File paths
    rover_obs_file = "okujo_test/main.obs"
    base_obs_file = "okujo_test/base.obs"
    nav_file = "okujo_test/rover.nav"
    
    # Check files
    for file_path in [rover_obs_file, base_obs_file, nav_file]:
        if not Path(file_path).exists():
            logger.error(f"File not found: {file_path}")
            return None
    
    # Read RINEX files
    logger.info("Reading RINEX files...")
    rover_reader = RinexObsReader(rover_obs_file)
    base_reader = RinexObsReader(base_obs_file)
    nav_reader = RinexNavReader(nav_file)
    
    # Read all data
    rover_obs_all = rover_reader.read()
    base_obs_all = base_reader.read()
    nav_data = nav_reader.read()
    
    logger.info(f"Read {len(rover_obs_all)} rover epochs, {len(base_obs_all)} base epochs")
    
    # Get base position from header
    base_header = base_reader.header
    if hasattr(base_header, 'approx_position') and base_header.approx_position is not None:
        base_pos_ecef = np.array(base_header.approx_position)
    else:
        # Use a default position if not in header (Tokyo area)
        base_pos_ecef = np.array([-3963730.2015, 3355699.6439, 3690265.5217])
    
    base_pos_llh = ecef2llh(base_pos_ecef)
    logger.info(f"Base position (LLH): {base_pos_llh}")
    
    # Initialize processors
    dd_processor = DoubleDifferenceProcessor()
    dd_solver = DDLeastSquares(dd_processor)
    
    # Process epochs
    max_epochs = min(100, len(rover_obs_all), len(base_obs_all))
    results = []
    
    logger.info(f"Processing {max_epochs} epochs...")
    
    for epoch_idx in range(max_epochs):
        if epoch_idx % 10 == 0:
            logger.info(f"Processing epoch {epoch_idx}...")
        
        rover_epoch = rover_obs_all[epoch_idx]
        base_epoch = base_obs_all[epoch_idx]
        
        # Get observations
        rover_obs = rover_epoch['observations']
        base_obs = base_epoch['observations']
        gps_time = rover_epoch['gps_time']
        
        # Add time to observations for satpos
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
                radius = np.linalg.norm(sat_pos_array[i])
                if 20000e3 < radius < 30000e3:
                    sat_positions[obs.sat] = sat_pos_array[i]
        
        # Form double differences
        dd_pr, dd_cp, sat_pairs, ref_sats = dd_processor.form_double_differences(
            rover_obs, base_obs, frequency_idx=0
        )
        
        if epoch_idx == 0:
            logger.info(f"First epoch: {len(dd_pr)} DD formed, {len(sat_positions)} sat positions available")
        
        if len(dd_pr) < 4:
            logger.debug(f"Not enough DD at epoch {epoch_idx}: {len(dd_pr)}")
            continue
        
        # Get initial position (use SPP for first epoch, previous solution for others)
        if len(results) == 0:
            # Try SPP for initial position
            spp_solution, _ = single_point_positioning(rover_obs, nav_data)
            if spp_solution and spp_solution.rr is not None:
                initial_pos = spp_solution.rr
            else:
                # Use approximate position from header
                rover_header = rover_reader.header
                if hasattr(rover_header, 'approx_position') and rover_header.approx_position is not None:
                    initial_pos = np.array(rover_header.approx_position)
                else:
                    initial_pos = base_pos_ecef + np.array([10, 10, 5])  # Offset from base
        else:
            initial_pos = results[-1]['position_ecef']
        
        # Prepare satellite positions for solver
        ref_positions = []
        other_positions = []
        valid_dd_pr = []
        
        for i, (ref_sat, other_sat) in enumerate(sat_pairs[:len(dd_pr)]):
            if ref_sat in sat_positions and other_sat in sat_positions:
                # Check for same system (important for DD)
                if sat2sys(ref_sat) == sat2sys(other_sat):
                    ref_positions.append(sat_positions[ref_sat])
                    other_positions.append(sat_positions[other_sat])
                    valid_dd_pr.append(dd_pr[i])
        
        if len(valid_dd_pr) < 4:
            logger.debug(f"Not enough valid DD pairs at epoch {epoch_idx}: {len(valid_dd_pr)}")
            continue
        
        # Solve for position
        try:
            rover_position, residuals = dd_solver.solve_baseline(
                np.array(valid_dd_pr),
                np.array(ref_positions),
                np.array(other_positions),
                base_pos_ecef,
                initial_pos
            )
            
            # Convert to LLH and ENU
            rover_llh = ecef2llh(rover_position)
            baseline_enu = ecef2enu(rover_position, base_pos_llh)
            
            results.append({
                'epoch': epoch_idx,
                'time': gps_time,
                'position_ecef': rover_position,
                'position_llh': rover_llh,
                'baseline_enu': baseline_enu,
                'n_dd': len(valid_dd_pr),
                'residuals': residuals
            })
            
        except Exception as e:
            logger.info(f"Solution failed at epoch {epoch_idx}: {e}")
            continue
    
    logger.info(f"Successfully processed {len(results)} epochs")
    return results, base_pos_llh


def plot_results(results, base_pos_llh):
    """Plot the DD positioning results"""
    
    if not results:
        logger.error("No results to plot")
        return
    
    # Extract data
    times = np.array([r['time'] for r in results])
    times = times - times[0]  # Relative time in seconds
    
    positions_llh = np.array([r['position_llh'] for r in results])
    baselines_enu = np.array([r['baseline_enu'] for r in results])
    n_dd = np.array([r['n_dd'] for r in results])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Horizontal position (E-N plot)
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(baselines_enu[:, 0], baselines_enu[:, 1], c=times, cmap='viridis', s=20)
    ax1.set_xlabel('East (m)')
    ax1.set_ylabel('North (m)')
    ax1.set_title('Horizontal Position (ENU)')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    plt.colorbar(ax1.scatter(baselines_enu[:, 0], baselines_enu[:, 1], c=times, cmap='viridis', s=20), 
                 ax=ax1, label='Time (s)')
    
    # 2. Position time series (ENU)
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(times, baselines_enu[:, 0], 'r-', label='East', linewidth=0.5)
    ax2.plot(times, baselines_enu[:, 1], 'g-', label='North', linewidth=0.5)
    ax2.plot(times, baselines_enu[:, 2], 'b-', label='Up', linewidth=0.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (m)')
    ax2.set_title('Position Time Series (ENU)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Height variation
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(times, baselines_enu[:, 2], 'b-', linewidth=0.5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Height (m)')
    ax3.set_title('Height Variation')
    ax3.grid(True, alpha=0.3)
    
    # 4. Position standard deviation (moving window)
    window_size = min(10, len(baselines_enu) // 4)
    if window_size > 2:
        std_e = np.array([np.std(baselines_enu[max(0, i-window_size):i+1, 0]) 
                         for i in range(len(baselines_enu))])
        std_n = np.array([np.std(baselines_enu[max(0, i-window_size):i+1, 1]) 
                         for i in range(len(baselines_enu))])
        std_u = np.array([np.std(baselines_enu[max(0, i-window_size):i+1, 2]) 
                         for i in range(len(baselines_enu))])
        
        ax4 = plt.subplot(2, 3, 4)
        ax4.plot(times, std_e, 'r-', label='East STD', linewidth=1)
        ax4.plot(times, std_n, 'g-', label='North STD', linewidth=1)
        ax4.plot(times, std_u, 'b-', label='Up STD', linewidth=1)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('STD (m)')
        ax4.set_title(f'Position STD (window={window_size})')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # 5. Number of DD measurements
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(times, n_dd, 'k-', linewidth=1)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Number of DD')
    ax5.set_title('DD Measurements Used')
    ax5.grid(True, alpha=0.3)
    
    # 6. Baseline length
    baseline_length = np.linalg.norm(baselines_enu, axis=1)
    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(times, baseline_length, 'purple', linewidth=0.5)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Baseline Length (m)')
    ax6.set_title('Baseline Length')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Double Difference Positioning Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_file = 'dd_position_plots.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"Plots saved to {output_file}")
    plt.show()
    
    # Print statistics
    logger.info("\n=== Position Statistics ===")
    logger.info(f"Mean baseline (ENU): [{np.mean(baselines_enu[:, 0]):.3f}, "
                f"{np.mean(baselines_enu[:, 1]):.3f}, {np.mean(baselines_enu[:, 2]):.3f}] m")
    logger.info(f"STD baseline (ENU): [{np.std(baselines_enu[:, 0]):.3f}, "
                f"{np.std(baselines_enu[:, 1]):.3f}, {np.std(baselines_enu[:, 2]):.3f}] m")
    logger.info(f"Mean baseline length: {np.mean(baseline_length):.3f} m")
    logger.info(f"Average DD measurements: {np.mean(n_dd):.1f}")


def main():
    """Main function"""
    logger.info("=" * 60)
    logger.info("Double Difference Positioning with Plots")
    logger.info("=" * 60)
    
    # Process DD and calculate positions
    results, base_pos_llh = process_dd_with_positions()
    
    if results:
        # Save results to CSV
        output_csv = 'dd_positions.csv'
        with open(output_csv, 'w') as f:
            f.write("epoch,time,lat,lon,height,east,north,up,n_dd,baseline_length\n")
            for r in results:
                baseline_length = np.linalg.norm(r['baseline_enu'])
                f.write(f"{r['epoch']},{r['time']:.3f},")
                f.write(f"{r['position_llh'][0]:.8f},{r['position_llh'][1]:.8f},{r['position_llh'][2]:.3f},")
                f.write(f"{r['baseline_enu'][0]:.3f},{r['baseline_enu'][1]:.3f},{r['baseline_enu'][2]:.3f},")
                f.write(f"{r['n_dd']},{baseline_length:.3f}\n")
        logger.info(f"Results saved to {output_csv}")
        
        # Plot results
        plot_results(results, base_pos_llh)
    else:
        logger.error("No results obtained from DD processing")


if __name__ == "__main__":
    main()