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

#!/usr/bin/env python3
"""
Example of Double Difference RTK processing with PyINS

This example demonstrates:
1. Reading rover and base station RINEX files
2. Handling time synchronization between receivers  
3. Forming double differences with per-system reference satellites
4. Solving for baseline using DD least squares
5. Working with modern pyins API including satpos function
"""

import numpy as np
from pyins.io.rinex import RinexObsReader, RinexNavReader
from pyins.rtk.double_difference import DoubleDifferenceProcessor
from pyins.rtk.carrier_phase_dd import CarrierPhaseDD
from pyins.coordinate.transforms import ecef2llh, ecef2enu, enu2ecef, llh2ecef
from pyins.gnss.ephemeris import satpos, seleph, eph2pos, eph2clk
from pyins.gnss.spp import spp_solve
from pyins.core.constants import CLIGHT, sat2sys, sat2prn, SYS_GPS, SYS_GAL, SYS_BDS, SYS_QZS
from pyins.core.time import gps_seconds_to_week_tow
import matplotlib.pyplot as plt
import logging


def main():
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='PyINS RTK Double Difference Example')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory containing observation and navigation files')
    parser.add_argument('--rover-obs', type=str, default='rover.obs',
                        help='Rover observation file name')
    parser.add_argument('--base-obs', type=str, default='base.obs',
                        help='Base observation file name')
    parser.add_argument('--nav-file', type=str, default='navigation.nav',
                        help='Navigation file name')
    args = parser.parse_args()
    
    print("PyINS RTK Double Difference Example")
    print("="*60)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # File paths
    if args.data_dir:
        from pathlib import Path
        data_dir = Path(args.data_dir)
        rover_obs_file = str(data_dir / args.rover_obs)
        base_obs_file = str(data_dir / args.base_obs)
        nav_file = str(data_dir / args.nav_file)
    else:
        # Default paths for demonstration
        rover_obs_file = 'path/to/rover.obs'
        base_obs_file = 'path/to/base.obs'
        nav_file = 'path/to/navigation.nav'
    
    # For demonstration, we'll show the expected file format
    print("\nExpected RINEX file structure:")
    print("- Rover observation file (.obs)")
    print("- Base station observation file (.obs)")
    print("- Navigation file (.nav, .gnav, or mixed navigation)")
    print("\nSupported GNSS systems: GPS, GLONASS, Galileo, BeiDou, QZSS")
    
    try:
        # Read RINEX files
        print("\nReading RINEX files...")
        rover_obs_reader = RinexObsReader(rover_obs_file)
        base_obs_reader = RinexObsReader(base_obs_file)
        nav_reader = RinexNavReader(nav_file)
        
        rover_data = rover_obs_reader.read()
        base_data = base_obs_reader.read()
        nav_data = nav_reader.read()
        
        logger.info(f"Loaded {len(rover_data)} rover epochs")
        logger.info(f"Loaded {len(base_data)} base epochs")
        logger.info(f"Loaded {len(nav_data)} navigation messages")
        
    except FileNotFoundError:
        print("\n[Demo Mode] RINEX files not found. Showing example code structure.")
        demo_mode()
        return
    
    # Base station position (usually from RINEX header)
    base_pos = np.array([-3961905.0273, 3348993.7525, 3698211.7835])  # Example
    base_llh = ecef2llh(base_pos)
    
    print(f"\nBase station position:")
    print(f"  ECEF: [{base_pos[0]:.4f}, {base_pos[1]:.4f}, {base_pos[2]:.4f}] m")
    print(f"  LLH: [{np.rad2deg(base_llh[0]):.8f}, {np.rad2deg(base_llh[1]):.8f}, {base_llh[2]:.4f}]")
    
    # Create DD processor and carrier phase processor
    dd_processor = DoubleDifferenceProcessor()
    carrier_processor = CarrierPhaseDD()
    
    # Process multiple epochs
    print("\nProcessing epochs...")
    baselines = []
    solutions = []
    
    # Process first epoch to get initial position using SPP
    if len(rover_data) > 0:
        rover_obs = rover_data[0]
        
        # Use SPP for initial rover position
        print("\nComputing initial rover position using SPP...")
        spp_solution, used_sats = spp_solve(
            rover_obs,
            nav_data,
            max_iter=10,
            converge_threshold=1e-4,
            systems_to_use=['G', 'E', 'C', 'J']  # Exclude GLONASS due to IFB
        )
        
        if spp_solution and spp_solution.rr is not None:
            initial_rover_pos = spp_solution.rr
            rover_llh = ecef2llh(initial_rover_pos)
            print(f"Initial rover position (SPP):")
            print(f"  ECEF: [{initial_rover_pos[0]:.4f}, {initial_rover_pos[1]:.4f}, {initial_rover_pos[2]:.4f}] m")
            print(f"  LLH: [{np.rad2deg(rover_llh[0]):.8f}, {np.rad2deg(rover_llh[1]):.8f}, {rover_llh[2]:.4f}]")
        else:
            logger.warning("SPP failed, using approximate position")
            initial_rover_pos = base_pos + np.array([10.0, 10.0, 0.0])  # Approximate
    
    # Process epochs with double differences
    for i in range(min(10, len(rover_data))):
        rover_epoch = rover_data[i]
        
        # Handle time synchronization
        # Find matching base epoch or interpolate
        base_epoch = find_matching_base_epoch(rover_epoch, base_data)
        
        if base_epoch is None:
            logger.warning(f"No matching base epoch for rover epoch {i+1}")
            continue
        
        try:
            # Form double differences
            dd_pr, dd_cp, sat_pairs, ref_sats = dd_processor.form_double_differences(
                rover_epoch, base_epoch
            )
            
            if len(dd_pr) == 0:
                logger.warning(f"No valid double differences for epoch {i+1}")
                continue
            
            # Compute satellite positions using satpos
            gps_time = rover_epoch.get('gps_time', rover_epoch.get('time'))
            sat_positions, sat_clocks, sat_vars, sat_healths = satpos(
                rover_epoch, nav_data
            )
            
            # Solve for baseline using least squares
            baseline = solve_dd_baseline(
                dd_pr, sat_pairs, ref_sats, sat_positions,
                base_pos, initial_rover_pos
            )
            
            baselines.append(baseline)
            baseline_norm = np.linalg.norm(baseline)
            
            print(f"Epoch {i+1}: Baseline = {baseline_norm:.4f} m")
            logger.info(f"  DD obs: {len(dd_pr)} pairs, Ref sats: {ref_sats}")
            
        except Exception as e:
            logger.error(f"Epoch {i+1}: Failed - {str(e)}")
    
    # Plot results
    if len(baselines) > 0:
        plot_baselines(baselines, base_llh)
        
    print("\nExample completed!")


def find_matching_base_epoch(rover_epoch, base_data):
    """Find or interpolate matching base epoch"""
    rover_time = rover_epoch.get('gps_time', rover_epoch.get('time'))
    
    # Look for exact match
    for base_epoch in base_data:
        base_time = base_epoch.get('gps_time', base_epoch.get('time'))
        if abs(base_time - rover_time) < 0.01:
            return base_epoch
    
    # Try interpolation (simplified - in practice would interpolate observations)
    for i in range(len(base_data) - 1):
        t1 = base_data[i].get('gps_time', base_data[i].get('time'))
        t2 = base_data[i+1].get('gps_time', base_data[i+1].get('time'))
        if t1 <= rover_time <= t2:
            # For simplicity, return nearest epoch
            if abs(rover_time - t1) < abs(rover_time - t2):
                return base_data[i]
            else:
                return base_data[i+1]
    
    return None


def solve_dd_baseline(dd_obs, sat_pairs, ref_sats, sat_positions, base_pos, rover_pos_init):
    """Solve for baseline using double difference observations
    
    Parameters
    ----------
    dd_obs : dict
        Double difference observations
    sat_pairs : list
        Satellite pairs for each DD observation
    ref_sats : dict
        Reference satellites for each system
    sat_positions : dict
        Satellite positions in ECEF
    base_pos : np.ndarray
        Base station position in ECEF
    rover_pos_init : np.ndarray
        Initial rover position in ECEF
        
    Returns
    -------
    baseline : np.ndarray
        Baseline vector in ECEF
    """
    # Build observation equations
    n_obs = len(dd_obs)
    H = np.zeros((n_obs, 3))  # Design matrix
    y = np.zeros(n_obs)       # Observation vector
    
    rover_pos = rover_pos_init.copy()
    
    # Iterative least squares
    for iteration in range(5):
        for i, (sat_i, sat_ref) in enumerate(sat_pairs):
            if sat_i not in sat_positions or sat_ref not in sat_positions:
                continue
                
            # Unit vectors from receiver to satellites
            r_i = sat_positions[sat_i] - rover_pos
            r_ref = sat_positions[sat_ref] - rover_pos
            e_i = r_i / np.linalg.norm(r_i)
            e_ref = r_ref / np.linalg.norm(r_ref)
            
            # Double difference geometry matrix row
            H[i, :] = e_ref - e_i
            
            # Double difference observation
            rho_rover_i = np.linalg.norm(r_i)
            rho_rover_ref = np.linalg.norm(r_ref)
            rho_base_i = np.linalg.norm(sat_positions[sat_i] - base_pos)
            rho_base_ref = np.linalg.norm(sat_positions[sat_ref] - base_pos)
            
            dd_computed = (rho_rover_i - rho_rover_ref) - (rho_base_i - rho_base_ref)
            y[i] = dd_obs[i] - dd_computed
        
        # Solve normal equations
        try:
            dx = np.linalg.solve(H.T @ H, H.T @ y)
            rover_pos += dx
            
            if np.linalg.norm(dx) < 1e-4:
                break
        except np.linalg.LinAlgError:
            logging.warning("Singular matrix in DD solution")
            break
    
    baseline = rover_pos - base_pos
    return baseline


def plot_baselines(baselines, base_llh):
    """Plot baseline components"""
    if len(baselines) == 0:
        print("No baselines to plot")
        return
        
    baselines = np.array(baselines)
    
    # Convert to ENU coordinates relative to base station
    enu_baselines = []
    for baseline in baselines:
        # For baseline vector, we need to rotate it to ENU frame
        # The baseline is already relative to base, so we just rotate
        enu = ecef2enu(baseline, base_llh, base_ecef=np.zeros(3))
        enu_baselines.append(enu)
    enu_baselines = np.array(enu_baselines)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Horizontal components
    ax1.plot(enu_baselines[:, 0], enu_baselines[:, 1], '.-')
    ax1.set_xlabel('East (m)')
    ax1.set_ylabel('North (m)')
    ax1.set_title('Horizontal Baseline Components')
    ax1.grid(True)
    ax1.axis('equal')
    
    # Vertical component
    epochs = range(1, len(baselines) + 1)
    ax2.plot(epochs, enu_baselines[:, 2], '.-')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Up (m)')
    ax2.set_title('Vertical Baseline Component')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('rtk_baseline_results.png')
    print("\nResults saved to rtk_baseline_results.png")


def demo_mode():
    """Demonstrate the algorithm structure"""
    print("\n" + "="*60)
    print("DOUBLE DIFFERENCE RTK ALGORITHM")
    print("="*60)
    
    print("\n1. OBSERVATION EQUATION:")
    print("   DD = H × baseline + ε")
    print("   where:")
    print("   - DD: Double difference observations")
    print("   - H: Geometry matrix")
    print("   - baseline: Unknown baseline vector (what we solve for)")
    print("   - ε: Measurement errors")
    
    print("\n2. DOUBLE DIFFERENCE FORMATION:")
    print("   DD = [ΔP_rover - ΔP_base]")
    print("   where ΔP = P(sat_i) - P(sat_ref)")
    print("   - Uses per-system reference satellites (GPS, BeiDou, Galileo, QZSS)")
    print("   - Eliminates satellite clocks, receiver clocks, atmospheric delays")
    
    print("\n3. LEAST SQUARES SOLUTION:")
    print("   baseline = (H^T × H)^(-1) × H^T × DD")
    print("   - Iterative solution for better accuracy")
    print("   - Typically converges in 3-5 iterations")
    
    print("\n4. KEY FEATURES:")
    print("   ✓ Per-system reference satellite selection")
    print("   ✓ Automatic time synchronization handling")
    print("   ✓ Multi-GNSS support (GPS, Galileo, BeiDou, QZSS)")
    print("   ✓ GLONASS can be included with IFB estimation")
    print("   ✓ Uses modern satpos function for satellite positions")
    print("   ✓ SPP initialization for better convergence")
    print("   ✓ Residual-based quality assessment")
    
    print("\n5. EXPECTED ACCURACY:")
    print("   - Pseudorange DD: 1-5 m")
    print("   - Carrier phase DD (float solution): 10-50 cm")
    print("   - Carrier phase DD (fixed ambiguities): 1-2 cm")
    
    print("\n6. MODERN PYINS API USAGE:")
    print("   - satpos(): Compute all satellite positions at once")
    print("   - spp_solve(): Robust SPP with multi-GNSS support")
    print("   - DoubleDifferenceProcessor: Form DD observations")
    print("   - CarrierPhaseDD: Handle carrier phase DD")
    print("   - Coordinate transforms: ecef2llh, ecef2enu, etc.")


if __name__ == '__main__':
    main()