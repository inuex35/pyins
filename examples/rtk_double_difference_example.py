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
"""

import numpy as np
from pyins.io.rinex import RinexObsReader, RinexNavReader
from pyins.rtk import DoubleDifferenceProcessor, DDLeastSquares, interpolate_epoch
from pyins.coordinate import ecef2llh, ecef2enu
import matplotlib.pyplot as plt


def main():
    print("PyINS RTK Double Difference Example")
    print("="*60)
    
    # File paths - update these to your RINEX files
    rover_obs_file = 'path/to/rover.obs'
    base_obs_file = 'path/to/base.obs'
    nav_file = 'path/to/navigation.nav'
    
    # For demonstration, we'll show the expected file format
    print("\nExpected RINEX file structure:")
    print("- Rover observation file (.obs)")
    print("- Base station observation file (.obs)")
    print("- Navigation file (.nav or .gnav)")
    
    try:
        # Read RINEX files
        print("\nReading RINEX files...")
        rover_obs_reader = RinexObsReader(rover_obs_file)
        base_obs_reader = RinexObsReader(base_obs_file)
        nav_reader = RinexNavReader(nav_file)
        
        rover_data = rover_obs_reader.read()
        base_data = base_obs_reader.read()
        nav_data = nav_reader.read()
        
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
    
    # Create DD processor and solver
    dd_processor = DoubleDifferenceProcessor()
    dd_solver = DDLeastSquares(dd_processor)
    
    # Process multiple epochs
    print("\nProcessing epochs...")
    baselines = []
    
    for i in range(min(10, len(rover_data))):
        rover_epoch = rover_data[i]
        
        # Handle time synchronization
        # Find matching base epoch or interpolate
        base_epoch = find_matching_base_epoch(rover_epoch, base_data)
        
        if base_epoch is None:
            continue
        
        try:
            # Solve for baseline
            baseline, rms, n_iter = dd_solver.solve_baseline(
                rover_epoch['observations'],
                base_epoch['observations'],
                nav_data,
                base_pos,
                max_iter=5
            )
            
            baselines.append(baseline)
            print(f"Epoch {i+1}: Baseline = {np.linalg.norm(baseline):.4f} m, RMS = {rms:.3f} m")
            
        except Exception as e:
            print(f"Epoch {i+1}: Failed - {str(e)}")
    
    # Plot results
    if len(baselines) > 0:
        plot_baselines(baselines, base_llh)
        
    print("\nExample completed!")


def find_matching_base_epoch(rover_epoch, base_data):
    """Find or interpolate matching base epoch"""
    rover_time = rover_epoch['gps_time']
    
    # Look for exact match
    for base_epoch in base_data:
        if abs(base_epoch['gps_time'] - rover_time) < 0.01:
            return base_epoch
    
    # Try interpolation
    for i in range(len(base_data) - 1):
        if base_data[i]['gps_time'] <= rover_time <= base_data[i+1]['gps_time']:
            return interpolate_epoch(base_data[i], base_data[i+1], rover_time)
    
    return None


def plot_baselines(baselines, base_llh):
    """Plot baseline components"""
    baselines = np.array(baselines)
    
    # Convert to ENU
    base_ecef = np.array([0, 0, 0])  # Reference at base
    enu_baselines = []
    for baseline in baselines:
        enu = ecef2enu(baseline, base_llh)
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
    print("   ✓ GLONASS excluded by default (inter-frequency bias)")
    print("   ✓ Residual-based quality assessment")
    
    print("\n5. EXPECTED ACCURACY:")
    print("   - Pseudorange DD: 1-5 m")
    print("   - Carrier phase DD (with fixed ambiguities): 1-2 cm")


if __name__ == '__main__':
    main()