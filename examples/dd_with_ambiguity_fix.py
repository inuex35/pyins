#!/usr/bin/env python3
"""
Double Difference Positioning with Ambiguity Resolution
========================================================

This example demonstrates RTK positioning using LAMBDA for ambiguity resolution.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyins.io.rinex import RinexObsReader, RinexNavReader
from pyins.gnss.spp import single_point_positioning
from pyins.gnss.ephemeris import satpos
from pyins.rtk.lambda_rtklib import mlambda
from pyins.rtk.ambiguity_resolution import RTKAmbiguityManager
from pyins.core.constants import CLIGHT, FREQ_L1, FREQ_L2
import warnings
warnings.filterwarnings('ignore')


def compute_dd_geometry_matrix(rover_pos, base_pos, sat_pos_ref, sat_pos_other):
    """
    Compute DD geometry matrix row
    
    Fixed version with correct formulation
    """
    # Unit vectors from rover
    u_rover_ref = (sat_pos_ref - rover_pos) / np.linalg.norm(sat_pos_ref - rover_pos)
    u_rover_other = (sat_pos_other - rover_pos) / np.linalg.norm(sat_pos_other - rover_pos)
    
    # DD geometry row: derivative of DD with respect to rover position
    # ∂DD/∂x = -u_j + u_i (other - reference)
    h_row = -u_rover_other + u_rover_ref
    
    return h_row


def process_epoch_with_ambiguity(rover_obs, base_obs, nav, rover_pos0, base_pos):
    """Process single epoch with ambiguity resolution"""
    
    # Get satellite positions
    sat_positions = satpos(rover_obs, nav)
    
    # Find common satellites
    common_sats = []
    for sat, obs_data in rover_obs.items():
        if sat in base_obs and sat in sat_positions:
            if sat.startswith('G'):  # GPS only
                sat_pos = sat_positions[sat]['pos']
                if sat_pos is not None and np.linalg.norm(sat_pos) > 0:
                    # Check if we have phase observations
                    if 'L1' in obs_data and 'L1' in base_obs[sat]:
                        if obs_data['L1'] > 0 and base_obs[sat]['L1'] > 0:
                            common_sats.append(sat)
    
    if len(common_sats) < 5:
        return None, None, None, None
    
    # Select reference satellite (highest elevation)
    elevations = {}
    for sat in common_sats:
        sat_pos = sat_positions[sat]['pos']
        dx = sat_pos - rover_pos0
        r = np.linalg.norm(dx)
        if r > 0:
            elev = np.arcsin(dx[2] / r) * 180 / np.pi
            elevations[sat] = elev
    
    ref_sat = max(elevations, key=elevations.get)
    other_sats = [s for s in common_sats if s != ref_sat]
    
    # Build DD system
    H = []  # Geometry matrix
    dd_phase = []  # DD phase observations (meters)
    dd_code = []  # DD code observations (meters)
    dd_phase_cycles = []  # DD phase in cycles
    
    lambda_l1 = CLIGHT / FREQ_L1  # ~0.19m
    
    for sat in other_sats:
        # Get satellite positions
        sat_pos_ref = sat_positions[ref_sat]['pos']
        sat_pos_other = sat_positions[sat]['pos']
        
        # Compute DD geometry
        h_row = compute_dd_geometry_matrix(rover_pos0, base_pos, sat_pos_ref, sat_pos_other)
        H.append(h_row)
        
        # Get observations
        L1_rover_ref = rover_obs[ref_sat]['L1']
        L1_rover_other = rover_obs[sat]['L1']
        L1_base_ref = base_obs[ref_sat]['L1']
        L1_base_other = base_obs[sat]['L1']
        
        # Code observations
        P1_rover_ref = rover_obs[ref_sat].get('P1', rover_obs[ref_sat].get('C1', 0))
        P1_rover_other = rover_obs[sat].get('P1', rover_obs[sat].get('C1', 0))
        P1_base_ref = base_obs[ref_sat].get('P1', base_obs[ref_sat].get('C1', 0))
        P1_base_other = base_obs[sat].get('P1', base_obs[sat].get('C1', 0))
        
        # Single differences
        sd_phase_ref = (L1_rover_ref - L1_base_ref) * lambda_l1  # Convert to meters
        sd_phase_other = (L1_rover_other - L1_base_other) * lambda_l1
        sd_code_ref = P1_rover_ref - P1_base_ref
        sd_code_other = P1_rover_other - P1_base_other
        
        # Double differences
        dd_L1 = sd_phase_other - sd_phase_ref  # in meters
        dd_P1 = sd_code_other - sd_code_ref  # in meters
        dd_L1_cycles = (sd_phase_other - sd_phase_ref) / lambda_l1  # in cycles
        
        dd_phase.append(dd_L1)
        dd_code.append(dd_P1)
        dd_phase_cycles.append(dd_L1_cycles)
    
    if len(dd_phase) < 4:
        return None, None, None, None
    
    H = np.array(H)
    dd_phase = np.array(dd_phase)
    dd_code = np.array(dd_code)
    dd_phase_cycles = np.array(dd_phase_cycles)
    
    # Float solution (using code only for simplicity)
    try:
        # Least squares with code observations
        dx = np.linalg.lstsq(H, dd_code, rcond=None)[0]
        float_position = rover_pos0 + dx
        
        # Compute float ambiguities
        residuals = dd_code - H @ dx
        float_ambiguities = residuals / lambda_l1
        
        # Create covariance matrix based on elevations
        n = len(dd_phase_cycles)
        Q = np.eye(n) * 0.001
        for i, sat in enumerate(other_sats):
            if sat in elevations:
                elev = elevations[sat]
                weight = np.sin(np.radians(max(elev, 10)))
                Q[i, i] = 1.0 / (weight**2) * 0.001
        
        # Try to fix ambiguities
        manager = RTKAmbiguityManager()
        fixed_amb, info = manager.resolve(float_ambiguities, Q)
        
        if fixed_amb is not None and info.get('ratio', 0) > 2.5:
            # Fixed solution
            dd_fixed = dd_phase - fixed_amb * lambda_l1
            dx_fixed = np.linalg.lstsq(H, dd_fixed, rcond=None)[0]
            fixed_position = rover_pos0 + dx_fixed
            
            return fixed_position, float_position, fixed_amb, info
        else:
            return None, float_position, None, None
            
    except Exception as e:
        return None, None, None, None


def main():
    """Main processing function"""
    print("=" * 80)
    print("Double Difference RTK with Ambiguity Resolution")
    print("=" * 80)
    
    # File paths
    base_dir = "/home/ubuntu/graph_ins"
    rover_obs_file = f"{base_dir}/okujo_test/main.obs"
    base_obs_file = f"{base_dir}/okujo_test/base.obs"
    nav_file = f"{base_dir}/okujo_test/rover.nav"
    
    print(f"\nReading RINEX files...")
    
    # Read RINEX files
    rover_reader = RinexObsReader(rover_obs_file)
    rover_obs = rover_reader.read()
    
    base_reader = RinexObsReader(base_obs_file)
    base_obs = base_reader.read()
    
    nav_reader = RinexNavReader(nav_file)
    nav = nav_reader.read()
    
    print(f"  Rover epochs: {len(rover_obs)}")
    print(f"  Base epochs: {len(base_obs)}")
    
    # Known base position (approximate)
    base_pos = np.array([-3962167.673, 3349401.645, 3698226.118])
    rover_pos0 = base_pos + np.array([10, 10, 10])  # Initial guess
    
    # Process epochs
    print("\n" + "-" * 60)
    print("Processing with Ambiguity Resolution")
    print("-" * 60)
    
    results = {
        'fixed': [],
        'float': [],
        'ratios': [],
        'epochs': []
    }
    
    n_fixed = 0
    n_total = 0
    
    # Process subset of epochs
    for epoch_idx in range(100, min(200, len(rover_obs))):
        if epoch_idx >= len(base_obs):
            continue
        
        fixed_pos, float_pos, fixed_amb, info = process_epoch_with_ambiguity(
            rover_obs[epoch_idx],
            base_obs[epoch_idx],
            nav,
            rover_pos0,
            base_pos
        )
        
        n_total += 1
        
        if fixed_pos is not None:
            n_fixed += 1
            baseline = fixed_pos - base_pos
            baseline_length = np.linalg.norm(baseline)
            ratio = info.get('ratio', 0)
            
            results['fixed'].append(baseline)
            results['ratios'].append(ratio)
            
            if n_fixed <= 5:  # Print first few
                print(f"\nEpoch {epoch_idx} - FIXED:")
                print(f"  Baseline: [{baseline[0]:.3f}, {baseline[1]:.3f}, {baseline[2]:.3f}] m")
                print(f"  Length: {baseline_length:.3f} m")
                print(f"  Ratio test: {ratio:.2f}")
                print(f"  Ambiguities: {fixed_amb[:3].astype(int)}...")
        
        elif float_pos is not None:
            baseline = float_pos - base_pos
            results['float'].append(baseline)
            
            if n_total <= 10 and n_fixed == 0:
                print(f"\nEpoch {epoch_idx} - FLOAT only:")
                print(f"  Baseline: [{baseline[0]:.3f}, {baseline[1]:.3f}, {baseline[2]:.3f}] m")
        
        # Update rover position for next epoch
        if fixed_pos is not None:
            rover_pos0 = fixed_pos
        elif float_pos is not None:
            rover_pos0 = float_pos
    
    # Statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    
    print(f"\nAmbiguity Resolution:")
    print(f"  Total epochs: {n_total}")
    print(f"  Fixed solutions: {n_fixed}")
    print(f"  Fix rate: {n_fixed/n_total*100:.1f}%")
    
    if len(results['fixed']) > 0:
        fixed_array = np.array(results['fixed'])
        mean_baseline = np.mean(fixed_array, axis=0)
        std_baseline = np.std(fixed_array, axis=0)
        
        print(f"\nFixed Solution Statistics:")
        print(f"  Mean baseline: [{mean_baseline[0]:.3f}, {mean_baseline[1]:.3f}, {mean_baseline[2]:.3f}] m")
        print(f"  Std dev: [{std_baseline[0]:.3f}, {std_baseline[1]:.3f}, {std_baseline[2]:.3f}] m")
        print(f"  Mean ratio: {np.mean(results['ratios']):.2f}")
        print(f"  Min ratio: {np.min(results['ratios']):.2f}")
    
    if len(results['float']) > 0:
        float_array = np.array(results['float'])
        mean_float = np.mean(float_array, axis=0)
        std_float = np.std(float_array, axis=0)
        
        print(f"\nFloat Solution Statistics:")
        print(f"  Mean baseline: [{mean_float[0]:.3f}, {mean_float[1]:.3f}, {mean_float[2]:.3f}] m")
        print(f"  Std dev: [{std_float[0]:.3f}, {std_float[1]:.3f}, {std_float[2]:.3f}] m")
    
    # Compare improvement
    if len(results['fixed']) > 0 and len(results['float']) > 0:
        print(f"\nImprovement with Ambiguity Fixing:")
        print(f"  East std:  {std_float[0]:.3f} → {std_baseline[0]:.3f} m ({std_float[0]/std_baseline[0]:.1f}x improvement)")
        print(f"  North std: {std_float[1]:.3f} → {std_baseline[1]:.3f} m ({std_float[1]/std_baseline[1]:.1f}x improvement)")
        print(f"  Up std:    {std_float[2]:.3f} → {std_baseline[2]:.3f} m ({std_float[2]/std_baseline[2]:.1f}x improvement)")
    
    print("\n" + "=" * 80)
    print("Test Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()