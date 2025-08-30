#!/usr/bin/env python3
"""Test LAMBDA with real RINEX data"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyins.io.rinex import RinexObsReader, RinexNavReader
from pyins.gnss.ephemeris import Ephemeris
from pyins.gnss.spp import single_point_positioning
from pyins.gnss.double_difference import DoubleDifference
from pyins.rtk.lambda_rtklib import mlambda
from pyins.rtk.ambiguity_resolution import RTKAmbiguityManager
from pyins.const import CLIGHT, FREQ_L1, FREQ_L2
import matplotlib.pyplot as plt

def compute_dd_ambiguities(dd_proc, obs_rover, obs_base, nav, epoch_idx):
    """Compute DD float ambiguities for one epoch"""
    
    # Get observations at this epoch
    rover_obs = obs_rover[epoch_idx]
    base_obs = obs_base[epoch_idx]
    
    if not rover_obs or not base_obs:
        return None, None, None, None
    
    # Find common satellites
    common_sats = set(rover_obs.keys()) & set(base_obs.keys())
    common_sats = sorted([s for s in common_sats if s.startswith('G')])  # GPS only
    
    if len(common_sats) < 5:
        return None, None, None, None
    
    # Select reference satellite (highest elevation)
    elevations = {}
    for sat in common_sats:
        if 'elev' in rover_obs[sat]:
            elevations[sat] = rover_obs[sat]['elev']
    
    if not elevations:
        return None, None, None, None
    
    ref_sat = max(elevations, key=elevations.get)
    other_sats = [s for s in common_sats if s != ref_sat]
    
    # Compute DD phase observations (in cycles)
    dd_phase_l1 = []
    dd_phase_l2 = []
    dd_code = []
    sat_list = []
    
    # Get reference satellite observations
    ref_L1_rover = rover_obs[ref_sat].get('L1', 0)
    ref_L1_base = base_obs[ref_sat].get('L1', 0)
    ref_L2_rover = rover_obs[ref_sat].get('L2', 0)
    ref_L2_base = base_obs[ref_sat].get('L2', 0)
    ref_P1_rover = rover_obs[ref_sat].get('P1', rover_obs[ref_sat].get('C1', 0))
    ref_P1_base = base_obs[ref_sat].get('P1', base_obs[ref_sat].get('C1', 0))
    
    if ref_L1_rover == 0 or ref_L1_base == 0:
        return None, None, None, None
    
    for sat in other_sats:
        # L1 phase
        L1_rover = rover_obs[sat].get('L1', 0)
        L1_base = base_obs[sat].get('L1', 0)
        
        # L2 phase
        L2_rover = rover_obs[sat].get('L2', 0)
        L2_base = base_obs[sat].get('L2', 0)
        
        # Code
        P1_rover = rover_obs[sat].get('P1', rover_obs[sat].get('C1', 0))
        P1_base = base_obs[sat].get('P1', base_obs[sat].get('C1', 0))
        
        if L1_rover == 0 or L1_base == 0:
            continue
        
        # Single differences
        sd_L1_ref = ref_L1_rover - ref_L1_base
        sd_L1_other = L1_rover - L1_base
        sd_L2_ref = ref_L2_rover - ref_L2_base if ref_L2_rover and ref_L2_base else 0
        sd_L2_other = L2_rover - L2_base if L2_rover and L2_base else 0
        sd_P1_ref = ref_P1_rover - ref_P1_base
        sd_P1_other = P1_rover - P1_base
        
        # Double differences
        dd_L1 = sd_L1_other - sd_L1_ref  # in cycles
        dd_L2 = sd_L2_other - sd_L2_ref if sd_L2_ref != 0 else dd_L1
        dd_P1 = sd_P1_other - sd_P1_ref  # in meters
        
        dd_phase_l1.append(dd_L1)
        dd_phase_l2.append(dd_L2)
        dd_code.append(dd_P1)
        sat_list.append(sat)
    
    if len(dd_phase_l1) < 4:
        return None, None, None, None
    
    # Convert to numpy arrays
    dd_phase_l1 = np.array(dd_phase_l1)
    dd_phase_l2 = np.array(dd_phase_l2)
    dd_code = np.array(dd_code)
    
    # Estimate float ambiguities (simple approach)
    lambda_l1 = CLIGHT / FREQ_L1
    float_amb_l1 = (dd_code / lambda_l1) - dd_phase_l1
    
    # Create covariance matrix based on elevations
    n = len(dd_phase_l1)
    Q = np.eye(n) * 0.01  # Default variance
    
    for i, sat in enumerate(sat_list):
        if sat in elevations:
            elev = elevations[sat]
            weight = np.sin(np.radians(elev))
            Q[i, i] = 1.0 / (weight**2) * 0.001
    
    return float_amb_l1, dd_phase_l1, Q, sat_list

def main():
    """Test LAMBDA with real RINEX data"""
    print("=" * 80)
    print("LAMBDA Test with Real RINEX Data")
    print("=" * 80)
    
    # File paths - using okujo_test data
    base_dir = "/home/ubuntu/graph_ins"
    rover_obs_file = f"{base_dir}/okujo_test/main.obs"
    base_obs_file = f"{base_dir}/okujo_test/base.obs"
    nav_file = f"{base_dir}/okujo_test/main.nav"
    
    if not os.path.exists(rover_obs_file):
        print(f"Error: Rover observation file not found: {rover_obs_file}")
        return
    
    print(f"\nReading RINEX files...")
    print(f"  Rover: {rover_obs_file}")
    print(f"  Base:  {base_obs_file}")
    print(f"  Nav:   {nav_file}")
    
    # Read RINEX files
    try:
        rover_reader = RinexObsReader(rover_obs_file)
        rover_epochs, rover_obs = rover_reader.read()
        
        base_reader = RinexObsReader(base_obs_file)
        base_epochs, base_obs = base_reader.read()
        
        nav_reader = RinexNavReader(nav_file)
        nav = nav_reader.read()
        
        print(f"\nData loaded:")
        print(f"  Rover epochs: {len(rover_epochs)}")
        print(f"  Base epochs:  {len(base_epochs)}")
        print(f"  Navigation messages: {len(nav)}")
        
    except Exception as e:
        print(f"Error reading RINEX files: {e}")
        return
    
    # Initialize processors
    ephemeris = Ephemeris()
    dd_proc = DoubleDifference()
    manager = RTKAmbiguityManager()
    
    # Process multiple epochs
    print("\n" + "-" * 60)
    print("Processing DD Ambiguities")
    print("-" * 60)
    
    n_epochs_to_process = min(50, len(rover_epochs))
    fixed_solutions = []
    float_solutions = []
    success_count = 0
    
    for epoch_idx in range(10, min(10 + n_epochs_to_process, len(rover_epochs))):
        # Compute DD float ambiguities
        float_amb, dd_phase, Q, sat_list = compute_dd_ambiguities(
            dd_proc, rover_obs, base_obs, nav, epoch_idx
        )
        
        if float_amb is None:
            continue
        
        # Try to fix ambiguities
        try:
            # Method 1: Standard LAMBDA
            fixed_amb, info = manager.resolve(float_amb, Q)
            
            if fixed_amb is not None and info.get('ratio', 0) > 2.5:
                success_count += 1
                fixed_solutions.append(fixed_amb)
                float_solutions.append(float_amb)
                
                if epoch_idx < 15:  # Print first few results
                    print(f"\nEpoch {epoch_idx} (Time: {rover_epochs[epoch_idx]}):")
                    print(f"  Satellites: {sat_list}")
                    print(f"  Float amb: {float_amb[:4].round(2)}...")
                    print(f"  Fixed amb: {fixed_amb[:4].astype(int)}...")
                    print(f"  Ratio test: {info.get('ratio', 0):.2f}")
                    print(f"  Valid: {info.get('is_valid', False)}")
            
        except Exception as e:
            if epoch_idx < 15:
                print(f"  Error at epoch {epoch_idx}: {e}")
    
    print(f"\n" + "-" * 60)
    print(f"Summary:")
    print(f"  Epochs processed: {n_epochs_to_process}")
    print(f"  Successful fixes: {success_count}")
    print(f"  Success rate: {success_count/n_epochs_to_process*100:.1f}%")
    
    # Test with RTKLIB mlambda
    print("\n" + "-" * 60)
    print("Testing RTKLIB mlambda Implementation")
    print("-" * 60)
    
    rtklib_success = 0
    for epoch_idx in range(10, min(10 + n_epochs_to_process, len(rover_epochs))):
        float_amb, dd_phase, Q, sat_list = compute_dd_ambiguities(
            dd_proc, rover_obs, base_obs, nav, epoch_idx
        )
        
        if float_amb is None:
            continue
        
        try:
            # RTKLIB mlambda
            candidates, ratios = mlambda(float_amb, Q, m=2)
            fixed_amb = candidates[:, 0]
            ratio = ratios[0] if len(ratios) > 0 else 0
            
            if ratio > 2.5:
                rtklib_success += 1
                
                if epoch_idx < 15:
                    print(f"\nEpoch {epoch_idx}:")
                    print(f"  Fixed: {fixed_amb[:4].astype(int)}...")
                    print(f"  Ratio: {ratio:.2f}")
            
        except Exception as e:
            pass
    
    print(f"\nRTKLIB Results:")
    print(f"  Success rate: {rtklib_success/n_epochs_to_process*100:.1f}%")
    
    # Analyze ambiguity stability over time
    if len(fixed_solutions) > 5:
        print("\n" + "-" * 60)
        print("Ambiguity Stability Analysis")
        print("-" * 60)
        
        fixed_array = np.array(fixed_solutions)
        n_sats = fixed_array.shape[1]
        
        for i in range(min(4, n_sats)):
            amb_values = fixed_array[:, i]
            unique_values = np.unique(amb_values)
            
            print(f"\nSatellite {i+1}:")
            print(f"  Unique ambiguities: {unique_values}")
            print(f"  Most common: {np.bincount(amb_values.astype(int) - amb_values.min().astype(int)).argmax() + amb_values.min()}")
            print(f"  Stability: {np.sum(amb_values == amb_values[0])/len(amb_values)*100:.1f}%")
    
    print("\n" + "=" * 80)
    print("Test Complete")
    print("=" * 80)

if __name__ == "__main__":
    main()