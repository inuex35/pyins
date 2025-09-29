#!/usr/bin/env python3
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

"""
Example of GNSS frequency processing with real RINEX data
"""

import numpy as np
import sys
import os
from datetime import datetime, timedelta
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pyins.gnss.frequency import sat2freq
from pyins.io import RinexObsReader
from pyins.core.constants import (
    CLIGHT, SYS_GPS, SYS_GLO, SYS_GAL, SYS_BDS, SYS_QZS,
    FREQ_L1, FREQ_L2, FREQ_L5
)


def example_frequency_calculations():
    """Example of frequency calculations for different GNSS systems"""
    print("\n=== Example: GNSS Frequency Calculations ===")
    
    # Load real RINEX data
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    rover_file = os.path.join(data_dir, "main.obs")
    
    if not os.path.exists(rover_file):
        print(f"RINEX file not found: {rover_file}")
        return
    
    print(f"Loading RINEX file: {rover_file}")
    reader = RinexObsReader(rover_file)
    epochs = reader.read()[:1]  # Read first epoch
    
    if not epochs:
        print("No epochs found in RINEX file")
        return
    
    observations = epochs[0]['observations']
    
    # Group by system
    systems = {}
    for obs in observations:
        if obs.system not in systems:
            systems[obs.system] = []
        systems[obs.system].append(obs)
    
    print("\n=== Frequencies by GNSS System ===")
    sys_names = {1: "GPS", 2: "GLONASS", 4: "Galileo", 8: "QZSS", 16: "BeiDou"}
    
    for sys_id, obs_list in sorted(systems.items()):
        sys_name = sys_names.get(sys_id, f"System{sys_id}")
        print(f"\n{sys_name} (system={sys_id}):")
        
        # Show first satellite with data
        for obs in obs_list[:3]:
            if np.any(obs.L != 0):
                print(f"  PRN {obs.sat}:")
                
                # Calculate frequency for L1
                freq_l1 = sat2freq(obs.sat, 0)  # Index 0 for L1
                wavelength_l1 = CLIGHT / freq_l1 if freq_l1 > 0 else 0
                
                if obs.L[0] != 0:
                    print(f"    L1: {obs.L[0]:.3f} cycles")
                    print(f"        Frequency: {freq_l1/1e6:.2f} MHz")
                    print(f"        Wavelength: {wavelength_l1:.3f} m")
                
                # Calculate frequency for L2
                if len(obs.L) > 1 and obs.L[1] != 0:
                    freq_l2 = sat2freq(obs.sat, 1)  # Index 1 for L2
                    wavelength_l2 = CLIGHT / freq_l2 if freq_l2 > 0 else 0
                    print(f"    L2: {obs.L[1]:.3f} cycles")
                    print(f"        Frequency: {freq_l2/1e6:.2f} MHz")
                    print(f"        Wavelength: {wavelength_l2:.3f} m")
                
                break  # Show only first satellite per system


def example_carrier_phase_to_distance():
    """Example: Convert carrier phase to distance"""
    print("\n=== Example: Carrier Phase to Distance Conversion ===")
    
    # Load real RINEX data
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    rover_file = os.path.join(data_dir, "main.obs")
    
    if not os.path.exists(rover_file):
        print(f"RINEX file not found: {rover_file}")
        return
    
    reader = RinexObsReader(rover_file)
    epochs = reader.read()[:1]
    
    if not epochs:
        return
    
    observations = epochs[0]['observations']
    
    # Find GPS satellites
    gps_obs = [o for o in observations if o.system == 1 and o.L[0] != 0]
    
    if gps_obs:
        print("\nGPS Carrier Phase to Distance:")
        for obs in gps_obs[:3]:
            # L1 frequency and wavelength
            freq_l1 = sat2freq(obs.sat, 0)
            wavelength_l1 = CLIGHT / freq_l1
            
            # Convert carrier phase to distance
            phase_cycles = obs.L[0]
            distance = phase_cycles * wavelength_l1
            
            print(f"  PRN {obs.sat}:")
            print(f"    L1 phase: {phase_cycles:.3f} cycles")
            print(f"    Distance: {distance:.3f} m")
            print(f"    Pseudorange: {obs.P[0]:.3f} m")
            print(f"    Phase - Code: {distance - obs.P[0]:.3f} m")


def main():
    """Run all GNSS processing examples"""
    print("=" * 70)
    print("GNSS Processing Examples with Real RINEX Data")
    print("=" * 70)
    
    example_frequency_calculations()
    example_carrier_phase_to_distance()
    
    print("\n" + "=" * 70)
    print("All GNSS examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
