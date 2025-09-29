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
Example of satellite position calculations with real RINEX data
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pyins.io import RinexObsReader, RinexNavReader
from pyins.gnss.satellite_positions import compute_satellite_info
from pyins.core.constants import CLIGHT


def example_satellite_positions():
    """Example of computing satellite positions from navigation data"""
    print("\n=== Example: Satellite Position Calculation ===")
    
    # Load real RINEX data
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    obs_file = os.path.join(data_dir, "main.obs")
    nav_file = os.path.join(data_dir, "main.nav")
    
    if not os.path.exists(obs_file):
        print(f"Observation file not found: {obs_file}")
        return
    
    print(f"Loading observation file: {obs_file}")
    obs_reader = RinexObsReader(obs_file)
    epochs = obs_reader.read()[:1]
    
    if not epochs:
        print("No epochs found")
        return
    
    observations = epochs[0]['observations']
    epoch_time = epochs[0]['gps_time']
    
    # Check if navigation file exists
    if os.path.exists(nav_file):
        print(f"Loading navigation file: {nav_file}")
        nav_reader = RinexNavReader(nav_file)
        nav_data = nav_reader.read()
        
        print(f"\nEpoch time (GPS seconds): {epoch_time}")
        
        # Try to compute positions for GPS satellites
        gps_obs = [o for o in observations if o.system == 1][:5]
        
        for obs in gps_obs:
            try:
                # Compute satellite position
                sat_info = compute_satellite_info(
                    nav_data, obs.sat, epoch_time
                )
                sat_pos = sat_info['position']
                sat_clk = sat_info['clock_bias']
                sat_vel = sat_info['velocity']
                
                if sat_pos is not None:
                    distance = np.linalg.norm(sat_pos)
                    print(f"\nPRN {obs.sat}:")
                    print(f"  Position: [{sat_pos[0]:.3f}, {sat_pos[1]:.3f}, {sat_pos[2]:.3f}] m")
                    print(f"  Distance from Earth center: {distance:.3f} m")
                    print(f"  Clock bias: {sat_clk:.9f} s ({sat_clk * CLIGHT:.3f} m)")
                    print(f"  Velocity: [{sat_vel[0]:.3f}, {sat_vel[1]:.3f}, {sat_vel[2]:.3f}] m/s")
            except Exception as e:
                print(f"\nPRN {obs.sat}: Could not compute position ({e})")
    else:
        print(f"\nNavigation file not found: {nav_file}")
        print("Showing observation statistics instead:")
        
        # Show observation statistics
        systems = {}
        for obs in observations:
            if obs.system not in systems:
                systems[obs.system] = []
            systems[obs.system].append(obs.sat)
        
        sys_names = {1: "GPS", 2: "GLONASS", 4: "Galileo", 8: "QZSS", 16: "BeiDou"}
        
        for sys_id, sats in sorted(systems.items()):
            sys_name = sys_names.get(sys_id, f"System{sys_id}")
            print(f"\n{sys_name}: {len(sats)} satellites")
            print(f"  PRNs: {sorted(set(sats))[:10]}")  # Show first 10


def main():
    """Run satellite position examples"""
    print("=" * 70)
    print("Satellite Position Calculation Examples")
    print("=" * 70)
    
    example_satellite_positions()
    
    print("\n" + "=" * 70)
    print("All satellite position examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
