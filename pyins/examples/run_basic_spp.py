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
"""Run basic SPP with real RINEX data"""

import sys
import numpy as np
from pathlib import Path

# Add pyins to path
sys.path.insert(0, str(Path(__file__).parent))

from pyins.io.rinex import RinexObsReader, RinexNavReader
from pyins.examples.basic_spp import single_point_positioning
from pyins.coordinate import ecef2llh
from pyins.core.constants import sat2sys, sys2char

def main():
    # RINEX files
    obs_file = "../../opensky_debug/kaiyodai.obs"
    nav_file = "../../opensky_debug/kaiyodai.nav"
    
    print("Reading RINEX files...")
    
    # Read observation data
    obs_reader = RinexObsReader(obs_file)
    obs_data = obs_reader.read()
    print(f"Read {len(obs_data)} epochs of observation data")
    
    # Read navigation data  
    nav_reader = RinexNavReader(nav_file)
    nav_data = nav_reader.read()
    print(f"Read navigation data with {len(nav_data.eph)} ephemerides")
    
    # Process first few epochs
    num_epochs_to_process = min(5, len(obs_data))
    
    for idx in range(num_epochs_to_process):
        epoch = obs_data[idx]
        obs_list = epoch['observations']
        
        print(f"\n{'='*60}")
        print(f"Processing epoch {idx+1}/{num_epochs_to_process}:")
        print(f"  Time: {epoch['time']}")
        print(f"  Observations: {len(obs_list)}")
        
        # Count by system
        sys_counts = {}
        valid_pr = 0
        for obs in obs_list:
            if obs.P[0] > 0 or obs.P[1] > 0:
                valid_pr += 1
                sys = sat2sys(obs.sat)
                sys_char = sys2char(sys)
                if sys_char not in sys_counts:
                    sys_counts[sys_char] = 0
                sys_counts[sys_char] += 1
        
        print(f"  Valid pseudoranges: {valid_pr}")
        print(f"  By system: {sys_counts}")
        
        # Try SPP with different system combinations
        # Use approximate initial position from reference file
        # lat=35.666°N, lon=139.792°E, h=65m
        from pyins.coordinate import llh2ecef
        initial_pos = llh2ecef(np.array([np.deg2rad(35.666), np.deg2rad(139.792), 65.0]))
        print(f"Using initial position: ECEF={initial_pos/1e3} km")
        
        # Try multi-GNSS first
        print("\nTrying multi-GNSS (GPS, Galileo, QZSS) - no BeiDou...")
        solution, used_sats = single_point_positioning(
            obs_list, nav_data, initial_pos=initial_pos, systems_to_use=['G', 'E', 'J']
        )
        
        # Also try with BeiDou included to test the fix
        print("\nTrying multi-GNSS with BeiDou (GPS, Galileo, BeiDou, QZSS)...")
        solution_with_bds, used_sats_bds = single_point_positioning(
            obs_list, nav_data, initial_pos=initial_pos, systems_to_use=['G', 'E', 'C', 'J']
        )
        if solution_with_bds:
            llh_bds = ecef2llh(solution_with_bds.rr)
            print(f"With BeiDou: h={llh_bds[2]:.2f} m (should be ~100m if fixed)")
        
        if not solution:
            print("Multi-GNSS failed, trying GPS only...")
            solution, used_sats = single_point_positioning(
                obs_list, nav_data, initial_pos=initial_pos, systems_to_use=['G']
            )
        
        if not solution:
            print("Multi-GNSS (no BeiDou) failed, trying each system individually...")
            # Try each system individually to see which ones work
            for sys_char in ['E', 'C', 'J']:  # Galileo, BeiDou, QZSS
                if sys_char in sys_counts and sys_counts[sys_char] >= 4:
                    print(f"\nTrying {sys_char} only ({sys_counts[sys_char]} satellites)...")
                    solution, used_sats = single_point_positioning(
                        obs_list, nav_data, initial_pos=initial_pos, systems_to_use=[sys_char]
                    )
                    if solution:
                        print(f"{sys_char}-only solution found!")
                        break
        
        # Also try BeiDou alone to debug the issue
        if 'C' in sys_counts and sys_counts['C'] >= 4:
            print(f"\nTrying BeiDou only to check time issue ({sys_counts['C']} satellites)...")
            bds_solution, bds_sats = single_point_positioning(
                obs_list, nav_data, initial_pos=initial_pos, systems_to_use=['C']
            )
            if bds_solution:
                bds_llh = ecef2llh(bds_solution.rr)
                print(f"BeiDou-only solution:")
                print(f"  LLH: lat={np.rad2deg(bds_llh[0]):.6f}°, lon={np.rad2deg(bds_llh[1]):.6f}°, h={bds_llh[2]:.2f} m")
                print(f"  Clock bias: {bds_solution.dtr[0]*1e9:.1f} ns")
        
        if solution:
            # Convert to LLH
            llh = ecef2llh(solution.rr)
            
            print(f"\nSPP Solution found:")
            print(f"  ECEF: [{solution.rr[0]:.3f}, {solution.rr[1]:.3f}, {solution.rr[2]:.3f}] m")
            print(f"  LLH: lat={np.rad2deg(llh[0]):.6f}°, lon={np.rad2deg(llh[1]):.6f}°, h={llh[2]:.2f} m")
            print(f"  Clock bias (GPS): {solution.dtr[0]*1e9:.1f} ns")
            print(f"  Used satellites: {len(used_sats)}")
            
            # Show which satellites were used
            sys_used = {}
            for sat in used_sats:
                sys_char = sys2char(sat2sys(sat))
                if sys_char not in sys_used:
                    sys_used[sys_char] = 0
                sys_used[sys_char] += 1
            
            print(f"  Satellites by system: {sys_used}")
        else:
            print("SPP failed to converge")
    
    print(f"\n{'='*60}")
    print("Processing complete")

if __name__ == "__main__":
    main()