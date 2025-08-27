# Copyright 2024 The PyIns Authors. All Rights Reserved.
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

"""Basic Single Point Positioning (SPP) example"""

from datetime import datetime
import numpy as np

from pyins.coordinate import ecef2llh
from pyins.core.constants import SYS_GPS
from pyins.core.data_structures import NavigationData, Observation
from pyins.core.unified_time import TimeCore
from pyins.gnss.spp import single_point_positioning
from pyins.io.rinex import RinexNavReader, RinexObsReader


def main():
    """Example usage of SPP with RINEX data"""
    
    print("Single Point Positioning Example")
    print("=" * 40)
    
    # Example 1: Load real RINEX data (if available)
    try:
        # Load navigation data
        nav_reader = RinexNavReader('/home/ubuntu/graph_ins/okujo_test/rover.nav')
        nav_data = nav_reader.read()
        
        # Load observation data
        obs_reader = RinexObsReader('/home/ubuntu/graph_ins/okujo_test/main.obs')
        obs_epochs = obs_reader.read()
        
        if obs_epochs:
            print("\nProcessing real RINEX data...")
            first_epoch = obs_epochs[0]
            observations = first_epoch['observations']
            
            # Run SPP with RAIM
            solution, used_sats = single_point_positioning(
                observations,
                nav_data,
                initial_pos=np.array([-3954867.0, 3353972.0, 3701263.0]),  # Tokyo area
                systems_to_use=['G', 'R'],  # GPS and GLONASS
                use_raim=True,
                raim_threshold=30.0  # 30 meters
            )
            
            if solution:
                llh = ecef2llh(solution.rr)
                print("\nSolution found:")
                print(f"  Position ECEF: {solution.rr}")
                print(f"  Position LLH: {np.rad2deg(llh[0]):.6f}°, {np.rad2deg(llh[1]):.6f}°, {llh[2]:.1f}m")
                print(f"  Clock bias: {solution.dtr[0]*1e9:.1f} ns")
                print(f"  Satellites used: {solution.ns}")
            else:
                print("No solution found")
                
    except Exception as e:
        print(f"Could not process RINEX data: {e}")
        print("\nFalling back to simulated data...")
        
        # Example 2: Simulated data
        current_tc = TimeCore.from_datetime(datetime.now())
        
        # Create sample observations
        observations = []
        for prn in [1, 3, 7, 11, 15, 20, 25, 30]:
            obs = Observation(
                time=current_tc.get_gps_seconds(),
                sat=prn,
                system=SYS_GPS,
                P=np.array([20e6 + np.random.uniform(0, 5e6), 0, 0]),
                SNR=np.array([40 + np.random.uniform(0, 10), 0, 0])
            )
            observations.append(obs)
        
        # Empty navigation data for example
        nav_data = NavigationData()
        
        # Run SPP
        solution, used_sats = single_point_positioning(observations, nav_data)
        
        if solution:
            llh = ecef2llh(solution.rr)
            print("\nSimulated solution:")
            print(f"  Position: {solution.rr}")
            print(f"  LLH: {np.rad2deg(llh[0]):.6f}°, {np.rad2deg(llh[1]):.6f}°, {llh[2]:.1f}m")
            print(f"  Clock bias: {solution.dtr[0]*1e9:.1f} ns")
            print(f"  Satellites used: {len(used_sats)}")
        else:
            print("No solution found")


if __name__ == "__main__":
    main()