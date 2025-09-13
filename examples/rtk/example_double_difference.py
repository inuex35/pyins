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
Working double difference example with synthetic data
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pyins.rtk.double_difference import DoubleDifferenceProcessor
from pyins.core.data_structures import Observation


def main():
    """Simple double difference example with synthetic data"""
    print("=" * 70)
    print("Double Difference Example (Synthetic Data)")
    print("=" * 70)
    
    # Create synthetic observations
    print("\n=== Creating Synthetic Observations ===")
    
    # Common time
    time = 1755584891.0
    
    # Create rover observations (4 GPS satellites)
    rover_obs = []
    for i, sat in enumerate([5, 10, 15, 24]):
        obs = Observation(time=time, sat=sat, system=1)  # GPS system
        # Synthetic L1 phase (cycles)
        obs.L[0] = 100000000 + i * 10000 + np.random.randn() * 10
        # Synthetic L1 pseudorange (meters)
        obs.P[0] = 20000000 + i * 1000 + np.random.randn() * 2
        rover_obs.append(obs)
    
    # Create base observations (same satellites, different values)
    base_obs = []
    for i, sat in enumerate([5, 10, 15, 24]):
        obs = Observation(time=time, sat=sat, system=1)
        # Slightly different values for base
        obs.L[0] = 100000000 + i * 10000 + np.random.randn() * 10 + 100
        obs.P[0] = 20000000 + i * 1000 + np.random.randn() * 2 + 20
        base_obs.append(obs)
    
    print(f"Created {len(rover_obs)} rover observations")
    print(f"Created {len(base_obs)} base observations")
    
    # Form double differences
    print("\n=== Forming Double Differences ===")
    dd_processor = DoubleDifferenceProcessor()
    dd_data = dd_processor.form_double_differences(rover_obs, base_obs, frequency_idx=0)
    
    if dd_data:
        dd_pr, dd_cp, pairs, refs = dd_data
        print(f"Reference satellite: {refs}")
        print(f"Number of DD observations: {len(dd_pr)}")
        
        print("\nDouble Difference Results:")
        for i in range(len(dd_cp)):
            if i < len(pairs):
                ref_sat, other_sat = pairs[i]
                print(f"  Pair ({other_sat}-{ref_sat}):")
                print(f"    Phase DD: {dd_cp[i]:.3f} cycles")
                print(f"    Code DD: {dd_pr[i]:.3f} m")
    else:
        print("Failed to form double differences")
    
    print("\n" + "=" * 70)
    print("Example completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()