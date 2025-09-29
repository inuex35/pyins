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
Example of GNSS observables processing with real RINEX data
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pyins.io import RinexObsReader
from pyins.core.constants import CLIGHT, FREQ_L1, FREQ_L2


def example_observable_combinations():
    """Example of observable combinations (ionosphere-free, wide-lane, etc.)"""
    print("\n=== Example: Observable Combinations ===")
    
    # Load real RINEX data
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    rover_file = os.path.join(data_dir, "main.obs")
    
    if not os.path.exists(rover_file):
        print(f"RINEX file not found: {rover_file}")
        return
    
    print(f"Loading RINEX file: {rover_file}")
    reader = RinexObsReader(rover_file)
    epochs = reader.read()[:1]
    
    if not epochs:
        print("No epochs found")
        return
    
    observations = epochs[0]['observations']
    
    # Find GPS satellites with L1 and L2
    gps_dual_freq = [o for o in observations 
                     if o.system == 1 and o.L[0] != 0 and len(o.L) > 1 and o.L[1] != 0]
    
    if not gps_dual_freq:
        print("No dual-frequency GPS observations found")
        return
    
    print(f"Found {len(gps_dual_freq)} dual-frequency GPS satellites")
    
    for obs in gps_dual_freq[:3]:
        print(f"\nPRN {obs.sat}:")
        
        # Get frequencies
        f1 = FREQ_L1
        f2 = FREQ_L2
        
        # Wavelengths
        lambda1 = CLIGHT / f1
        lambda2 = CLIGHT / f2
        
        # Phase observations in meters
        L1_m = obs.L[0] * lambda1
        L2_m = obs.L[1] * lambda2
        
        # Pseudorange observations
        P1 = obs.P[0]
        P2 = obs.P[1] if len(obs.P) > 1 else P1
        
        # Ionosphere-free combination
        alpha = (f1**2) / (f1**2 - f2**2)
        beta = (f2**2) / (f1**2 - f2**2)
        
        IF_phase = alpha * L1_m - beta * L2_m
        IF_code = alpha * P1 - beta * P2
        
        print(f"  L1 phase: {obs.L[0]:.3f} cycles ({L1_m:.3f} m)")
        print(f"  L2 phase: {obs.L[1]:.3f} cycles ({L2_m:.3f} m)")
        print(f"  Ionosphere-free phase: {IF_phase:.3f} m")
        print(f"  Ionosphere-free code: {IF_code:.3f} m")
        
        # Wide-lane combination
        WL_phase = (f1 * L1_m - f2 * L2_m) / (f1 - f2)
        WL_wavelength = CLIGHT / (f1 - f2)
        
        print(f"  Wide-lane phase: {WL_phase:.3f} m")
        print(f"  Wide-lane wavelength: {WL_wavelength:.3f} m")
        
        # Narrow-lane combination
        NL_phase = (f1 * L1_m + f2 * L2_m) / (f1 + f2)
        NL_wavelength = CLIGHT / (f1 + f2)
        
        print(f"  Narrow-lane phase: {NL_phase:.3f} m")
        print(f"  Narrow-lane wavelength: {NL_wavelength:.3f} m")


def main():
    """Run observable examples"""
    print("=" * 70)
    print("GNSS Observable Processing Examples")
    print("=" * 70)
    
    example_observable_combinations()
    
    print("\n" + "=" * 70)
    print("All observable examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
