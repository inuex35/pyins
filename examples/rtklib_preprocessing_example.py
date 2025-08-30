#!/usr/bin/env python3
"""
RTKLIB-style preprocessing example for pyins
============================================

Demonstrates RTKLIB-compatible:
- Cycle slip detection 
- Initial phase bias removal
- Ambiguity resolution
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyins.rtk import RTKLIBPreprocessor
from pyins.rtk.mlambda import mlambda, ratio_test


def example_basic_preprocessing():
    """Basic preprocessing example"""
    print("=" * 70)
    print("RTKLIB Preprocessing Example")
    print("=" * 70)
    
    # Initialize preprocessor
    preprocessor = RTKLIBPreprocessor()
    
    # Example 1: Cycle slip detection
    print("\n1. Cycle Slip Detection:")
    print("-" * 40)
    
    # Simulate observations with cycle slip
    observations = [
        # (sat, L1, L2, P1, P2)
        ("G01", 100000.5, 78000.5, 20000000.0, 20000100.0),
        ("G01", 100010.5, 78008.5, 20000050.0, 20000150.0),  # Normal
        ("G01", 100025.5, 78018.5, 20000100.0, 20000200.0),  # Jump in phase
    ]
    
    for i, (sat, L1, L2, P1, P2) in enumerate(observations):
        slip = preprocessor.detect_cycle_slip(sat, L1, L2, P1, P2)
        print(f"  Epoch {i}: Sat {sat}, Slip detected: {slip}")
    
    # Example 2: Initial bias estimation
    print("\n2. Initial Phase Bias Estimation:")
    print("-" * 40)
    
    # Rover and base observations
    rover_obs = {
        "G01": {"L1": 100000.5, "P1": 19114700.0},
        "G02": {"L1": 105000.3, "P1": 20069800.0},
        "G03": {"L1": 98000.8, "P1": 18732400.0},
    }
    
    base_obs = {
        "G01": {"L1": 99500.2, "P1": 19114600.0},
        "G02": {"L1": 104500.1, "P1": 20069700.0},
        "G03": {"L1": 97500.6, "P1": 18732300.0},
    }
    
    # Compute single difference biases
    for sat in rover_obs:
        if sat in base_obs:
            bias = preprocessor.init_phase_bias(
                rover_obs[sat]["L1"], base_obs[sat]["L1"],
                rover_obs[sat]["P1"], base_obs[sat]["P1"]
            )
            print(f"  Sat {sat}: Initial bias = {bias:.1f} cycles")
    
    # Example 3: Double difference bias computation
    print("\n3. Double Difference Biases:")
    print("-" * 40)
    
    dd_biases, sats = preprocessor.udbias(rover_obs, base_obs)
    
    print(f"  Reference satellite: {sats[0] if len(sats) > 0 else 'None'}")
    print(f"  DD biases: {dd_biases}")
    
    # Example 4: Ambiguity fixing
    print("\n4. Ambiguity Resolution:")
    print("-" * 40)
    
    if len(dd_biases) > 0:
        # Method 1: Simple rounding
        fixed_round = preprocessor.fix_amb_ROUND(dd_biases)
        print(f"  ROUND method: {fixed_round}")
        
        # Method 2: LAMBDA
        if len(dd_biases) >= 2:
            Q = np.eye(len(dd_biases)) * 0.001
            F, s = mlambda(dd_biases, Q, m=2)
            passed, ratio = ratio_test(s, threshold=3.0)
            
            print(f"  LAMBDA method: {F[:, 0].astype(int)}")
            print(f"  Ratio test: {ratio:.2f} ({'PASS' if passed else 'FAIL'})")
    
    # Example 5: Position validation
    print("\n5. Position Validation:")
    print("-" * 40)
    
    # Simulate DD residuals
    good_residuals = np.array([0.02, -0.03, 0.01, -0.02, 0.03])
    bad_residuals = np.array([0.02, -0.03, 0.5, -0.02, 0.03])
    
    valid_good = preprocessor.valpos(good_residuals, threshold=0.1)
    valid_bad = preprocessor.valpos(bad_residuals, threshold=0.1)
    
    print(f"  Good residuals: RMS={np.sqrt(np.mean(good_residuals**2)):.3f}m, Valid={valid_good}")
    print(f"  Bad residuals: RMS={np.sqrt(np.mean(bad_residuals**2)):.3f}m, Valid={valid_bad}")


def example_continuous_processing():
    """Example of continuous epoch processing"""
    print("\n" + "=" * 70)
    print("Continuous Processing Example")
    print("=" * 70)
    
    preprocessor = RTKLIBPreprocessor()
    
    # Simulate multiple epochs
    n_epochs = 10
    n_sats = 5
    
    print("\nSimulating RTK processing over multiple epochs:")
    print("-" * 50)
    
    for epoch in range(n_epochs):
        # Generate synthetic observations
        rover_obs = {}
        base_obs = {}
        
        for i in range(1, n_sats + 1):
            sat = f"G{i:02d}"
            
            # Add some noise and bias
            phase_bias = 1000 * i
            code_range = 20000000 + 100000 * i
            
            rover_obs[sat] = {
                "L1": phase_bias + epoch * 10 + np.random.randn() * 0.1,
                "L2": phase_bias * 0.78 + epoch * 8 + np.random.randn() * 0.1,
                "P1": code_range + epoch * 50 + np.random.randn() * 1,
                "P2": code_range + epoch * 50 + np.random.randn() * 1
            }
            
            base_obs[sat] = {
                "L1": phase_bias + epoch * 10 + np.random.randn() * 0.1,
                "L2": phase_bias * 0.78 + epoch * 8 + np.random.randn() * 0.1,
                "P1": code_range + epoch * 50 + np.random.randn() * 1,
                "P2": code_range + epoch * 50 + np.random.randn() * 1
            }
        
        # Process epoch
        dd_biases, sats = preprocessor.udbias(rover_obs, base_obs)
        
        if len(dd_biases) >= 4:
            # Try to fix ambiguities
            Q = np.eye(len(dd_biases)) * 0.001
            F, s = mlambda(dd_biases, Q, m=2)
            passed, ratio = ratio_test(s, threshold=3.0)
            
            status = "FIXED" if passed else "FLOAT"
            print(f"  Epoch {epoch:2d}: {len(dd_biases)} DD obs, Ratio={ratio:.2f}, Status={status}")


if __name__ == "__main__":
    example_basic_preprocessing()
    example_continuous_processing()
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)