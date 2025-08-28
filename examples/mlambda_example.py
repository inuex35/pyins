#!/usr/bin/env python3
"""
MLAMBDA Example - Integer Ambiguity Resolution
==============================================

This example demonstrates the usage of the MLAMBDA algorithm for
GNSS integer ambiguity resolution in various scenarios.

MLAMBDA (Modified LAMBDA) is the standard method for solving the
integer least-squares problem in RTK positioning.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyins.rtk.mlambda import mlambda, ratio_test, success_rate, mlambda_with_partial
from pyins.io.rinex import RinexObsReader
import matplotlib.pyplot as plt
import time


def example_basic():
    """Basic MLAMBDA usage example"""
    print("=" * 70)
    print("Example 1: Basic MLAMBDA Usage")
    print("=" * 70)
    
    # Create float ambiguities
    n = 6  # Number of ambiguities
    
    # True integer values (unknown in real scenario)
    z_true = np.array([5, -3, 2, -7, 4, 1])
    print(f"\nTrue integer ambiguities: {z_true}")
    
    # Float ambiguities with noise
    np.random.seed(42)
    noise_level = 0.05
    a_float = z_true + np.random.normal(0, noise_level, n)
    print(f"Float ambiguities:        {a_float.round(3)}")
    
    # Covariance matrix (diagonal for simplicity)
    variances = np.array([0.01, 0.015, 0.02, 0.01, 0.025, 0.01])
    Q = np.diag(variances)
    
    # Resolve ambiguities
    print("\nResolving ambiguities with MLAMBDA...")
    start_time = time.time()
    F, s = mlambda(a_float, Q, m=2)
    elapsed = (time.time() - start_time) * 1000
    
    # Results
    print(f"\nResults:")
    print(f"  Best solution:     {F[:, 0].astype(int)}")
    print(f"  Second best:       {F[:, 1].astype(int)}")
    print(f"  Residuals:         {s.round(4)}")
    print(f"  Execution time:    {elapsed:.2f} ms")
    
    # Validation
    passed, ratio = ratio_test(s)
    print(f"\nValidation:")
    print(f"  Ratio test:        {ratio:.2f}")
    print(f"  Test result:       {'PASSED' if passed else 'FAILED'}")
    print(f"  Threshold:         3.0")
    
    # Check if correct
    errors = np.abs(F[:, 0] - z_true)
    print(f"\nAccuracy:")
    print(f"  Errors:            {errors.astype(int)}")
    print(f"  All correct:       {np.all(errors == 0)}")


def example_correlated():
    """MLAMBDA with correlated covariance matrix"""
    print("\n" + "=" * 70)
    print("Example 2: MLAMBDA with Correlated Covariance")
    print("=" * 70)
    
    n = 5
    z_true = np.array([2, -4, 3, 1, -2])
    a_float = z_true + np.random.normal(0, 0.03, n)
    
    # Create correlated covariance matrix
    print("\nCreating correlated covariance matrix...")
    L = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.3, 1.0, 0.0, 0.0, 0.0],
        [0.1, 0.2, 1.0, 0.0, 0.0],
        [0.2, 0.1, 0.3, 1.0, 0.0],
        [0.1, 0.1, 0.2, 0.1, 1.0]
    ])
    Q = L @ L.T * 0.01
    
    print(f"Condition number: {np.linalg.cond(Q):.2f}")
    
    # Resolve
    F, s = mlambda(a_float, Q, m=3)
    
    print(f"\nResults:")
    print(f"  True:              {z_true}")
    print(f"  Best solution:     {F[:, 0].astype(int)}")
    print(f"  2nd best:          {F[:, 1].astype(int)}")
    print(f"  3rd best:          {F[:, 2].astype(int)}")
    
    # Ratios
    ratio_12 = np.sqrt(s[1] / s[0]) if s[0] > 0 else 0
    ratio_13 = np.sqrt(s[2] / s[0]) if s[0] > 0 else 0
    print(f"\nRatio tests:")
    print(f"  Ratio (2nd/1st):   {ratio_12:.2f}")
    print(f"  Ratio (3rd/1st):   {ratio_13:.2f}")


def example_partial_resolution():
    """Partial ambiguity resolution example"""
    print("\n" + "=" * 70)
    print("Example 3: Partial Ambiguity Resolution")
    print("=" * 70)
    
    # Create difficult scenario with mixed quality
    n = 10
    z_true = np.random.randint(-20, 20, n)
    
    # Different noise levels for different satellites
    noise_levels = np.array([0.01, 0.01, 0.02, 0.15, 0.01, 
                             0.20, 0.02, 0.01, 0.25, 0.03])
    a_float = z_true.astype(float)
    for i in range(n):
        a_float[i] += np.random.normal(0, noise_levels[i])
    
    Q = np.diag(noise_levels**2)
    
    print(f"\nScenario:")
    print(f"  Total satellites:  {n}")
    print(f"  Good satellites:   {np.sum(noise_levels < 0.05)}")
    print(f"  Poor satellites:   {np.sum(noise_levels >= 0.15)}")
    
    # Try full resolution
    print("\nFull resolution attempt:")
    F_full, s_full = mlambda(a_float, Q)
    passed_full, ratio_full = ratio_test(s_full)
    print(f"  Ratio:             {ratio_full:.2f}")
    print(f"  Status:            {'PASSED' if passed_full else 'FAILED'}")
    
    # Try partial resolution
    print("\nPartial resolution:")
    fixed, ratio_partial, flags = mlambda_with_partial(a_float, Q, min_satellites=4)
    
    print(f"  Fixed satellites:  {np.sum(flags)}/{n}")
    print(f"  Fixed indices:     {np.where(flags)[0]}")
    print(f"  Ratio:             {ratio_partial:.2f}")
    
    if np.sum(flags) > 0:
        errors = np.abs(fixed[flags] - z_true[flags])
        success_rate = np.mean(errors < 0.5)
        print(f"  Success rate:      {success_rate:.1%}")


def example_real_dd_data():
    """MLAMBDA with real Double Difference data"""
    print("\n" + "=" * 70)
    print("Example 4: MLAMBDA with Real DD Data")
    print("=" * 70)
    
    # Load real data
    print("\nLoading real RINEX data...")
    base_dir = "/home/ubuntu/graph_ins"
    rover_file = f"{base_dir}/okujo_test/main.obs"
    base_file = f"{base_dir}/okujo_test/base.obs"
    
    try:
        rover_reader = RinexObsReader(rover_file)
        rover_epochs = rover_reader.read()
        
        base_reader = RinexObsReader(base_file)
        base_epochs = base_reader.read()
        
        print(f"  Loaded {len(rover_epochs)} rover epochs")
        print(f"  Loaded {len(base_epochs)} base epochs")
        
        # Process one epoch
        epoch_idx = 150
        rover_obs = rover_epochs[epoch_idx]['observations']
        base_obs = base_epochs[epoch_idx]['observations']
        
        # Extract DD observations (simplified)
        dd_ambiguities = []
        snr_values = []
        
        for r_obs in rover_obs:
            if r_obs.system == 8:  # GPS
                for b_obs in base_obs:
                    if b_obs.system == 8 and b_obs.sat == r_obs.sat:
                        if r_obs.L[0] > 0 and b_obs.L[0] > 0:
                            # Simple DD (not proper, just for demo)
                            dd = (r_obs.L[0] - b_obs.L[0]) - (r_obs.P[0] - b_obs.P[0]) * 5.255
                            dd_ambiguities.append(dd)
                            snr_values.append(r_obs.SNR[0])
                            break
        
        if len(dd_ambiguities) > 5:
            dd_ambiguities = np.array(dd_ambiguities[:8])  # Limit to 8
            snr_values = np.array(snr_values[:8])
            
            # Create covariance based on SNR
            weights = snr_values / np.max(snr_values)
            Q = np.diag(1.0 / (weights**2 + 0.1)) * 0.001
            
            print(f"\nDD observations: {len(dd_ambiguities)} satellites")
            print(f"Float ambiguities: {dd_ambiguities[:5].round(2)}...")
            
            # Resolve
            F, s = mlambda(dd_ambiguities, Q, m=3)
            
            print(f"\nMLAMBDA Results:")
            print(f"  Best solution:     {F[:5, 0].astype(int)}...")
            print(f"  2nd best:          {F[:5, 1].astype(int)}...")
            
            passed, ratio = ratio_test(s)
            print(f"  Ratio test:        {ratio:.2f}")
            print(f"  Status:            {'PASSED' if passed else 'FAILED'}")
            
    except Exception as e:
        print(f"Note: Real data example requires RINEX files")
        print(f"Error: {e}")


def example_success_rate():
    """Success rate estimation example"""
    print("\n" + "=" * 70)
    print("Example 5: Success Rate Estimation")
    print("=" * 70)
    
    print("\nComparing different covariance scenarios:")
    
    scenarios = [
        ("High precision", np.diag([0.001, 0.001, 0.001, 0.001])),
        ("Medium precision", np.diag([0.01, 0.01, 0.01, 0.01])),
        ("Low precision", np.diag([0.1, 0.1, 0.1, 0.1])),
        ("Mixed precision", np.diag([0.001, 0.01, 0.1, 0.001])),
    ]
    
    for name, Q in scenarios:
        p_boot = success_rate(Q, method='bootstrapping')
        p_ils = success_rate(Q, method='ils')
        
        print(f"\n{name}:")
        print(f"  Bootstrapping success rate: {p_boot:.1%}")
        print(f"  ILS success rate:          {p_ils:.1%}")
        print(f"  Condition number:          {np.linalg.cond(Q):.2f}")


def example_performance_test():
    """Performance testing"""
    print("\n" + "=" * 70)
    print("Example 6: Performance Testing")
    print("=" * 70)
    
    print("\nTesting MLAMBDA performance with different problem sizes:")
    
    sizes = [3, 5, 10, 15, 20]
    times = []
    
    for n in sizes:
        # Create problem
        z_true = np.random.randint(-100, 100, n)
        a_float = z_true + np.random.normal(0, 0.05, n)
        Q = np.diag(np.random.uniform(0.01, 0.05, n))
        
        # Time resolution
        start = time.time()
        F, s = mlambda(a_float, Q, m=2)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
        
        # Check accuracy
        errors = np.abs(F[:, 0] - z_true)
        accuracy = np.mean(errors == 0) * 100
        
        passed, ratio = ratio_test(s)
        
        print(f"  n={n:2d}: Time={elapsed:6.2f}ms, Accuracy={accuracy:5.1f}%, Ratio={ratio:6.2f}")
    
    # Plot if matplotlib available
    try:
        plt.figure(figsize=(8, 5))
        plt.plot(sizes, times, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Number of Ambiguities')
        plt.ylabel('Execution Time (ms)')
        plt.title('MLAMBDA Performance Scaling')
        plt.grid(True, alpha=0.3)
        plt.savefig('/tmp/mlambda_performance.png', dpi=100, bbox_inches='tight')
        print(f"\n  Performance plot saved to /tmp/mlambda_performance.png")
    except:
        pass


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("MLAMBDA - Modified LAMBDA Algorithm Examples")
    print("=" * 70)
    print("\nDemonstrating integer ambiguity resolution for RTK GNSS")
    
    # Run examples
    example_basic()
    example_correlated()
    example_partial_resolution()
    example_real_dd_data()
    example_success_rate()
    example_performance_test()
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
MLAMBDA is the standard method for GNSS integer ambiguity resolution.

Key features demonstrated:
1. Basic ambiguity resolution with validation
2. Handling correlated covariance matrices
3. Partial ambiguity resolution for difficult cases
4. Processing real DD observations
5. Success rate estimation
6. Performance scaling with problem size

Usage:
    from pyins.rtk.mlambda import mlambda, ratio_test
    
    F, s = mlambda(float_ambiguities, covariance, m=2)
    passed, ratio = ratio_test(s)
    
    if passed:
        fixed_ambiguities = F[:, 0]
""")


if __name__ == "__main__":
    main()