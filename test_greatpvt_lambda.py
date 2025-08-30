#!/usr/bin/env python3
"""Test GREAT-PVT LAMBDA implementation"""

import numpy as np
import sys
sys.path.insert(0, '.')

from pyins.rtk.lambda_greatpvt import GreatPVTLambdaResolver
from pyins.rtk.lambda_rtklib import RTKLIBAmbiguityResolver
from pyins.rtk.ambiguity_resolution import RTKAmbiguityManager
import time

def generate_test_with_elevations(n=5, noise_level=0.1):
    """Generate test case with elevations"""
    # True integer ambiguities
    a_true = np.random.randint(-10, 10, n)
    
    # Add noise to create float ambiguities
    a_float = a_true + np.random.randn(n) * noise_level
    
    # Generate positive definite covariance matrix
    L = np.tril(np.random.randn(n, n))
    np.fill_diagonal(L, np.abs(np.diagonal(L)) + 0.1)
    Q = L @ L.T
    
    # Generate elevations (some low, some high)
    elevations = np.random.uniform(5, 85, n)
    
    return a_true, a_float, Q, elevations

def test_implementations():
    """Compare GREAT-PVT with other implementations"""
    
    print("=" * 70)
    print("GREAT-PVT LAMBDA Implementation Test")
    print("=" * 70)
    
    # Test configurations
    test_configs = [
        (5, 0.1, "Small, low noise"),
        (8, 0.15, "Medium, moderate noise"),
        (12, 0.2, "Large, higher noise"),
    ]
    
    for n, noise, description in test_configs:
        print(f"\n{description} (n={n}, noise={noise})")
        print("-" * 50)
        
        # Generate test case
        a_true, a_float, Q, elevations = generate_test_with_elevations(n, noise)
        
        # Test GREAT-PVT
        resolver_gp = GreatPVTLambdaResolver(ratio_threshold=3.0)
        t1 = time.time()
        a_fixed_gp, ratio_gp, is_fixed_gp, info_gp = resolver_gp.resolve(a_float, Q, elevations)
        time_gp = time.time() - t1
        
        # Test RTKLIB
        resolver_rt = RTKLIBAmbiguityResolver(ratio_threshold=3.0)
        t2 = time.time()
        a_fixed_rt, info_rt = resolver_rt.resolve(a_float, Q)
        time_rt = time.time() - t2
        
        # Test current pyins
        resolver_pi = RTKAmbiguityManager(ratio_threshold=3.0)
        t3 = time.time()
        a_fixed_pi, info_pi = resolver_pi.resolve(a_float, Q)
        time_pi = time.time() - t3
        
        # Print results
        print(f"True ambiguities: {a_true[:min(5,n)]}")
        print(f"Float ambiguities: {a_float[:min(5,n)].round(2)}")
        print(f"Elevations: {elevations[:min(5,n)].round(0)}°")
        
        print(f"\nGREAT-PVT:")
        print(f"  Fixed: {is_fixed_gp}, Ratio: {ratio_gp:.3f}")
        if is_fixed_gp:
            errors_gp = np.sum(a_fixed_gp.astype(int) != a_true)
            print(f"  Errors: {errors_gp}/{n}")
        print(f"  Time: {time_gp*1000:.2f} ms")
        print(f"  Info: {info_gp}")
        
        print(f"\nRTKLIB:")
        print(f"  Fixed: {info_rt['fixed']}, Ratio: {info_rt['ratio']:.3f}")
        if info_rt['fixed']:
            errors_rt = np.sum(a_fixed_rt != a_true)
            print(f"  Errors: {errors_rt}/{n}")
        print(f"  Time: {time_rt*1000:.2f} ms")
        
        print(f"\npyins (current):")
        print(f"  Fixed: {info_pi['fixed']}, Ratio: {info_pi['ratio']:.3f}")
        if info_pi['fixed']:
            errors_pi = np.sum(a_fixed_pi != a_true)
            print(f"  Errors: {errors_pi}/{n}")
        print(f"  Time: {time_pi*1000:.2f} ms")

def test_partial_resolution():
    """Test partial ambiguity resolution"""
    print("\n" + "=" * 70)
    print("Partial Ambiguity Resolution Test")
    print("=" * 70)
    
    n = 10
    a_true = np.random.randint(-5, 5, n)
    
    # Create mixed precision
    a_float = a_true.astype(float)
    a_float[:5] += np.random.randn(5) * 0.05  # Good precision (high elevation)
    a_float[5:] += np.random.randn(5) * 0.3   # Poor precision (low elevation)
    
    # Covariance with varying precision
    Q = np.eye(n)
    Q[:5, :5] *= 0.001  # Good
    Q[5:, 5:] *= 0.1    # Poor
    
    # Elevations
    elevations = np.array([60, 70, 50, 80, 65, 10, 12, 8, 15, 20])
    
    print(f"Elevations: {elevations}°")
    print(f"High elevation (>15°): {np.where(elevations > 15)[0]}")
    
    # Test GREAT-PVT partial resolution
    resolver = GreatPVTLambdaResolver(elevation_threshold=15.0, min_satellites=4)
    fixed_amb, fixed_mask, ratio, success = resolver.resolve_partial(
        a_float, Q, elevations, satellite_ids=list(range(n))
    )
    
    print(f"\nPartial Resolution:")
    print(f"  Success: {success}, Ratio: {ratio:.3f}")
    print(f"  Fixed satellites: {np.where(fixed_mask)[0]}")
    if success:
        print(f"  Fixed count: {np.sum(fixed_mask)}/{n}")
        errors = np.sum(fixed_amb[fixed_mask].astype(int) != a_true[fixed_mask])
        print(f"  Errors in fixed: {errors}/{np.sum(fixed_mask)}")

def test_validation():
    """Test validation functionality"""
    print("\n" + "=" * 70)
    print("Validation Test")
    print("=" * 70)
    
    n = 5
    resolver = GreatPVTLambdaResolver(max_deviation=0.25)
    
    # Test 1: Good solution (should pass)
    a_true = np.array([1, 2, -1, 0, 3])
    a_float = a_true + np.random.randn(n) * 0.05
    Q = np.eye(n) * 0.01
    
    a_fixed, ratio, is_fixed, info = resolver.resolve(a_float, Q)
    print(f"Test 1 (good): Fixed={is_fixed}, Validated={info.get('validated', False)}")
    
    # Test 2: Bad solution (should fail validation)
    a_float_bad = a_true + np.array([0.4, 0.1, -0.3, 0.05, 0.02])
    a_fixed, ratio, is_fixed, info = resolver.resolve(a_float_bad, Q)
    print(f"Test 2 (bad): Fixed={is_fixed}, Validated={info.get('validated', False)}")

if __name__ == "__main__":
    test_implementations()
    test_partial_resolution()
    test_validation()
    
    print("\n" + "=" * 70)
    print("GREAT-PVT implementation provides:")
    print("- Simpler, faster decorrelation")
    print("- Elevation-based partial resolution")
    print("- Built-in validation")
    print("- Practical robustness for RTK")
    print("=" * 70)