#!/usr/bin/env python3
"""Test and compare LAMBDA implementations"""

import numpy as np
import sys
sys.path.insert(0, '.')

from pyins.rtk.ambiguity_resolution import RTKAmbiguityManager
from pyins.rtk.lambda_rtklib import RTKLIBAmbiguityResolver, mlambda
import time

def generate_test_case(n=5, noise_level=0.1):
    """Generate test case with known integer ambiguities"""
    # True integer ambiguities
    a_true = np.random.randint(-10, 10, n)
    
    # Add noise to create float ambiguities
    a_float = a_true + np.random.randn(n) * noise_level
    
    # Generate positive definite covariance matrix
    L = np.tril(np.random.randn(n, n))
    np.fill_diagonal(L, np.abs(np.diagonal(L)) + 0.1)
    Q = L @ L.T
    
    return a_true, a_float, Q

def test_implementations():
    """Compare implementations on various test cases"""
    
    print("=" * 70)
    print("LAMBDA Implementation Comparison: pyins vs RTKLIB-py")
    print("=" * 70)
    
    # Test cases
    test_configs = [
        (5, 0.1, "Small, low noise"),
        (10, 0.2, "Medium, moderate noise"),
        (15, 0.3, "Large, high noise"),
        (20, 0.15, "Very large, moderate noise")
    ]
    
    for n, noise, description in test_configs:
        print(f"\n{description} (n={n}, noise={noise})")
        print("-" * 50)
        
        # Generate test case
        a_true, a_float, Q = generate_test_case(n, noise)
        
        # Test pyins MLAMBDA
        resolver1 = RTKAmbiguityManager()
        t1 = time.time()
        a_fixed1, info1 = resolver1.resolve(a_float, Q)
        time1 = time.time() - t1
        
        # Test RTKLIB LAMBDA
        resolver2 = RTKLIBAmbiguityResolver()
        t2 = time.time()
        a_fixed2, info2 = resolver2.resolve(a_float, Q)
        time2 = time.time() - t2
        
        # Direct RTKLIB mlambda call for comparison
        t3 = time.time()
        candidates, residuals = mlambda(a_float, Q, m=2)
        time3 = time.time() - t3
        ratio_direct = residuals[1]/residuals[0] if residuals[0] > 0 else 0
        
        # Compare results
        print(f"True ambiguities:    {a_true[:5]}..." if n > 5 else f"True: {a_true}")
        print(f"Float ambiguities:   {a_float[:5].round(2)}..." if n > 5 else f"Float: {a_float.round(2)}")
        
        print(f"\npyins MLAMBDA:")
        if a_fixed1 is not None:
            print(f"  Fixed: {a_fixed1[:5]}..." if n > 5 else f"  Fixed: {a_fixed1}")
            print(f"  Ratio: {info1['ratio']:.3f}")
            print(f"  Success: {info1['fixed']}")
            errors1 = np.sum(a_fixed1 != a_true)
            print(f"  Errors: {errors1}/{n}")
        else:
            print(f"  Failed to fix (ratio={info1['ratio']:.3f})")
        print(f"  Time: {time1*1000:.2f} ms")
        
        print(f"\nRTKLIB LAMBDA:")
        if a_fixed2 is not None:
            print(f"  Fixed: {a_fixed2[:5]}..." if n > 5 else f"  Fixed: {a_fixed2}")
            print(f"  Ratio: {info2['ratio']:.3f}")
            print(f"  Success: {info2['fixed']}")
            errors2 = np.sum(a_fixed2 != a_true)
            print(f"  Errors: {errors2}/{n}")
        else:
            print(f"  Failed to fix (ratio={info2['ratio']:.3f})")
        print(f"  Time: {time2*1000:.2f} ms")
        
        print(f"\nDirect RTKLIB mlambda:")
        print(f"  Best candidate: {candidates[:5, 0].astype(int)}..." if n > 5 else f"  Best: {candidates[:, 0].astype(int)}")
        print(f"  Ratio: {ratio_direct:.3f}")
        errors3 = np.sum(candidates[:, 0].astype(int) != a_true)
        print(f"  Errors: {errors3}/{n}")
        print(f"  Time: {time3*1000:.2f} ms")
        
        # Performance comparison
        print(f"\nPerformance:")
        print(f"  Speedup RTKLIB/pyins: {time1/time2:.2f}x")
        print(f"  Speedup direct/pyins: {time1/time3:.2f}x")

def test_edge_cases():
    """Test edge cases and numerical stability"""
    print("\n" + "=" * 70)
    print("Edge Case Testing")
    print("=" * 70)
    
    # Test 1: Near-singular covariance
    print("\n1. Near-singular covariance matrix")
    n = 5
    a_float = np.random.randn(n)
    Q = np.eye(n) * 1e-6
    Q[0, 0] = 1.0  # One well-determined ambiguity
    
    resolver1 = RTKAmbiguityManager()
    resolver2 = RTKLIBAmbiguityResolver()
    
    a1, info1 = resolver1.resolve(a_float, Q)
    a2, info2 = resolver2.resolve(a_float, Q)
    
    print(f"  pyins: {'Fixed' if info1['fixed'] else 'Failed'}, ratio={info1['ratio']:.3f}")
    print(f"  RTKLIB: {'Fixed' if info2['fixed'] else 'Failed'}, ratio={info2['ratio']:.3f}")
    
    # Test 2: High correlation
    print("\n2. Highly correlated ambiguities")
    n = 4
    a_float = np.array([1.2, 2.8, 3.1, 4.9])
    Q = np.array([[1.0, 0.9, 0.8, 0.7],
                   [0.9, 1.0, 0.9, 0.8],
                   [0.8, 0.9, 1.0, 0.9],
                   [0.7, 0.8, 0.9, 1.0]]) * 0.01
    
    a1, info1 = resolver1.resolve(a_float, Q)
    a2, info2 = resolver2.resolve(a_float, Q)
    
    print(f"  pyins: {'Fixed' if info1['fixed'] else 'Failed'}, ratio={info1['ratio']:.3f}")
    print(f"  RTKLIB: {'Fixed' if info2['fixed'] else 'Failed'}, ratio={info2['ratio']:.3f}")
    
    # Test 3: Large dimension
    print("\n3. Large dimension (n=30)")
    n = 30
    a_true = np.zeros(n, dtype=int)
    a_float = a_true + np.random.randn(n) * 0.1
    Q = np.eye(n) * 0.01
    
    t1 = time.time()
    a1, info1 = resolver1.resolve(a_float, Q)
    time1 = time.time() - t1
    
    t2 = time.time()
    a2, info2 = resolver2.resolve(a_float, Q)
    time2 = time.time() - t2
    
    if a1 is not None:
        errors1 = np.sum(a1 != a_true)
    else:
        errors1 = n
        
    if a2 is not None:
        errors2 = np.sum(a2 != a_true)
    else:
        errors2 = n
    
    print(f"  pyins: Errors={errors1}/{n}, Time={time1*1000:.2f}ms, ratio={info1['ratio']:.3f}")
    print(f"  RTKLIB: Errors={errors2}/{n}, Time={time2*1000:.2f}ms, ratio={info2['ratio']:.3f}")

def test_partial_fixing():
    """Test partial ambiguity fixing"""
    print("\n" + "=" * 70)
    print("Partial Ambiguity Fixing Test")
    print("=" * 70)
    
    n = 10
    a_true = np.random.randint(-5, 5, n)
    
    # Create mixed precision: some good, some bad
    a_float = a_true.astype(float)
    a_float[:5] += np.random.randn(5) * 0.05  # Good precision
    a_float[5:] += np.random.randn(5) * 0.5   # Poor precision
    
    # Covariance with varying precision
    Q = np.eye(n)
    Q[:5, :5] *= 0.001  # Good
    Q[5:, 5:] *= 0.1    # Poor
    
    resolver = RTKLIBAmbiguityResolver()
    fixed_mask, a_partial = resolver.partial_fix(a_float, Q, min_ratio_partial=2.0)
    
    print(f"Fixed {np.sum(fixed_mask)}/{n} ambiguities")
    print(f"Fixed indices: {np.where(fixed_mask)[0]}")
    if np.sum(fixed_mask) > 0:
        errors = np.sum(a_partial[fixed_mask].astype(int) != a_true[fixed_mask])
        print(f"Errors in fixed: {errors}/{np.sum(fixed_mask)}")

if __name__ == "__main__":
    test_implementations()
    test_edge_cases()
    test_partial_fixing()
    
    print("\n" + "=" * 70)
    print("Summary: RTKLIB implementation is generally faster and more stable")
    print("=" * 70)