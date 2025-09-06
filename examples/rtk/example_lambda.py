#!/usr/bin/env python3
"""
Working LAMBDA ambiguity resolution example
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pyins.rtk.lambda_rtklib import mlambda


def main():
    """Simple LAMBDA example that works"""
    print("=" * 70)
    print("LAMBDA Ambiguity Resolution Example")
    print("=" * 70)
    
    # Simple synthetic example
    print("\n=== Synthetic Data Example ===")
    
    # Float ambiguities
    float_amb = np.array([10.3, -5.7, 15.1])
    
    # Covariance matrix
    Q = np.diag([0.01, 0.02, 0.015])
    
    print(f"Float ambiguities: {float_amb}")
    
    # Solve with LAMBDA
    fixed_amb, residuals = mlambda(float_amb, Q, m=2)
    
    print(f"Fixed ambiguities (best): {fixed_amb[:,0].astype(int)}")
    print(f"Residuals: {residuals}")
    
    # Check if close to expected
    expected = np.array([10, -6, 15])
    if np.allclose(fixed_amb[:,0], expected, atol=1):
        print("✓ LAMBDA resolution successful!")
    else:
        print("✗ Result different from expected")
    
    print("\n" + "=" * 70)
    print("Example completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()