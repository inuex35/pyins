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
MLAMBDA - Modified LAMBDA Integer Ambiguity Resolution
======================================================

This module provides the core MLAMBDA implementation for integer ambiguity resolution.
MLAMBDA (Modified LAMBDA) is the standard method for solving integer least-squares
problems in GNSS RTK positioning.

The actual implementation is in lambda_rtklib.py. This module provides a clean
interface and additional utilities.

References:
    [1] P.J.G. Teunissen, The least-squares ambiguity decorrelation adjustment:
        a method for fast GPS ambiguity estimation, Journal of Geodesy, 1995
    [2] X.-W. Chang, X. Yang, T. Zhou, MLAMBDA: A modified LAMBDA method for
        integer least-squares estimation, Journal of Geodesy, 2005

Author: Based on RTKLIB by T. Takasu
Python implementation for pyins
"""

import numpy as np
from typing import Tuple, Optional
import logging

# Import the actual implementation from lambda_rtklib
from pyins.rtk.lambda_rtklib import mlambda as _mlambda_core
from pyins.rtk.lambda_rtklib import LD, reduction, search

logger = logging.getLogger(__name__)

# Re-export core functions
__all__ = ['mlambda', 'ratio_test', 'success_rate', 'mlambda_with_partial']


def mlambda(a: np.ndarray, Q: np.ndarray, m: int = 2,
            threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    MLAMBDA (Modified LAMBDA) integer ambiguity resolution
    
    Solves the integer least-squares problem:
        min ||a - z||²_Q
    where z is an integer vector and ||x||²_Q = x'*Q^(-1)*x
    
    Parameters
    ----------
    a : np.ndarray
        Float ambiguity vector (n,)
    Q : np.ndarray
        Covariance matrix of float ambiguities (n, n)
        Must be positive definite
    m : int, optional
        Number of candidate solutions to return (default: 2)
    threshold : float, optional
        Search threshold (not used in RTKLIB implementation)
    
    Returns
    -------
    F : np.ndarray
        Integer ambiguity candidates (n, m)
        F[:, 0] is the best solution
        F[:, 1] is the second-best solution, etc.
    s : np.ndarray
        Sum of squared residuals for each candidate (m,)
        s[0] is the residual for the best solution
        Ratio test = sqrt(s[1]/s[0])
    
    Examples
    --------
    >>> import numpy as np
    >>> from pyins.rtk.mlambda import mlambda
    >>> 
    >>> # Float ambiguities
    >>> a = np.array([1.2, -2.8, 3.1])
    >>> 
    >>> # Covariance matrix
    >>> Q = np.array([[0.01, 0.002, 0.001],
    ...               [0.002, 0.02, 0.003],
    ...               [0.001, 0.003, 0.015]])
    >>> 
    >>> # Resolve ambiguities
    >>> F, s = mlambda(a, Q, m=2)
    >>> 
    >>> # Best solution
    >>> print(f"Best: {F[:, 0]}")
    >>> print(f"Ratio: {np.sqrt(s[1]/s[0]):.2f}")
    
    Notes
    -----
    The algorithm consists of three main steps:
    1. LD factorization: Q = L' * diag(D) * L
    2. Reduction (decorrelation): Minimize off-diagonal elements
    3. Search: Find m best integer solutions
    
    This is a wrapper around the RTKLIB implementation which has been
    extensively tested and optimized.
    """
    return _mlambda_core(a, Q, m)


def ratio_test(s: np.ndarray, threshold: float = 3.0) -> Tuple[bool, float]:
    """
    Ratio test for ambiguity validation
    
    The ratio test checks the ratio between the second-best and best solutions.
    A high ratio indicates reliable ambiguity resolution.
    
    Parameters
    ----------
    s : np.ndarray
        Sum of squared residuals from mlambda
    threshold : float
        Minimum ratio for acceptance (default: 3.0)
        Common values:
        - 3.0: Conservative (recommended for safety-critical applications)
        - 2.5: Moderate (good balance)
        - 2.0: Liberal (may accept incorrect fixes)
    
    Returns
    -------
    passed : bool
        Whether the ratio test passed
    ratio : float
        The actual ratio value
    
    Examples
    --------
    >>> s = np.array([1.0, 9.0])  # Residuals from mlambda
    >>> passed, ratio = ratio_test(s)
    >>> print(f"Ratio: {ratio:.2f}, Passed: {passed}")
    Ratio: 3.00, Passed: True
    
    Notes
    -----
    The ratio test is crucial for validating ambiguity resolution:
    - Ratio > 3.0: Very reliable fix
    - 2.5 < Ratio < 3.0: Good fix
    - 2.0 < Ratio < 2.5: Marginal fix
    - Ratio < 2.0: Unreliable, should reject
    """
    if len(s) < 2 or s[0] <= 0:
        return False, 0.0
    
    # RTKLIB uses s[1]/s[0], but standard ratio is sqrt(s[1]/s[0])
    ratio = np.sqrt(s[1] / s[0])
    passed = ratio >= threshold
    
    return passed, ratio


def success_rate(Q: np.ndarray, method: str = 'bootstrapping') -> float:
    """
    Estimate success rate of ambiguity resolution
    
    Parameters
    ----------
    Q : np.ndarray
        Covariance matrix of float ambiguities
    method : str
        Method for estimation:
        - 'bootstrapping': Use bootstrapping success rate formula
        - 'ils': Use ILS (Integer Least Squares) approximation
        - 'adop': Use ADOP (Ambiguity Dilution of Precision)
    
    Returns
    -------
    p_s : float
        Estimated success rate (0 to 1)
    
    Examples
    --------
    >>> Q = np.diag([0.01, 0.01, 0.01])
    >>> p_s = success_rate(Q)
    >>> print(f"Success rate: {p_s:.1%}")
    Success rate: 99.9%
    """
    from scipy.stats import norm, chi2
    
    n = Q.shape[0]
    
    if method == 'bootstrapping':
        # Bootstrapping success rate (Teunissen, 1998)
        L, d = LD(Q)
        if np.any(d <= 0):
            return 0.0
        
        p_s = 1.0
        for i in range(n):
            sigma = np.sqrt(d[i])
            # Probability of correct rounding for each conditional ambiguity
            p_i = 2 * norm.cdf(0.5 / sigma) - 1
            p_s *= max(0, p_i)
        
        return max(0, min(1, p_s))
    
    elif method == 'adop':
        # ADOP-based approximation (Teunissen & Odijk, 1997)
        det_Q = np.linalg.det(Q)
        if det_Q <= 0:
            return 0.0
        
        # ADOP = (det(Q))^(1/2n) in cycles
        adop = np.power(det_Q, 1.0 / (2.0 * n))
        
        # Empirical formula: P_s ≈ (1 + exp(a * ADOP - b))^(-1)
        # with a ≈ 157, b ≈ 5 for typical cases
        p_s = 1.0 / (1.0 + np.exp(157.0 * adop - 5.0))
        
        return max(0, min(1, p_s))
    
    else:  # 'ils' or default
        # Simple ILS approximation based on volume
        det_Q = np.linalg.det(Q)
        if det_Q <= 0:
            return 0.0
        
        # Volume of confidence ellipsoid
        from scipy.special import gamma
        volume = np.power(np.pi, n/2) / gamma(n/2 + 1) * np.sqrt(det_Q)
        
        # Approximate success rate (simplified)
        # P_s ≈ P(χ²_n < critical_value)
        critical_value = chi2.ppf(0.99, df=n)
        p_s = min(1.0, critical_value / (volume * n))
        
        return max(0, min(1, p_s))


def mlambda_with_partial(a: np.ndarray, Q: np.ndarray, 
                         min_satellites: int = 4,
                         partial_threshold: float = 2.5) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    MLAMBDA with partial ambiguity resolution fallback
    
    If full ambiguity resolution fails the ratio test, tries to fix
    a subset of the most reliable ambiguities.
    
    Parameters
    ----------
    a : np.ndarray
        Float ambiguity vector
    Q : np.ndarray
        Covariance matrix
    min_satellites : int
        Minimum number of ambiguities to fix (default: 4)
    partial_threshold : float
        Ratio threshold for partial fixing (default: 2.5, lower than full)
    
    Returns
    -------
    fixed : np.ndarray
        Fixed ambiguities (NaN for unfixed)
    ratio : float
        Ratio test value
    fixed_flags : np.ndarray
        Boolean array indicating which ambiguities were fixed
    
    Examples
    --------
    >>> # Difficult case with large variance
    >>> a = np.array([1.2, -2.8, 3.1, 5.9, -1.1])
    >>> Q = np.diag([0.01, 0.02, 0.1, 0.15, 0.01])  # Mixed quality
    >>> 
    >>> fixed, ratio, flags = mlambda_with_partial(a, Q)
    >>> print(f"Fixed {np.sum(flags)}/{len(a)} ambiguities")
    >>> print(f"Ratio: {ratio:.2f}")
    
    Notes
    -----
    Partial ambiguity resolution is useful when:
    - Some satellites have poor geometry or low SNR
    - Full resolution fails ratio test
    - Better to fix subset reliably than none at all
    """
    n = len(a)
    
    # Try full resolution first
    F, s = mlambda(a, Q)
    passed, ratio = ratio_test(s, threshold=3.0)
    
    if passed:
        # Full resolution successful
        return F[:, 0], ratio, np.ones(n, dtype=bool)
    
    # Full resolution failed, try partial
    logger.info(f"Full AR failed (ratio={ratio:.2f}), trying partial resolution")
    
    # Initialize output
    fixed = np.full(n, np.nan)
    fixed_flags = np.zeros(n, dtype=bool)
    best_ratio = 0.0
    
    # Sort by variance (most reliable first)
    variances = np.diag(Q)
    order = np.argsort(variances)
    
    # Try subsets of decreasing size
    for subset_size in range(n, min_satellites - 1, -1):
        subset = order[:subset_size]
        
        # Extract subset
        a_sub = a[subset]
        Q_sub = Q[np.ix_(subset, subset)]
        
        # Try to fix subset
        F_sub, s_sub = mlambda(a_sub, Q_sub)
        passed_sub, ratio_sub = ratio_test(s_sub, threshold=partial_threshold)
        
        if passed_sub:
            # Partial resolution successful
            fixed[subset] = F_sub[:, 0]
            fixed_flags[subset] = True
            best_ratio = ratio_sub
            
            logger.info(f"Partial AR successful: {subset_size}/{n} ambiguities, ratio={ratio_sub:.2f}")
            break
    
    if not np.any(fixed_flags):
        # No subset could be fixed, return float solution
        logger.warning("Partial AR also failed, returning float solution")
        return a, 0.0, fixed_flags
    
    return fixed, best_ratio, fixed_flags


if __name__ == "__main__":
    """Test MLAMBDA implementation"""
    
    print("MLAMBDA Test")
    print("=" * 50)
    
    # Test case 1: Simple example
    np.random.seed(42)
    n = 5
    
    # True integer ambiguities
    z_true = np.array([3, -2, 5, 1, -4])
    
    # Add noise to create float ambiguities
    noise = np.random.normal(0, 0.01, n)
    a = z_true + noise
    
    # Create covariance matrix
    Q = np.diag([0.001, 0.002, 0.0015, 0.001, 0.0025])
    
    print("Test 1: Basic MLAMBDA")
    print(f"True ambiguities:  {z_true}")
    print(f"Float ambiguities: {a.round(3)}")
    
    # Resolve
    F, s = mlambda(a, Q, m=2)
    
    print(f"Best solution:     {F[:, 0].astype(int)}")
    print(f"Second best:       {F[:, 1].astype(int)}")
    print(f"Residuals:         {s.round(3)}")
    
    passed, ratio = ratio_test(s)
    print(f"Ratio test:        {ratio:.2f} ({'PASS' if passed else 'FAIL'})")
    print(f"Correct:           {np.all(F[:, 0] == z_true)}")
    
    # Test case 2: Success rate
    print("\nTest 2: Success Rate Estimation")
    p_boot = success_rate(Q, method='bootstrapping')
    p_adop = success_rate(Q, method='adop')
    p_ils = success_rate(Q, method='ils')
    print(f"Bootstrapping success rate: {p_boot:.1%}")
    print(f"ADOP success rate:         {p_adop:.1%}")
    print(f"ILS success rate:          {p_ils:.1%}")
    
    # Test case 3: Partial resolution
    print("\nTest 3: Partial Ambiguity Resolution")
    
    # Make problem harder with mixed quality
    Q_hard = np.diag([0.01, 0.02, 0.1, 0.01, 0.15])  # Some bad satellites
    a_hard = z_true + np.random.normal(0, np.sqrt(np.diag(Q_hard)), n)
    
    fixed, ratio, flags = mlambda_with_partial(a_hard, Q_hard)
    print(f"Float ambiguities: {a_hard.round(2)}")
    print(f"Fixed ambiguities: {fixed[flags].astype(int) if np.any(flags) else 'None'}")
    print(f"Number fixed:      {np.sum(flags)}/{n}")
    print(f"Ratio:             {ratio:.2f}")
    
    if np.any(flags):
        errors = np.abs(fixed[flags] - z_true[flags])
        print(f"Errors in fixed:   {errors.astype(int)}")
        print(f"All correct:       {np.all(errors < 0.5)}")
    
    print("\n" + "=" * 50)
    print("mlambda module ready for use!")
    print("\nUsage:")
    print("  from pyins.rtk.mlambda import mlambda, ratio_test")
    print("  F, s = mlambda(float_ambiguities, covariance)")
    print("  passed, ratio = ratio_test(s)")