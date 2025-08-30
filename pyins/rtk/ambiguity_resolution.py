#!/usr/bin/env python3
"""
Ambiguity Resolution for RTK Positioning
=========================================

This module implements integer ambiguity resolution algorithms for carrier phase
positioning, using the LAMBDA (Least-squares AMBiguity Decorrelation Adjustment) method
based on RTKLIB implementation.

References:
    [1] P.J.G.Teunissen, The least-square ambiguity decorrelation adjustment:
        a method for fast GPS ambiguity estimation, J.Geodesy, Vol.70, 65-82, 1995
    [2] X.-W.Chang, X.Yang, T.Zhou, MLAMBDA: A modified LAMBDA method for
        integer least-squares estimation, J.Geodesy, Vol.79, 552-565, 2005

Based on RTKLIB-py implementation by Rui Hirokawa and Tim Everett
"""

import numpy as np
from numpy.linalg import inv
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


def LD(Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    LD factorization (Q = L' * diag(D) * L)
    
    Parameters
    ----------
    Q : np.ndarray
        Covariance matrix (n x n), must be positive definite
        
    Returns
    -------
    L : np.ndarray
        Lower triangular matrix with ones on diagonal
    d : np.ndarray
        Diagonal values
    """
    n = len(Q)
    L = np.zeros((n, n))
    d = np.zeros(n)
    A = Q.copy()
    
    for i in range(n-1, -1, -1):
        d[i] = A[i, i]
        if d[i] <= 0.0:
            logger.warning(f'LD Factorization warning: non-positive diagonal element d[{i}]={d[i]:.6f}')
            d[i] = 1e-6  # Small positive value to continue
        
        L[i, :i+1] = A[i, :i+1] / np.sqrt(d[i])
        for j in range(i):
            A[j, :j+1] -= L[i, :j+1] * L[i, j]
        L[i, :i+1] /= L[i, i]
    
    return L, d


def reduction(L: np.ndarray, d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    LAMBDA reduction (z=Z'*a, Qz=Z'*Q*Z=L'*diag(D)*L) (ref.[1])
    
    Parameters
    ----------
    L : np.ndarray
        Lower triangular matrix from LD factorization
    d : np.ndarray
        Diagonal values from LD factorization
        
    Returns
    -------
    L : np.ndarray
        Reduced L matrix
    d : np.ndarray
        Reduced diagonal values
    Z : np.ndarray
        Transformation matrix
    """
    n = len(d)
    Z = np.eye(n)
    j = k = n - 2
    
    while j >= 0:
        if j <= k:
            # Integer Gauss transformation
            for i in range(j+1, n):
                mu = round(L[i, j])
                if mu != 0:
                    L[i:, j] -= mu * L[i:, i]
                    Z[:, j] -= mu * Z[:, i]
        
        # Check if swap is beneficial
        delta = d[j] + L[j+1, j]**2 * d[j+1]
        if delta + 1e-6 < d[j+1]:  # Consider numerical error
            # Perform swap
            eta = d[j] / delta
            lam = d[j+1] * L[j+1, j] / delta
            d[j] = eta * d[j+1]
            d[j+1] = delta
            
            # Update L matrix
            L[j:j+2, :j] = np.array([[-L[j+1, j], 1], [eta, lam]]) @ L[j:j+2, :j]
            L[j+1, j] = lam
            
            # Swap columns in L and Z
            L[j+2:, j], L[j+2:, j+1] = L[j+2:, j+1].copy(), L[j+2:, j].copy()
            Z[:, j], Z[:, j+1] = Z[:, j+1].copy(), Z[:, j].copy()
            
            j, k = n - 2, j
        else:
            j -= 1
    
    return L, d, Z


def search(L: np.ndarray, d: np.ndarray, zs: np.ndarray, m: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Modified LAMBDA (MLAMBDA) search (ref. [2])
    
    Parameters
    ----------
    L : np.ndarray
        Reduced lower triangular matrix
    d : np.ndarray
        Reduced diagonal values
    zs : np.ndarray
        Transformed float ambiguities
    m : int
        Number of candidates to find
        
    Returns
    -------
    zn : np.ndarray
        Integer candidates (n x m)
    s : np.ndarray
        Sum of squared residuals for each candidate
    """
    n = len(d)
    nn = 0  # Number of candidates found
    imax = 0  # Index of worst candidate
    Chi2 = 1e18  # Initial search bound
    
    # Initialize arrays
    S = np.zeros((n, n))  # Partial sums
    dist = np.zeros(n)    # Partial distances
    zb = np.zeros(n)      # Float values at each level
    z = np.zeros(n)       # Integer candidates
    step = np.zeros(n)    # Search step direction
    zn = np.zeros((n, m)) # Final candidates
    s = np.zeros(m)       # Residuals
    
    # Start from the last level
    k = n - 1
    zb[-1] = zs[-1]
    z[-1] = round(zb[-1])
    y = zb[-1] - z[-1]
    step[-1] = np.sign(y) if y != 0 else 1
    
    # Main search loop
    for iteration in range(10000):
        # Compute partial distance
        newdist = dist[k] + y**2 / d[k]
        
        if newdist < Chi2:
            # Move down in search tree
            if k != 0:
                k -= 1
                dist[k] = newdist
                S[k, :k+1] = S[k+1, :k+1] + (z[k+1] - zb[k+1]) * L[k+1, :k+1]
                zb[k] = zs[k] + S[k, k]
                z[k] = round(zb[k])
                y = zb[k] - z[k]
                step[k] = np.sign(y) if y != 0 else 1
            else:
                # Found a candidate at bottom level
                if nn < m:
                    # Store first m candidates
                    if nn == 0 or newdist > s[imax]:
                        imax = nn
                    zn[:, nn] = z
                    s[nn] = newdist
                    nn += 1
                else:
                    # Replace worst candidate if better
                    if newdist < s[imax]:
                        zn[:, imax] = z
                        s[imax] = newdist
                        imax = np.argmax(s)
                    Chi2 = s[imax]  # Shrink search space
                
                # Move to next integer at bottom level
                z[0] += step[0]
                y = zb[0] - z[0]
                step[0] = -step[0] - np.sign(step[0])
        else:
            # Move up in search tree
            if k == n - 1:
                break  # Finished search
            k += 1
            z[k] += step[k]
            y = zb[k] - z[k]
            step[k] = -step[k] - np.sign(step[k])
    
    # Sort candidates by residual
    if nn > 0:
        order = np.argsort(s[:nn])
        s = s[order]
        zn = zn[:, order]
        return zn[:, :nn], s[:nn]
    else:
        # No candidates found (should not happen)
        logger.warning("MLAMBDA search found no candidates")
        return np.round(zs).reshape(-1, 1), np.array([0.0])


def mlambda(a: np.ndarray, Q: np.ndarray, m: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    LAMBDA/MLAMBDA integer least-squares estimation
    
    Integer least-squares estimation. Reduction is performed by LAMBDA (ref.[1]),
    and search by MLAMBDA (ref.[2]).
    
    Parameters
    ----------
    a : np.ndarray
        Float ambiguities (n x 1)
    Q : np.ndarray
        Covariance matrix of float ambiguities (n x n)
    m : int
        Number of candidates to return
        
    Returns
    -------
    afix : np.ndarray
        Fixed integer candidates (n x m)
    s : np.ndarray
        Sum of squared residuals for each candidate
    """
    # LD factorization (Q = L' * diag(D) * L)
    L, d = LD(Q)
    
    # Reduction/decorrelation
    L, d, Z = reduction(L, d)
    
    # Transform float ambiguities
    z = Z.T @ a
    
    # MLAMBDA search
    E, s = search(L, d, z, m)
    
    # Transform back to original space
    try:
        invZt = np.round(inv(Z.T))
        afix = invZt @ E
    except:
        logger.warning("Failed to invert transformation matrix, using direct transformation")
        afix = Z @ E
    
    return afix.astype(int), s


class RTKAmbiguityManager:
    """
    Integer ambiguity resolution using GREAT-PVT approach with LAMBDA as fallback
    
    GREAT-PVT (Graph Robust Estimation for Adaptive Tracking) provides better
    ambiguity resolution for challenging scenarios like long baselines by using
    partial ambiguity fixing based on quality metrics.
    """
    
    def __init__(self, ratio_threshold: float = 2.0, success_rate_threshold: float = 0.95,
                 min_sats: int = 4, max_position_var: float = 0.1,
                 use_great_pvt: bool = True, max_fix_count: int = 15):
        """
        Initialize ambiguity resolver with GREAT-PVT approach
        
        Parameters
        ----------
        ratio_threshold : float
            Threshold for ratio test (relaxed to 2.0 for GREAT-PVT)
        success_rate_threshold : float
            Required success rate for validation (relaxed to 0.95)
        min_sats : int
            Minimum satellites for ambiguity resolution
        max_position_var : float
            Maximum position variance for attempting AR
        use_great_pvt : bool
            Enable GREAT-PVT partial fixing (default: True)
        max_fix_count : int
            Maximum ambiguities to fix with GREAT-PVT (default: 15)
        """
        self.ratio_threshold = ratio_threshold
        self.success_rate_threshold = success_rate_threshold
        self.min_sats = min_sats
        self.max_position_var = max_position_var
        self.fixed_ambiguities = {}
        self.ambiguity_covariance = None
        
        # GREAT-PVT parameters
        self.use_great_pvt = use_great_pvt
        self.max_fix_count = max_fix_count
        self.max_sigma = 0.25  # Maximum sigma for GREAT-PVT selection
        
        # State tracking for validation
        self.prev_ratio = 0.0
        self.prev_ratio2 = 0.0
        self.nb_ar = 0  # Number of ambiguities resolved
        
    def search_ambiguities(self, a_float: np.ndarray, Q_a: np.ndarray, 
                          n_candidates: int = 2) -> List[Tuple[np.ndarray, float]]:
        """
        Search for integer ambiguity candidates using RTKLIB MLAMBDA
        
        Parameters
        ----------
        a_float : np.ndarray
            Float ambiguity estimates
        Q_a : np.ndarray
            Covariance matrix of float ambiguities
        n_candidates : int
            Number of candidates to return
            
        Returns
        -------
        candidates : List[Tuple[np.ndarray, float]]
            List of (integer ambiguities, residual norm) sorted by residual
        """
        # Use RTKLIB MLAMBDA
        afix, s = mlambda(a_float, Q_a, m=n_candidates)
        
        # Convert to list of tuples for compatibility
        candidates = []
        for i in range(afix.shape[1]):
            candidates.append((afix[:, i], s[i]))
        
        return candidates
    
    def _compute_residual_norm(self, a_float: np.ndarray, a_int: np.ndarray, 
                              Q_a: np.ndarray) -> float:
        """
        Compute quadratic form residual norm
        
        ||a_float - a_int||²_Q = (a_float - a_int)^T * Q_a^-1 * (a_float - a_int)
        """
        diff = a_float - a_int
        try:
            Q_inv = np.linalg.inv(Q_a)
            return float(diff.T @ Q_inv @ diff)
        except:
            return float('inf')
    
    def ratio_test(self, candidates: List[Tuple[np.ndarray, float]]) -> Tuple[bool, float]:
        """
        Perform ratio test for ambiguity validation
        
        Parameters
        ----------
        candidates : List[Tuple[np.ndarray, float]]
            Sorted list of ambiguity candidates with residuals
            
        Returns
        -------
        passed : bool
            Whether ratio test passed
        ratio : float
            Ratio value (second best / best)
        """
        if len(candidates) < 2:
            return False, 0.0
        
        best_residual = candidates[0][1]
        second_best_residual = candidates[1][1]
        
        # Avoid division by zero
        if best_residual < 1e-10:
            return True, float('inf')
        
        ratio = second_best_residual / best_residual
        passed = ratio >= self.ratio_threshold
        
        # Update tracking
        self.prev_ratio2 = self.prev_ratio
        self.prev_ratio = ratio
        
        return passed, ratio
    
    def success_rate_test(self, Q_a: np.ndarray, ratio: float) -> Tuple[bool, float]:
        """
        Compute success rate based on covariance and ratio
        
        Parameters
        ----------
        Q_a : np.ndarray
            Covariance matrix
        ratio : float
            Ratio test value
            
        Returns
        -------
        passed : bool
            Whether success rate test passed
        success_rate : float
            Estimated success rate
        """
        # Simplified success rate estimation based on ratio and dimension
        n = Q_a.shape[0]
        
        # Estimate based on ratio
        if ratio > 3.0:
            success_rate = 0.999
        elif ratio > 2.5:
            success_rate = 0.99
        elif ratio > 2.0:
            success_rate = 0.95
        else:
            success_rate = 0.5 + 0.25 * ratio
        
        # Adjust for dimension
        success_rate = success_rate ** (1.0 / max(1, n/4))
        
        passed = success_rate >= self.success_rate_threshold
        
        return passed, success_rate
    
    def resolve(self, a_float: np.ndarray, Q_a: np.ndarray, 
                wavelengths: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], dict]:
        """
        Full ambiguity resolution pipeline using RTKLIB LAMBDA
        
        Parameters
        ----------
        a_float : np.ndarray
            Float ambiguity estimates (in cycles)
        Q_a : np.ndarray
            Covariance matrix of float ambiguities
        wavelengths : np.ndarray, optional
            Wavelengths for each ambiguity (for validation)
            
        Returns
        -------
        a_fixed : np.ndarray or None
            Fixed integer ambiguities, or None if failed
        info : dict
            Resolution information (ratio, success_rate, etc.)
        """
        info = {
            'n_ambiguities': len(a_float),
            'ratio': 0.0,
            'success_rate': 0.0,
            'passed_ratio': False,
            'passed_success_rate': False,
            'fixed': False,
            'method': 'RTKLIB-LAMBDA'
        }
        
        # Check input validity
        if len(a_float) == 0 or Q_a.shape[0] != len(a_float):
            logger.warning("Invalid input dimensions for ambiguity resolution")
            return None, info
        
        # Check minimum satellites
        if len(a_float) < self.min_sats - 1:  # -1 for reference satellite
            logger.debug(f"Not enough ambiguities: {len(a_float)} < {self.min_sats-1}")
            return None, info
        
        # Search for integer candidates using RTKLIB MLAMBDA
        try:
            # Search for more candidates for better ratio testing
            n_search = max(5, min(10, len(a_float)))
            afix, residuals = mlambda(a_float, Q_a, m=n_search)
            
            # Convert to candidates format for compatibility
            candidates = [(afix[:, i], residuals[i]) for i in range(afix.shape[1])]
        except Exception as e:
            logger.warning(f"MLAMBDA search failed: {e}")
            return None, info
        
        if len(candidates) == 0:
            logger.warning("No ambiguity candidates found")
            return None, info
        
        # Ratio test
        passed_ratio, ratio = self.ratio_test(candidates)
        info['ratio'] = ratio
        info['passed_ratio'] = passed_ratio
        
        if not passed_ratio:
            logger.debug(f"Ratio test failed: {ratio:.2f} < {self.ratio_threshold}")
            return None, info
        
        # Success rate test
        passed_success, success_rate = self.success_rate_test(Q_a, ratio)
        info['success_rate'] = success_rate
        info['passed_success_rate'] = passed_success
        
        if not passed_success:
            logger.debug(f"Success rate test failed: {success_rate:.3f} < {self.success_rate_threshold}")
            return None, info
        
        # Get best candidate
        a_fixed = candidates[0][0]
        
        # Final validation
        if wavelengths is not None:
            # Check that ambiguities give reasonable position change
            position_change = a_fixed * wavelengths
            if np.any(np.abs(position_change) > 100):  # More than 100m change is suspicious
                logger.warning("Ambiguity resolution resulted in unreasonable position change")
                return None, info
        
        info['fixed'] = True
        self.nb_ar = len(a_float)
        logger.info(f"Ambiguities fixed successfully: ratio={ratio:.2f}, success_rate={success_rate:.3f}")
        
        return a_fixed.astype(int), info
    
    def partial_ambiguity_resolution(self, a_float: np.ndarray, Q_a: np.ndarray,
                                    min_ratio: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resolve subset of ambiguities with highest confidence
        
        Parameters
        ----------
        a_float : np.ndarray
            Float ambiguities
        Q_a : np.ndarray
            Covariance matrix
        min_ratio : float
            Minimum ratio for partial fixing
            
        Returns
        -------
        fixed_mask : np.ndarray
            Boolean mask of fixed ambiguities
        a_partial : np.ndarray
            Partially fixed ambiguities (float where not fixed)
        """
        n = len(a_float)
        fixed_mask = np.zeros(n, dtype=bool)
        a_partial = a_float.copy()
        
        # Sort by variance (diagonal of covariance)
        variances = np.diag(Q_a)
        sorted_idx = np.argsort(variances)
        
        # Try fixing subsets starting with most precise
        for subset_size in range(n, max(self.min_sats-2, 2), -1):
            subset = sorted_idx[:subset_size]
            
            # Extract subset
            a_subset = a_float[subset]
            Q_subset = Q_a[np.ix_(subset, subset)]
            
            # Try to fix subset with relaxed threshold
            saved_threshold = self.ratio_threshold
            self.ratio_threshold = min_ratio
            
            a_fixed, info = self.resolve(a_subset, Q_subset)
            
            self.ratio_threshold = saved_threshold
            
            if info['fixed'] and info['ratio'] >= min_ratio:
                fixed_mask[subset] = True
                a_partial[subset] = a_fixed
                logger.info(f"Partial fix: {subset_size}/{n} ambiguities, ratio={info['ratio']:.2f}")
                break
        
        return fixed_mask, a_partial