#!/usr/bin/env python3
"""
Ambiguity Resolution for RTK Positioning
=========================================

This module implements integer ambiguity resolution algorithms for carrier phase
positioning, including the LAMBDA (Least-squares AMBiguity Decorrelation Adjustment) method.
"""

import numpy as np
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class RTKAmbiguityManager:
    """
    Integer ambiguity resolution using LAMBDA method and validation
    """
    
    def __init__(self, ratio_threshold: float = 3.0, success_rate_threshold: float = 0.999):
        """
        Initialize ambiguity resolver
        
        Parameters
        ----------
        ratio_threshold : float
            Threshold for ratio test (second best / best solution)
        success_rate_threshold : float
            Required success rate for validation
        """
        self.ratio_threshold = ratio_threshold
        self.success_rate_threshold = success_rate_threshold
        self.fixed_ambiguities = {}
        self.ambiguity_covariance = None
        
    def lambda_reduction(self, Q: np.ndarray, a: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        LAMBDA decorrelation and reduction
        
        Parameters
        ----------
        Q : np.ndarray
            Covariance matrix of float ambiguities (n x n)
        a : np.ndarray
            Float ambiguity vector (n,)
            
        Returns
        -------
        Z : np.ndarray
            Transformation matrix
        L : np.ndarray
            Diagonal of reduced covariance matrix
        D : np.ndarray
            Decorrelated ambiguities
        """
        n = len(a)
        Z = np.eye(n)
        L = Q.copy()
        
        # LDL decomposition
        for i in range(n):
            for j in range(i):
                L[i, j] = L[i, j] / L[j, j]
                for k in range(j):
                    L[i, j] -= L[i, k] * L[j, k] * L[k, k] / L[j, j]
            
            for j in range(i):
                L[i, i] -= L[i, j]**2 * L[j, j]
        
        # Extract D and L
        D = np.diag(np.diag(L))
        for i in range(n):
            for j in range(i+1, n):
                L[j, i] = 0
        
        # Decorrelation
        k = n - 1
        while k > 0:
            k_old = k
            for i in range(k, 0, -1):
                if D[i-1, i-1] > D[i, i]:
                    # Swap
                    D[i-1, i-1], D[i, i] = D[i, i], D[i-1, i-1]
                    Z[:, [i-1, i]] = Z[:, [i, i-1]]
                    L[[i-1, i], :] = L[[i, i-1], :]
                    L[:, [i-1, i]] = L[:, [i, i-1]]
                    k = i
            
            if k == k_old:
                k -= 1
        
        # Transform ambiguities
        a_decorr = Z.T @ a
        
        return Z, np.diag(D), a_decorr
    
    def search_ambiguities(self, a_float: np.ndarray, Q_a: np.ndarray, 
                          n_candidates: int = 2) -> List[Tuple[np.ndarray, float]]:
        """
        Search for integer ambiguity candidates using MLAMBDA
        
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
        # Use MLAMBDA for more robust search
        return self.mlambda_search(a_float, Q_a, n_candidates)
    
    def mlambda_search(self, a_float: np.ndarray, Q_a: np.ndarray, 
                      n_max: int = 2, chi2_init: Optional[float] = None) -> List[Tuple[np.ndarray, float]]:
        """
        Modified LAMBDA (MLAMBDA) integer least squares search
        
        This implements the efficient integer search algorithm from:
        Chang X-W, Yang X, Zhou T (2005) MLAMBDA: a modified LAMBDA method for 
        integer least-squares estimation. J Geod 79:552-565
        
        Parameters
        ----------
        a_float : np.ndarray
            Float ambiguity vector (n,)
        Q_a : np.ndarray
            Covariance matrix (n x n)
        n_max : int
            Maximum number of candidates to find
            
        Returns
        -------
        candidates : List[Tuple[np.ndarray, float]]
            List of integer candidates with their residuals
        """
        n = len(a_float)
        
        # LtDL decomposition of Q_a
        L, D, Z, zt = self._ltdl_decomp(Q_a, a_float)
        
        # Initialize search
        candidates = []
        
        # Calculate better initial search radius using chi-square distribution
        # For n_max candidates, we want high probability of finding them
        # Use bootstrapping estimate based on covariance
        if n_max == 2:
            # For ratio test, need at least 2 candidates
            # Start with larger radius to ensure we find multiple candidates
            chi2_max = max(n * 3.0, 20.0)  # Chi-square with high confidence
        else:
            chi2_max = max(n * 2.0, 10.0)
        
        # Alternative: compute initial radius from rounded solution
        z_round = np.round(zt).astype(int)
        initial_dist = np.sum((zt - z_round)**2 / D)
        chi2_max = max(chi2_max, initial_dist * 10)  # Ensure we search wide enough
        
        # Search tree parameters
        S = np.zeros((n, n))  # Partial sums
        dist = np.zeros(n)    # Partial distances
        zb = np.zeros(n, dtype=int)  # Integer candidates
        z = np.zeros(n)       # Current search point
        step = np.zeros(n, dtype=int)  # Search steps
        
        # Initialize first level
        k = n - 1
        dist[k] = 0
        zb[k] = round(zt[k])
        z[k] = zt[k] - zb[k]
        step[k] = 1 if z[k] < 0 else -1
        
        # Main search loop
        n_found = 0
        max_iter = 10000  # Prevent infinite loops
        iter_count = 0
        
        while n_found < n_max and iter_count < max_iter:
            iter_count += 1
            
            # Compute partial distance
            new_dist = dist[k] + z[k]**2 / D[k]
            
            if new_dist < chi2_max:
                # Move down in search tree
                if k > 0:
                    # Compute partial sums
                    for i in range(k):
                        S[k-1, i] = S[k, i] + z[k] * L[k, i]
                    
                    dist[k-1] = new_dist
                    k -= 1
                    
                    # Compute next level
                    zb[k] = round(zt[k] + S[k, k])
                    z[k] = zt[k] + S[k, k] - zb[k]
                    step[k] = 1 if z[k] < 0 else -1
                else:
                    # Found a candidate at bottom level
                    if n_found < n_max:
                        # Store candidate
                        candidate_z = zb.copy()
                        # Transform back to original space
                        candidate_a = Z @ candidate_z
                        residual = new_dist
                        candidates.append((candidate_a.astype(int), residual))
                        n_found += 1
                        
                        # Update search radius for shrinking (key optimization)
                        if n_found >= 2:
                            # Shrink search space to focus on better candidates
                            candidates.sort(key=lambda x: x[1])
                            if n_found >= n_max:
                                # Keep only best n_max candidates
                                candidates = candidates[:n_max]
                                chi2_max = candidates[-1][1] * 1.0001  # Small margin
                            else:
                                # Adaptively shrink based on found candidates
                                chi2_max = min(chi2_max, candidates[-1][1] * 2.0)
                    
                    # Move to next integer at bottom level
                    zb[0] += step[0]
                    z[0] = zt[0] + S[0, 0] - zb[0]
                    step[0] = -step[0] - (1 if step[0] > 0 else -1)
            else:
                # Move up in search tree
                if k == n - 1:
                    break  # Finished search
                else:
                    k += 1
                    # Move to next integer
                    zb[k] += step[k]
                    z[k] = zt[k] + S[k, k] - zb[k]
                    step[k] = -step[k] - (1 if step[k] > 0 else -1)
        
        # Sort candidates by residual
        candidates.sort(key=lambda x: x[1])
        
        # If no candidates found, use simple rounding as fallback
        if len(candidates) == 0:
            a_int = np.round(a_float).astype(int)
            residual = self._compute_residual_norm(a_float, a_int, Q_a)
            candidates.append((a_int, residual))
            
            # Add second best by perturbing worst dimension
            if n_max > 1:
                variances = np.diag(Q_a)
                worst_idx = np.argmax(variances)
                a_int2 = a_int.copy()
                if a_float[worst_idx] - a_int[worst_idx] > 0:
                    a_int2[worst_idx] += 1
                else:
                    a_int2[worst_idx] -= 1
                residual2 = self._compute_residual_norm(a_float, a_int2, Q_a)
                candidates.append((a_int2, residual2))
                candidates.sort(key=lambda x: x[1])
        
        return candidates[:n_max]
    
    def _ltdl_decomp(self, Q: np.ndarray, a: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        LtDL decomposition with decorrelation for MLAMBDA
        
        Returns L (unit lower triangular), D (diagonal), 
        Z (transformation matrix), and transformed ambiguities
        """
        n = len(a)
        L = np.eye(n)
        D = np.zeros(n)
        Z = np.eye(n)
        
        # Copy Q to avoid modifying
        Q_work = Q.copy()
        
        # LDL factorization
        for i in range(n):
            D[i] = Q_work[i, i]
            for j in range(i+1, n):
                L[j, i] = Q_work[j, i] / D[i]
                for k in range(j, n):
                    Q_work[k, j] -= L[k, i] * L[j, i] * D[i]
        
        # Decorrelation (partial pivoting)
        k = n - 2
        while k >= 0:
            k_old = k
            for i in range(k, -1, -1):
                # Check if swap would reduce correlation
                if abs(L[i+1, i]) > 0.5:
                    # Integer rounding to reduce correlation
                    delta = round(L[i+1, i])
                    L[i+1, i] -= delta
                    Z[:, i+1] -= delta * Z[:, i]
                    for j in range(i):
                        L[i+1, j] -= delta * L[i, j]
                    
                # Check if swap would help
                if D[i] > 2 * D[i+1]:
                    # Swap columns i and i+1
                    D[i], D[i+1] = D[i+1], D[i]
                    Z[:, [i, i+1]] = Z[:, [i+1, i]]
                    
                    # Update L
                    for j in range(i):
                        L[i, j], L[i+1, j] = L[i+1, j], L[i, j]
                    
                    lambda_val = L[i+1, i]
                    eta = D[i] / D[i+1]
                    L[i+1, i] = eta * L[i+1, i]
                    
                    for j in range(i+2, n):
                        temp = L[j, i]
                        L[j, i] = L[j, i+1] - lambda_val * temp
                        L[j, i+1] = temp + L[i+1, i] * L[j, i+1]
                    
                    k = i
                    break
            
            if k == k_old:
                k -= 1
        
        # Transform float ambiguities
        zt = np.linalg.solve(Z.T, a)
        
        return L, D, Z, zt
    
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
        # Simplified success rate estimation
        # In practice, use more sophisticated methods (e.g., based on bootstrapping)
        n = Q_a.shape[0]
        
        # Estimate based on ratio and dimension
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
        Full ambiguity resolution pipeline
        
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
            'fixed': False
        }
        
        # Check input validity
        if len(a_float) == 0 or Q_a.shape[0] != len(a_float):
            logger.warning("Invalid input dimensions for ambiguity resolution")
            return None, info
        
        # Decorrelate (optional but improves search)
        try:
            Z, D, a_decorr = self.lambda_reduction(Q_a, a_float)
            Q_decorr = np.diag(D)
        except:
            # If decorrelation fails, use original
            a_decorr = a_float
            Q_decorr = Q_a
            Z = np.eye(len(a_float))
        
        # Search for integer candidates
        # For better ratio test, search for more candidates
        n_search = max(5, min(10, len(a_float)))  # Search for 5-10 candidates
        candidates = self.search_ambiguities(a_decorr, Q_decorr, n_candidates=n_search)
        
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
        
        # Transform back to original space
        a_fixed_decorr = candidates[0][0]
        a_fixed = Z @ a_fixed_decorr
        
        # Final validation
        if wavelengths is not None:
            # Check that ambiguities give reasonable position change
            position_change = a_fixed * wavelengths
            if np.any(np.abs(position_change) > 100):  # More than 100m change is suspicious
                logger.warning("Ambiguity resolution resulted in unreasonable position change")
                return None, info
        
        info['fixed'] = True
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
        sorted_indices = np.argsort(variances)
        
        # Try to fix ambiguities starting from most precise
        for idx in sorted_indices:
            if variances[idx] > 0.5:  # Too uncertain
                continue
            
            # Try fixing this ambiguity
            a_test = a_float.copy()
            a_test[idx] = np.round(a_float[idx])
            
            # Simple validation
            if np.abs(a_float[idx] - a_test[idx]) < 0.25:  # Close to integer
                fixed_mask[idx] = True
                a_partial[idx] = a_test[idx]
        
        return fixed_mask, a_partial