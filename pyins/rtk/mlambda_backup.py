#!/usr/bin/env python3
"""
LAMBDA Implementation Based on RTKLIB-py
=========================================

This module implements the LAMBDA (Least-squares AMBiguity Decorrelation Adjustment)
method following the RTKLIB-py approach, which is simpler and more efficient than
the classical implementation.

References:
    [1] P.J.G.Teunissen, The least-square ambiguity decorrelation adjustment:
        a method for fast GPS ambiguity estimation, J.Geodesy, Vol.70, 65-82, 1995
    [2] X.-W.Chang, X.Yang, T.Zhou, MLAMBDA: A modified LAMBDA method for
        integer least-squares estimation, J.Geodesy, Vol.79, 552-565, 2005

Based on RTKLIB-py implementation by Rui Hirokawa and Tim Everett
"""

import numpy as np
from numpy.linalg import inv
from typing import Tuple, Optional, List
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


class RTKLIBAmbiguityResolver:
    """
    Ambiguity resolver using RTKLIB-py LAMBDA implementation
    """
    
    def __init__(self, ratio_threshold: float = 3.0, 
                 min_sats: int = 4,
                 max_position_var: float = 0.1):
        """
        Initialize RTKLIB-style ambiguity resolver
        
        Parameters
        ----------
        ratio_threshold : float
            Minimum ratio for validation (second/first residual)
        min_sats : int
            Minimum satellites for ambiguity resolution
        max_position_var : float
            Maximum position variance for attempting AR
        """
        self.ratio_threshold = ratio_threshold
        self.min_sats = min_sats
        self.max_position_var = max_position_var
        
        # State tracking
        self.prev_ratio = 0.0
        self.prev_ratio2 = 0.0
        self.nb_ar = 0  # Number of ambiguities resolved
        
    def resolve(self, a_float: np.ndarray, Q_a: np.ndarray,
                validate: bool = True) -> Tuple[Optional[np.ndarray], dict]:
        """
        Resolve integer ambiguities using RTKLIB LAMBDA
        
        Parameters
        ----------
        a_float : np.ndarray
            Float ambiguity estimates (cycles)
        Q_a : np.ndarray
            Covariance matrix
        validate : bool
            Whether to perform validation
            
        Returns
        -------
        a_fixed : np.ndarray or None
            Fixed integer ambiguities
        info : dict
            Resolution information
        """
        info = {
            'n_ambiguities': len(a_float),
            'ratio': 0.0,
            'fixed': False,
            'method': 'RTKLIB-LAMBDA'
        }
        
        # Check minimum satellites
        if len(a_float) < self.min_sats - 1:  # -1 for reference satellite
            logger.debug(f"Not enough ambiguities: {len(a_float)} < {self.min_sats-1}")
            return None, info
        
        # MLAMBDA search for 2 best candidates
        try:
            candidates, residuals = mlambda(a_float, Q_a, m=2)
        except Exception as e:
            logger.warning(f"MLAMBDA failed: {e}")
            return None, info
        
        # Compute ratio
        if len(residuals) >= 2 and residuals[0] > 0:
            ratio = residuals[1] / residuals[0]
        else:
            ratio = 0.0
        
        info['ratio'] = ratio
        self.nb_ar = len(a_float)
        
        # Validation
        if validate:
            if residuals[0] <= 0.0 or ratio < self.ratio_threshold:
                logger.debug(f"Validation failed: ratio={ratio:.2f} < {self.ratio_threshold}")
                self.prev_ratio2 = self.prev_ratio
                self.prev_ratio = ratio
                return None, info
        
        # Success
        info['fixed'] = True
        self.prev_ratio2 = self.prev_ratio
        self.prev_ratio = ratio
        
        logger.info(f"Ambiguities fixed: ratio={ratio:.2f}, residual={residuals[0]:.3f}")
        
        return candidates[:, 0].astype(int), info
    
    def partial_fix(self, a_float: np.ndarray, Q_a: np.ndarray,
                    min_ratio_partial: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Partial ambiguity fixing strategy
        
        Fix subset of ambiguities with best precision
        
        Parameters
        ----------
        a_float : np.ndarray
            Float ambiguities
        Q_a : np.ndarray
            Covariance matrix
        min_ratio_partial : float
            Minimum ratio for partial fixing
            
        Returns
        -------
        fixed_mask : np.ndarray
            Boolean mask of fixed ambiguities
        a_partial : np.ndarray
            Partially fixed ambiguities
        """
        n = len(a_float)
        fixed_mask = np.zeros(n, dtype=bool)
        a_partial = a_float.copy()
        
        # Sort by precision (diagonal of covariance)
        variances = np.diag(Q_a)
        sorted_idx = np.argsort(variances)
        
        # Try fixing subsets starting with most precise
        for subset_size in range(n, self.min_sats-2, -1):
            subset = sorted_idx[:subset_size]
            
            # Extract subset
            a_subset = a_float[subset]
            Q_subset = Q_a[np.ix_(subset, subset)]
            
            # Try to fix subset
            a_fixed, info = self.resolve(a_subset, Q_subset, validate=True)
            
            if info['fixed'] and info['ratio'] >= min_ratio_partial:
                fixed_mask[subset] = True
                a_partial[subset] = a_fixed
                logger.info(f"Partial fix: {subset_size}/{n} ambiguities, ratio={info['ratio']:.2f}")
                break
        
        return fixed_mask, a_partial