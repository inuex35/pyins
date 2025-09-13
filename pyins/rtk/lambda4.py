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
LAMBDA4 Algorithm Implementation
=================================

Implementation of LAMBDA4 algorithm from GreatPVT for integer ambiguity resolution.
Includes bootstrapping success rate calculation and partial fixing support.
"""

import numpy as np
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class LAMBDA4:
    """
    LAMBDA4 implementation based on GreatPVT
    
    Enhanced version with bootstrapping and partial fixing support.
    """
    
    def __init__(self):
        """Initialize LAMBDA4 solver"""
        self.pDia = None  # Diagonal elements for partial fixing
        self.bootstrapping_rate = 0.0
        
    def LAMBDA4(self, maxcan: int, namb: int, Q: np.ndarray, fbias: np.ndarray,
               ncan_out: List[int], ipos_out: List[int],
               cands: np.ndarray, disall: np.ndarray,
               boot: List[float]) -> int:
        """
        LAMBDA4 main function
        
        Parameters
        ----------
        maxcan : int
            Maximum number of candidates (typically 2)
        namb : int
            Number of ambiguities
        Q : np.ndarray
            Lower triangular covariance matrix (namb x namb)
        fbias : np.ndarray
            Float ambiguity residuals (namb,)
        ncan_out : List[int]
            Output: number of candidates found
        ipos_out : List[int]
            Output: position index
        cands : np.ndarray
            Output: candidate solutions (namb x maxcan)
        disall : np.ndarray
            Output: squared residuals for each candidate
        boot : List[float]
            Output: bootstrapping success rate
            
        Returns
        -------
        status : int
            0 for success, -1 for failure
        """
        try:
            # Initialize
            self.pDia = np.zeros(namb)
            ncan_out[0] = 0
            ipos_out[0] = 0
            
            # LD factorization
            L, D = self._ld_factorization(Q)
            
            # Store diagonal for partial fixing
            self.pDia = D.copy()
            
            # Z-transformation (decorrelation)
            Z, Zt, L_trans, D_trans = self._z_transformation(L, D)
            
            # Transform float solution
            a_trans = Zt @ fbias
            
            # Search for integer candidates
            candidates, residuals = self._search_candidates(
                L_trans, D_trans, a_trans, maxcan
            )
            
            # Calculate bootstrapping success rate
            boot[0] = self._calculate_bootstrapping(D_trans)
            self.bootstrapping_rate = boot[0]
            
            # Transform candidates back
            for i in range(len(candidates)):
                if i < maxcan:
                    cands[:, i] = Z @ candidates[i]
                    disall[i] = residuals[i]
            
            ncan_out[0] = len(candidates)
            
            return 0
            
        except Exception as e:
            logger.error(f"LAMBDA4 failed: {e}")
            return -1
    
    def _ld_factorization(self, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        LDL^T factorization of covariance matrix
        
        Parameters
        ----------
        Q : np.ndarray
            Symmetric positive definite matrix
            
        Returns
        -------
        L : np.ndarray
            Lower triangular matrix with 1s on diagonal
        D : np.ndarray
            Diagonal values
        """
        n = Q.shape[0]
        L = np.eye(n)
        D = np.zeros(n)
        
        for i in range(n):
            D[i] = Q[i, i]
            for j in range(i):
                D[i] -= L[i, j]**2 * D[j]
            
            if D[i] <= 0:
                logger.warning(f"Non-positive diagonal element D[{i}]={D[i]}")
                D[i] = 1e-6
            
            for j in range(i+1, n):
                L[j, i] = Q[j, i]
                for k in range(i):
                    L[j, i] -= L[j, k] * L[i, k] * D[k]
                L[j, i] /= D[i]
        
        return L, D
    
    def _z_transformation(self, L: np.ndarray, D: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Z-transformation for decorrelation
        
        Parameters
        ----------
        L : np.ndarray
            Lower triangular matrix from LD factorization
        D : np.ndarray
            Diagonal values
            
        Returns
        -------
        Z : np.ndarray
            Transformation matrix
        Zt : np.ndarray
            Transpose of Z
        L_trans : np.ndarray
            Transformed L
        D_trans : np.ndarray
            Transformed D
        """
        n = len(D)
        Z = np.eye(n)
        L_trans = L.copy()
        D_trans = D.copy()
        
        # LAMBDA reduction
        k = n - 1
        while k > 0:
            k_prev = k
            k -= 1
            
            # Integer Gaussian elimination
            for i in range(k+1):
                mu = round(L_trans[k, i])
                if mu != 0:
                    for j in range(k+1):
                        L_trans[k, j] -= mu * L_trans[i, j]
                    for j in range(n):
                        Z[j, k] -= mu * Z[j, i]
            
            # Permutation
            delta = D_trans[k] + L_trans[k+1, k]**2 * D_trans[k+1]
            
            if delta < D_trans[k+1]:
                # Swap k and k+1
                lam = D_trans[k+1] * L_trans[k+1, k] / delta
                eta = D_trans[k] / delta
                
                D_trans[k] = eta * D_trans[k+1]
                D_trans[k+1] = delta
                
                # Update L matrix
                for i in range(k):
                    a = L_trans[k, i]
                    b = L_trans[k+1, i]
                    L_trans[k, i] = -L_trans[k+1, k] * a + b
                    L_trans[k+1, i] = eta * a + lam * b
                
                L_trans[k+1, k] = lam
                
                # Update transformation matrix
                for i in range(n):
                    Z[i, k], Z[i, k+1] = Z[i, k+1], Z[i, k]
                
                k = n - 1  # Restart from end
        
        Zt = Z.T
        return Z, Zt, L_trans, D_trans
    
    def _search_candidates(self, L: np.ndarray, D: np.ndarray, 
                          a: np.ndarray, maxcan: int) -> Tuple[List[np.ndarray], List[float]]:
        """
        Search for integer candidates
        
        Parameters
        ----------
        L : np.ndarray
            Transformed lower triangular matrix
        D : np.ndarray
            Transformed diagonal
        a : np.ndarray
            Transformed float ambiguities
        maxcan : int
            Maximum candidates to find
            
        Returns
        -------
        candidates : List[np.ndarray]
            Integer candidate vectors
        residuals : List[float]
            Squared residuals
        """
        n = len(a)
        candidates = []
        residuals = []
        
        # Initialize search
        Chi2 = 1e10  # Initial search radius
        dist = np.zeros(n)
        
        # Bootstrap search
        z = np.zeros(n, dtype=int)
        z_float = a.copy()
        
        # Start from last ambiguity
        k = n - 1
        z[k] = round(z_float[k])
        y = z_float[k] - z[k]
        step = np.sign(y) if y != 0 else 1
        
        for _ in range(10000):  # Maximum iterations
            # Calculate partial distance
            newdist = dist[k] + y**2 / D[k] if k < n else 0
            
            if newdist < Chi2:
                if k == 0:
                    # Found a candidate
                    if len(candidates) < maxcan:
                        candidates.append(z.copy())
                        residuals.append(newdist)
                        
                        if len(candidates) == maxcan:
                            Chi2 = max(residuals) * 1.1  # Shrink search space
                    
                    # Next integer at level 0
                    z[0] += step
                    y = z_float[0] - z[0]
                    step = -step - np.sign(step)
                else:
                    # Move down
                    k -= 1
                    dist[k] = newdist
                    
                    # Calculate conditional estimate
                    s = 0
                    for j in range(k+1, n):
                        s += L[j, k] * (z[j] - z_float[j])
                    z_float[k] = a[k] + s
                    
                    z[k] = round(z_float[k])
                    y = z_float[k] - z[k]
                    step = np.sign(y) if y != 0 else 1
            else:
                # Move up
                if k == n - 1:
                    break  # Done
                
                k += 1
                z[k] += step
                y = z_float[k] - z[k]
                step = -step - np.sign(step)
        
        return candidates, residuals
    
    def _calculate_bootstrapping(self, D: np.ndarray) -> float:
        """
        Calculate bootstrapping success rate
        
        Parameters
        ----------
        D : np.ndarray
            Diagonal values after decorrelation
            
        Returns
        -------
        boot_rate : float
            Bootstrapping success rate [0, 1]
        """
        n = len(D)
        P_total = 1.0
        
        for i in range(n):
            # Success probability for each ambiguity
            sigma = np.sqrt(D[i])
            # Probability of correct rounding
            P_i = 2 * self._normal_cdf(0.5 / sigma) - 1
            P_total *= P_i
        
        return P_total
    
    def _normal_cdf(self, x: float) -> float:
        """
        Cumulative distribution function of standard normal
        
        Parameters
        ----------
        x : float
            Input value
            
        Returns
        -------
        cdf : float
            CDF value
        """
        # Approximation of normal CDF
        try:
            from scipy.special import erf
            return 0.5 * (1 + erf(x / np.sqrt(2)))
        except ImportError:
            # Simple approximation if scipy not available
            t = 1 / (1 + 0.2316419 * abs(x))
            d = 0.3989423 * np.exp(-x * x / 2)
            p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
            return p if x < 0 else 1 - p
    
    def get_partial_fixing_order(self) -> np.ndarray:
        """
        Get order for partial fixing based on diagonal elements
        
        Returns
        -------
        order : np.ndarray
            Indices sorted by reliability (smallest variance first)
        """
        if self.pDia is None:
            return np.array([])
        
        # Sort by variance (smaller is better)
        return np.argsort(self.pDia)


class EnhancedLAMBDA4:
    """
    Enhanced LAMBDA4 with partial fixing and adaptive strategies
    """
    
    def __init__(self, enable_partial: bool = True,
                 min_partial_ratio: float = 2.0,
                 min_partial_boot: float = 0.95):
        """
        Initialize enhanced LAMBDA4
        
        Parameters
        ----------
        enable_partial : bool
            Enable partial ambiguity fixing
        min_partial_ratio : float
            Minimum ratio for partial fixing
        min_partial_boot : float
            Minimum bootstrapping rate for partial fixing
        """
        self.lambda4 = LAMBDA4()
        self.enable_partial = enable_partial
        self.min_partial_ratio = min_partial_ratio
        self.min_partial_boot = min_partial_boot
    
    def resolve_with_partial(self, a_float: np.ndarray, Q: np.ndarray,
                            ratio_threshold: float = 3.0,
                            boot_threshold: float = 0.99) -> Tuple[np.ndarray, float, float, np.ndarray]:
        """
        Resolve with partial fixing capability
        
        Parameters
        ----------
        a_float : np.ndarray
            Float ambiguities
        Q : np.ndarray
            Covariance matrix
        ratio_threshold : float
            Full fixing ratio threshold
        boot_threshold : float
            Full fixing bootstrapping threshold
            
        Returns
        -------
        a_fixed : np.ndarray
            Fixed ambiguities (NaN for unfixed)
        ratio : float
            Ratio test value
        boot : float
            Bootstrapping rate
        fixed_mask : np.ndarray
            Boolean mask of fixed ambiguities
        """
        n = len(a_float)
        
        # Prepare LAMBDA4 inputs
        maxcan = 2
        fbias = a_float - np.round(a_float)
        Q_lower = np.tril(Q)
        
        ncan = [0]
        ipos = [0]
        cands = np.zeros((n, maxcan))
        disall = np.zeros(maxcan)
        boot = [0.0]
        
        # Run LAMBDA4
        status = self.lambda4.LAMBDA4(
            maxcan, n, Q_lower, fbias,
            ncan, ipos, cands, disall, boot
        )
        
        if status != 0 or ncan[0] == 0:
            logger.warning("LAMBDA4 search failed")
            return np.full(n, np.nan), 0.0, 0.0, np.zeros(n, dtype=bool)
        
        # Calculate ratio
        if ncan[0] >= 2 and disall[0] > 0:
            ratio = disall[1] / disall[0] if disall[1] > 0 else 999.0
        else:
            ratio = 999.0 if ncan[0] == 1 else 0.0
        
        boot_rate = boot[0]
        
        # Check for full fixing
        if ratio >= ratio_threshold and boot_rate >= boot_threshold:
            # Full fixing successful
            a_fixed = np.round(a_float) + cands[:, 0]
            return a_fixed, ratio, boot_rate, np.ones(n, dtype=bool)
        
        # Try partial fixing if enabled
        if self.enable_partial and ratio >= self.min_partial_ratio:
            return self._partial_fixing(
                a_float, Q, cands[:, 0], ratio, boot_rate
            )
        
        # No fixing
        return np.full(n, np.nan), ratio, boot_rate, np.zeros(n, dtype=bool)
    
    def _partial_fixing(self, a_float: np.ndarray, Q: np.ndarray,
                       best_cand: np.ndarray, ratio: float, boot: float) -> Tuple[np.ndarray, float, float, np.ndarray]:
        """
        Perform partial ambiguity fixing
        
        Parameters
        ----------
        a_float : np.ndarray
            Float ambiguities
        Q : np.ndarray
            Covariance matrix
        best_cand : np.ndarray
            Best candidate from search
        ratio : float
            Current ratio
        boot : float
            Current bootstrapping rate
            
        Returns
        -------
        a_partial : np.ndarray
            Partially fixed ambiguities
        ratio : float
            Updated ratio
        boot : float
            Updated bootstrapping rate
        fixed_mask : np.ndarray
            Mask of fixed ambiguities
        """
        n = len(a_float)
        fixed_mask = np.zeros(n, dtype=bool)
        a_partial = a_float.copy()
        
        # Get fixing order based on reliability
        order = self.lambda4.get_partial_fixing_order()
        
        if len(order) == 0:
            return a_partial, ratio, boot, fixed_mask
        
        # Try fixing subset
        for num_fix in range(n, 0, -1):
            subset = order[:num_fix]
            
            # Extract subset covariance
            Q_sub = Q[np.ix_(subset, subset)]
            a_sub = a_float[subset]
            
            # Resolve subset
            a_sub_fixed, sub_ratio, sub_boot, sub_mask = self.resolve_with_partial(
                a_sub, Q_sub, self.min_partial_ratio, self.min_partial_boot
            )
            
            if sub_ratio >= self.min_partial_ratio and sub_boot >= self.min_partial_boot:
                # Accept partial fixing
                a_partial[subset] = a_sub_fixed
                fixed_mask[subset] = sub_mask
                
                logger.info(f"Partial fixing: {np.sum(fixed_mask)}/{n} ambiguities, "
                          f"ratio={sub_ratio:.2f}, boot={sub_boot:.3f}")
                
                return a_partial, sub_ratio, sub_boot, fixed_mask
        
        # No partial fixing successful
        return a_partial, ratio, boot, fixed_mask