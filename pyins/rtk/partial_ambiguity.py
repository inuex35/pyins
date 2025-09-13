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
Partial Ambiguity Resolution (PAR)
===================================

PAR methods fix only a subset of ambiguities with high confidence,
leaving uncertain ones as float values.

References:
    [1] Teunissen et al. (1999) The success rate and precision of GPS ambiguities
    [2] Cao et al. (2008) A new method for partial ambiguity resolution
    [3] Li & Shen (2010) Improved partial ambiguity resolution
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PARResult:
    """Result from partial ambiguity resolution"""
    fixed_ambiguities: np.ndarray
    fixed_indices: np.ndarray
    float_indices: np.ndarray
    success_rate: float
    ratio_test: float
    n_fixed: int
    n_total: int


class PartialAmbiguityResolver:
    """
    Partial Ambiguity Resolution (PAR)
    
    Fixes only the most reliable subset of ambiguities
    """
    
    def __init__(self, 
                 min_success_rate: float = 0.99,
                 min_ratio: float = 3.0,
                 max_subset_size: int = 10):
        """
        Initialize PAR resolver
        
        Parameters
        ----------
        min_success_rate : float
            Minimum required success rate for subset
        min_ratio : float
            Minimum ratio test value
        max_subset_size : int
            Maximum size of subset to search
        """
        self.min_success_rate = min_success_rate
        self.min_ratio = min_ratio
        self.max_subset_size = max_subset_size
    
    def resolve(self, float_amb: np.ndarray, Q: np.ndarray,
                elevations: Optional[np.ndarray] = None) -> PARResult:
        """
        Resolve partial ambiguities
        
        Parameters
        ----------
        float_amb : np.ndarray
            Float ambiguity vector (n,)
        Q : np.ndarray
            Covariance matrix (n, n)
        elevations : np.ndarray, optional
            Satellite elevation angles for selection
            
        Returns
        -------
        result : PARResult
            Partial resolution result
        """
        n = len(float_amb)
        
        # Determine selection order based on reliability
        if elevations is not None:
            # Prefer high elevation satellites
            reliability_order = np.argsort(-elevations)
        else:
            # Use variance as reliability measure
            variances = np.diag(Q)
            reliability_order = np.argsort(variances)
        
        # Try different subset sizes
        best_result = None
        best_score = 0
        
        for subset_size in range(min(4, n), min(n + 1, self.max_subset_size + 1)):
            # Select subset
            subset_indices = reliability_order[:subset_size]
            
            # Extract subset
            float_subset = float_amb[subset_indices]
            Q_subset = Q[np.ix_(subset_indices, subset_indices)]
            
            # Try to fix subset using LAMBDA-like approach
            fixed_subset, ratio, success_rate = self._fix_subset(float_subset, Q_subset)
            
            if ratio > self.min_ratio and success_rate > self.min_success_rate:
                # Compute overall score (prefer larger subsets with high reliability)
                score = subset_size * success_rate * min(ratio / self.min_ratio, 2.0)
                
                if score > best_score:
                    best_score = score
                    
                    # Build full result
                    fixed_full = np.array(float_amb, copy=True)  # Ensure writable copy
                    fixed_full[subset_indices] = fixed_subset
                    
                    float_indices = np.setdiff1d(np.arange(n), subset_indices)
                    
                    best_result = PARResult(
                        fixed_ambiguities=fixed_full,
                        fixed_indices=subset_indices,
                        float_indices=float_indices,
                        success_rate=success_rate,
                        ratio_test=ratio,
                        n_fixed=subset_size,
                        n_total=n
                    )
        
        # If no good subset found, return float solution
        if best_result is None:
            best_result = PARResult(
                fixed_ambiguities=float_amb,
                fixed_indices=np.array([]),
                float_indices=np.arange(n),
                success_rate=0.0,
                ratio_test=0.0,
                n_fixed=0,
                n_total=n
            )
        
        return best_result
    
    def _fix_subset(self, float_subset: np.ndarray, Q_subset: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Fix ambiguity subset using simplified LAMBDA
        
        Returns fixed ambiguities, ratio test, success rate
        """
        n = len(float_subset)
        
        # Simple rounding for small subsets
        if n <= 3:
            fixed = np.round(float_subset)
            
            # Compute ratio
            residual1 = np.linalg.norm(float_subset - fixed)
            second_best = fixed.copy()
            worst_idx = np.argmax(np.abs(float_subset - fixed))
            second_best[worst_idx] += 1 if float_subset[worst_idx] - fixed[worst_idx] > 0 else -1
            residual2 = np.linalg.norm(float_subset - second_best)
            ratio = residual2 / (residual1 + 1e-10)
            
            # Estimate success rate
            success_rate = self._compute_success_rate(float_subset, Q_subset, fixed)
            
            return fixed.astype(int), ratio, success_rate
        
        # For larger subsets, use decorrelation
        Z, L, D = self._decorrelate(Q_subset)
        z_float = Z.T @ float_subset
        
        # Sequential rounding
        z_fixed = np.round(z_float)
        
        # Back-transform
        fixed = Z @ z_fixed
        fixed = np.round(fixed)  # Ensure integers
        
        # Compute ratio and success rate
        ratio = self._compute_ratio(float_subset, fixed, Q_subset)
        success_rate = self._compute_success_rate(float_subset, Q_subset, fixed)
        
        return fixed.astype(int), ratio, success_rate
    
    def _decorrelate(self, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simple decorrelation"""
        n = Q.shape[0]
        Z = np.eye(n)
        L = np.eye(n)
        D = np.diag(Q).copy()  # Make writable copy
        Q_work = Q.copy()  # Work on a copy
        
        # Basic Gram-Schmidt orthogonalization
        for i in range(1, n):
            for j in range(i):
                if D[j] > 0:
                    mu = Q_work[i, j] / D[j]
                    L[i, j] = mu
                    for k in range(j + 1, i):
                        Q_work[i, k] -= mu * Q_work[j, k]
                    D[i] -= mu * Q_work[i, j]
        
        return Z, L, D
    
    def _compute_ratio(self, float_amb: np.ndarray, fixed: np.ndarray, Q: np.ndarray) -> float:
        """Compute ratio test value"""
        # Best solution residual
        residual1 = (float_amb - fixed).T @ np.linalg.inv(Q) @ (float_amb - fixed)
        
        # Second best (perturb worst component)
        second = fixed.copy()
        worst_idx = np.argmax(np.abs(float_amb - fixed))
        second[worst_idx] += 1 if float_amb[worst_idx] - fixed[worst_idx] > 0 else -1
        residual2 = (float_amb - second).T @ np.linalg.inv(Q) @ (float_amb - second)
        
        return np.sqrt(residual2 / (residual1 + 1e-10))
    
    def _compute_success_rate(self, float_amb: np.ndarray, Q: np.ndarray, fixed: np.ndarray) -> float:
        """Estimate success rate"""
        from scipy.stats import chi2
        
        n = len(float_amb)
        residual = (float_amb - fixed).T @ np.linalg.inv(Q) @ (float_amb - fixed)
        
        # Chi-square test
        p_value = 1 - chi2.cdf(residual, df=n)
        
        return p_value


class OptimalSubsetSelector:
    """
    Select optimal subset for partial ambiguity resolution
    """
    
    def __init__(self, min_satellites: int = 4):
        """
        Initialize selector
        
        Parameters
        ----------
        min_satellites : int
            Minimum number of satellites to fix
        """
        self.min_satellites = min_satellites
    
    def select_subset(self, Q: np.ndarray, elevations: np.ndarray,
                     snr: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Select optimal subset based on multiple criteria
        
        Parameters
        ----------
        Q : np.ndarray
            Covariance matrix
        elevations : np.ndarray
            Satellite elevation angles
        snr : np.ndarray, optional
            Signal-to-noise ratios
            
        Returns
        -------
        indices : np.ndarray
            Indices of selected satellites
        """
        n = len(elevations)
        
        # Compute reliability scores
        scores = np.zeros(n)
        
        # Factor 1: Elevation (higher is better)
        scores += elevations / 90.0
        
        # Factor 2: Variance (lower is better)
        variances = np.diag(Q)
        scores += (1.0 - variances / np.max(variances))
        
        # Factor 3: SNR if available
        if snr is not None:
            scores += snr / np.max(snr)
        
        # Factor 4: Correlation (less correlated is better)
        for i in range(n):
            off_diag = np.sum(np.abs(Q[i, :])) - Q[i, i]
            scores[i] -= off_diag / (n * np.max(np.abs(Q)))
        
        # Select top satellites
        sorted_indices = np.argsort(-scores)
        
        # Start with minimum set
        selected = sorted_indices[:self.min_satellites]
        
        # Add more if they improve conditioning
        for idx in sorted_indices[self.min_satellites:]:
            test_set = np.append(selected, idx)
            Q_test = Q[np.ix_(test_set, test_set)]
            
            # Check condition number
            try:
                cond = np.linalg.cond(Q_test)
                if cond < 1000:  # Reasonable threshold
                    selected = test_set
                else:
                    break
            except:
                break
        
        return selected


def test_partial_ambiguity():
    """Test partial ambiguity resolution"""
    print("Testing Partial Ambiguity Resolution")
    print("=" * 50)
    
    # Generate test data with mixed quality
    np.random.seed(42)
    n = 10
    
    # Some ambiguities are well-determined, others are not
    true_amb = np.random.randint(-10, 10, n)
    noise_levels = np.array([0.01, 0.01, 0.02, 0.1, 0.01, 
                             0.15, 0.02, 0.01, 0.2, 0.03])
    float_amb = true_amb + np.random.normal(0, noise_levels, n)
    
    # Covariance matrix
    Q = np.diag(noise_levels**2)
    
    # Elevation angles
    elevations = np.array([60, 45, 50, 20, 55, 15, 40, 70, 10, 35])
    
    # Test PAR
    resolver = PartialAmbiguityResolver()
    result = resolver.resolve(float_amb, Q, elevations)
    
    print(f"Total ambiguities: {result.n_total}")
    print(f"Fixed ambiguities: {result.n_fixed}")
    print(f"Fixed indices: {result.fixed_indices}")
    print(f"Success rate: {result.success_rate:.3f}")
    print(f"Ratio test: {result.ratio_test:.2f}")
    
    print(f"\nTrue values:  {true_amb[result.fixed_indices]}")
    print(f"Fixed values: {result.fixed_ambiguities[result.fixed_indices].astype(int)}")
    
    # Check accuracy of fixed subset
    if len(result.fixed_indices) > 0:
        errors = np.abs(result.fixed_ambiguities[result.fixed_indices] - 
                       true_amb[result.fixed_indices])
        print(f"Errors in fixed subset: {errors}")
        print(f"Success rate: {np.mean(errors == 0)*100:.1f}%")
    
    # Test optimal subset selector
    print("\nTesting Optimal Subset Selection")
    print("-" * 50)
    
    selector = OptimalSubsetSelector()
    selected = selector.select_subset(Q, elevations)
    
    print(f"Selected satellites: {selected}")
    print(f"Selected elevations: {elevations[selected]}")
    print(f"Selected variances: {np.diag(Q)[selected].round(4)}")


if __name__ == "__main__":
    test_partial_ambiguity()