#!/usr/bin/env python3
"""
Bootstrapping Ambiguity Resolution
===================================

Bootstrapping is a simple sequential rounding method that fixes ambiguities
one by one based on their conditional variances.

References:
    [1] Teunissen P.J.G. (1998) Success probability of integer GPS ambiguity rounding
        and bootstrapping, Journal of Geodesy 72:606-612
    [2] Verhagen S. (2004) Integer ambiguity validation: an open problem?
        GPS Solutions 8:36-43
"""

import numpy as np
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class BootstrappingResolver:
    """
    Bootstrapping ambiguity resolution
    
    This is a simpler alternative to LAMBDA that sequentially rounds
    ambiguities based on their conditional variances.
    """
    
    def __init__(self, success_rate_threshold: float = 0.95):
        """
        Initialize bootstrapping resolver
        
        Parameters
        ----------
        success_rate_threshold : float
            Minimum required success rate for accepting solution
        """
        self.success_threshold = success_rate_threshold
    
    def resolve(self, float_amb: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Resolve integer ambiguities using bootstrapping
        
        Parameters
        ----------
        float_amb : np.ndarray
            Float ambiguity vector (n,)
        Q : np.ndarray
            Covariance matrix (n, n)
            
        Returns
        -------
        fixed_amb : np.ndarray
            Fixed integer ambiguities
        success_rate : float
            Bootstrapping success rate
        ratio : float
            Ratio test value (for compatibility)
        """
        n = len(float_amb)
        
        # Perform Z-transformation (decorrelation)
        Z, L, D = self._z_transform(Q)
        
        # Transform float ambiguities
        z_float = Z.T @ float_amb
        
        # Sequential rounding (bootstrapping)
        z_fixed = np.zeros(n)
        success_rate = 1.0
        
        for i in range(n):
            # Conditional estimate
            z_cond = z_float[i]
            for j in range(i):
                z_cond -= L[i, j] * (z_fixed[j] - z_float[j])
            
            # Round to nearest integer
            z_fixed[i] = np.round(z_cond)
            
            # Compute success rate for this ambiguity
            sigma_i = np.sqrt(D[i])
            residual = abs(z_cond - z_fixed[i])
            
            # Success rate using normal CDF approximation
            from scipy.stats import norm
            p_i = 2 * norm.cdf(0.5, loc=residual, scale=sigma_i) - 1
            success_rate *= max(p_i, 0.01)  # Avoid zero
        
        # Back-transform to original space
        fixed_amb = Z @ z_fixed
        
        # Compute ratio for compatibility (simplified)
        residual_float = np.linalg.norm(float_amb - np.round(float_amb))
        residual_fixed = np.linalg.norm(float_amb - fixed_amb)
        ratio = residual_float / (residual_fixed + 1e-10)
        
        return fixed_amb.astype(int), success_rate, ratio
    
    def _z_transform(self, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Z-transformation for decorrelation
        
        Returns Z, L, D where Q = Z * L * D * L' * Z'
        """
        n = Q.shape[0]
        Q_work = Q.copy()
        Z = np.eye(n)
        L = np.zeros((n, n))
        D = np.zeros(n)
        
        # LDL decomposition
        for i in range(n):
            D[i] = Q_work[i, i]
            L[i, i] = 1.0
            
            for j in range(i):
                L[i, j] = Q_work[i, j] / D[j]
                for k in range(j):
                    L[i, j] -= L[i, k] * L[j, k] * D[k] / D[j]
            
            for j in range(i + 1, n):
                for k in range(i):
                    Q_work[j, i] -= L[j, k] * Q_work[i, k]
                Q_work[i, j] = Q_work[j, i]
        
        # Integer decorrelation
        for i in range(n - 1, 0, -1):
            for j in range(i):
                mu = np.round(L[i, j])
                if mu != 0:
                    L[i, :] -= mu * L[j, :]
                    Z[:, i] -= mu * Z[:, j]
        
        return Z, L, D
    
    def resolve_with_validation(self, float_amb: np.ndarray, Q: np.ndarray,
                               code_residuals: Optional[np.ndarray] = None) -> Dict:
        """
        Resolve with additional validation checks
        
        Parameters
        ----------
        float_amb : np.ndarray
            Float ambiguity vector
        Q : np.ndarray
            Covariance matrix
        code_residuals : np.ndarray, optional
            Code observation residuals for validation
            
        Returns
        -------
        result : dict
            Dictionary with fixed ambiguities and statistics
        """
        # Basic resolution
        fixed_amb, success_rate, ratio = self.resolve(float_amb, Q)
        
        # Validation checks
        is_valid = success_rate > self.success_threshold
        
        # Check integer constraint
        if is_valid and code_residuals is not None:
            # Ambiguities should be consistent with code observations
            lambda_l1 = 0.1903  # GPS L1 wavelength (m)
            phase_residuals = (float_amb - fixed_amb) * lambda_l1
            
            # Check if phase residuals are reasonable
            if np.max(np.abs(phase_residuals)) > 0.5:  # 50cm threshold
                is_valid = False
        
        return {
            'fixed': fixed_amb,
            'success_rate': success_rate,
            'ratio': ratio,
            'is_valid': is_valid,
            'method': 'bootstrapping'
        }


class SequentialResolver:
    """
    Sequential ambiguity resolution
    
    Fixes ambiguities one by one in order of reliability
    """
    
    def __init__(self, confidence_threshold: float = 0.99):
        """
        Initialize sequential resolver
        
        Parameters
        ----------
        confidence_threshold : float
            Confidence level required for fixing
        """
        self.confidence_threshold = confidence_threshold
    
    def resolve(self, float_amb: np.ndarray, Q: np.ndarray,
                elevations: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Sequential resolution based on reliability
        
        Parameters
        ----------
        float_amb : np.ndarray
            Float ambiguity vector
        Q : np.ndarray
            Covariance matrix
        elevations : np.ndarray, optional
            Satellite elevation angles for ordering
            
        Returns
        -------
        fixed_amb : np.ndarray
            Fixed ambiguities (NaN for unfixed)
        fixed_flags : np.ndarray
            Boolean flags indicating which were fixed
        confidence : float
            Overall confidence level
        """
        n = len(float_amb)
        fixed_amb = np.full(n, np.nan)
        fixed_flags = np.zeros(n, dtype=bool)
        
        # Determine fixing order
        if elevations is not None:
            # Fix high elevation satellites first
            order = np.argsort(-elevations)  # Descending order
        else:
            # Fix based on variance (smallest first)
            variances = np.diag(Q)
            order = np.argsort(variances)
        
        overall_confidence = 1.0
        
        for idx in order:
            # Get conditional variance
            var_cond = Q[idx, idx]
            
            # Adjust for already fixed ambiguities
            for j in range(n):
                if fixed_flags[j] and j != idx:
                    var_cond -= Q[idx, j]**2 / Q[j, j]
            
            if var_cond <= 0:
                continue
            
            # Compute confidence for this ambiguity
            sigma = np.sqrt(var_cond)
            residual = abs(float_amb[idx] - np.round(float_amb[idx]))
            
            from scipy.stats import norm
            confidence = 2 * norm.cdf(0.5, loc=residual, scale=sigma) - 1
            
            if confidence > self.confidence_threshold:
                fixed_amb[idx] = np.round(float_amb[idx])
                fixed_flags[idx] = True
                overall_confidence *= confidence
        
        # Fill unfixed with float values
        fixed_amb[~fixed_flags] = float_amb[~fixed_flags]
        
        return fixed_amb.astype(int), fixed_flags, overall_confidence


def test_bootstrapping():
    """Test bootstrapping resolver"""
    print("Testing Bootstrapping Ambiguity Resolution")
    print("=" * 50)
    
    # Generate test data
    np.random.seed(42)
    n = 5
    true_amb = np.array([3, -2, 5, 1, -4])
    float_amb = true_amb + np.random.normal(0, 0.05, n)
    
    # Create covariance matrix
    Q = np.diag([0.01, 0.02, 0.015, 0.01, 0.025])
    
    # Test bootstrapping
    resolver = BootstrappingResolver()
    fixed_amb, success_rate, ratio = resolver.resolve(float_amb, Q)
    
    print(f"True ambiguities:  {true_amb}")
    print(f"Float ambiguities: {float_amb.round(3)}")
    print(f"Fixed ambiguities: {fixed_amb}")
    print(f"Success rate: {success_rate:.3f}")
    print(f"Ratio: {ratio:.2f}")
    print(f"Errors: {np.abs(fixed_amb - true_amb)}")
    
    # Test sequential resolver
    print("\nTesting Sequential Resolution")
    print("-" * 50)
    
    seq_resolver = SequentialResolver()
    elevations = np.array([45, 30, 60, 25, 50])  # degrees
    fixed_seq, flags, confidence = seq_resolver.resolve(float_amb, Q, elevations)
    
    print(f"Fixed ambiguities: {fixed_seq}")
    print(f"Fixed flags: {flags}")
    print(f"Confidence: {confidence:.3f}")
    print(f"Number fixed: {np.sum(flags)}/{n}")


if __name__ == "__main__":
    test_bootstrapping()