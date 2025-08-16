"""LAMBDA method implementation for ambiguity resolution"""

import numpy as np
from scipy.linalg import cholesky, qr
from typing import Tuple, List


def lambda_reduction(Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    LAMBDA reduction of ambiguity covariance matrix
    
    Parameters:
    -----------
    Q : np.ndarray
        Ambiguity covariance matrix
        
    Returns:
    --------
    Z : np.ndarray
        Integer transformation matrix
    L : np.ndarray
        Lower triangular matrix
    D : np.ndarray
        Diagonal matrix
    """
    n = Q.shape[0]
    Z = np.eye(n)
    
    # LDL decomposition
    L, D = _ldl_decomposition(Q)
    
    # LLL reduction
    Z, L, D = _lll_reduction(L, D, Z)
    
    return Z, L, D


def lambda_search(a_float: np.ndarray, L: np.ndarray, D: np.ndarray, 
                 ncands: int = 2) -> Tuple[List[np.ndarray], List[float]]:
    """
    Integer least squares search using LAMBDA method
    
    Parameters:
    -----------
    a_float : np.ndarray
        Float ambiguity estimates
    L : np.ndarray
        Lower triangular matrix from LDL decomposition
    D : np.ndarray
        Diagonal matrix from LDL decomposition
    ncands : int
        Number of candidates to return
        
    Returns:
    --------
    candidates : List[np.ndarray]
        Integer candidates
    residuals : List[float]
        Residuals for each candidate
    """
    n = len(a_float)
    candidates = []
    residuals = []
    
    # Initialize search
    chi2_max = np.inf
    
    # Search bounds
    left = np.zeros(n)
    right = np.zeros(n)
    
    # Start recursive search
    _search_recursive(a_float, L, D, 0, 0.0, np.zeros(n), 
                     candidates, residuals, chi2_max, ncands, left, right)
    
    # Sort by residual
    if len(candidates) > 1:
        sorted_indices = np.argsort(residuals)
        candidates = [candidates[i] for i in sorted_indices]
        residuals = [residuals[i] for i in sorted_indices]
    
    return candidates[:ncands], residuals[:ncands]


def _ldl_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """LDL decomposition A = L*D*L'"""
    n = A.shape[0]
    L = np.eye(n)
    D = np.zeros(n)
    
    for i in range(n):
        # Diagonal element
        D[i] = A[i, i] - np.sum(L[i, :i]**2 * D[:i])
        
        # Below diagonal elements
        for j in range(i + 1, n):
            L[j, i] = (A[j, i] - np.sum(L[j, :i] * L[i, :i] * D[:i])) / D[i]
    
    return L, D


def _lll_reduction(L: np.ndarray, D: np.ndarray, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LLL reduction for improved conditioning"""
    n = L.shape[0]
    delta = 0.75  # LLL parameter
    
    k = 1
    while k < n:
        # Size reduction
        for j in range(k - 1, -1, -1):
            if abs(L[k, j]) > 0.5:
                mu = round(L[k, j])
                L[k, j] -= mu
                
                # Update L and Z
                for i in range(j):
                    L[k, i] -= mu * L[j, i]
                for i in range(n):
                    Z[i, k] -= mu * Z[i, j]
        
        # Lovász condition
        if D[k] >= (delta - L[k, k-1]**2) * D[k-1]:
            k += 1
        else:
            # Swap columns k-1 and k
            _swap_columns(L, D, Z, k-1, k)
            k = max(1, k - 1)
    
    return Z, L, D


def _swap_columns(L: np.ndarray, D: np.ndarray, Z: np.ndarray, i: int, j: int):
    """Swap columns in LDL decomposition"""
    n = L.shape[0]
    
    # Swap in Z
    Z[:, [i, j]] = Z[:, [j, i]]
    
    # Update L and D
    if i == j - 1:  # Adjacent swap
        mu = L[j, i]
        delta = D[i] + mu**2 * D[j]
        
        if abs(delta) > 1e-12:
            D[i] = D[j] * D[i] / delta
            D[j] = delta
            L[j, i] = mu * D[j] / delta
            
            # Update other elements
            for k in range(j + 1, n):
                temp = L[k, i]
                L[k, i] = L[k, j] - mu * temp
                L[k, j] = temp


def _search_recursive(a_float: np.ndarray, L: np.ndarray, D: np.ndarray,
                     level: int, chi2: float, a_int: np.ndarray,
                     candidates: List[np.ndarray], residuals: List[float],
                     chi2_max: float, ncands: int, left: np.ndarray, right: np.ndarray):
    """Recursive integer search"""
    n = len(a_float)
    
    if level == n:
        # Found candidate
        if len(candidates) < ncands:
            candidates.append(a_int.copy())
            residuals.append(chi2)
        elif chi2 < max(residuals):
            # Replace worst candidate
            worst_idx = np.argmax(residuals)
            candidates[worst_idx] = a_int.copy()
            residuals[worst_idx] = chi2
        return
    
    # Current level from top
    k = n - 1 - level
    
    # Compute conditional mean
    s = a_float[k]
    for j in range(k + 1, n):
        s -= L[k, j] * a_int[j]
    
    # Search bounds
    if D[k] > 0:
        sqrt_D = np.sqrt(D[k])
        range_max = np.sqrt(chi2_max - chi2) / sqrt_D
        left[k] = int(np.ceil(s - range_max))
        right[k] = int(np.floor(s + range_max))
    else:
        left[k] = right[k] = int(round(s))
    
    # Search interval
    for a_k in range(left[k], right[k] + 1):
        a_int[k] = a_k
        diff = a_k - s
        chi2_new = chi2 + diff**2 * D[k]
        
        if chi2_new < chi2_max:
            _search_recursive(a_float, L, D, level + 1, chi2_new, a_int,
                            candidates, residuals, chi2_max, ncands, left, right)


def bootstrap_success_rate(Q: np.ndarray) -> float:
    """
    Estimate success rate using bootstrapping method
    
    Parameters:
    -----------
    Q : np.ndarray
        Ambiguity covariance matrix
        
    Returns:
    --------
    success_rate : float
        Estimated success rate
    """
    n = Q.shape[0]
    if n == 0:
        return 0.0
    
    # Conditional variances
    diag_Q = np.diag(Q)
    success_rate = 1.0
    
    for i in range(n):
        # Simplified bootstrap success rate calculation
        sigma = np.sqrt(diag_Q[i])
        if sigma > 0:
            # Probability of correct integer
            p_correct = 2 * _normal_cdf(0.5 / sigma) - 1
            success_rate *= max(0.01, p_correct)  # Avoid zero
        else:
            success_rate *= 0.01
    
    return min(success_rate, 0.999)


def _normal_cdf(x: float) -> float:
    """Approximation of normal CDF"""
    # Abramowitz and Stegun approximation
    if x < 0:
        return 1 - _normal_cdf(-x)
    
    # Constants
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    
    return y


def ratio_test(residuals: List[float], threshold: float = 3.0) -> Tuple[bool, float]:
    """
    Perform ratio test for ambiguity validation
    
    Parameters:
    -----------
    residuals : List[float]
        Residuals from integer search
    threshold : float
        Ratio threshold
        
    Returns:
    --------
    passed : bool
        Whether ratio test passed
    ratio : float
        Computed ratio
    """
    if len(residuals) < 2:
        return False, 0.0
    
    if residuals[0] <= 0:
        return False, 0.0
    
    ratio = residuals[1] / residuals[0]
    return ratio >= threshold, ratio