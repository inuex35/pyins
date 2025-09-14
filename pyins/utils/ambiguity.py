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

"""LAMBDA method implementation for ambiguity resolution"""


import numpy as np


def lambda_reduction(Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
                 ncands: int = 2) -> tuple[list[np.ndarray], list[float]]:
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


def _ldl_decomposition(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    LDL decomposition of a symmetric positive definite matrix.

    Decomposes matrix A into A = L*D*L' where L is lower triangular
    with unit diagonal and D is diagonal.

    Parameters
    ----------
    A : np.ndarray
        Symmetric positive definite matrix to decompose

    Returns
    -------
    L : np.ndarray
        Lower triangular matrix with unit diagonal
    D : np.ndarray
        Diagonal elements as 1D array

    Notes
    -----
    This decomposition is numerically more stable than Cholesky
    decomposition for ill-conditioned matrices and is preferred
    in the LAMBDA method implementation.

    The algorithm follows the standard LDL decomposition:
    - D[i] = A[i,i] - sum(L[i,k]^2 * D[k] for k < i)
    - L[j,i] = (A[j,i] - sum(L[j,k] * L[i,k] * D[k] for k < i)) / D[i]
    """
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


def _lll_reduction(L: np.ndarray, D: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    LLL (Lenstra-Lenstra-Lovász) reduction for improved conditioning.

    Performs LLL reduction on the LDL decomposition to improve the
    condition number and reduce the search space for integer solutions.

    Parameters
    ----------
    L : np.ndarray
        Lower triangular matrix from LDL decomposition
    D : np.ndarray
        Diagonal elements from LDL decomposition
    Z : np.ndarray
        Integer transformation matrix (initially identity)

    Returns
    -------
    Z : np.ndarray
        Updated integer transformation matrix
    L : np.ndarray
        Updated lower triangular matrix
    D : np.ndarray
        Updated diagonal elements

    Notes
    -----
    The LLL algorithm uses two operations:
    1. Size reduction: Ensures |L[i,j]| ≤ 0.5 for all i > j
    2. Column swapping: Based on Lovász condition with δ = 0.75

    The Lovász condition is: D[k] ≥ (δ - L[k,k-1]²) * D[k-1]
    where δ = 0.75 is the standard LLL parameter.

    This reduction significantly improves the efficiency of the
    subsequent integer search by creating a more orthogonal basis.
    """
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
    """
    Swap columns in LDL decomposition during LLL reduction.

    Performs the column swap operation required by the LLL algorithm
    while maintaining the LDL decomposition structure.

    Parameters
    ----------
    L : np.ndarray
        Lower triangular matrix (modified in-place)
    D : np.ndarray
        Diagonal elements array (modified in-place)
    Z : np.ndarray
        Integer transformation matrix (modified in-place)
    i : int
        First column index
    j : int
        Second column index

    Notes
    -----
    This function implements the column swap operation for adjacent
    columns (i = j-1) in the LDL decomposition. The swap involves:
    1. Swapping columns in the transformation matrix Z
    2. Updating the diagonal elements D
    3. Updating the lower triangular matrix L

    The update formulas ensure that the decomposition A = L*D*L'
    remains valid after the column permutation.

    For numerical stability, the swap is only performed if the
    computed delta value is sufficiently large (> 1e-12).
    """
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
                     candidates: list[np.ndarray], residuals: list[float],
                     chi2_max: float, ncands: int, left: np.ndarray, right: np.ndarray):
    """
    Recursive integer search for LAMBDA method.

    Performs a depth-first search through the integer lattice to find
    the best integer candidates that minimize the quadratic form.

    Parameters
    ----------
    a_float : np.ndarray
        Float ambiguity estimates
    L : np.ndarray
        Lower triangular matrix from LDL decomposition
    D : np.ndarray
        Diagonal elements from LDL decomposition
    level : int
        Current recursion level (0 = start)
    chi2 : float
        Accumulated chi-square value
    a_int : np.ndarray
        Current integer candidate being constructed
    candidates : List[np.ndarray]
        List to store found integer candidates
    residuals : List[float]
        List to store residuals for each candidate
    chi2_max : float
        Maximum allowed chi-square value
    ncands : int
        Maximum number of candidates to keep
    left : np.ndarray
        Left bounds for search intervals
    right : np.ndarray
        Right bounds for search intervals

    Notes
    -----
    This function implements the core integer search of the LAMBDA method:
    1. Computes conditional mean for current ambiguity
    2. Determines search bounds based on chi-square constraint
    3. Recursively searches integer values within bounds
    4. Maintains list of best candidates found so far

    The search works from the last ambiguity (level n-1) down to the
    first (level 0), building up the integer solution incrementally.

    The chi-square value tracks the quadratic form:
    chi2 = (a_int - a_float)' * Q^(-1) * (a_int - a_float)
    where Q is the ambiguity covariance matrix.
    """
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
    """
    Approximation of the normal cumulative distribution function.

    Uses the Abramowitz and Stegun approximation for computing
    the CDF of the standard normal distribution.

    Parameters
    ----------
    x : float
        Input value

    Returns
    -------
    float
        Approximate value of Φ(x) = P(Z ≤ x) where Z ~ N(0,1)

    Notes
    -----
    This function implements the Abramowitz and Stegun approximation:
    Φ(x) ≈ 1 - φ(x) * (a1*t + a2*t² + a3*t³ + a4*t⁴ + a5*t⁵)
    where t = 1/(1 + p*x) and φ(x) is the standard normal PDF.

    The approximation has a maximum error of 7.5 × 10^(-8).

    For negative values, uses the symmetry property:
    Φ(-x) = 1 - Φ(x)

    References
    ----------
    Abramowitz, M. and Stegun, I. A. (1964). Handbook of Mathematical
    Functions. Dover Publications.
    """
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


def ratio_test(residuals: list[float], threshold: float = 3.0) -> tuple[bool, float]:
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
