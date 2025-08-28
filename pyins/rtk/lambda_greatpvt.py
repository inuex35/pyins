#!/usr/bin/env python3
"""
LAMBDA Implementation Based on GREAT-PVT
=========================================

This module implements the LAMBDA method following GREAT-PVT's approach,
which emphasizes simplicity and practical robustness over theoretical completeness.

Based on GREAT-PVT's implementation with enhancements for pyins.
"""

import numpy as np
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class GreatPVTLambdaResolver:
    """
    LAMBDA resolver using GREAT-PVT's simplified approach
    
    This implementation prioritizes:
    - Computational efficiency
    - Practical robustness
    - Ease of understanding
    """
    
    def __init__(self, ratio_threshold: float = 3.0,
                 max_candidates: int = 2,
                 elevation_threshold: float = 15.0,
                 min_satellites: int = 4,
                 max_deviation: float = 0.25):
        """
        Initialize GREAT-PVT LAMBDA resolver
        
        Parameters
        ----------
        ratio_threshold : float
            Minimum ratio test threshold for validation
        max_candidates : int
            Number of integer candidates to search
        elevation_threshold : float
            Minimum elevation angle in degrees for partial AR
        min_satellites : int
            Minimum number of satellites to fix
        max_deviation : float
            Maximum allowed deviation in cycles for validation
        """
        self.ratio_threshold = ratio_threshold
        self.max_candidates = max_candidates
        self.elevation_threshold = elevation_threshold
        self.min_satellites = min_satellites
        self.max_deviation = max_deviation
        
    def resolve(self, float_ambiguities: np.ndarray, 
                covariance: np.ndarray,
                elevations: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Resolve integer ambiguities using GREAT-PVT approach
        
        Parameters
        ----------
        float_ambiguities : np.ndarray
            Float ambiguity estimates (n,)
        covariance : np.ndarray
            Covariance matrix of float ambiguities (n, n)
        elevations : np.ndarray, optional
            Satellite elevation angles in degrees
            
        Returns
        -------
        fixed_ambiguities : np.ndarray
            Integer ambiguity solution
        ratio : float
            Ratio test value (sqrt(second_best / best))
        is_fixed : bool
            Whether ambiguities were successfully fixed
        info : dict
            Additional information about resolution
        """
        n = len(float_ambiguities)
        info = {
            'n_ambiguities': n,
            'method': 'GREAT-PVT',
            'decorrelated': False,
            'validated': False
        }
        
        # Apply elevation mask if provided
        if elevations is not None and n > self.min_satellites:
            mask, subset_indices = self._apply_elevation_mask(elevations)
            if len(subset_indices) >= self.min_satellites:
                # Work with high-elevation subset
                float_subset = float_ambiguities[subset_indices]
                cov_subset = covariance[np.ix_(subset_indices, subset_indices)]
                info['subset_size'] = len(subset_indices)
                info['elevation_filtered'] = True
            else:
                # Not enough high-elevation satellites
                float_subset = float_ambiguities
                cov_subset = covariance
                info['elevation_filtered'] = False
        else:
            float_subset = float_ambiguities
            cov_subset = covariance
            subset_indices = np.arange(n)
        
        # Decorrelate ambiguities
        Z, L, D = self._decorrelate(cov_subset)
        info['decorrelated'] = True
        
        # Transform float ambiguities
        float_decorr = Z.T @ float_subset
        
        # Integer search
        candidates, residuals = self._integer_search(float_decorr, L, D)
        
        if len(candidates) < 2:
            logger.warning("Not enough candidates for ratio test")
            return float_ambiguities, 0.0, False, info
        
        # Transform back to original space
        fixed_candidates = []
        for candidate in candidates:
            fixed_candidates.append(Z @ candidate)
        
        # Ratio test (GREAT-PVT style with square root)
        ratio = np.sqrt(residuals[1] / residuals[0]) if residuals[0] > 0 else 0
        info['ratio'] = ratio
        
        # Check if ratio test passes
        is_fixed = ratio >= self.ratio_threshold
        info['ratio_passed'] = is_fixed
        
        if is_fixed:
            # Validate solution
            fixed_subset = np.round(fixed_candidates[0]).astype(int)
            is_valid = self._validate_solution(fixed_subset, float_subset, cov_subset)
            info['validated'] = is_valid
            
            if is_valid:
                # Build full solution
                fixed_ambiguities = float_ambiguities.copy()
                fixed_ambiguities[subset_indices] = fixed_subset
                
                logger.info(f"Ambiguities fixed with ratio: {ratio:.2f}")
                return fixed_ambiguities.astype(int), ratio, True, info
            else:
                logger.warning("Fixed solution failed validation")
                return float_ambiguities, ratio, False, info
        else:
            logger.debug(f"Ratio test failed: {ratio:.2f} < {self.ratio_threshold}")
            return float_ambiguities, ratio, False, info
    
    def _apply_elevation_mask(self, elevations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply elevation mask to select high-elevation satellites"""
        mask = elevations >= self.elevation_threshold
        indices = np.where(mask)[0]
        return mask, indices
    
    def _decorrelate(self, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Decorrelate covariance matrix using GREAT-PVT's simplified approach
        
        Returns Z transformation matrix, L and D from Q = Z * L * D * L' * Z'
        """
        n = Q.shape[0]
        Z = np.eye(n)
        Q_work = Q.copy()
        
        # Simple decorrelation with integer rounding
        for i in range(n - 1):
            for j in range(i + 1, n):
                if Q_work[i, i] > 1e-10:  # Avoid division by small numbers
                    # Compute decorrelation factor
                    mu = np.round(Q_work[i, j] / Q_work[i, i])
                    if mu != 0:
                        # Update transformation matrix
                        Z[j, :] -= mu * Z[i, :]
                        # Update covariance
                        Q_work[j, :] -= mu * Q_work[i, :]
                        Q_work[:, j] -= mu * Q_work[:, i]
        
        # Extract L and D (simplified)
        D = np.diag(np.diag(Q_work))
        L = np.eye(n)
        for i in range(n):
            for j in range(i):
                if Q_work[j, j] > 1e-10:
                    L[i, j] = Q_work[i, j] / Q_work[j, j]
        
        return Z, L, D
    
    def _integer_search(self, float_amb: np.ndarray, L: np.ndarray, 
                       D: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        """
        Integer search using GREAT-PVT's simple approach
        
        Returns list of integer candidates and their residuals
        """
        n = len(float_amb)
        candidates = []
        residuals = []
        
        # First candidate: simple rounding
        candidate1 = np.round(float_amb)
        residual1 = self._compute_residual(candidate1, float_amb, D)
        candidates.append(candidate1)
        residuals.append(residual1)
        
        # Second candidate: try ±1 perturbations
        best_alt_residual = float('inf')
        best_alt_candidate = None
        
        for i in range(n):
            for delta in [-1, 1]:
                candidate = candidate1.copy()
                candidate[i] += delta
                residual = self._compute_residual(candidate, float_amb, D)
                
                # Check if this is better than current alternative
                if residual < best_alt_residual and not np.array_equal(candidate, candidate1):
                    best_alt_residual = residual
                    best_alt_candidate = candidate.copy()
        
        # Add second best if found
        if best_alt_candidate is not None:
            candidates.append(best_alt_candidate)
            residuals.append(best_alt_residual)
        
        # If requested more candidates, try combinations
        if self.max_candidates > 2 and n > 1:
            for i in range(n-1):
                for j in range(i+1, n):
                    for delta_i in [-1, 1]:
                        for delta_j in [-1, 1]:
                            candidate = candidate1.copy()
                            candidate[i] += delta_i
                            candidate[j] += delta_j
                            
                            # Check if unique
                            is_unique = True
                            for existing in candidates:
                                if np.array_equal(candidate, existing):
                                    is_unique = False
                                    break
                            
                            if is_unique:
                                residual = self._compute_residual(candidate, float_amb, D)
                                candidates.append(candidate)
                                residuals.append(residual)
                                
                                if len(candidates) >= self.max_candidates:
                                    break
                        if len(candidates) >= self.max_candidates:
                            break
                    if len(candidates) >= self.max_candidates:
                        break
                if len(candidates) >= self.max_candidates:
                    break
        
        # Sort by residual
        sorted_indices = np.argsort(residuals)
        candidates = [candidates[i] for i in sorted_indices[:self.max_candidates]]
        residuals = [residuals[i] for i in sorted_indices[:self.max_candidates]]
        
        return candidates, residuals
    
    def _compute_residual(self, integer_amb: np.ndarray, float_amb: np.ndarray,
                         D: np.ndarray) -> float:
        """
        Compute weighted residual for integer candidate
        
        Uses diagonal weighting from D matrix
        """
        diff = integer_amb - float_amb
        d_diag = np.diag(D)
        
        # Weighted sum of squared differences
        weighted_residual = 0.0
        for i in range(len(diff)):
            if d_diag[i] > 1e-10:  # Avoid division by small numbers
                weighted_residual += diff[i]**2 / d_diag[i]
        
        return weighted_residual
    
    def _validate_solution(self, fixed_amb: np.ndarray, float_amb: np.ndarray,
                          covariance: np.ndarray) -> bool:
        """
        Validate fixed solution using GREAT-PVT's criteria
        
        Checks both absolute and standardized deviations
        """
        # Compute differences
        diff = fixed_amb - float_amb
        
        # Check absolute deviation
        max_abs_diff = np.max(np.abs(diff))
        if max_abs_diff > self.max_deviation:
            logger.debug(f"Validation failed: max absolute diff {max_abs_diff:.3f} > {self.max_deviation}")
            return False
        
        # Check standardized residuals
        std_devs = np.sqrt(np.diag(covariance))
        valid_std = std_devs > 1e-9  # Avoid division by very small numbers
        
        if np.any(valid_std):
            standardized = np.abs(diff[valid_std]) / std_devs[valid_std]
            max_standardized = np.max(standardized)
            
            if max_standardized > 3.0:
                logger.debug(f"Validation failed: max standardized residual {max_standardized:.2f} > 3.0")
                return False
        
        return True
    
    def resolve_partial(self, float_ambiguities: np.ndarray,
                       covariance: np.ndarray,
                       elevations: np.ndarray,
                       satellite_ids: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray, float, bool]:
        """
        Partial ambiguity resolution with elevation-based selection
        
        This is the preferred method for GREAT-PVT approach
        
        Parameters
        ----------
        float_ambiguities : np.ndarray
            Float ambiguity estimates
        covariance : np.ndarray
            Covariance matrix
        elevations : np.ndarray
            Satellite elevation angles in degrees
        satellite_ids : List[int], optional
            Satellite IDs for logging
            
        Returns
        -------
        fixed_ambiguities : np.ndarray
            Fixed ambiguities (unchanged for unfixed)
        fixed_mask : np.ndarray
            Boolean mask of fixed ambiguities
        ratio : float
            Ratio test value
        success : bool
            Whether partial fixing succeeded
        """
        n = len(float_ambiguities)
        
        # Apply elevation mask
        mask, indices = self._apply_elevation_mask(elevations)
        n_high_elev = len(indices)
        
        if n_high_elev < self.min_satellites:
            logger.warning(f"Not enough high-elevation satellites: {n_high_elev} < {self.min_satellites}")
            return float_ambiguities, np.zeros(n, dtype=bool), 0.0, False
        
        # Log selected satellites
        if satellite_ids is not None:
            selected_sats = [satellite_ids[i] for i in indices]
            logger.debug(f"Selected {n_high_elev} high-elevation satellites: {selected_sats}")
        
        # Extract subset
        subset_float = float_ambiguities[indices]
        subset_cov = covariance[np.ix_(indices, indices)]
        
        # Resolve subset
        fixed_subset, ratio, is_fixed, info = self.resolve(subset_float, subset_cov)
        
        # Build full result
        fixed_ambiguities = float_ambiguities.copy()
        fixed_mask = np.zeros(n, dtype=bool)
        
        if is_fixed:
            # Only update fixed satellites
            fixed_ambiguities[indices] = fixed_subset[:n_high_elev]
            fixed_mask[indices] = True
            logger.info(f"Partially fixed {n_high_elev}/{n} ambiguities with ratio {ratio:.2f}")
        
        return fixed_ambiguities, fixed_mask, ratio, is_fixed