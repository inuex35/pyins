"""Ambiguity resolution for RTK using LAMBDA method"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from scipy.linalg import cholesky, solve_triangular
from ..utils.ambiguity import lambda_search, lambda_reduction


class AmbiguityResolver:
    """Integer ambiguity resolution using LAMBDA method"""
    
    def __init__(self, 
                 ratio_threshold: float = 3.0,
                 success_rate_threshold: float = 0.999):
        """
        Initialize ambiguity resolver
        
        Parameters:
        -----------
        ratio_threshold : float
            Minimum ratio test threshold
        success_rate_threshold : float
            Minimum success rate for fixing
        """
        self.ratio_threshold = ratio_threshold
        self.success_rate_threshold = success_rate_threshold
        
    def resolve_ambiguities(self, 
                           float_ambiguities: np.ndarray,
                           covariance_matrix: np.ndarray) -> Tuple[Optional[np.ndarray], float, bool]:
        """
        Resolve integer ambiguities using LAMBDA method
        
        Parameters:
        -----------
        float_ambiguities : np.ndarray
            Float ambiguity estimates
        covariance_matrix : np.ndarray
            Ambiguity covariance matrix
            
        Returns:
        --------
        fixed_ambiguities : np.ndarray or None
            Fixed integer ambiguities (None if fixing failed)
        ratio : float
            Ratio test value
        success : bool
            Whether ambiguity fixing was successful
        """
        n = len(float_ambiguities)
        if n == 0:
            return None, 0.0, False
            
        try:
            # LAMBDA reduction
            Z, L, D = lambda_reduction(covariance_matrix)
            
            # Transform float solution
            transformed_float = Z.T @ float_ambiguities
            
            # Integer search
            candidates, residuals = lambda_search(transformed_float, L, D, ncands=2)
            
            if len(candidates) < 2:
                return None, 0.0, False
                
            # Ratio test
            ratio = residuals[1] / residuals[0] if residuals[0] > 0 else 0.0
            
            # Success rate estimation
            success_rate = self._estimate_success_rate(residuals[0], residuals[1], n)
            
            # Check fixing criteria
            fixing_success = (ratio >= self.ratio_threshold and 
                            success_rate >= self.success_rate_threshold)
            
            if fixing_success:
                # Transform back to original space
                fixed_ambiguities = Z @ candidates[0]
                return fixed_ambiguities.astype(int), ratio, True
            else:
                return None, ratio, False
                
        except Exception as e:
            print(f"Ambiguity resolution failed: {e}")
            return None, 0.0, False
            
    def _estimate_success_rate(self, 
                             min_residual: float, 
                             second_residual: float, 
                             n_amb: int) -> float:
        """
        Estimate success rate using bootstrapping approximation
        
        Parameters:
        -----------
        min_residual : float
            Minimum residual
        second_residual : float
            Second minimum residual
        n_amb : int
            Number of ambiguities
            
        Returns:
        --------
        success_rate : float
            Estimated success rate
        """
        if min_residual <= 0 or second_residual <= min_residual:
            return 0.0
            
        # Simplified bootstrapping success rate
        # Based on ratio of residuals and number of ambiguities
        ratio = second_residual / min_residual
        
        # Empirical formula (simplified)
        if ratio < 1.5:
            return 0.1
        elif ratio < 2.0:
            return 0.5
        elif ratio < 3.0:
            return 0.9
        else:
            return min(0.999, 1.0 - np.exp(-ratio / 2.0))
            
    def partial_ambiguity_resolution(self, 
                                   float_ambiguities: np.ndarray,
                                   covariance_matrix: np.ndarray,
                                   min_subset_size: int = 2) -> Tuple[Optional[np.ndarray], np.ndarray, float]:
        """
        Partial ambiguity resolution - fix subset of most reliable ambiguities
        
        Parameters:
        -----------
        float_ambiguities : np.ndarray
            Float ambiguity estimates
        covariance_matrix : np.ndarray
            Ambiguity covariance matrix
        min_subset_size : int
            Minimum subset size to attempt fixing
            
        Returns:
        --------
        fixed_subset : np.ndarray or None
            Fixed ambiguities for selected subset
        subset_indices : np.ndarray
            Indices of selected subset
        ratio : float
            Ratio test value for subset
        """
        n = len(float_ambiguities)
        if n < min_subset_size:
            return None, np.array([]), 0.0
            
        best_ratio = 0.0
        best_fixed = None
        best_indices = np.array([])
        
        # Try different subset sizes, starting from largest
        for subset_size in range(n, min_subset_size - 1, -1):
            # Select subset based on diagonal covariance (most precise)
            diag_cov = np.diag(covariance_matrix)
            indices = np.argsort(diag_cov)[:subset_size]
            
            # Extract subset
            subset_float = float_ambiguities[indices]
            subset_cov = covariance_matrix[np.ix_(indices, indices)]
            
            # Try to resolve subset
            fixed_subset, ratio, success = self.resolve_ambiguities(subset_float, subset_cov)
            
            if success and ratio > best_ratio:
                best_ratio = ratio
                best_fixed = fixed_subset
                best_indices = indices
                break  # Use largest successful subset
                
        return best_fixed, best_indices, best_ratio


class WidelaneAmbiguityResolver:
    """Widelane ambiguity resolution for dual-frequency RTK"""
    
    def __init__(self):
        self.l1_freq = 1575.42e6
        self.l2_freq = 1227.60e6
        self.wl_wavelength = 299792458.0 / (self.l1_freq - self.l2_freq)  # ~86 cm
        
    def resolve_widelane(self, 
                        l1_observations: np.ndarray,
                        l2_observations: np.ndarray,
                        l1_pseudorange: np.ndarray,
                        l2_pseudorange: np.ndarray) -> np.ndarray:
        """
        Resolve widelane ambiguities using Melbourne-Wubbena combination
        
        Parameters:
        -----------
        l1_observations : np.ndarray
            L1 carrier phase observations (cycles)
        l2_observations : np.ndarray
            L2 carrier phase observations (cycles)
        l1_pseudorange : np.ndarray
            L1 pseudorange observations (m)
        l2_pseudorange : np.ndarray
            L2 pseudorange observations (m)
            
        Returns:
        --------
        widelane_ambiguities : np.ndarray
            Resolved widelane ambiguities
        """
        l1_wavelength = 299792458.0 / self.l1_freq
        l2_wavelength = 299792458.0 / self.l2_freq
        
        # Melbourne-Wubbena combination
        mw_combination = (l1_observations - l2_observations) - (
            (self.l1_freq * l1_pseudorange - self.l2_freq * l2_pseudorange) /
            (299792458.0 * (self.l1_freq - self.l2_freq))
        )
        
        # Round to nearest integer
        widelane_ambiguities = np.round(mw_combination)
        
        return widelane_ambiguities
        
    def constrain_l1_ambiguities(self, 
                               l1_float: np.ndarray,
                               l2_float: np.ndarray,
                               widelane_fixed: np.ndarray) -> np.ndarray:
        """
        Constrain L1 ambiguities using fixed widelane ambiguities
        
        Parameters:
        -----------
        l1_float : np.ndarray
            L1 float ambiguities
        l2_float : np.ndarray
            L2 float ambiguities
        widelane_fixed : np.ndarray
            Fixed widelane ambiguities
            
        Returns:
        --------
        l2_constrained : np.ndarray
            L2 ambiguities constrained by widelane
        """
        # L2 = L1 - widelane
        l2_constrained = l1_float - widelane_fixed
        
        return l2_constrained


class RTKAmbiguityManager:
    """Manage ambiguities for RTK processing"""
    
    def __init__(self):
        self.resolver = AmbiguityResolver()
        self.wl_resolver = WidelaneAmbiguityResolver()
        self.fixed_ambiguities = {}  # sat -> ambiguity
        self.ambiguity_history = {}  # sat -> list of estimates
        self.fix_status = {}  # sat -> bool
        
    def update_ambiguities(self, 
                          satellites: List[int],
                          float_estimates: np.ndarray,
                          covariance: np.ndarray) -> Tuple[Dict[int, int], Dict[int, bool]]:
        """
        Update and resolve ambiguities
        
        Parameters:
        -----------
        satellites : List[int]
            Satellite numbers
        float_estimates : np.ndarray
            Float ambiguity estimates
        covariance : np.ndarray
            Ambiguity covariance matrix
            
        Returns:
        --------
        ambiguities : Dict[int, int]
            Fixed ambiguities by satellite
        fix_status : Dict[int, bool]
            Fix status by satellite
        """
        # Try full resolution first
        fixed, ratio, success = self.resolver.resolve_ambiguities(
            float_estimates, covariance)
        
        if success:
            # Update all ambiguities
            for i, sat in enumerate(satellites):
                self.fixed_ambiguities[sat] = int(fixed[i])
                self.fix_status[sat] = True
        else:
            # Try partial resolution
            fixed_subset, indices, ratio = self.resolver.partial_ambiguity_resolution(
                float_estimates, covariance)
            
            if fixed_subset is not None:
                # Update subset
                for i, idx in enumerate(indices):
                    sat = satellites[idx]
                    self.fixed_ambiguities[sat] = int(fixed_subset[i])
                    self.fix_status[sat] = True
                    
                # Mark others as float
                for i, sat in enumerate(satellites):
                    if i not in indices:
                        self.fix_status[sat] = False
            else:
                # All remain float
                for sat in satellites:
                    self.fix_status[sat] = False
                    
        return self.fixed_ambiguities.copy(), self.fix_status.copy()
        
    def validate_ambiguities(self, 
                           satellites: List[int],
                           residuals: np.ndarray,
                           threshold: float = 0.1) -> Dict[int, bool]:
        """
        Validate fixed ambiguities using residuals
        
        Parameters:
        -----------
        satellites : List[int]
            Satellite numbers
        residuals : np.ndarray
            Post-fix residuals
        threshold : float
            Residual threshold for validation
            
        Returns:
        --------
        validation_status : Dict[int, bool]
            Validation status by satellite
        """
        validation_status = {}
        
        for i, sat in enumerate(satellites):
            if i < len(residuals):
                validation_status[sat] = abs(residuals[i]) < threshold
            else:
                validation_status[sat] = False
                
        return validation_status