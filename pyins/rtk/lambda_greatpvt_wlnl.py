#!/usr/bin/env python3
"""
GREAT-PVT with Wide-Lane/Narrow-Lane Integration
================================================

This module extends GREAT-PVT's LAMBDA implementation with WL/NL
(Wide-Lane/Narrow-Lane) combination method for improved multi-frequency
ambiguity resolution.

The WL/NL approach provides:
- Better performance for long baselines
- Sequential fixing strategy (WL first, then NL)
- More robust ambiguity resolution for dual-frequency data
"""

import numpy as np
from typing import Tuple, Optional, Dict
import logging
from .lambda_greatpvt import GreatPVTLambdaResolver

logger = logging.getLogger(__name__)


class GreatPVTWLNLResolver(GreatPVTLambdaResolver):
    """
    GREAT-PVT LAMBDA resolver with Wide-Lane/Narrow-Lane support
    
    This class extends the basic GREAT-PVT resolver with WL/NL capabilities
    for dual-frequency GNSS data.
    """
    
    # GPS frequencies (Hz)
    FREQ_L1 = 1575.42e6
    FREQ_L2 = 1227.60e6
    CLIGHT = 299792458.0
    
    def __init__(self, 
                 ratio_threshold: float = 3.0,
                 wl_ratio_threshold: float = 2.0,
                 max_candidates: int = 2,
                 elevation_threshold: float = 15.0,
                 min_satellites: int = 4,
                 wl_round_threshold: float = 0.25,
                 use_wlnl: bool = True):
        """
        Initialize GREAT-PVT WL/NL resolver
        
        Parameters
        ----------
        ratio_threshold : float
            Ratio test threshold for narrow-lane/L1
        wl_ratio_threshold : float
            Ratio test threshold for wide-lane (can be lower)
        max_candidates : int
            Number of integer candidates to search
        elevation_threshold : float
            Minimum elevation angle in degrees
        min_satellites : int
            Minimum number of satellites to fix
        wl_round_threshold : float
            Maximum deviation from integer for WL rounding
        use_wlnl : bool
            Enable WL/NL processing
        """
        super().__init__(ratio_threshold, max_candidates, 
                        elevation_threshold, min_satellites)
        
        self.wl_ratio_threshold = wl_ratio_threshold
        self.wl_round_threshold = wl_round_threshold
        self.use_wlnl = use_wlnl
        
        # Compute wavelengths
        self.lambda_l1 = self.CLIGHT / self.FREQ_L1  # ~0.19m
        self.lambda_l2 = self.CLIGHT / self.FREQ_L2  # ~0.24m
        self.lambda_wl = self.CLIGHT / (self.FREQ_L1 - self.FREQ_L2)  # ~0.86m
        self.lambda_nl = self.CLIGHT / (self.FREQ_L1 + self.FREQ_L2)  # ~0.11m
        
        # Frequency ratios for combinations
        self.f1 = self.FREQ_L1 / (self.FREQ_L1 - self.FREQ_L2)  # L1 coeff for WL
        self.f2 = self.FREQ_L2 / (self.FREQ_L1 - self.FREQ_L2)  # L2 coeff for WL
        
        logger.info(f"WL/NL wavelengths: WL={self.lambda_wl:.3f}m, NL={self.lambda_nl:.3f}m")
    
    def resolve_wlnl(self,
                     l1_ambiguities: np.ndarray,
                     l2_ambiguities: np.ndarray,
                     covariance_l1l2: np.ndarray,
                     elevations: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, float, bool, Dict]:
        """
        Resolve ambiguities using WL/NL strategy
        
        Parameters
        ----------
        l1_ambiguities : np.ndarray
            L1 float ambiguities (cycles)
        l2_ambiguities : np.ndarray
            L2 float ambiguities (cycles)
        covariance_l1l2 : np.ndarray
            Combined covariance matrix for L1 and L2 (2n x 2n)
        elevations : np.ndarray, optional
            Satellite elevation angles
            
        Returns
        -------
        fixed_l1 : np.ndarray
            Fixed L1 ambiguities
        fixed_l2 : np.ndarray
            Fixed L2 ambiguities
        ratio : float
            Final ratio test value
        is_fixed : bool
            Whether ambiguities were successfully fixed
        info : dict
            Additional information
        """
        n_amb = len(l1_ambiguities)
        info = {
            'method': 'GREAT-PVT-WLNL',
            'n_ambiguities': n_amb,
            'wl_fixed': False,
            'nl_fixed': False
        }
        
        if not self.use_wlnl:
            # Fall back to standard GREAT-PVT for L1
            fixed_l1, ratio, is_fixed, sub_info = super().resolve(
                l1_ambiguities, covariance_l1l2[:n_amb, :n_amb], elevations)
            fixed_l2 = np.round(l2_ambiguities)  # Simple rounding for L2
            return fixed_l1, fixed_l2, ratio, is_fixed, info
        
        # Step 1: Form Wide-Lane combination
        # WL = L1 - L2 (in cycles)
        wl_float = l1_ambiguities - l2_ambiguities
        
        # WL covariance from error propagation
        # Cov(WL) = Cov(L1) + Cov(L2) - 2*Cov(L1,L2)
        cov_l1 = covariance_l1l2[:n_amb, :n_amb]
        cov_l2 = covariance_l1l2[n_amb:, n_amb:]
        cov_l1l2 = covariance_l1l2[:n_amb, n_amb:]
        
        cov_wl = cov_l1 + cov_l2 - 2 * cov_l1l2
        
        # Make sure covariance is positive definite
        eigvals = np.linalg.eigvalsh(cov_wl)
        if np.min(eigvals) < 1e-10:
            cov_wl += np.eye(n_amb) * (1e-9 - np.min(eigvals))
        
        # Step 2: Fix Wide-Lane ambiguities
        # Check if WL can be fixed by simple rounding (geometry-free)
        wl_deviations = np.abs(wl_float - np.round(wl_float))
        
        if np.max(wl_deviations) < self.wl_round_threshold:
            # Simple rounding for WL (works well due to long wavelength)
            wl_fixed = np.round(wl_float).astype(int)
            wl_ratio = 10.0  # High confidence
            wl_is_fixed = True
            info['wl_method'] = 'rounding'
            logger.debug(f"WL fixed by rounding, max deviation: {np.max(wl_deviations):.3f}")
        else:
            # Use GREAT-PVT LAMBDA for WL
            wl_fixed, wl_ratio, wl_is_fixed, wl_info = super().resolve(
                wl_float, cov_wl, elevations)
            info['wl_method'] = 'LAMBDA'
            
            # Use lower threshold for WL due to long wavelength
            if wl_ratio < self.wl_ratio_threshold:
                wl_is_fixed = False
        
        info['wl_fixed'] = wl_is_fixed
        info['wl_ratio'] = wl_ratio
        info['wl_deviations'] = wl_deviations
        
        if not wl_is_fixed:
            # If WL not fixed, fall back to L1-only fixing
            logger.debug("WL not fixed, falling back to L1-only")
            fixed_l1, ratio, is_fixed, sub_info = super().resolve(
                l1_ambiguities, cov_l1, elevations)
            fixed_l2 = np.round(l2_ambiguities)
            return fixed_l1, fixed_l2, ratio, is_fixed, info
        
        # Step 3: Use fixed WL to constrain L1 fixing
        # With WL fixed, we have: L1 - L2 = WL_fixed
        # So: L2 = L1 - WL_fixed
        
        # Update L1 covariance considering WL constraint
        # This effectively reduces the search space
        constrained_l1 = l1_ambiguities.copy()
        
        # Apply WL constraint as pseudo-observations
        # This reduces the effective noise in the L1 estimates
        wl_constraint_weight = 1.0 / np.mean(np.diag(cov_wl))
        
        # Modified covariance with WL constraint
        cov_l1_constrained = cov_l1.copy()
        
        # Reduce variance based on WL fixing confidence
        if wl_ratio > 5.0:  # Very confident WL fix
            variance_reduction = 0.5
        elif wl_ratio > 3.0:  # Confident WL fix
            variance_reduction = 0.7
        else:  # Marginal WL fix
            variance_reduction = 0.9
        
        cov_l1_constrained *= variance_reduction
        
        # Step 4: Fix L1 with WL constraint
        fixed_l1, nl_ratio, nl_is_fixed, nl_info = super().resolve(
            constrained_l1, cov_l1_constrained, elevations)
        
        info['nl_fixed'] = nl_is_fixed
        info['nl_ratio'] = nl_ratio
        
        # Step 5: Recover L2 from fixed WL and L1
        if nl_is_fixed:
            fixed_l2 = fixed_l1 - wl_fixed
            is_fixed = True
            ratio = min(wl_ratio, nl_ratio)  # Conservative ratio
            
            # Validation check
            l2_check = np.round(l2_ambiguities)
            if np.max(np.abs(fixed_l2 - l2_check)) > 2:
                logger.warning("L2 validation check failed, large deviation from float solution")
                # Keep the fix but note the issue
                info['l2_validation'] = 'warning'
        else:
            # L1 not fixed, but we have WL
            # Try alternative: fix L2 using WL
            l2_from_wl = l1_ambiguities - wl_fixed
            fixed_l2 = np.round(l2_from_wl).astype(int)
            fixed_l1 = fixed_l2 + wl_fixed
            
            # Check if this gives reasonable results
            l1_dev = np.max(np.abs(fixed_l1 - l1_ambiguities))
            l2_dev = np.max(np.abs(fixed_l2 - l2_ambiguities))
            
            if l1_dev < 0.5 and l2_dev < 0.5:
                is_fixed = True
                ratio = wl_ratio
                info['fixing_method'] = 'WL+L2'
            else:
                is_fixed = False
                ratio = 0.0
                fixed_l1 = l1_ambiguities
                fixed_l2 = l2_ambiguities
        
        # Log results
        if is_fixed:
            logger.info(f"WL/NL fixed: WL ratio={wl_ratio:.2f}, NL ratio={nl_ratio:.2f}")
        else:
            logger.debug(f"WL/NL failed: WL ratio={wl_ratio:.2f}, NL ratio={nl_ratio:.2f}")
        
        return fixed_l1, fixed_l2, ratio, is_fixed, info
    
    def resolve_partial_wlnl(self,
                            l1_ambiguities: np.ndarray,
                            l2_ambiguities: np.ndarray,
                            covariance_l1l2: np.ndarray,
                            elevations: np.ndarray,
                            max_deviation: float = 0.3) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
        """
        Partial ambiguity resolution with WL/NL
        
        Fix only the subset of ambiguities that have:
        1. High elevation angles
        2. Confident WL fixes
        3. Small deviations from integers
        
        Parameters
        ----------
        l1_ambiguities : np.ndarray
            L1 float ambiguities
        l2_ambiguities : np.ndarray
            L2 float ambiguities
        covariance_l1l2 : np.ndarray
            Combined covariance matrix
        elevations : np.ndarray
            Satellite elevation angles
        max_deviation : float
            Maximum allowed deviation from integer
            
        Returns
        -------
        fixed_l1 : np.ndarray
            Fixed L1 ambiguities (unchanged for unfixed)
        fixed_l2 : np.ndarray
            Fixed L2 ambiguities (unchanged for unfixed)
        fixed_mask : np.ndarray
            Boolean mask of fixed ambiguities
        ratio : float
            Ratio test value
        success : bool
            Whether partial fixing succeeded
        """
        n_amb = len(l1_ambiguities)
        
        # Initialize outputs
        fixed_l1 = l1_ambiguities.copy()
        fixed_l2 = l2_ambiguities.copy()
        fixed_mask = np.zeros(n_amb, dtype=bool)
        
        # Form WL combination
        wl_float = l1_ambiguities - l2_ambiguities
        wl_deviations = np.abs(wl_float - np.round(wl_float))
        
        # Select subset based on multiple criteria
        good_elevation = elevations > self.elevation_threshold
        good_wl = wl_deviations < max_deviation
        good_subset = good_elevation & good_wl
        
        subset_indices = np.where(good_subset)[0]
        n_subset = len(subset_indices)
        
        if n_subset < self.min_satellites:
            return fixed_l1, fixed_l2, fixed_mask, 0.0, False
        
        # Extract subset
        l1_subset = l1_ambiguities[subset_indices]
        l2_subset = l2_ambiguities[subset_indices]
        elev_subset = elevations[subset_indices]
        
        # Build subset covariance
        cov_subset = np.zeros((2*n_subset, 2*n_subset))
        for i in range(n_subset):
            for j in range(n_subset):
                ii, jj = subset_indices[i], subset_indices[j]
                # L1-L1
                cov_subset[i, j] = covariance_l1l2[ii, jj]
                # L2-L2
                cov_subset[n_subset+i, n_subset+j] = covariance_l1l2[n_amb+ii, n_amb+jj]
                # L1-L2
                cov_subset[i, n_subset+j] = covariance_l1l2[ii, n_amb+jj]
                cov_subset[n_subset+i, j] = covariance_l1l2[n_amb+ii, jj]
        
        # Resolve subset with WL/NL
        fixed_l1_subset, fixed_l2_subset, ratio, is_fixed, info = self.resolve_wlnl(
            l1_subset, l2_subset, cov_subset, elev_subset)
        
        if is_fixed:
            # Apply fixed values to subset
            fixed_l1[subset_indices] = fixed_l1_subset
            fixed_l2[subset_indices] = fixed_l2_subset
            fixed_mask[subset_indices] = True
            
            logger.info(f"Partial WL/NL fixed {n_subset}/{n_amb} ambiguities, ratio={ratio:.2f}")
        
        return fixed_l1, fixed_l2, fixed_mask, ratio, is_fixed