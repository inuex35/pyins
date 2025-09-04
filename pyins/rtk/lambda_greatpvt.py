#!/usr/bin/env python3
"""
Multi-Frequency LAMBDA Implementation Based on GREAT-PVT
=========================================================

This module implements the LAMBDA method with multi-frequency support,
following GREAT-PVT's approach with EWL->WL->NL cascaded resolution.

Based on GREAT-PVT's implementation with multi-frequency enhancements.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Union
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class FrequencyType(Enum):
    """Frequency/combination types"""
    L1 = "L1"
    L2 = "L2" 
    L5 = "L5"
    WL = "WL"    # Wide-Lane (L1-L2)
    NL = "NL"    # Narrow-Lane ((L1+L2)/2)
    EWL = "EWL"  # Extra-Wide-Lane (L2-L5)
    IF = "IF"    # Ionosphere-Free


@dataclass
class MultiFreqData:
    """Container for multi-frequency ambiguity data"""
    float_ambiguities: Dict[FrequencyType, np.ndarray]
    covariances: Dict[FrequencyType, np.ndarray]
    wavelengths: Dict[FrequencyType, float]
    satellite_ids: Optional[List[str]] = None
    elevations: Optional[np.ndarray] = None


class GreatPVTLambdaResolver:
    """
    Multi-frequency LAMBDA resolver using GREAT-PVT's simplified approach
    
    This implementation supports:
    - Multi-frequency cascaded resolution (EWL->WL->NL)
    - Multi-GNSS support
    - Automatic frequency detection
    - Backwards compatibility with single-frequency
    """
    
    def __init__(self, 
                 ratio_threshold: float = 3.0,
                 max_candidates: int = 2,
                 elevation_threshold: float = 15.0,
                 min_satellites: int = 4,
                 max_deviation: float = 0.25,
                 use_satellite_selection: bool = False,
                 # Multi-frequency specific parameters
                 wl_threshold: float = 0.25,  # cycles
                 nl_threshold: float = 0.15,  # cycles
                 ewl_threshold: float = 0.30,  # cycles
                 enable_multifreq: bool = True):
        """
        Initialize multi-frequency GREAT-PVT LAMBDA resolver
        
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
        use_satellite_selection : bool
            Enable quality-based satellite selection
        wl_threshold : float
            Threshold for WL ambiguity decision (cycles)
        nl_threshold : float
            Threshold for NL ambiguity decision (cycles)
        ewl_threshold : float
            Threshold for EWL ambiguity decision (cycles)
        enable_multifreq : bool
            Enable multi-frequency processing
        """
        self.ratio_threshold = ratio_threshold
        self.max_candidates = max_candidates
        self.elevation_threshold = elevation_threshold
        self.min_satellites = min_satellites
        self.max_deviation = max_deviation
        self.use_satellite_selection = use_satellite_selection
        
        # Multi-frequency parameters
        self.wl_threshold = wl_threshold
        self.nl_threshold = nl_threshold
        self.ewl_threshold = ewl_threshold
        self.enable_multifreq = enable_multifreq
        
        # Store fixed ambiguities for cascaded resolution
        self.fixed_wl: Dict[str, int] = {}
        self.fixed_nl: Dict[str, int] = {}
        self.fixed_ewl: Dict[str, int] = {}
        
        # GPS L1/L2/L5 wavelengths (meters)
        self.wavelengths = {
            FrequencyType.L1: 0.19029367,
            FrequencyType.L2: 0.24421021,
            FrequencyType.L5: 0.25482064,
            FrequencyType.WL: 0.86190382,  # L1-L2
            FrequencyType.NL: 0.10695739,  # (L1+L2)/2
            FrequencyType.EWL: 5.861,       # L2-L5
        }
        
        # Import selection resolver if enabled
        if use_satellite_selection:
            try:
                from .lambda_greatpvt_with_selection import GreatPVTWithSelection
                self.selection_resolver = GreatPVTWithSelection(
                    ratio_threshold=ratio_threshold,
                    max_candidates=max_candidates,
                    elevation_threshold=elevation_threshold,
                    min_satellites=min_satellites
                )
            except ImportError:
                logger.warning("Selection resolver not available")
                self.selection_resolver = None
        else:
            self.selection_resolver = None
    
    def resolve(self, 
                float_ambiguities: Union[np.ndarray, Dict[str, np.ndarray]], 
                covariance: Union[np.ndarray, Dict[str, np.ndarray]],
                elevations: Optional[np.ndarray] = None,
                satellite_ids: Optional[List[str]] = None,
                frequency_info: Optional[Union[str, Dict]] = None) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Resolve integer ambiguities with multi-frequency support
        
        Parameters
        ----------
        float_ambiguities : np.ndarray or dict
            Float ambiguity estimates
            - np.ndarray: Single frequency (backward compatible)
            - dict: Multi-frequency {'L1': array, 'L2': array, ...}
        covariance : np.ndarray or dict
            Covariance matrix of float ambiguities
        elevations : np.ndarray, optional
            Satellite elevation angles in degrees
        satellite_ids : List[str], optional
            Satellite identifiers for quality-based selection
        frequency_info : str or dict, optional
            Frequency information
            - str: 'L1', 'L2', 'L1L2', etc.
            - dict: Detailed frequency mapping
            
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
        
        # Check if multi-frequency data is provided
        if isinstance(float_ambiguities, dict) and self.enable_multifreq:
            return self._resolve_multifreq(
                float_ambiguities, covariance, elevations, satellite_ids
            )
        
        # Single-frequency processing (backward compatible)
        if isinstance(float_ambiguities, np.ndarray):
            return self._resolve_single_freq(
                float_ambiguities, covariance, elevations, satellite_ids
            )
        
        # If dict but multifreq disabled, use L1 only
        if isinstance(float_ambiguities, dict):
            if 'L1' in float_ambiguities:
                return self._resolve_single_freq(
                    float_ambiguities['L1'], 
                    covariance['L1'] if isinstance(covariance, dict) else covariance,
                    elevations, satellite_ids
                )
            else:
                # Use first available frequency
                freq = list(float_ambiguities.keys())[0]
                return self._resolve_single_freq(
                    float_ambiguities[freq],
                    covariance[freq] if isinstance(covariance, dict) else covariance,
                    elevations, satellite_ids
                )
    
    def _resolve_multifreq(self, 
                          float_amb_dict: Dict[str, np.ndarray],
                          cov_dict: Dict[str, np.ndarray],
                          elevations: Optional[np.ndarray],
                          satellite_ids: Optional[List[str]]) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Multi-frequency cascaded ambiguity resolution
        """
        info = {
            'method': 'GREAT-PVT-MultiFreq',
            'frequencies_available': list(float_amb_dict.keys()),
            'cascaded_resolution': True
        }
        
        n = len(next(iter(float_amb_dict.values())))
        fixed_ambiguities = np.zeros(n)
        
        # Step 1: Resolve EWL if L2 and L5 are available
        if 'L2' in float_amb_dict and 'L5' in float_amb_dict:
            ewl_float = self._form_ewl(float_amb_dict['L2'], float_amb_dict['L5'])
            ewl_cov = self._combine_covariance(cov_dict.get('L2'), cov_dict.get('L5'))
            
            ewl_fixed = self._resolve_ewl(ewl_float, ewl_cov, satellite_ids)
            info['ewl_fixed'] = len(ewl_fixed)
            info['ewl_success_rate'] = len(ewl_fixed) / n if n > 0 else 0
        
        # Step 2: Resolve WL if L1 and L2 are available
        if 'L1' in float_amb_dict and 'L2' in float_amb_dict:
            wl_float = self._form_wl(float_amb_dict['L1'], float_amb_dict['L2'])
            wl_cov = self._combine_covariance(cov_dict.get('L1'), cov_dict.get('L2'))
            
            wl_fixed = self._resolve_wl(wl_float, wl_cov, satellite_ids)
            info['wl_fixed'] = len(wl_fixed)
            info['wl_success_rate'] = len(wl_fixed) / n if n > 0 else 0
            
            # Step 3: Resolve NL using WL constraint
            if len(wl_fixed) >= self.min_satellites:
                nl_float = self._form_nl(float_amb_dict['L1'], float_amb_dict['L2'])
                nl_cov = self._combine_covariance(cov_dict.get('L1'), cov_dict.get('L2'))
                
                nl_fixed, ratio = self._resolve_nl_with_wl(
                    nl_float, nl_cov, wl_fixed, satellite_ids
                )
                
                # Recover L1 and L2 from WL and NL
                if len(nl_fixed) >= self.min_satellites:
                    for i, sat in enumerate(satellite_ids or range(n)):
                        sat_key = str(sat)
                        if sat_key in wl_fixed and sat_key in nl_fixed:
                            # L1 = NL + WL/2, L2 = NL - WL/2
                            fixed_ambiguities[i] = nl_fixed[sat_key] + wl_fixed[sat_key] / 2.0
                    
                    is_fixed = True
                    info['nl_fixed'] = len(nl_fixed)
                    info['ratio'] = ratio
                else:
                    is_fixed = False
                    ratio = 0.0
            else:
                is_fixed = False
                ratio = 0.0
        else:
            # Fallback to single frequency if only L1 available
            if 'L1' in float_amb_dict:
                return self._resolve_single_freq(
                    float_amb_dict['L1'], 
                    cov_dict.get('L1', np.eye(n)),
                    elevations, satellite_ids
                )
            is_fixed = False
            ratio = 0.0
        
        return fixed_ambiguities, ratio, is_fixed, info
    
    def _resolve_single_freq(self, float_ambiguities: np.ndarray, 
                            covariance: np.ndarray,
                            elevations: Optional[np.ndarray],
                            satellite_ids: Optional[List[str]]) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Original single-frequency resolution (backward compatible)
        """
        n = len(float_ambiguities)
        info = {
            'n_ambiguities': n,
            'method': 'GREAT-PVT-SingleFreq',
            'decorrelated': False,
            'validated': False
        }
        
        # Use satellite selection if enabled
        if self.use_satellite_selection and self.selection_resolver and satellite_ids is not None:
            return self.selection_resolver.resolve(
                float_ambiguities, covariance, elevations, satellite_ids)
        
        # Apply elevation mask if provided
        if elevations is not None and n > self.min_satellites:
            mask, subset_indices = self._apply_elevation_mask(elevations)
            if len(subset_indices) >= self.min_satellites:
                float_subset = float_ambiguities[subset_indices]
                cov_subset = covariance[np.ix_(subset_indices, subset_indices)]
                info['subset_size'] = len(subset_indices)
                info['elevation_filtered'] = True
            else:
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
        
        # Ratio test
        ratio = np.sqrt(residuals[1] / residuals[0]) if residuals[0] > 0 else 0
        info['ratio'] = ratio
        
        # Check if ratio test passes
        is_fixed = ratio > self.ratio_threshold
        info['ratio_passed'] = is_fixed
        
        if is_fixed:
            fixed_subset = np.round(fixed_candidates[0]).astype(int)
            info['validated'] = True
            
            # Build full solution
            fixed_ambiguities = float_ambiguities.copy()
            fixed_ambiguities[subset_indices] = fixed_subset
            
            logger.info(f"Ambiguities fixed with ratio: {ratio:.2f}")
            return fixed_ambiguities.astype(int), ratio, True, info
        else:
            logger.debug(f"Ratio test failed: {ratio:.2f} < {self.ratio_threshold}")
            return float_ambiguities, ratio, False, info
    
    def _form_wl(self, l1_amb: np.ndarray, l2_amb: np.ndarray) -> np.ndarray:
        """Form Wide-Lane combination (L1 - L2)"""
        return l1_amb - l2_amb
    
    def _form_nl(self, l1_amb: np.ndarray, l2_amb: np.ndarray) -> np.ndarray:
        """Form Narrow-Lane combination ((L1 + L2) / 2)"""
        return (l1_amb + l2_amb) / 2.0
    
    def _form_ewl(self, l2_amb: np.ndarray, l5_amb: np.ndarray) -> np.ndarray:
        """Form Extra-Wide-Lane combination (L2 - L5)"""
        return l2_amb - l5_amb
    
    def _combine_covariance(self, cov1: Optional[np.ndarray], 
                           cov2: Optional[np.ndarray]) -> np.ndarray:
        """Combine covariances for linear combinations"""
        if cov1 is None or cov2 is None:
            n = len(cov1) if cov1 is not None else len(cov2)
            return np.eye(n) * 0.01  # Default small covariance
        # For difference: Cov(L1-L2) = Cov(L1) + Cov(L2) - 2*Cov(L1,L2)
        # Simplified assuming independence
        return cov1 + cov2
    
    def _resolve_ewl(self, ewl_float: np.ndarray, ewl_cov: np.ndarray,
                    satellite_ids: Optional[List[str]]) -> Dict[str, int]:
        """Resolve EWL ambiguities (easiest with ~5.86m wavelength)"""
        fixed = {}
        ewl_sigma = np.sqrt(np.diag(ewl_cov))
        
        for i, sat in enumerate(satellite_ids or range(len(ewl_float))):
            if ewl_sigma[i] < self.ewl_threshold:
                fixed_val = round(ewl_float[i])
                residual = abs(ewl_float[i] - fixed_val)
                if residual < self.ewl_threshold:
                    fixed[str(sat)] = fixed_val
                    self.fixed_ewl[str(sat)] = fixed_val
        
        return fixed
    
    def _resolve_wl(self, wl_float: np.ndarray, wl_cov: np.ndarray,
                   satellite_ids: Optional[List[str]]) -> Dict[str, int]:
        """Resolve WL ambiguities (easier with ~86cm wavelength)"""
        fixed = {}
        wl_sigma = np.sqrt(np.diag(wl_cov))
        
        for i, sat in enumerate(satellite_ids or range(len(wl_float))):
            sat_key = str(sat)
            
            # Use EWL to constrain if available
            constraint = 0.0
            if sat_key in self.fixed_ewl:
                constraint = self.fixed_ewl[sat_key] * 0.1  # Scaling factor
            
            adjusted_float = wl_float[i] - constraint
            
            if wl_sigma[i] < self.wl_threshold:
                fixed_val = round(adjusted_float)
                residual = abs(adjusted_float - fixed_val)
                if residual < self.wl_threshold:
                    fixed[sat_key] = fixed_val
                    self.fixed_wl[sat_key] = fixed_val
        
        return fixed
    
    def _resolve_nl_with_wl(self, nl_float: np.ndarray, nl_cov: np.ndarray,
                           wl_fixed: Dict[str, int],
                           satellite_ids: Optional[List[str]]) -> Tuple[Dict[str, int], float]:
        """Resolve NL using WL constraint and LAMBDA"""
        # Filter to satellites with fixed WL
        indices = []
        for i, sat in enumerate(satellite_ids or range(len(nl_float))):
            if str(sat) in wl_fixed:
                indices.append(i)
        
        if len(indices) < self.min_satellites:
            return {}, 0.0
        
        indices = np.array(indices)
        nl_subset = nl_float[indices]
        cov_subset = nl_cov[np.ix_(indices, indices)]
        
        # Apply WL constraint to NL
        for i, idx in enumerate(indices):
            sat_key = str(satellite_ids[idx] if satellite_ids else idx)
            if sat_key in wl_fixed:
                # NL is constrained by WL
                nl_subset[i] -= wl_fixed[sat_key] * 0.05  # Scaling factor
        
        # Use LAMBDA for final NL resolution
        Z, L, D = self._decorrelate(cov_subset)
        float_decorr = Z.T @ nl_subset
        candidates, residuals = self._integer_search(float_decorr, L, D)
        
        if len(candidates) < 2:
            return {}, 0.0
        
        # Ratio test
        ratio = np.sqrt(residuals[1] / residuals[0]) if residuals[0] > 0 else 0
        
        if ratio > self.ratio_threshold:
            fixed_nl = {}
            fixed_candidate = Z @ candidates[0]
            for i, idx in enumerate(indices):
                sat_key = str(satellite_ids[idx] if satellite_ids else idx)
                fixed_nl[sat_key] = round(fixed_candidate[i])
                self.fixed_nl[sat_key] = fixed_nl[sat_key]
            return fixed_nl, ratio
        
        return {}, ratio
    
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
        
        # Sort by residual
        if len(candidates) >= 2:
            sorted_indices = np.argsort(residuals[:2])
            candidates = [candidates[i] for i in sorted_indices]
            residuals = [residuals[i] for i in sorted_indices]
        
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