#!/usr/bin/env python3
"""
GREAT-PVT LAMBDA with Satellite Selection
==========================================

Enhanced GREAT-PVT implementation with quality-based satellite selection
for improved ambiguity resolution, especially for zero-baseline scenarios.

This module extends the basic GREAT-PVT with:
- DD quality-based satellite selection
- Multi-GNSS support with known good/bad satellite lists
- Adaptive threshold based on satellite quality
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
import logging
from ..utils.ambiguity import lambda_reduction, lambda_search

logger = logging.getLogger(__name__)


class GreatPVTWithSelection:
    """
    GREAT-PVT LAMBDA resolver with satellite quality-based selection
    
    This implementation includes satellite quality assessment to improve
    ambiguity resolution performance, particularly for zero-baseline and
    short-baseline scenarios.
    """
    
    # Disabled hardcoded satellite lists - they are dataset-specific
    # These were from kaiyodai zero-baseline analysis and don't apply to other datasets
    GOOD_SATELLITES = set()  # Empty - rely on dynamic metrics
    BAD_SATELLITES = set()   # Empty - rely on dynamic metrics
    
    # System-specific quality weights (based on zero-baseline analysis)
    SYSTEM_WEIGHTS = {
        'C': 1.5,  # BeiDou - best performance
        'J': 1.4,  # QZSS - excellent for Japan
        'G': 1.0,  # GPS - baseline
        'E': 0.9,  # Galileo
        'R': 0.8   # GLONASS
    }
    
    def __init__(self, 
                 ratio_threshold: float = 2.5,
                 max_candidates: int = 2,
                 elevation_threshold: float = 15.0,
                 min_satellites: int = 4,
                 use_satellite_selection: bool = True,
                 max_deviation_threshold: float = 0.25):
        """
        Initialize GREAT-PVT resolver with satellite selection
        
        Parameters
        ----------
        ratio_threshold : float
            Minimum ratio for accepting fixed solution
        max_candidates : int
            Maximum number of integer candidates to search
        elevation_threshold : float
            Minimum elevation angle in degrees
        min_satellites : int
            Minimum number of satellites required for AR
        use_satellite_selection : bool
            Enable quality-based satellite selection
        max_deviation_threshold : float
            Maximum allowed deviation from integer for satellite selection
        """
        self.ratio_threshold = ratio_threshold
        self.max_candidates = max_candidates
        self.elevation_threshold = elevation_threshold
        self.min_satellites = min_satellites
        self.use_satellite_selection = use_satellite_selection
        self.max_deviation_threshold = max_deviation_threshold
        
        # Statistics tracking
        self.total_epochs = 0
        self.fixed_epochs = 0
        self.satellite_usage = {}
        
    def assess_satellite_quality(self,
                                 satellite_ids: List[str],
                                 float_ambiguities: np.ndarray,
                                 elevations: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Assess quality of satellites based on various metrics
        
        Parameters
        ----------
        satellite_ids : List[str]
            Satellite identifiers (e.g., 'G14', 'C04')
        float_ambiguities : np.ndarray
            Float ambiguities for each satellite
        elevations : np.ndarray, optional
            Elevation angles in degrees
            
        Returns
        -------
        np.ndarray
            Quality scores for each satellite (higher is better)
        """
        n = len(satellite_ids)
        quality_scores = np.ones(n)
        
        # Factor 1: Known good/bad satellites (disabled for dataset-agnostic operation)
        # Only apply if lists are not empty
        for i, sat_id in enumerate(satellite_ids):
            if self.GOOD_SATELLITES and sat_id in self.GOOD_SATELLITES:
                quality_scores[i] *= 2.0  # Boost good satellites
            elif self.BAD_SATELLITES and sat_id in self.BAD_SATELLITES:
                quality_scores[i] *= 0.3  # Penalize bad satellites
            
            # Factor 2: System quality
            if sat_id and len(sat_id) > 0:
                system = sat_id[0]
                quality_scores[i] *= self.SYSTEM_WEIGHTS.get(system, 1.0)
        
        # Factor 3: Deviation from integer
        deviations = np.abs(float_ambiguities - np.round(float_ambiguities))
        for i, dev in enumerate(deviations):
            if dev < 0.1:
                quality_scores[i] *= 1.5
            elif dev < 0.2:
                quality_scores[i] *= 1.2
            elif dev > 0.4:
                quality_scores[i] *= 0.5
        
        # Factor 4: Elevation angle
        if elevations is not None:
            for i, elev in enumerate(elevations):
                if elev > 60:
                    quality_scores[i] *= 1.3
                elif elev > 30:
                    quality_scores[i] *= 1.1
                elif elev < 20:
                    quality_scores[i] *= 0.7
        
        return quality_scores
    
    def select_satellites(self,
                         satellite_ids: List[str],
                         float_ambiguities: np.ndarray,
                         covariance: np.ndarray,
                         elevations: Optional[np.ndarray] = None,
                         max_satellites: int = 10) -> Tuple[np.ndarray, List[str]]:
        """
        Select best subset of satellites for ambiguity resolution
        
        Parameters
        ----------
        satellite_ids : List[str]
            All available satellite identifiers
        float_ambiguities : np.ndarray
            Float ambiguities for all satellites
        covariance : np.ndarray
            Covariance matrix for all ambiguities
        elevations : np.ndarray, optional
            Elevation angles for all satellites
        max_satellites : int
            Maximum number of satellites to use
            
        Returns
        -------
        selected_indices : np.ndarray
            Indices of selected satellites
        selected_ids : List[str]
            IDs of selected satellites
        """
        n = len(satellite_ids)
        
        if n <= self.min_satellites:
            # Use all if we have minimum or fewer
            return np.arange(n), satellite_ids
        
        # Assess quality
        quality_scores = self.assess_satellite_quality(
            satellite_ids, float_ambiguities, elevations)
        
        # Filter by deviation threshold
        deviations = np.abs(float_ambiguities - np.round(float_ambiguities))
        valid_mask = deviations < self.max_deviation_threshold
        
        # Combine with quality scores
        combined_scores = quality_scores.copy()
        combined_scores[~valid_mask] *= 0.1  # Heavily penalize high deviation
        
        # Sort by score and select best
        sorted_indices = np.argsort(combined_scores)[::-1]  # Descending order
        
        # Select top satellites up to max_satellites
        n_select = min(max_satellites, n)
        
        # Ensure minimum satellites with relaxed threshold if needed
        selected_indices = []
        selected_ids = []
        
        for idx in sorted_indices:
            if len(selected_indices) < self.min_satellites:
                # Must include to meet minimum
                selected_indices.append(idx)
                selected_ids.append(satellite_ids[idx])
            elif len(selected_indices) < n_select:
                # Include if quality is good enough
                if combined_scores[idx] > 0.5:  # Quality threshold
                    selected_indices.append(idx)
                    selected_ids.append(satellite_ids[idx])
            else:
                break
        
        selected_indices = np.array(selected_indices)
        
        # Log selection
        logger.debug(f"Selected {len(selected_indices)}/{n} satellites")
        logger.debug(f"Selected: {selected_ids}")
        logger.debug(f"Mean deviation of selected: {np.mean(deviations[selected_indices]):.3f}")
        
        return selected_indices, selected_ids
    
    def resolve(self,
                float_ambiguities: np.ndarray,
                covariance: np.ndarray,
                elevations: Optional[np.ndarray] = None,
                satellite_ids: Optional[List[str]] = None) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Resolve integer ambiguities using GREAT-PVT with satellite selection
        
        Parameters
        ----------
        float_ambiguities : np.ndarray
            Float ambiguity estimates
        covariance : np.ndarray
            Covariance matrix of float ambiguities
        elevations : np.ndarray, optional
            Satellite elevation angles in degrees
        satellite_ids : List[str], optional
            Satellite identifiers for quality assessment
            
        Returns
        -------
        fixed_ambiguities : np.ndarray
            Integer ambiguity estimates
        ratio : float
            Ratio test statistic
        is_fixed : bool
            Whether ambiguities were successfully fixed
        info : dict
            Additional information about the resolution
        """
        self.total_epochs += 1
        n_total = len(float_ambiguities)
        
        info = {
            'n_total_ambiguities': n_total,
            'n_selected': n_total,
            'selected_satellites': satellite_ids,
            'mean_deviation': 0.0,
            'selection_applied': False
        }
        
        # Apply satellite selection if enabled and satellite IDs provided
        if self.use_satellite_selection and satellite_ids is not None:
            selected_indices, selected_ids = self.select_satellites(
                satellite_ids, float_ambiguities, covariance, elevations)
            
            if len(selected_indices) < self.min_satellites:
                logger.warning(f"Too few satellites after selection: {len(selected_indices)}")
                return float_ambiguities, 0.0, False, info
            
            # Extract selected subset
            float_amb_subset = float_ambiguities[selected_indices]
            cov_subset = covariance[np.ix_(selected_indices, selected_indices)]
            elev_subset = elevations[selected_indices] if elevations is not None else None
            
            info['n_selected'] = len(selected_indices)
            info['selected_satellites'] = selected_ids
            info['selection_applied'] = True
            
            # Track usage
            for sat_id in selected_ids:
                if sat_id not in self.satellite_usage:
                    self.satellite_usage[sat_id] = 0
                self.satellite_usage[sat_id] += 1
        else:
            float_amb_subset = float_ambiguities
            cov_subset = covariance
            elev_subset = elevations
            selected_indices = np.arange(n_total)
        
        # Compute deviations from integers
        deviations = np.abs(float_amb_subset - np.round(float_amb_subset))
        info['mean_deviation'] = np.mean(deviations)
        
        # Check condition number
        try:
            cond = np.linalg.cond(cov_subset)
            if cond > 1e10:
                logger.warning(f"Poor conditioning: {cond:.2e}")
                return float_ambiguities, 0.0, False, info
        except:
            return float_ambiguities, 0.0, False, info
        
        # Use proper LAMBDA search for integer ambiguities
        try:
            # LAMBDA reduction
            Z, L, D = lambda_reduction(cov_subset)
            
            # Transform float ambiguities
            z_hat = Z.T @ float_amb_subset
            
            # Integer search (get top 2 candidates for ratio test)
            candidates, scores = lambda_search(z_hat, L, D, ncands=2)
            
            if len(candidates) >= 2:
                # Transform back to original space
                z_fixed = Z @ candidates[0]
                z_fixed = np.round(z_fixed).astype(int)  # Ensure integers
                
                # Compute ratio
                if scores[0] > 0:
                    ratio = np.sqrt(scores[1] / scores[0])
                else:
                    ratio = float('inf')
            elif len(candidates) == 1:
                # Only one candidate found
                z_fixed = Z @ candidates[0]
                z_fixed = np.round(z_fixed).astype(int)
                ratio = float('inf')  # No second candidate for ratio
            else:
                # No candidates found, fall back to rounding
                z_fixed = np.round(float_amb_subset).astype(int)
                ratio = 0.0
                
        except Exception as e:
            logger.debug(f"LAMBDA search failed: {e}, falling back to rounding")
            # Fall back to simple rounding if LAMBDA fails
            z_fixed = np.round(float_amb_subset).astype(int)
            
            try:
                # Compute basic ratio test
                Q_inv = np.linalg.inv(cov_subset)
                residual = float_amb_subset - z_fixed
                score_best = residual.T @ Q_inv @ residual
                
                # Simple second-best
                z_second = z_fixed.copy()
                worst_idx = np.argmax(np.abs(residual))
                z_second[worst_idx] += 1 if residual[worst_idx] > 0 else -1
                
                residual_second = float_amb_subset - z_second
                score_second = residual_second.T @ Q_inv @ residual_second
                
                ratio = np.sqrt(score_second / score_best) if score_best > 0 else 0.0
            except:
                ratio = 0.0
        
        # Accept if ratio exceeds threshold
        is_fixed = ratio >= self.ratio_threshold
        
        if is_fixed:
            self.fixed_epochs += 1
            logger.info(f"Ambiguities fixed with ratio: {ratio:.2f}")
            
            # Map back to full vector if selection was applied
            if info['selection_applied']:
                fixed_full = float_ambiguities.copy()
                fixed_full[selected_indices] = z_fixed
                return fixed_full, ratio, True, info
            else:
                return z_fixed, ratio, True, info
        else:
            logger.debug(f"Ratio test failed: {ratio:.2f} < {self.ratio_threshold}")
            return float_ambiguities, ratio, False, info
    
    def get_statistics(self) -> dict:
        """Get resolver statistics"""
        stats = {
            'total_epochs': self.total_epochs,
            'fixed_epochs': self.fixed_epochs,
            'fix_rate': self.fixed_epochs / self.total_epochs if self.total_epochs > 0 else 0,
            'satellite_usage': self.satellite_usage
        }
        
        # Find most/least used satellites
        if self.satellite_usage:
            sorted_usage = sorted(self.satellite_usage.items(), key=lambda x: x[1], reverse=True)
            stats['most_used_satellites'] = sorted_usage[:5]
            stats['least_used_satellites'] = sorted_usage[-5:]
        
        return stats


# Convenience function for backward compatibility
def lambda_greatpvt_with_selection(float_ambiguities: np.ndarray,
                                   covariance: np.ndarray,
                                   ratio_threshold: float = 2.5,
                                   satellite_ids: Optional[List[str]] = None,
                                   elevations: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, bool]:
    """
    Resolve ambiguities using GREAT-PVT with satellite selection
    
    Parameters
    ----------
    float_ambiguities : np.ndarray
        Float ambiguity estimates
    covariance : np.ndarray
        Covariance matrix
    ratio_threshold : float
        Ratio test threshold
    satellite_ids : List[str], optional
        Satellite identifiers for selection
    elevations : np.ndarray, optional
        Elevation angles
        
    Returns
    -------
    fixed_ambiguities : np.ndarray
        Fixed integer ambiguities
    ratio : float
        Ratio test value
    is_fixed : bool
        Whether fixing succeeded
    """
    resolver = GreatPVTWithSelection(ratio_threshold=ratio_threshold)
    fixed, ratio, is_fixed, _ = resolver.resolve(
        float_ambiguities, covariance, elevations, satellite_ids)
    return fixed, ratio, is_fixed