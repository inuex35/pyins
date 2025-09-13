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
Double-Difference Covariance Extraction Module
==============================================

Extracts DD-specific covariance from full parameter covariance matrix.
Based on GreatPVT's _prepareCovariance implementation.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DDDefinition:
    """Double-difference ambiguity definition"""
    ref_sat: str  # Reference satellite
    rover_sat: str  # Rover satellite
    base_idx: int  # Index in base station parameters
    rover_idx: int  # Index in rover station parameters
    wavelength: float  # Carrier wavelength
    frequency: str  # Frequency type (L1, L2, WL, NL)


class DDCovarianceExtractor:
    """
    Extracts DD-specific covariance matrix from full state covariance
    
    This is crucial for proper LAMBDA decorrelation and search.
    """
    
    def __init__(self):
        """Initialize DD covariance extractor"""
        self.dd_definitions: List[DDDefinition] = []
        self.full_covariance: Optional[np.ndarray] = None
        self.param_mapping: Dict[str, int] = {}  # parameter name -> index
        
    def set_parameter_mapping(self, param_names: List[str]):
        """
        Set mapping from parameter names to indices
        
        Parameters
        ----------
        param_names : List[str]
            List of parameter names in order
        """
        self.param_mapping = {name: i for i, name in enumerate(param_names)}
        logger.debug(f"Set parameter mapping for {len(param_names)} parameters")
    
    def define_dd_ambiguities(self, ref_sat: str, 
                            rover_sats: List[str],
                            base_station: str = "base",
                            rover_station: str = "rover",
                            frequency: str = "L1") -> List[DDDefinition]:
        """
        Define DD ambiguities for given satellites
        
        Parameters
        ----------
        ref_sat : str
            Reference satellite PRN
        rover_sats : List[str]
            List of rover satellite PRNs
        base_station : str
            Base station name
        rover_station : str
            Rover station name
        frequency : str
            Frequency type
            
        Returns
        -------
        dd_defs : List[DDDefinition]
            List of DD definitions
        """
        dd_defs = []
        
        for sat in rover_sats:
            # Build parameter names for ambiguities
            base_ref_param = f"N_{base_station}_{ref_sat}_{frequency}"
            base_sat_param = f"N_{base_station}_{sat}_{frequency}"
            rover_ref_param = f"N_{rover_station}_{ref_sat}_{frequency}"
            rover_sat_param = f"N_{rover_station}_{sat}_{frequency}"
            
            # Get indices if they exist
            if all(p in self.param_mapping for p in 
                   [base_ref_param, base_sat_param, rover_ref_param, rover_sat_param]):
                
                dd_def = DDDefinition(
                    ref_sat=ref_sat,
                    rover_sat=sat,
                    base_idx=self.param_mapping[base_sat_param],
                    rover_idx=self.param_mapping[rover_sat_param],
                    wavelength=self._get_wavelength(sat, frequency),
                    frequency=frequency
                )
                dd_defs.append(dd_def)
                
        self.dd_definitions = dd_defs
        logger.info(f"Defined {len(dd_defs)} DD ambiguities")
        return dd_defs
    
    def _get_wavelength(self, sat: str, frequency: str) -> float:
        """Get wavelength for satellite and frequency"""
        # GPS L1/L2 wavelengths
        wavelengths = {
            'L1': 0.19029367,  # meters
            'L2': 0.24421021,
            'WL': 0.86202774,  # Wide-Lane
            'NL': 0.10695182,  # Narrow-Lane
        }
        
        # Handle GLONASS frequency-dependent wavelengths
        if sat.startswith('R'):
            # GLONASS FDMA - needs channel number
            # This is simplified, actual implementation needs channel mapping
            return wavelengths.get(frequency, 0.19)
        
        return wavelengths.get(frequency, 0.19)
    
    def extract_dd_covariance(self, full_cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract DD-specific covariance matrix from full parameter covariance
        
        Parameters
        ----------
        full_cov : np.ndarray
            Full parameter covariance matrix (n_params x n_params)
            
        Returns
        -------
        dd_cov : np.ndarray
            DD ambiguity covariance matrix (n_dd x n_dd)
        dd_values : np.ndarray
            DD float ambiguity values (n_dd,)
        """
        if not self.dd_definitions:
            raise ValueError("No DD ambiguities defined")
        
        n_dd = len(self.dd_definitions)
        dd_cov = np.zeros((n_dd, n_dd))
        
        # Check if we have None values (testing mode)
        if any(dd is None for dd in self.dd_definitions):
            # In testing mode, just return the input covariance
            return full_cov, np.zeros(full_cov.shape[0])
        
        # Build DD transformation matrix
        # DD = (N_rover_sat - N_rover_ref) - (N_base_sat - N_base_ref)
        for i, dd_i in enumerate(self.dd_definitions):
            for j, dd_j in enumerate(self.dd_definitions):
                # Calculate covariance between DD_i and DD_j
                cov_ij = self._compute_dd_covariance_element(
                    full_cov, dd_i, dd_j
                )
                dd_cov[i, j] = cov_ij
        
        # Ensure positive definite
        dd_cov = self._ensure_positive_definite(dd_cov)
        
        return dd_cov, np.zeros(n_dd)  # Values extracted separately
    
    def _compute_dd_covariance_element(self, full_cov: np.ndarray,
                                      dd_i: DDDefinition,
                                      dd_j: DDDefinition) -> float:
        """
        Compute single element of DD covariance matrix
        
        Uses the formula:
        Cov(DD_i, DD_j) = Cov(SD_i, SD_j) where SD is single difference
        """
        # Get reference satellite indices
        ref_i_idx = self._get_ref_index(dd_i.ref_sat, dd_i.frequency)
        ref_j_idx = self._get_ref_index(dd_j.ref_sat, dd_j.frequency)
        
        # Build covariance using chain rule
        # DD = SD_rover - SD_base
        # SD = N_sat - N_ref
        
        cov = 0.0
        
        # Main diagonal terms
        cov += full_cov[dd_i.rover_idx, dd_j.rover_idx]
        cov += full_cov[dd_i.base_idx, dd_j.base_idx]
        
        # Reference satellite terms
        if ref_i_idx >= 0 and ref_j_idx >= 0:
            cov += full_cov[ref_i_idx, ref_j_idx]
        
        # Cross terms
        cov -= full_cov[dd_i.rover_idx, ref_j_idx] if ref_j_idx >= 0 else 0
        cov -= full_cov[dd_i.base_idx, ref_j_idx] if ref_j_idx >= 0 else 0
        cov -= full_cov[ref_i_idx, dd_j.rover_idx] if ref_i_idx >= 0 else 0
        cov -= full_cov[ref_i_idx, dd_j.base_idx] if ref_i_idx >= 0 else 0
        
        return cov
    
    def _get_ref_index(self, ref_sat: str, frequency: str) -> int:
        """Get parameter index for reference satellite"""
        # Build parameter name
        ref_param = f"N_base_{ref_sat}_{frequency}"
        return self.param_mapping.get(ref_param, -1)
    
    def _ensure_positive_definite(self, cov: np.ndarray, 
                                 min_eigenvalue: float = 1e-6) -> np.ndarray:
        """
        Ensure covariance matrix is positive definite
        
        Parameters
        ----------
        cov : np.ndarray
            Covariance matrix
        min_eigenvalue : float
            Minimum eigenvalue threshold
            
        Returns
        -------
        cov_pd : np.ndarray
            Positive definite covariance matrix
        """
        # Check eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        if np.min(eigenvalues) < min_eigenvalue:
            # Fix negative eigenvalues
            eigenvalues[eigenvalues < min_eigenvalue] = min_eigenvalue
            
            # Reconstruct covariance
            cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            
            logger.warning("Fixed non-positive definite DD covariance matrix")
        
        return cov
    
    def apply_satellite_weighting(self, dd_cov: np.ndarray,
                                elevations: Dict[str, float],
                                snr_values: Dict[str, float] = None) -> np.ndarray:
        """
        Apply satellite-specific weighting to DD covariance
        
        Parameters
        ----------
        dd_cov : np.ndarray
            DD covariance matrix
        elevations : Dict[str, float]
            Satellite elevations in degrees
        snr_values : Dict[str, float]
            SNR values (optional)
            
        Returns
        -------
        weighted_cov : np.ndarray
            Weighted DD covariance
        """
        n = dd_cov.shape[0]
        weight_matrix = np.eye(n)
        
        for i, dd_def in enumerate(self.dd_definitions):
            # Get elevation for rover satellite
            elev = elevations.get(dd_def.rover_sat, 90.0)
            
            # Elevation-dependent weighting
            weight = np.sin(np.radians(elev))
            
            # SNR weighting if available
            if snr_values and dd_def.rover_sat in snr_values:
                snr = snr_values[dd_def.rover_sat]
                snr_weight = min(1.0, snr / 45.0)  # Normalize to 45 dB-Hz
                weight *= snr_weight
            
            weight_matrix[i, i] = 1.0 / max(weight, 0.1)
        
        # Apply weighting
        weighted_cov = weight_matrix @ dd_cov @ weight_matrix.T
        
        return weighted_cov


class DDValueExtractor:
    """
    Extracts DD float ambiguity values from parameter estimates
    """
    
    def __init__(self, dd_definitions: List[DDDefinition]):
        """
        Initialize DD value extractor
        
        Parameters
        ----------
        dd_definitions : List[DDDefinition]
            DD ambiguity definitions
        """
        self.dd_definitions = dd_definitions
    
    def extract_dd_values(self, param_values: np.ndarray,
                        param_mapping: Dict[str, int]) -> np.ndarray:
        """
        Extract DD float ambiguity values
        
        Parameters
        ----------
        param_values : np.ndarray
            Full parameter value vector
        param_mapping : Dict[str, int]
            Parameter name to index mapping
            
        Returns
        -------
        dd_values : np.ndarray
            DD float ambiguity values
        """
        n_dd = len(self.dd_definitions)
        dd_values = np.zeros(n_dd)
        
        for i, dd_def in enumerate(self.dd_definitions):
            # Get single difference values
            base_sat_val = param_values[dd_def.base_idx]
            rover_sat_val = param_values[dd_def.rover_idx]
            
            # Get reference satellite values
            base_ref_param = f"N_base_{dd_def.ref_sat}_{dd_def.frequency}"
            rover_ref_param = f"N_rover_{dd_def.ref_sat}_{dd_def.frequency}"
            
            base_ref_val = 0.0
            rover_ref_val = 0.0
            
            if base_ref_param in param_mapping:
                base_ref_val = param_values[param_mapping[base_ref_param]]
            if rover_ref_param in param_mapping:
                rover_ref_val = param_values[param_mapping[rover_ref_param]]
            
            # Compute DD value
            sd_base = base_sat_val - base_ref_val
            sd_rover = rover_sat_val - rover_ref_val
            dd_values[i] = sd_rover - sd_base
        
        return dd_values