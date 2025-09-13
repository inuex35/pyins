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
Elevation-Dependent Weighting Module
====================================

Implements elevation and SNR-dependent weighting for improved ambiguity resolution.
Based on GreatPVT's weighting strategies.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class WeightingModel(Enum):
    """Weighting model types"""
    ELEVATION = "elevation"  # Elevation-only
    SNR = "snr"  # SNR-only
    COMBINED = "combined"  # Elevation + SNR
    EXPONENTIAL = "exponential"  # Exponential elevation model
    SINE = "sine"  # Sine elevation model


class ElevationWeighting:
    """
    Elevation and SNR-dependent weighting for GNSS measurements
    
    Improves ambiguity resolution by properly weighting observations
    based on satellite geometry and signal quality.
    """
    
    def __init__(self, 
                 model: WeightingModel = WeightingModel.SINE,
                 min_elevation: float = 10.0,
                 sigma_0: float = 0.003,  # meters for phase
                 snr_threshold: float = 35.0):  # dB-Hz
        """
        Initialize elevation weighting
        
        Parameters
        ----------
        model : WeightingModel
            Weighting model to use
        min_elevation : float
            Minimum elevation angle (degrees)
        sigma_0 : float
            Base standard deviation (meters for phase)
        snr_threshold : float
            SNR threshold for good signal (dB-Hz)
        """
        self.model = model
        self.min_elevation = min_elevation
        self.sigma_0 = sigma_0
        self.snr_threshold = snr_threshold
        
        # Model-specific parameters
        self.elevation_factor = 1.0  # Scale factor
        self.snr_factor = 0.5  # SNR contribution weight
        
    def compute_weight(self, elevation: float, 
                      snr: Optional[float] = None) -> float:
        """
        Compute observation weight
        
        Parameters
        ----------
        elevation : float
            Satellite elevation angle (degrees)
        snr : float, optional
            Signal-to-noise ratio (dB-Hz)
            
        Returns
        -------
        weight : float
            Observation weight (higher is better)
        """
        # Ensure minimum elevation
        elevation = max(elevation, self.min_elevation)
        
        # Base elevation weight
        if self.model == WeightingModel.SINE:
            elev_weight = self._sine_weight(elevation)
        elif self.model == WeightingModel.EXPONENTIAL:
            elev_weight = self._exponential_weight(elevation)
        else:
            elev_weight = self._sine_weight(elevation)
        
        # SNR weight if available
        snr_weight = 1.0
        if snr is not None and self.model in [WeightingModel.SNR, WeightingModel.COMBINED]:
            snr_weight = self._snr_weight(snr)
        
        # Combine weights
        if self.model == WeightingModel.SNR:
            return snr_weight
        elif self.model == WeightingModel.COMBINED:
            return elev_weight * (1 - self.snr_factor) + snr_weight * self.snr_factor
        else:
            return elev_weight
    
    def _sine_weight(self, elevation: float) -> float:
        """
        Sine elevation weighting model
        
        Weight = sin(elevation)
        """
        elev_rad = np.radians(elevation)
        return np.sin(elev_rad)
    
    def _exponential_weight(self, elevation: float) -> float:
        """
        Exponential elevation weighting model
        
        Weight = exp(-k / sin(elevation))
        """
        elev_rad = np.radians(elevation)
        sin_elev = np.sin(elev_rad)
        
        if sin_elev < 0.1:
            sin_elev = 0.1
        
        k = 0.1  # Exponential factor
        return np.exp(-k / sin_elev)
    
    def _snr_weight(self, snr: float) -> float:
        """
        SNR-based weighting
        
        Parameters
        ----------
        snr : float
            Signal-to-noise ratio (dB-Hz)
            
        Returns
        -------
        weight : float
            SNR weight [0, 1]
        """
        if snr <= 0:
            return 0.1
        
        # Normalize to threshold
        normalized_snr = snr / self.snr_threshold
        
        # Sigmoid-like function
        weight = 1.0 / (1.0 + np.exp(-2.0 * (normalized_snr - 1.0)))
        
        return max(0.1, min(1.0, weight))
    
    def compute_std_deviation(self, elevation: float,
                             snr: Optional[float] = None,
                             measurement_type: str = 'phase') -> float:
        """
        Compute standard deviation for measurement
        
        Parameters
        ----------
        elevation : float
            Satellite elevation (degrees)
        snr : float, optional
            Signal-to-noise ratio (dB-Hz)
        measurement_type : str
            'phase' or 'code'
            
        Returns
        -------
        sigma : float
            Standard deviation (meters)
        """
        # Base sigma
        if measurement_type == 'phase':
            base_sigma = self.sigma_0  # 3mm typical
        else:  # code
            base_sigma = self.sigma_0 * 100  # 30cm typical
        
        # Get weight
        weight = self.compute_weight(elevation, snr)
        
        # Sigma inversely proportional to weight
        sigma = base_sigma / np.sqrt(weight)
        
        return sigma
    
    def build_weight_matrix(self, elevations: Dict[str, float],
                           snr_values: Optional[Dict[str, float]] = None,
                           satellites: Optional[List[str]] = None) -> np.ndarray:
        """
        Build diagonal weight matrix for satellite observations
        
        Parameters
        ----------
        elevations : Dict[str, float]
            Satellite elevations
        snr_values : Dict[str, float], optional
            SNR values
        satellites : List[str], optional
            Ordered list of satellites
            
        Returns
        -------
        W : np.ndarray
            Diagonal weight matrix
        """
        if satellites is None:
            satellites = list(elevations.keys())
        
        n = len(satellites)
        weights = np.zeros(n)
        
        for i, sat in enumerate(satellites):
            elev = elevations.get(sat, 90.0)
            snr = snr_values.get(sat) if snr_values else None
            weights[i] = self.compute_weight(elev, snr)
        
        # Convert to weight matrix (diagonal)
        W = np.diag(weights)
        
        return W
    
    def build_covariance_matrix(self, elevations: Dict[str, float],
                               snr_values: Optional[Dict[str, float]] = None,
                               satellites: Optional[List[str]] = None,
                               measurement_type: str = 'phase') -> np.ndarray:
        """
        Build covariance matrix based on elevation/SNR
        
        Parameters
        ----------
        elevations : Dict[str, float]
            Satellite elevations
        snr_values : Dict[str, float], optional
            SNR values
        satellites : List[str], optional
            Ordered list of satellites
        measurement_type : str
            'phase' or 'code'
            
        Returns
        -------
        R : np.ndarray
            Diagonal covariance matrix
        """
        if satellites is None:
            satellites = list(elevations.keys())
        
        n = len(satellites)
        variances = np.zeros(n)
        
        for i, sat in enumerate(satellites):
            elev = elevations.get(sat, 90.0)
            snr = snr_values.get(sat) if snr_values else None
            sigma = self.compute_std_deviation(elev, snr, measurement_type)
            variances[i] = sigma ** 2
        
        # Build diagonal covariance
        R = np.diag(variances)
        
        return R


class AdaptiveWeighting:
    """
    Adaptive weighting based on residuals and solution quality
    """
    
    def __init__(self, base_weighting: ElevationWeighting):
        """
        Initialize adaptive weighting
        
        Parameters
        ----------
        base_weighting : ElevationWeighting
            Base weighting model
        """
        self.base_weighting = base_weighting
        self.residual_history: Dict[str, List[float]] = {}
        self.adaptive_factors: Dict[str, float] = {}
        
    def update_from_residuals(self, residuals: Dict[str, float],
                             threshold: float = 0.1):
        """
        Update weights based on residuals
        
        Parameters
        ----------
        residuals : Dict[str, float]
            Post-fit residuals by satellite
        threshold : float
            Threshold for outlier detection (meters)
        """
        for sat, res in residuals.items():
            # Update history
            if sat not in self.residual_history:
                self.residual_history[sat] = []
            self.residual_history[sat].append(abs(res))
            
            # Keep only recent history
            if len(self.residual_history[sat]) > 10:
                self.residual_history[sat].pop(0)
            
            # Calculate adaptive factor
            mean_res = np.mean(self.residual_history[sat])
            
            if mean_res < threshold:
                # Good satellite, increase weight
                self.adaptive_factors[sat] = min(1.5, 1.0 + (threshold - mean_res) / threshold)
            else:
                # Poor satellite, decrease weight
                self.adaptive_factors[sat] = max(0.5, threshold / mean_res)
    
    def get_adaptive_weight(self, satellite: str, elevation: float,
                           snr: Optional[float] = None) -> float:
        """
        Get adaptive weight for satellite
        
        Parameters
        ----------
        satellite : str
            Satellite PRN
        elevation : float
            Elevation angle (degrees)
        snr : float, optional
            Signal-to-noise ratio
            
        Returns
        -------
        weight : float
            Adaptive weight
        """
        # Base weight
        base_weight = self.base_weighting.compute_weight(elevation, snr)
        
        # Apply adaptive factor
        adaptive_factor = self.adaptive_factors.get(satellite, 1.0)
        
        return base_weight * adaptive_factor
    
    def build_adaptive_weight_matrix(self, elevations: Dict[str, float],
                                    snr_values: Optional[Dict[str, float]] = None,
                                    satellites: Optional[List[str]] = None) -> np.ndarray:
        """
        Build adaptive weight matrix
        
        Parameters
        ----------
        elevations : Dict[str, float]
            Satellite elevations
        snr_values : Dict[str, float], optional
            SNR values
        satellites : List[str], optional
            Ordered list of satellites
            
        Returns
        -------
        W : np.ndarray
            Adaptive weight matrix
        """
        if satellites is None:
            satellites = list(elevations.keys())
        
        n = len(satellites)
        weights = np.zeros(n)
        
        for i, sat in enumerate(satellites):
            elev = elevations.get(sat, 90.0)
            snr = snr_values.get(sat) if snr_values else None
            weights[i] = self.get_adaptive_weight(sat, elev, snr)
        
        return np.diag(weights)