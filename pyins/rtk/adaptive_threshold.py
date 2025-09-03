#!/usr/bin/env python3
"""
Adaptive Threshold Module
=========================

Implements adaptive threshold adjustment for ambiguity resolution based on
baseline length, observation quality, and environmental conditions.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class BaselineType(Enum):
    """Baseline length categories"""
    SHORT = "short"  # < 10 km
    MEDIUM = "medium"  # 10-50 km
    LONG = "long"  # 50-100 km
    VERY_LONG = "very_long"  # > 100 km


@dataclass
class ThresholdConfig:
    """Threshold configuration for different conditions"""
    ratio_threshold: float
    boot_threshold: float
    success_rate_threshold: float
    wl_threshold: float  # cycles
    nl_threshold: float  # cycles
    min_satellites: int
    max_pdop: float


class AdaptiveThresholdManager:
    """
    Manages adaptive thresholds for ambiguity resolution
    
    Adjusts thresholds based on:
    - Baseline length
    - Number of satellites
    - DOP values
    - Atmospheric activity
    - Time since last fix
    """
    
    def __init__(self):
        """Initialize adaptive threshold manager"""
        # Default configurations for different baseline types
        self.configs = {
            BaselineType.SHORT: ThresholdConfig(
                ratio_threshold=2.0,
                boot_threshold=0.95,
                success_rate_threshold=0.95,
                wl_threshold=0.25,
                nl_threshold=0.15,
                min_satellites=4,
                max_pdop=4.0
            ),
            BaselineType.MEDIUM: ThresholdConfig(
                ratio_threshold=2.5,
                boot_threshold=0.98,
                success_rate_threshold=0.98,
                wl_threshold=0.20,
                nl_threshold=0.12,
                min_satellites=5,
                max_pdop=3.5
            ),
            BaselineType.LONG: ThresholdConfig(
                ratio_threshold=3.0,
                boot_threshold=0.99,
                success_rate_threshold=0.99,
                wl_threshold=0.15,
                nl_threshold=0.10,
                min_satellites=6,
                max_pdop=3.0
            ),
            BaselineType.VERY_LONG: ThresholdConfig(
                ratio_threshold=3.5,
                boot_threshold=0.995,
                success_rate_threshold=0.995,
                wl_threshold=0.12,
                nl_threshold=0.08,
                min_satellites=7,
                max_pdop=2.5
            )
        }
        
        # Dynamic adjustment factors
        self.time_since_last_fix = 0.0
        self.consecutive_failures = 0
        self.ionospheric_activity = 1.0  # 1.0 = quiet, >1.0 = active
        self.tropospheric_variance = 1.0
        
        # Current configuration
        self.current_config = self.configs[BaselineType.SHORT]
        self.baseline_length = 0.0
        
    def classify_baseline(self, length_km: float) -> BaselineType:
        """
        Classify baseline length
        
        Parameters
        ----------
        length_km : float
            Baseline length in kilometers
            
        Returns
        -------
        baseline_type : BaselineType
            Baseline classification
        """
        if length_km < 10:
            return BaselineType.SHORT
        elif length_km < 50:
            return BaselineType.MEDIUM
        elif length_km < 100:
            return BaselineType.LONG
        else:
            return BaselineType.VERY_LONG
    
    def update_baseline(self, length_km: float):
        """
        Update baseline length and select appropriate config
        
        Parameters
        ----------
        length_km : float
            Baseline length in kilometers
        """
        self.baseline_length = length_km
        baseline_type = self.classify_baseline(length_km)
        self.current_config = self.configs[baseline_type]
        
        logger.info(f"Updated baseline: {length_km:.1f} km ({baseline_type.value})")
    
    def adjust_for_satellites(self, n_satellites: int, pdop: float):
        """
        Adjust thresholds based on satellite geometry
        
        Parameters
        ----------
        n_satellites : int
            Number of visible satellites
        pdop : float
            Position dilution of precision
        """
        # Satellite count adjustment
        if n_satellites > 10:
            # Many satellites, can be more strict
            sat_factor = 0.9
        elif n_satellites >= 7:
            sat_factor = 1.0
        else:
            # Few satellites, relax thresholds
            sat_factor = 1.1 + 0.05 * (7 - n_satellites)
        
        # PDOP adjustment
        if pdop < 2.0:
            pdop_factor = 0.9
        elif pdop < 3.0:
            pdop_factor = 1.0
        else:
            pdop_factor = 1.1 + 0.1 * (pdop - 3.0)
        
        # Combined factor
        combined_factor = sat_factor * pdop_factor
        
        # Apply adjustments
        self.current_config.ratio_threshold *= combined_factor
        self.current_config.wl_threshold *= combined_factor
        self.current_config.nl_threshold *= combined_factor
    
    def adjust_for_atmosphere(self, iono_index: Optional[float] = None,
                             tropo_ztd: Optional[float] = None):
        """
        Adjust for atmospheric conditions
        
        Parameters
        ----------
        iono_index : float, optional
            Ionospheric activity index (e.g., VTEC, Kp)
        tropo_ztd : float, optional
            Tropospheric zenith total delay (meters)
        """
        # Ionospheric adjustment
        if iono_index is not None:
            if iono_index < 20:  # Low activity
                self.ionospheric_activity = 1.0
            elif iono_index < 50:  # Moderate
                self.ionospheric_activity = 1.1
            else:  # High activity
                self.ionospheric_activity = 1.2
        
        # Tropospheric adjustment
        if tropo_ztd is not None:
            # Normal ZTD is about 2.3m
            self.tropospheric_variance = abs(tropo_ztd - 2.3) / 2.3 + 1.0
        
        # Apply atmospheric factors
        atmos_factor = np.sqrt(self.ionospheric_activity * self.tropospheric_variance)
        
        self.current_config.ratio_threshold *= atmos_factor
        self.current_config.boot_threshold = min(0.999, 
                                                 self.current_config.boot_threshold * atmos_factor)
    
    def adjust_for_time(self, time_since_fix: float):
        """
        Adjust thresholds based on time since last successful fix
        
        Parameters
        ----------
        time_since_fix : float
            Time since last successful fix (seconds)
        """
        self.time_since_last_fix = time_since_fix
        
        if time_since_fix < 10:
            # Recent fix, maintain strict thresholds
            time_factor = 1.0
        elif time_since_fix < 60:
            # Gradual relaxation
            time_factor = 1.0 + 0.005 * (time_since_fix - 10)
        else:
            # Long time without fix, relax more
            time_factor = 1.25 + 0.001 * (time_since_fix - 60)
            time_factor = min(time_factor, 1.5)  # Cap at 50% relaxation
        
        # Apply time factor (relax thresholds)
        self.current_config.ratio_threshold /= time_factor
        self.current_config.ratio_threshold = max(1.5, self.current_config.ratio_threshold)
    
    def update_failure_count(self, failed: bool):
        """
        Update consecutive failure counter
        
        Parameters
        ----------
        failed : bool
            Whether the last attempt failed
        """
        if failed:
            self.consecutive_failures += 1
            
            # Relax thresholds after failures
            if self.consecutive_failures > 3:
                relax_factor = 1.0 + 0.05 * (self.consecutive_failures - 3)
                relax_factor = min(relax_factor, 1.3)
                
                self.current_config.ratio_threshold /= relax_factor
                self.current_config.ratio_threshold = max(1.5, 
                                                          self.current_config.ratio_threshold)
        else:
            self.consecutive_failures = 0
    
    def get_thresholds(self) -> Dict[str, float]:
        """
        Get current threshold values
        
        Returns
        -------
        thresholds : Dict[str, float]
            Current threshold values
        """
        return {
            'ratio': self.current_config.ratio_threshold,
            'boot': self.current_config.boot_threshold,
            'success_rate': self.current_config.success_rate_threshold,
            'wl': self.current_config.wl_threshold,
            'nl': self.current_config.nl_threshold,
            'min_sats': self.current_config.min_satellites,
            'max_pdop': self.current_config.max_pdop
        }
    
    def should_attempt_fix(self, n_satellites: int, pdop: float) -> Tuple[bool, str]:
        """
        Determine if ambiguity fixing should be attempted
        
        Parameters
        ----------
        n_satellites : int
            Number of satellites
        pdop : float
            Position DOP
            
        Returns
        -------
        should_fix : bool
            Whether to attempt fixing
        reason : str
            Reason if not attempting
        """
        if n_satellites < self.current_config.min_satellites:
            return False, f"Too few satellites: {n_satellites} < {self.current_config.min_satellites}"
        
        if pdop > self.current_config.max_pdop:
            return False, f"PDOP too high: {pdop:.1f} > {self.current_config.max_pdop}"
        
        return True, "OK"
    
    def adapt_to_conditions(self, conditions: Dict):
        """
        Comprehensive adaptation to current conditions
        
        Parameters
        ----------
        conditions : Dict
            Dictionary containing:
            - baseline_length: float (km)
            - n_satellites: int
            - pdop: float
            - time_since_fix: float (seconds)
            - iono_index: float (optional)
            - tropo_ztd: float (optional)
        """
        # Update baseline
        if 'baseline_length' in conditions:
            self.update_baseline(conditions['baseline_length'])
        
        # Reset to base config for this baseline
        baseline_type = self.classify_baseline(self.baseline_length)
        self.current_config = self.configs[baseline_type]
        
        # Apply adjustments
        if 'n_satellites' in conditions and 'pdop' in conditions:
            self.adjust_for_satellites(conditions['n_satellites'], 
                                      conditions['pdop'])
        
        if 'iono_index' in conditions or 'tropo_ztd' in conditions:
            self.adjust_for_atmosphere(conditions.get('iono_index'),
                                      conditions.get('tropo_ztd'))
        
        if 'time_since_fix' in conditions:
            self.adjust_for_time(conditions['time_since_fix'])
        
        # Log adapted thresholds
        thresholds = self.get_thresholds()
        logger.debug(f"Adapted thresholds: ratio={thresholds['ratio']:.2f}, "
                    f"boot={thresholds['boot']:.3f}, "
                    f"wl={thresholds['wl']:.3f}, nl={thresholds['nl']:.3f}")