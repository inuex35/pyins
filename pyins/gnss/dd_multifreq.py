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
Multi-frequency Double Difference Processor for pyins
======================================================

General multi-frequency DD processor that handles L1+L2 (and L5 if available)
for all GNSS systems as standard functionality.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Standard wavelengths (meters)
WAVELENGTH_GPS_L1 = 0.19029367279836487
WAVELENGTH_GPS_L2 = 0.24421021342456828
WAVELENGTH_GPS_L5 = 0.25480066505892986

WAVELENGTH_GAL_E1 = 0.19029367279836487  # Same as GPS L1
WAVELENGTH_GAL_E5a = 0.25480066505892986  # Same as GPS L5
WAVELENGTH_GAL_E5b = 0.24834937054815966
WAVELENGTH_GAL_E6 = 0.23254598707972196

WAVELENGTH_BDS_B1I = 0.19203948801316013
WAVELENGTH_BDS_B1C = 0.19029367279836487
WAVELENGTH_BDS_B2a = 0.25480066505892986
WAVELENGTH_BDS_B2b = 0.24834937054815966
WAVELENGTH_BDS_B3 = 0.23633278865436135

WAVELENGTH_QZS_L1 = 0.19029367279836487
WAVELENGTH_QZS_L2 = 0.24421021342456828
WAVELENGTH_QZS_L5 = 0.25480066505892986

# Speed of light
CLIGHT = 299792458.0


@dataclass
class MultiFreqObservation:
    """Multi-frequency observation data"""
    prn: str
    l1_phase: float  # L1 carrier phase (cycles)
    l1_range: float  # L1 pseudorange (meters)
    l2_phase: Optional[float] = None  # L2 carrier phase (cycles)
    l2_range: Optional[float] = None  # L2 pseudorange (meters)
    l5_phase: Optional[float] = None  # L5/E5a carrier phase (cycles)
    l5_range: Optional[float] = None  # L5/E5a pseudorange (meters)
    elevation: float = 0.0  # Satellite elevation (degrees)
    azimuth: float = 0.0    # Satellite azimuth (degrees)
    
    @property
    def has_l2(self) -> bool:
        return self.l2_phase is not None and self.l2_range is not None
    
    @property
    def has_l5(self) -> bool:
        return self.l5_phase is not None and self.l5_range is not None
    
    @property
    def is_multifreq(self) -> bool:
        return self.has_l2 or self.has_l5


@dataclass
class DoubleDifference:
    """Double difference measurement"""
    ref_prn: str  # Reference satellite PRN
    sat_prn: str  # Other satellite PRN
    dd_l1_phase: float  # DD L1 carrier phase (cycles)
    dd_l1_range: float  # DD L1 pseudorange (meters)
    dd_l2_phase: Optional[float] = None  # DD L2 carrier phase
    dd_l2_range: Optional[float] = None  # DD L2 pseudorange
    dd_l5_phase: Optional[float] = None  # DD L5 carrier phase
    dd_l5_range: Optional[float] = None  # DD L5 pseudorange
    
    @property
    def has_l2(self) -> bool:
        return self.dd_l2_phase is not None and self.dd_l2_range is not None
    
    @property
    def has_l5(self) -> bool:
        return self.dd_l5_phase is not None and self.dd_l5_range is not None
    
    def get_float_ambiguity_l1(self, wavelength: float = WAVELENGTH_GPS_L1) -> float:
        """Calculate L1 float ambiguity"""
        return self.dd_l1_phase - self.dd_l1_range / wavelength
    
    def get_float_ambiguity_l2(self, wavelength: float = WAVELENGTH_GPS_L2) -> float:
        """Calculate L2 float ambiguity"""
        if not self.has_l2:
            return None
        return self.dd_l2_phase - self.dd_l2_range / wavelength
    
    def get_wide_lane(self) -> Optional[float]:
        """Calculate Wide-Lane combination (L1-L2)"""
        if not self.has_l2:
            return None
        return self.dd_l1_phase - self.dd_l2_phase
    
    def get_narrow_lane(self) -> Optional[float]:
        """Calculate Narrow-Lane combination ((L1+L2)/2)"""
        if not self.has_l2:
            return None
        return (self.dd_l1_phase + self.dd_l2_phase) / 2
    
    def get_ionosphere_free_phase(self, f1: float = 1575.42e6, f2: float = 1227.60e6) -> Optional[float]:
        """Calculate Ionosphere-Free combination for phase"""
        if not self.has_l2:
            return None
        alpha = f1**2 / (f1**2 - f2**2)
        beta = -f2**2 / (f1**2 - f2**2)
        return alpha * self.dd_l1_phase + beta * self.dd_l2_phase
    
    def get_ionosphere_free_range(self, f1: float = 1575.42e6, f2: float = 1227.60e6) -> Optional[float]:
        """Calculate Ionosphere-Free combination for pseudorange"""
        if not self.has_l2:
            return None
        alpha = f1**2 / (f1**2 - f2**2)
        beta = -f2**2 / (f1**2 - f2**2)
        return alpha * self.dd_l1_range + beta * self.dd_l2_range


class MultiFreqDDProcessor:
    """
    Multi-frequency Double Difference Processor
    
    Processes L1+L2 (and L5 if available) by default for improved ambiguity resolution
    """
    
    def __init__(self, use_l2: bool = True, use_l5: bool = True, 
                 min_elevation: float = 10.0):
        """
        Initialize multi-frequency DD processor
        
        Parameters
        ----------
        use_l2 : bool
            Use L2 frequency if available (default: True)
        use_l5 : bool
            Use L5 frequency if available (default: True)
        min_elevation : float
            Minimum satellite elevation in degrees (default: 10.0)
        """
        self.use_l2 = use_l2
        self.use_l5 = use_l5
        self.min_elevation = min_elevation
        
        logger.info(f"Multi-frequency DD initialized: L2={use_l2}, L5={use_l5}")
    
    def form_double_differences(self, 
                               base_obs: List[MultiFreqObservation],
                               rover_obs: List[MultiFreqObservation]) -> List[DoubleDifference]:
        """
        Form double differences from base and rover observations
        
        Parameters
        ----------
        base_obs : List[MultiFreqObservation]
            Base station observations
        rover_obs : List[MultiFreqObservation]
            Rover station observations
            
        Returns
        -------
        List[DoubleDifference]
            List of double difference measurements
        """
        # Find common satellites
        base_prns = {obs.prn for obs in base_obs}
        rover_prns = {obs.prn for obs in rover_obs}
        common_prns = base_prns & rover_prns
        
        if len(common_prns) < 2:
            logger.warning(f"Not enough common satellites: {len(common_prns)}")
            return []
        
        # Create observation dictionaries
        base_dict = {obs.prn: obs for obs in base_obs}
        rover_dict = {obs.prn: obs for obs in rover_obs}
        
        # Select reference satellite (highest elevation)
        ref_prn = max(common_prns, 
                     key=lambda prn: rover_dict[prn].elevation)
        
        # Form double differences
        dd_list = []
        for prn in common_prns:
            if prn == ref_prn:
                continue
            
            # Skip low elevation satellites
            if rover_dict[prn].elevation < self.min_elevation:
                continue
            
            # Get observations
            base_ref = base_dict[ref_prn]
            base_sat = base_dict[prn]
            rover_ref = rover_dict[ref_prn]
            rover_sat = rover_dict[prn]
            
            # Single differences
            sd_ref_l1_phase = rover_ref.l1_phase - base_ref.l1_phase
            sd_sat_l1_phase = rover_sat.l1_phase - base_sat.l1_phase
            sd_ref_l1_range = rover_ref.l1_range - base_ref.l1_range
            sd_sat_l1_range = rover_sat.l1_range - base_sat.l1_range
            
            # Double differences
            dd = DoubleDifference(
                ref_prn=ref_prn,
                sat_prn=prn,
                dd_l1_phase=sd_sat_l1_phase - sd_ref_l1_phase,
                dd_l1_range=sd_sat_l1_range - sd_ref_l1_range
            )
            
            # Add L2 if available and enabled
            if self.use_l2 and all([base_ref.has_l2, base_sat.has_l2, 
                                   rover_ref.has_l2, rover_sat.has_l2]):
                sd_ref_l2_phase = rover_ref.l2_phase - base_ref.l2_phase
                sd_sat_l2_phase = rover_sat.l2_phase - base_sat.l2_phase
                sd_ref_l2_range = rover_ref.l2_range - base_ref.l2_range
                sd_sat_l2_range = rover_sat.l2_range - base_sat.l2_range
                
                dd.dd_l2_phase = sd_sat_l2_phase - sd_ref_l2_phase
                dd.dd_l2_range = sd_sat_l2_range - sd_ref_l2_range
            
            # Add L5 if available and enabled
            if self.use_l5 and all([base_ref.has_l5, base_sat.has_l5,
                                   rover_ref.has_l5, rover_sat.has_l5]):
                sd_ref_l5_phase = rover_ref.l5_phase - base_ref.l5_phase
                sd_sat_l5_phase = rover_sat.l5_phase - base_sat.l5_phase
                sd_ref_l5_range = rover_ref.l5_range - base_ref.l5_range
                sd_sat_l5_range = rover_sat.l5_range - base_sat.l5_range
                
                dd.dd_l5_phase = sd_sat_l5_phase - sd_ref_l5_phase
                dd.dd_l5_range = sd_sat_l5_range - sd_ref_l5_range
            
            dd_list.append(dd)
        
        # Log statistics
        n_multifreq = sum(1 for dd in dd_list if dd.has_l2)
        logger.debug(f"Formed {len(dd_list)} DDs, {n_multifreq} with L2")
        
        return dd_list
    
    def get_float_ambiguities(self, dd_list: List[DoubleDifference]) -> Dict[str, np.ndarray]:
        """
        Extract float ambiguities from double differences
        
        Parameters
        ----------
        dd_list : List[DoubleDifference]
            List of double difference measurements
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with 'L1', 'L2', 'L5', 'WL', 'NL' ambiguities
        """
        result = {}
        
        # L1 ambiguities (always available)
        result['L1'] = np.array([dd.get_float_ambiguity_l1() for dd in dd_list])
        
        # L2 ambiguities
        l2_amb = [dd.get_float_ambiguity_l2() for dd in dd_list if dd.has_l2]
        if l2_amb:
            result['L2'] = np.array(l2_amb)
        
        # Wide-Lane and Narrow-Lane
        wl = [dd.get_wide_lane() for dd in dd_list if dd.has_l2]
        nl = [dd.get_narrow_lane() for dd in dd_list if dd.has_l2]
        if wl:
            result['WL'] = np.array(wl)
        if nl:
            result['NL'] = np.array(nl)
        
        return result
    
    def get_ionosphere_free_measurements(self, dd_list: List[DoubleDifference]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get Ionosphere-Free combination measurements
        
        Parameters
        ----------
        dd_list : List[DoubleDifference]
            List of double difference measurements
            
        Returns
        -------
        if_phase : np.ndarray
            Ionosphere-free phase measurements
        if_range : np.ndarray
            Ionosphere-free pseudorange measurements
        """
        if_phase = []
        if_range = []
        
        for dd in dd_list:
            if dd.has_l2:
                phase = dd.get_ionosphere_free_phase()
                range_val = dd.get_ionosphere_free_range()
                if phase is not None and range_val is not None:
                    if_phase.append(phase)
                    if_range.append(range_val)
        
        return np.array(if_phase), np.array(if_range)
    
    def get_wavelengths(self, system: str = 'G') -> Dict[str, float]:
        """
        Get wavelengths for different frequencies
        
        Parameters
        ----------
        system : str
            GNSS system ('G'=GPS, 'E'=Galileo, 'C'=BeiDou, 'J'=QZSS)
            
        Returns
        -------
        Dict[str, float]
            Wavelengths in meters
        """
        if system == 'G':  # GPS
            return {
                'L1': WAVELENGTH_GPS_L1,
                'L2': WAVELENGTH_GPS_L2,
                'L5': WAVELENGTH_GPS_L5,
                'WL': 0.862027742,  # Wide-Lane wavelength
                'NL': 0.106951823   # Narrow-Lane wavelength
            }
        elif system == 'E':  # Galileo
            return {
                'E1': WAVELENGTH_GAL_E1,
                'E5a': WAVELENGTH_GAL_E5a,
                'E5b': WAVELENGTH_GAL_E5b,
                'E6': WAVELENGTH_GAL_E6
            }
        elif system == 'C':  # BeiDou
            return {
                'B1I': WAVELENGTH_BDS_B1I,
                'B1C': WAVELENGTH_BDS_B1C,
                'B2a': WAVELENGTH_BDS_B2a,
                'B2b': WAVELENGTH_BDS_B2b,
                'B3': WAVELENGTH_BDS_B3
            }
        elif system == 'J':  # QZSS
            return {
                'L1': WAVELENGTH_QZS_L1,
                'L2': WAVELENGTH_QZS_L2,
                'L5': WAVELENGTH_QZS_L5
            }
        else:
            return {'L1': WAVELENGTH_GPS_L1}  # Default