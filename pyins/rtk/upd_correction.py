#!/usr/bin/env python3
"""
UPD (Uncalibrated Phase Delay) Correction Module
=================================================

Implements Wide-Lane and Narrow-Lane UPD corrections for ambiguity resolution.
Based on GreatPVT implementation.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class UPDData:
    """UPD data structure"""
    time: datetime
    satellite: str
    wl_upd: float  # Wide-Lane UPD
    wl_sigma: float  # Wide-Lane UPD sigma
    nl_upd: float  # Narrow-Lane UPD
    nl_sigma: float  # Narrow-Lane UPD sigma
    ewl_upd: Optional[float] = None  # Extra-Wide-Lane UPD
    ewl_sigma: Optional[float] = None


class UPDCorrector:
    """
    UPD Correction Manager
    
    Manages and applies UPD corrections to float ambiguities
    before integer ambiguity resolution.
    """
    
    def __init__(self):
        """Initialize UPD corrector"""
        self.upd_data: Dict[str, List[UPDData]] = {}  # satellite -> list of UPD data
        self.current_time: Optional[datetime] = None
        self.interpolation_window = 900  # seconds (15 minutes)
        
    def load_upd_file(self, filename: str) -> bool:
        """
        Load UPD data from file
        
        Parameters
        ----------
        filename : str
            Path to UPD file
            
        Returns
        -------
        success : bool
            Whether loading was successful
        """
        try:
            with open(filename, 'r') as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    
                    parts = line.strip().split()
                    if len(parts) < 6:
                        continue
                    
                    # Parse UPD data (format may vary)
                    # Example: TIME SAT WL_UPD WL_SIGMA NL_UPD NL_SIGMA [EWL_UPD EWL_SIGMA]
                    sat = parts[1]
                    upd = UPDData(
                        time=self._parse_time(parts[0]),
                        satellite=sat,
                        wl_upd=float(parts[2]),
                        wl_sigma=float(parts[3]),
                        nl_upd=float(parts[4]),
                        nl_sigma=float(parts[5])
                    )
                    
                    if len(parts) >= 8:
                        upd.ewl_upd = float(parts[6])
                        upd.ewl_sigma = float(parts[7])
                    
                    if sat not in self.upd_data:
                        self.upd_data[sat] = []
                    self.upd_data[sat].append(upd)
            
            # Sort by time for each satellite
            for sat in self.upd_data:
                self.upd_data[sat].sort(key=lambda x: x.time)
            
            logger.info(f"Loaded UPD data for {len(self.upd_data)} satellites")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load UPD file: {e}")
            return False
    
    def _parse_time(self, time_str: str) -> datetime:
        """Parse time string to datetime"""
        # Implement based on actual UPD file format
        # This is a placeholder
        return datetime.now()
    
    def get_upd(self, satellite: str, time: datetime, 
                mode: str = 'NL') -> Tuple[Optional[float], Optional[float]]:
        """
        Get UPD correction for satellite at given time
        
        Parameters
        ----------
        satellite : str
            Satellite PRN (e.g., 'G01')
        time : datetime
            Epoch time
        mode : str
            UPD type: 'WL', 'NL', or 'EWL'
            
        Returns
        -------
        upd_value : float or None
            UPD correction value
        upd_sigma : float or None
            UPD sigma
        """
        if satellite not in self.upd_data:
            return None, None
        
        upd_list = self.upd_data[satellite]
        if not upd_list:
            return None, None
        
        # Find nearest UPD data
        nearest_upd = None
        min_dt = float('inf')
        
        for upd in upd_list:
            dt = abs((upd.time - time).total_seconds())
            if dt < min_dt and dt < self.interpolation_window:
                min_dt = dt
                nearest_upd = upd
        
        if nearest_upd is None:
            return None, None
        
        # Return requested UPD type
        if mode == 'WL':
            return nearest_upd.wl_upd, nearest_upd.wl_sigma
        elif mode == 'NL':
            return nearest_upd.nl_upd, nearest_upd.nl_sigma
        elif mode == 'EWL' and nearest_upd.ewl_upd is not None:
            return nearest_upd.ewl_upd, nearest_upd.ewl_sigma
        else:
            return None, None
    
    def apply_upd_corrections(self, float_amb: np.ndarray, 
                            satellites: List[str],
                            time: datetime,
                            mode: str = 'NL') -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply UPD corrections to float ambiguities
        
        Parameters
        ----------
        float_amb : np.ndarray
            Float ambiguities (n,)
        satellites : List[str]
            List of satellite PRNs corresponding to ambiguities
        time : datetime
            Current epoch
        mode : str
            UPD type to apply
            
        Returns
        -------
        corrected_amb : np.ndarray
            UPD-corrected float ambiguities
        upd_sigmas : np.ndarray
            UPD sigmas for weighting
        """
        n = len(float_amb)
        corrected_amb = float_amb.copy()
        upd_sigmas = np.zeros(n)
        
        for i, sat in enumerate(satellites):
            upd_val, upd_sig = self.get_upd(sat, time, mode)
            
            if upd_val is not None:
                # Apply UPD correction
                corrected_amb[i] -= upd_val
                upd_sigmas[i] = upd_sig
                logger.debug(f"Applied {mode} UPD to {sat}: {upd_val:.3f} ± {upd_sig:.3f}")
            else:
                # No UPD available, use default sigma
                upd_sigmas[i] = 0.1  # cycles
                logger.debug(f"No {mode} UPD available for {sat}")
        
        return corrected_amb, upd_sigmas
    
    def apply_dd_upd_corrections(self, dd_amb: np.ndarray,
                                ref_sat: str,
                                rover_sats: List[str],
                                time: datetime,
                                mode: str = 'NL') -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply UPD corrections to double-difference ambiguities
        
        Parameters
        ----------
        dd_amb : np.ndarray
            Double-difference ambiguities (n,)
        ref_sat : str
            Reference satellite PRN
        rover_sats : List[str]
            Rover satellite PRNs
        time : datetime
            Current epoch
        mode : str
            UPD type
            
        Returns
        -------
        corrected_dd : np.ndarray
            Corrected DD ambiguities
        dd_upd_sigmas : np.ndarray
            Combined UPD sigmas for DD
        """
        n = len(dd_amb)
        corrected_dd = dd_amb.copy()
        dd_upd_sigmas = np.zeros(n)
        
        # Get reference satellite UPD
        ref_upd, ref_sigma = self.get_upd(ref_sat, time, mode)
        if ref_upd is None:
            ref_upd = 0.0
            ref_sigma = 0.1
        
        # Apply DD UPD corrections
        for i, sat in enumerate(rover_sats):
            sat_upd, sat_sigma = self.get_upd(sat, time, mode)
            
            if sat_upd is None:
                sat_upd = 0.0
                sat_sigma = 0.1
            
            # DD UPD = UPD(sat) - UPD(ref)
            dd_upd = sat_upd - ref_upd
            corrected_dd[i] -= dd_upd
            
            # Combined sigma for DD
            dd_upd_sigmas[i] = np.sqrt(sat_sigma**2 + ref_sigma**2)
            
            logger.debug(f"DD UPD for {sat}-{ref_sat}: {dd_upd:.3f} ± {dd_upd_sigmas[i]:.3f}")
        
        return corrected_dd, dd_upd_sigmas


class MultiFrequencyUPD:
    """
    Multi-frequency UPD manager for cascaded ambiguity resolution
    """
    
    def __init__(self):
        """Initialize multi-frequency UPD manager"""
        self.wl_corrector = UPDCorrector()
        self.nl_corrector = UPDCorrector()
        self.ewl_corrector = UPDCorrector()
        
    def load_all_upd_files(self, wl_file: str = None, 
                          nl_file: str = None,
                          ewl_file: str = None) -> bool:
        """
        Load UPD files for all frequencies
        
        Parameters
        ----------
        wl_file : str
            Wide-Lane UPD file
        nl_file : str
            Narrow-Lane UPD file
        ewl_file : str
            Extra-Wide-Lane UPD file
            
        Returns
        -------
        success : bool
            Whether all files loaded successfully
        """
        success = True
        
        if wl_file:
            success &= self.wl_corrector.load_upd_file(wl_file)
        if nl_file:
            success &= self.nl_corrector.load_upd_file(nl_file)
        if ewl_file:
            success &= self.ewl_corrector.load_upd_file(ewl_file)
        
        return success
    
    def apply_cascaded_upd(self, wl_amb: np.ndarray,
                          nl_amb: np.ndarray,
                          satellites: List[str],
                          time: datetime) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply cascaded UPD corrections for WL/NL ambiguities
        
        Parameters
        ----------
        wl_amb : np.ndarray
            Wide-Lane ambiguities
        nl_amb : np.ndarray
            Narrow-Lane ambiguities
        satellites : List[str]
            Satellite PRNs
        time : datetime
            Current epoch
            
        Returns
        -------
        corrected_wl : np.ndarray
            Corrected WL ambiguities
        corrected_nl : np.ndarray
            Corrected NL ambiguities
        """
        # Apply WL UPD
        corrected_wl, _ = self.wl_corrector.apply_upd_corrections(
            wl_amb, satellites, time, 'WL'
        )
        
        # Apply NL UPD
        corrected_nl, _ = self.nl_corrector.apply_upd_corrections(
            nl_amb, satellites, time, 'NL'
        )
        
        return corrected_wl, corrected_nl