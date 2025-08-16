"""Cycle slip detection for RTK processing"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from ..core.data_structures import Observation
from ..core.constants import CLIGHT


class CycleSlipDetector:
    """Detect cycle slips in carrier phase measurements"""
    
    def __init__(self, 
                 lln_threshold: float = 0.05,  # cycles
                 mw_threshold: float = 3.0,    # cycles  
                 gf_threshold: float = 0.1):   # cycles
        """
        Initialize cycle slip detector
        
        Parameters:
        -----------
        lln_threshold : float
            Lock time normalized threshold (cycles)
        mw_threshold : float
            Melbourne-Wubbena threshold (cycles)
        gf_threshold : float
            Geometry-free threshold (cycles)
        """
        self.lln_threshold = lln_threshold
        self.mw_threshold = mw_threshold  
        self.gf_threshold = gf_threshold
        
        # History for detection
        self.phase_history = {}  # sat -> list of (time, phase)
        self.mw_history = {}     # sat -> list of (time, mw)
        self.gf_history = {}     # sat -> list of (time, gf)
        self.lock_time_history = {}  # sat -> list of (time, lock_time)
        
        # Frequencies
        self.l1_freq = 1575.42e6
        self.l2_freq = 1227.60e6
        self.l1_wavelength = CLIGHT / self.l1_freq
        self.l2_wavelength = CLIGHT / self.l2_freq
        
    def detect_cycle_slips(self, 
                          observations: List[Observation],
                          time: float) -> Dict[int, bool]:
        """
        Detect cycle slips for all satellites
        
        Parameters:
        -----------
        observations : List[Observation]
            Current epoch observations
        time : float
            Current time
            
        Returns:
        --------
        slip_flags : Dict[int, bool]
            Cycle slip flags by satellite
        """
        slip_flags = {}
        
        for obs in observations:
            if obs.L1 == 0.0:  # Skip invalid observations
                continue
                
            sat = obs.sat
            slip_detected = False
            
            # Lock time normalized detection
            if hasattr(obs, 'lli1') and obs.lli1 > 0:
                slip_detected = True
                
            # Dual-frequency detection if L2 available
            if obs.L2 > 0.0:
                # Melbourne-Wubbena detection
                mw_slip = self._detect_mw_slip(obs, time)
                
                # Geometry-free detection  
                gf_slip = self._detect_gf_slip(obs, time)
                
                slip_detected = slip_detected or mw_slip or gf_slip
            else:
                # Single-frequency detection
                phase_slip = self._detect_phase_slip(obs, time)
                slip_detected = slip_detected or phase_slip
                
            slip_flags[sat] = slip_detected
            
            # Update history
            self._update_history(obs, time)
            
        return slip_flags
        
    def _detect_mw_slip(self, obs: Observation, time: float) -> bool:
        """Detect cycle slip using Melbourne-Wubbena combination"""
        sat = obs.sat
        
        # Compute Melbourne-Wubbena combination
        mw = self._compute_melbourne_wubbena(obs)
        
        if sat not in self.mw_history:
            self.mw_history[sat] = []
            return False
            
        # Check against recent history
        recent_mw = [entry for entry in self.mw_history[sat] if time - entry[0] < 30.0]
        
        if len(recent_mw) < 3:
            return False
            
        # Compute mean and check for outliers
        mw_values = [entry[1] for entry in recent_mw]
        mean_mw = np.mean(mw_values)
        
        return abs(mw - mean_mw) > self.mw_threshold
        
    def _detect_gf_slip(self, obs: Observation, time: float) -> bool:
        """Detect cycle slip using geometry-free combination"""
        sat = obs.sat
        
        # Compute geometry-free combination
        gf = obs.L1 * self.l1_wavelength - obs.L2 * self.l2_wavelength
        
        if sat not in self.gf_history:
            self.gf_history[sat] = []
            return False
            
        # Check recent history
        recent_gf = [entry for entry in self.gf_history[sat] if time - entry[0] < 30.0]
        
        if len(recent_gf) < 2:
            return False
            
        # Check for sudden jumps
        last_gf = recent_gf[-1][1]
        gf_diff = abs(gf - last_gf)
        
        return gf_diff > self.gf_threshold
        
    def _detect_phase_slip(self, obs: Observation, time: float) -> bool:
        """Detect cycle slip using single frequency phase"""
        sat = obs.sat
        
        if sat not in self.phase_history:
            self.phase_history[sat] = []
            return False
            
        # Check recent history  
        recent_phase = [entry for entry in self.phase_history[sat] if time - entry[0] < 10.0]
        
        if len(recent_phase) < 3:
            return False
            
        # Compute phase rate
        times = [entry[0] for entry in recent_phase]
        phases = [entry[1] for entry in recent_phase]
        
        if len(times) >= 2:
            dt = times[-1] - times[-2]
            if dt > 0:
                phase_rate = (phases[-1] - phases[-2]) / dt
                
                # Check if current rate is consistent
                current_rate = (obs.L1 - phases[-1]) / (time - times[-1])
                rate_diff = abs(current_rate - phase_rate)
                
                return rate_diff > 1.0  # cycles/s threshold
                
        return False
        
    def _compute_melbourne_wubbena(self, obs: Observation) -> float:
        """Compute Melbourne-Wubbena combination"""
        if obs.L2 == 0.0 or obs.P1 == 0.0 or obs.P2 == 0.0:
            return 0.0
            
        # MW combination
        phase_term = obs.L1 - obs.L2
        code_term = (self.l1_freq * obs.P1 - self.l2_freq * obs.P2) / (
            CLIGHT * (self.l1_freq - self.l2_freq))
            
        return phase_term - code_term
        
    def _update_history(self, obs: Observation, time: float):
        """Update observation history"""
        sat = obs.sat
        
        # Phase history
        if sat not in self.phase_history:
            self.phase_history[sat] = []
        self.phase_history[sat].append((time, obs.L1))
        
        # MW history
        if obs.L2 > 0.0:
            mw = self._compute_melbourne_wubbena(obs)
            if sat not in self.mw_history:
                self.mw_history[sat] = []
            self.mw_history[sat].append((time, mw))
            
            # GF history
            gf = obs.L1 * self.l1_wavelength - obs.L2 * self.l2_wavelength
            if sat not in self.gf_history:
                self.gf_history[sat] = []
            self.gf_history[sat].append((time, gf))
            
        # Clean old history (keep 5 minutes)
        self._clean_history(time, 300.0)
        
    def _clean_history(self, current_time: float, max_age: float):
        """Clean old history entries"""
        cutoff_time = current_time - max_age
        
        for sat in list(self.phase_history.keys()):
            self.phase_history[sat] = [
                entry for entry in self.phase_history[sat] 
                if entry[0] > cutoff_time
            ]
            if not self.phase_history[sat]:
                del self.phase_history[sat]
                
        for sat in list(self.mw_history.keys()):
            self.mw_history[sat] = [
                entry for entry in self.mw_history[sat]
                if entry[0] > cutoff_time
            ]
            if not self.mw_history[sat]:
                del self.mw_history[sat]
                
        for sat in list(self.gf_history.keys()):
            self.gf_history[sat] = [
                entry for entry in self.gf_history[sat]
                if entry[0] > cutoff_time
            ]
            if not self.gf_history[sat]:
                del self.gf_history[sat]


class TurboEditDetector:
    """TurboEdit cycle slip detection"""
    
    def __init__(self, threshold: float = 4.0):
        """
        Initialize TurboEdit detector
        
        Parameters:
        -----------
        threshold : float
            Detection threshold in standard deviations
        """
        self.threshold = threshold
        self.code_minus_phase_history = {}  # sat -> history
        
    def detect_slips(self, 
                    observations: List[Observation],
                    time: float) -> Dict[int, bool]:
        """
        Detect cycle slips using TurboEdit algorithm
        
        Parameters:
        -----------
        observations : List[Observation]
            Current observations
        time : float
            Current time
            
        Returns:
        --------
        slip_flags : Dict[int, bool]
            Cycle slip detection flags
        """
        slip_flags = {}
        
        for obs in observations:
            if obs.L1 == 0.0 or obs.P1 == 0.0:
                continue
                
            sat = obs.sat
            
            # Compute code-minus-phase
            wavelength = CLIGHT / self.l1_freq
            cmp = obs.P1 - obs.L1 * wavelength
            
            slip_detected = self._check_turboedit(sat, cmp, time)
            slip_flags[sat] = slip_detected
            
            # Update history
            if sat not in self.code_minus_phase_history:
                self.code_minus_phase_history[sat] = []
            self.code_minus_phase_history[sat].append((time, cmp))
            
            # Clean old history
            self.code_minus_phase_history[sat] = [
                entry for entry in self.code_minus_phase_history[sat]
                if time - entry[0] < 60.0  # Keep 1 minute
            ]
            
        return slip_flags
        
    def _check_turboedit(self, sat: int, cmp: float, time: float) -> bool:
        """Check for cycle slip using TurboEdit statistic"""
        if sat not in self.code_minus_phase_history:
            return False
            
        history = self.code_minus_phase_history[sat]
        if len(history) < 4:
            return False
            
        # Compute fourth-order difference
        values = [entry[1] for entry in history[-4:]]
        fourth_diff = values[3] - 4*values[2] + 6*values[1] - 4*values[0]
        
        # Estimate noise level from recent history
        if len(history) >= 10:
            recent_values = [entry[1] for entry in history[-10:]]
            noise_std = np.std(np.diff(recent_values))
        else:
            noise_std = 1.0  # Default noise level
            
        # TurboEdit test
        test_statistic = abs(fourth_diff) / (8 * noise_std)
        
        return test_statistic > self.threshold