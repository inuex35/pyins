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

"""Satellite clock computation and correction"""

import numpy as np
from typing import Tuple, Optional
from ..core.constants import *
from ..core.data_structures import Ephemeris
from ..core.time import timediff

def compute_satellite_clock(eph: Ephemeris, time: float) -> Tuple[float, float]:
    """
    Compute satellite clock bias and drift
    
    Parameters:
    -----------
    eph : Ephemeris
        Satellite ephemeris
    time : float
        Time of interest (GPST)
        
    Returns:
    --------
    dts : float
        Satellite clock bias (s)
    ddts : float
        Satellite clock drift (s/s)
    """
    # Time from clock reference epoch
    dt = timediff(time, eph.toc)
    
    # Clock bias (polynomial model)
    dts = eph.f0 + eph.f1 * dt + eph.f2 * dt**2
    
    # Clock drift
    ddts = eph.f1 + 2.0 * eph.f2 * dt
    
    # Add relativistic correction for non-GEO satellites
    if eph.A > 0:  # Check if orbital parameters are available
        # Compute eccentric anomaly
        n = np.sqrt(GME / eph.A**3)
        M = eph.M0 + n * timediff(time, eph.toe)
        E = M
        for _ in range(10):
            E_old = E
            E = M + eph.e * np.sin(E)
            if abs(E - E_old) < 1e-12:
                break
                
        # Relativistic correction
        F = -2.0 * np.sqrt(GME) / CLIGHT**2
        dtr = F * eph.e * np.sqrt(eph.A) * np.sin(E)
        dts += dtr
        
        # Relativistic drift correction
        dE_dt = n / (1.0 - eph.e * np.cos(E))
        ddtr = F * eph.e * np.sqrt(eph.A) * np.cos(E) * dE_dt
        ddts += ddtr
    
    return dts, ddts


def apply_tgd_correction(eph: Ephemeris, freq_idx: int) -> float:
    """
    Apply Time Group Delay (TGD) correction
    
    Parameters:
    -----------
    eph : Ephemeris
        Satellite ephemeris
    freq_idx : int
        Frequency index (0=L1, 1=L2, etc.)
        
    Returns:
    --------
    tgd : float
        TGD correction (s)
    """
    sys = sat2sys(eph.sat)
    
    if sys == SYS_GPS:
        if freq_idx == 0:  # L1
            return eph.tgd[0]
        elif freq_idx == 1:  # L2
            # TGD is defined for L1-L2, so L2 correction is different
            gamma = (FREQ_L1 / FREQ_L2)**2
            return eph.tgd[0] * gamma / (gamma - 1.0)
        elif freq_idx == 2:  # L5
            return eph.tgd[1] if len(eph.tgd) > 1 else 0.0
            
    elif sys == SYS_GAL:
        if freq_idx == 0:  # E1
            return eph.tgd[0]
        elif freq_idx == 1:  # E5b
            return eph.tgd[1] if len(eph.tgd) > 1 else 0.0
        elif freq_idx == 2:  # E5a
            return eph.tgd[2] if len(eph.tgd) > 2 else 0.0
            
    elif sys == SYS_BDS:
        # Check if TGD data is available in ephemeris
        if freq_idx == 0 and len(eph.tgd) > 0 and eph.tgd[0] != 0.0:  # B1
            return eph.tgd[0]
        elif freq_idx == 1 and len(eph.tgd) > 1 and eph.tgd[1] != 0.0:  # B2
            return eph.tgd[1]
        elif freq_idx == 2 and len(eph.tgd) > 2 and eph.tgd[2] != 0.0:  # B3
            return eph.tgd[2]
        else:
            # Use BeiDou bias table if ephemeris TGD is missing or zero
            from ..gnss.beidou_bias import get_beidou_tgd
            return get_beidou_tgd(eph.sat, freq_idx)
            
    return 0.0


def compute_clock_variance(eph: Ephemeris, time: float) -> float:
    """
    Compute satellite clock variance
    
    Parameters:
    -----------
    eph : Ephemeris
        Satellite ephemeris
    time : float
        Time of interest (GPST)
        
    Returns:
    --------
    var : float
        Clock variance (s^2)
    """
    # Base variance from SV accuracy
    from .satellite_position import ura_value
    std_base = ura_value(eph.sva) / CLIGHT  # Convert to time
    
    # Add age-dependent term
    age = abs(timediff(time, eph.toc))
    age_factor = 1.0 + age / 3600.0  # Increase uncertainty with age
    
    # Clock variance
    var = (std_base * age_factor)**2
    
    return var


class SatelliteClockManager:
    """Manage satellite clock corrections"""
    
    def __init__(self):
        self.clock_cache = {}  # (sat, time) -> (dts, ddts)
        self.cache_tolerance = 1.0  # seconds
        
    def get_clock_correction(self, eph: Ephemeris, time: float,
                            use_cache: bool = True) -> Tuple[float, float]:
        """
        Get satellite clock correction with caching
        
        Parameters:
        -----------
        eph : Ephemeris
            Satellite ephemeris
        time : float
            Time of interest (GPST)
        use_cache : bool
            Whether to use cached values
            
        Returns:
        --------
        dts : float
            Clock bias (s)
        ddts : float
            Clock drift (s/s)
        """
        # Check cache
        if use_cache:
            for (sat, t), (dts, ddts) in self.clock_cache.items():
                if sat == eph.sat and abs(time - t) < self.cache_tolerance:
                    return dts, ddts
        
        # Compute clock correction
        dts, ddts = compute_satellite_clock(eph, time)
        
        # Update cache
        self.clock_cache[(eph.sat, time)] = (dts, ddts)
        
        # Clean old cache entries
        if len(self.clock_cache) > 1000:
            self._clean_cache(time)
            
        return dts, ddts
    
    def _clean_cache(self, current_time: float):
        """Remove old cache entries"""
        max_age = 300.0  # 5 minutes
        self.clock_cache = {
            (sat, t): (dts, ddts)
            for (sat, t), (dts, ddts) in self.clock_cache.items()
            if abs(current_time - t) < max_age
        }