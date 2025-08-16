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

"""GLONASS Inter-Frequency Bias (IFB) correction"""

import numpy as np
from ..core.constants import FREQ_G1, CLIGHT


def compute_glonass_ifb_correction(freq_num: int, ifb_rate: float = 0.0) -> float:
    """
    Compute GLONASS inter-frequency bias correction
    
    Parameters
    ----------
    freq_num : int
        GLONASS frequency number (-7 to +6)
    ifb_rate : float
        Inter-frequency bias rate in m/MHz (receiver-specific)
        For u-blox receivers, this can be around -30 to -50 m/MHz
        
    Returns
    -------
    float
        Bias correction in meters to be subtracted from pseudorange
    """
    # GLONASS L1 frequency for this satellite
    f1 = FREQ_G1 + freq_num * 0.5625e6  # Hz
    
    # Reference frequency (usually k=0)
    f0 = FREQ_G1  # Hz
    
    # Frequency difference in MHz
    df_MHz = (f1 - f0) / 1e6
    
    # Bias correction
    bias = ifb_rate * df_MHz
    
    return bias


def estimate_glonass_ifb_rate(gps_clock_bias: float, 
                             glonass_residuals: list,
                             glonass_freq_nums: list) -> float:
    """
    Estimate GLONASS inter-frequency bias rate from residuals
    
    Parameters
    ----------
    gps_clock_bias : float
        GPS-derived receiver clock bias in meters
    glonass_residuals : list
        List of (observed - expected) pseudoranges for GLONASS satellites
    glonass_freq_nums : list
        Corresponding GLONASS frequency numbers
        
    Returns
    -------
    float
        Estimated IFB rate in m/MHz
    """
    if len(glonass_residuals) < 2:
        return 0.0
    
    # Remove GPS clock bias to get GLONASS-specific residuals
    residuals = np.array(glonass_residuals) - gps_clock_bias
    
    # Compute frequencies
    frequencies = np.array([FREQ_G1 + k * 0.5625e6 for k in glonass_freq_nums])
    
    # Linear regression: residual = a * freq + b
    A = np.vstack([frequencies, np.ones(len(frequencies))]).T
    try:
        m, c = np.linalg.lstsq(A, residuals, rcond=None)[0]
        
        # Convert slope to m/MHz
        ifb_rate = m * 1e6
        
        return ifb_rate
    except:
        return 0.0


class GlonassIFBCalibration:
    """
    GLONASS Inter-Frequency Bias calibration for specific receiver types
    """
    
    # Known IFB rates for different receiver types (m/MHz)
    # These are approximate values and should be calibrated for each receiver
    RECEIVER_IFB_RATES = {
        'ublox': -45.0,      # u-blox receivers typically have -30 to -50 m/MHz
        'novatel': -5.0,     # NovAtel receivers have smaller IFB
        'trimble': -8.0,     # Trimble receivers
        'septentrio': -3.0,  # Septentrio receivers
        'javad': -2.0,       # Javad receivers
        'default': 0.0       # Unknown receiver type
    }
    
    def __init__(self, receiver_type: str = 'default'):
        """
        Initialize with receiver type
        
        Parameters
        ----------
        receiver_type : str
            Receiver manufacturer/model
        """
        self.receiver_type = receiver_type.lower()
        self.ifb_rate = self.RECEIVER_IFB_RATES.get(self.receiver_type, 0.0)
        self.calibrated = False
        
    def set_ifb_rate(self, rate: float):
        """Manually set IFB rate"""
        self.ifb_rate = rate
        self.calibrated = True
        
    def calibrate_from_data(self, gps_clock: float, 
                           glonass_data: list):
        """
        Calibrate IFB rate from observation data
        
        Parameters
        ----------
        gps_clock : float
            GPS-derived clock bias in meters
        glonass_data : list
            List of dicts with 'residual' and 'freq_num' keys
        """
        if len(glonass_data) < 3:
            return False
            
        residuals = [d['residual'] for d in glonass_data]
        freq_nums = [d['freq_num'] for d in glonass_data]
        
        estimated_rate = estimate_glonass_ifb_rate(gps_clock, residuals, freq_nums)
        
        # Only update if estimation seems reasonable
        if abs(estimated_rate) < 1000.0:  # Less than 1000 m/MHz
            self.ifb_rate = estimated_rate
            self.calibrated = True
            return True
            
        return False
        
    def get_correction(self, freq_num: int) -> float:
        """
        Get IFB correction for a specific GLONASS satellite
        
        Parameters
        ----------
        freq_num : int
            GLONASS frequency number
            
        Returns
        -------
        float
            Correction in meters (to be subtracted from pseudorange)
        """
        return compute_glonass_ifb_correction(freq_num, self.ifb_rate)


# For the specific u-blox data we're analyzing
# Based on the analysis, the IFB rate is approximately -30,236 m/MHz
# This is unusually large and might indicate a problem with the RINEX conversion
UBLOX_KAIYODAI_IFB_RATE = -30236.5  # m/MHz