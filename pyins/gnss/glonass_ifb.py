"""GLONASS Inter-Frequency Bias (IFB) handling

GLONASS uses FDMA (Frequency Division Multiple Access), where each satellite
transmits on a different frequency. This causes receiver-dependent biases
known as Inter-Frequency Biases (IFB).
"""

import numpy as np
from typing import Dict, Optional
from ..core.constants import FREQ_G1, DFREQ_G1


class GLONASSBias:
    """Handle GLONASS Inter-Frequency Bias (IFB)"""
    
    def __init__(self):
        """Initialize GLONASS bias handler"""
        self.ifb_estimates = {}  # Per-channel IFB estimates
        self.reference_channel = 0  # Reference channel (usually 0)
        
    def get_glonass_frequency(self, channel: int, band: int = 1) -> float:
        """Get GLONASS frequency for given channel
        
        Parameters
        ----------
        channel : int
            GLONASS frequency channel number (-7 to +6)
        band : int
            Frequency band (1 for G1/L1, 2 for G2/L2)
            
        Returns
        -------
        float
            Frequency in Hz
        """
        if band == 1:
            return FREQ_G1 + channel * DFREQ_G1
        else:
            # G2 frequency
            from ..core.constants import FREQ_G2, DFREQ_G2
            return FREQ_G2 + channel * DFREQ_G2
    
    def estimate_ifb(self, residuals: Dict[int, float], channels: Dict[int, int]) -> Dict[int, float]:
        """Estimate Inter-Frequency Bias from residuals
        
        Parameters
        ----------
        residuals : dict
            Pseudorange residuals per satellite
        channels : dict
            GLONASS frequency channels per satellite
            
        Returns
        -------
        dict
            Estimated IFB per channel
        """
        # Group residuals by channel
        channel_residuals = {}
        for sat, res in residuals.items():
            if sat in channels:
                ch = channels[sat]
                if ch not in channel_residuals:
                    channel_residuals[ch] = []
                channel_residuals[ch].append(res)
        
        # Calculate mean residual per channel
        ifb = {}
        ref_bias = 0.0
        
        # Find reference channel bias
        if self.reference_channel in channel_residuals:
            ref_bias = np.mean(channel_residuals[self.reference_channel])
        
        # Calculate relative IFB
        for ch, res_list in channel_residuals.items():
            if len(res_list) > 0:
                ifb[ch] = np.mean(res_list) - ref_bias
        
        self.ifb_estimates = ifb
        return ifb
    
    def apply_ifb_correction(self, pseudorange: float, channel: int) -> float:
        """Apply IFB correction to pseudorange
        
        Parameters
        ----------
        pseudorange : float
            Raw pseudorange measurement
        channel : int
            GLONASS frequency channel
            
        Returns
        -------
        float
            Corrected pseudorange
        """
        if channel in self.ifb_estimates:
            return pseudorange - self.ifb_estimates[channel]
        return pseudorange
    
    def get_ifb_model(self, channel: int, receiver_type: str = "generic") -> float:
        """Get IFB model value for given channel
        
        Parameters
        ----------
        channel : int
            GLONASS frequency channel
        receiver_type : str
            Receiver type for specific models
            
        Returns
        -------
        float
            Model IFB value in meters
        """
        # Simple linear model: IFB proportional to frequency offset
        # This is a simplified model; real IFB is receiver-dependent
        if receiver_type == "generic":
            # Generic linear model: ~0.5 m per channel
            return channel * 0.5
        
        # Add specific receiver models here if known
        return 0.0


def get_glonass_channel(sat_id, nav_data=None) -> int:
    """Get GLONASS frequency channel from satellite ID
    
    Parameters
    ----------
    sat_id : int
        Satellite ID (70-90 for GLONASS)
    nav_data : NavigationData, optional
        Navigation data containing GLONASS ephemerides
        
    Returns
    -------
    int
        Frequency channel number (-7 to +6)
    """
    # If nav_data is provided, look up the frequency channel
    if nav_data is not None and hasattr(nav_data, 'geph'):
        for geph in nav_data.geph:
            if geph.sat == sat_id and hasattr(geph, 'frq'):
                return geph.frq
    
    # Default mapping if nav_data not available or channel not found
    # This is a simplified mapping and may not be accurate
    # Real channels should come from RINEX navigation file
    default_channels = {
        71: 5,   # R01
        72: 6,   # R02  
        73: -2,  # R03
        74: -7,  # R04
        75: 1,   # R05
        76: -4,  # R06
        77: 5,   # R07
        78: 6,   # R08
        79: -2,  # R09
        80: -1,  # R10
        81: 4,   # R11
        82: -3,  # R12
        83: 3,   # R13
        84: 2,   # R14
        85: 4,   # R15
        86: -3,  # R16
        87: 3,   # R17
        88: 2,   # R18
        89: -1,  # R19
        90: -4,  # R20
    }
    
    return default_channels.get(sat_id, 0)