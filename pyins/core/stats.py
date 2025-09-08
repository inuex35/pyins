#!/usr/bin/env python
# Copyright 2024 pyins
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
RTK Statistical Parameters and Constants
=========================================

RTKLIB-compatible statistical parameters for RTK/PPP processing.
Based on RTKLIB prcopt_t structure.
"""

import numpy as np

# ============================================================================
# CODE/PHASE ERROR RATIO
# ============================================================================
ERATIO_L1 = 300.0    # Code/phase error ratio for L1
ERATIO_L2 = 300.0    # Code/phase error ratio for L2
ERATIO_L5 = 300.0    # Code/phase error ratio for L5

# Default error ratios for each frequency
DEFAULT_ERATIO = [ERATIO_L1, ERATIO_L2, ERATIO_L5]

# ============================================================================
# OBSERVATION ERROR MODEL
# ============================================================================
ERR_RESERVED = 1.0      # [0] Reserved
ERR_CONSTANT = 0.003    # [1] Base error constant (m)
ERR_ELEVATION = 0.003   # [2] Error per sin(elevation) (m)
ERR_BASELINE = 0.0      # [3] Error per baseline (m/10km)
ERR_DOPPLER = 1.0       # [4] Doppler error factor
ERR_SNR_MAX = 0.0       # [5] SNR threshold for error model
ERR_SNR = 0.0           # [6] SNR-based error (m per 10 dB-Hz)
ERR_RCV_STD = 0.0       # [7] Receiver std deviation

# Default observation error terms
DEFAULT_ERR = [
    ERR_RESERVED,
    ERR_CONSTANT,
    ERR_ELEVATION,
    ERR_BASELINE,
    ERR_DOPPLER,
    ERR_SNR_MAX,
    ERR_SNR,
    ERR_RCV_STD
]

# ============================================================================
# INITIAL STATE STANDARD DEVIATIONS
# ============================================================================
STD_BIAS = 30.0      # Ambiguity bias initial std (cycles)
STD_IONO = 0.03      # Ionosphere initial std (m)
STD_TROP = 0.3       # Troposphere initial std (m)

# Default initial standard deviations
DEFAULT_STD = [STD_BIAS, STD_IONO, STD_TROP]

# ============================================================================
# PROCESS NOISE STANDARD DEVIATIONS
# ============================================================================
PRN_BIAS = 1e-4      # Ambiguity bias process noise (cycles/sqrt(s))
PRN_IONO = 1e-3      # Ionosphere process noise (m/sqrt(s))
PRN_TROP = 1e-4      # Troposphere process noise (m/sqrt(s))
PRN_ACCH = 1.0       # Horizontal acceleration process noise (m/s²/sqrt(s))
PRN_ACCV = 1.0       # Vertical acceleration process noise (m/s²/sqrt(s))
PRN_POS = 0.0        # Position process noise (m/sqrt(s))

# Default process noise
DEFAULT_PRN = [PRN_BIAS, PRN_IONO, PRN_TROP, PRN_ACCH, PRN_ACCV, PRN_POS]

# ============================================================================
# AMBIGUITY RESOLUTION THRESHOLDS
# ============================================================================
THRESAR_RATIO = 3.0      # Ratio test threshold
THRESAR_WL = 0.25        # Wide-lane ambiguity threshold (cycles)
THRESAR_NL = 0.15        # Narrow-lane ambiguity threshold (cycles)
THRESAR_CONF = 0.999     # Min confidence for fix

# Default AR thresholds
DEFAULT_THRESAR = [
    THRESAR_RATIO,
    THRESAR_WL,
    THRESAR_NL,
    0.0,              # Reserved
    THRESAR_CONF,
    0.0,              # Reserved
    0.0,              # Reserved
    0.0               # Reserved
]

# ============================================================================
# ELEVATION MASKS
# ============================================================================
ELMASK = 15.0            # Elevation mask angle (degrees)
ELMASKAR = 0.0           # Elevation mask for AR on rising satellites (degrees)
ELMASKHOLD = 0.0         # Elevation mask to hold ambiguity (degrees)

# ============================================================================
# CYCLE SLIP THRESHOLDS
# ============================================================================
THRESSLIP_GF = 0.05      # Geometry-free phase slip threshold (m)
THRESSLIP_DOP = 10.0     # Doppler-based slip threshold (Hz)

# ============================================================================
# SATELLITE CLOCK
# ============================================================================
SCLKSTAB = 5e-12         # Satellite clock stability (sec/sec)

# ============================================================================
# INNOVATION THRESHOLDS
# ============================================================================
MAXINNO_PHASE = 30.0     # Max innovation for phase (m)
MAXINNO_CODE = 30.0      # Max innovation for code (m)

# Default innovation thresholds
DEFAULT_MAXINNO = [MAXINNO_PHASE, MAXINNO_CODE]

# ============================================================================
# TIME SYNCHRONIZATION
# ============================================================================
MAXTDIFF = 3.0           # Max time difference between rover and base (sec)

# ============================================================================
# FIX AND HOLD
# ============================================================================
VARHOLDAMB = 0.1         # Variance for fix-and-hold pseudo measurements (cycle²)
GAINHOLDAMB = 0.01       # Gain for GLONASS/SBAS ambiguity adjustment

# ============================================================================
# BASELINE CONSTRAINT
# ============================================================================
BASELINE_CONST = 0.0     # Baseline length constraint constant (m)
BASELINE_SIGMA = 0.0     # Baseline length constraint sigma (m)

# Default baseline constraint
DEFAULT_BASELINE = [BASELINE_CONST, BASELINE_SIGMA]

# ============================================================================
# PROCESSING OPTIONS
# ============================================================================

# Dynamics model
DYNAMICS_NONE = 0        # No dynamics
DYNAMICS_VEL = 1         # Velocity dynamics
DYNAMICS_ACC = 2         # Acceleration dynamics

# AR mode
ARMODE_OFF = 0           # AR off
ARMODE_CONT = 1          # Continuous AR
ARMODE_INST = 2          # Instantaneous AR
ARMODE_FIXHOLD = 3       # Fix and hold AR

# GLONASS AR mode
GLOMODE_OFF = 0          # GLONASS AR off
GLOMODE_ON = 1           # GLONASS AR on
GLOMODE_AUTOCAL = 2      # GLONASS AR with auto calibration

# BeiDou AR mode
BDSMODE_OFF = 0          # BeiDou AR off
BDSMODE_ON = 1           # BeiDou AR on

# Ionosphere option
IONOOPT_OFF = 0          # Ionosphere correction off
IONOOPT_BRDC = 1         # Broadcast ionosphere model
IONOOPT_SBAS = 2         # SBAS ionosphere model
IONOOPT_IFLC = 3         # Ionosphere-free LC
IONOOPT_EST = 4          # Estimate ionosphere

# Troposphere option
TROPOPT_OFF = 0          # Troposphere correction off
TROPOPT_SAAS = 1         # Saastamoinen model
TROPOPT_SBAS = 2         # SBAS troposphere model
TROPOPT_EST = 3          # Estimate troposphere

# Earth tide correction
TIDECORR_OFF = 0         # Tide correction off
TIDECORR_SOLID = 1       # Solid earth tide
TIDECORR_OTL = 2         # + Ocean tide loading
TIDECORR_POLE = 3        # + Pole tide

# ============================================================================
# DEFAULT CONFIGURATIONS
# ============================================================================

DEFAULT_STATS = {
    'eratio': DEFAULT_ERATIO,
    'err': DEFAULT_ERR,
    'std': DEFAULT_STD,
    'prn': DEFAULT_PRN,
    'thresar': DEFAULT_THRESAR,
    'elmask': ELMASK,
    'elmaskar': ELMASKAR,
    'elmaskhold': ELMASKHOLD,
    'thresslip': THRESSLIP_GF,
    'thresdop': THRESSLIP_DOP,
    'sclkstab': SCLKSTAB,
    'maxinno': DEFAULT_MAXINNO,
    'maxtdiff': MAXTDIFF,
    'varholdamb': VARHOLDAMB,
    'gainholdamb': GAINHOLDAMB,
    'baseline': DEFAULT_BASELINE,
    'niter': 1,
    'dynamics': DYNAMICS_NONE,
    'armode': ARMODE_FIXHOLD,
    'glomodear': GLOMODE_ON,
    'bdsmodear': BDSMODE_ON,
    'armaxiter': 1,
    'ionoopt': IONOOPT_BRDC,
    'tropopt': TROPOPT_SAAS,
    'tidecorr': TIDECORR_OFF
}

# High precision configuration
HIGH_PRECISION_STATS = {
    'eratio': [100.0, 100.0, 100.0],
    'err': [1.0, 0.001, 0.001, 0.0, 1.0, 0.0, 0.0, 0.0],
    'std': [10.0, 0.01, 0.1],
    'prn': [5e-5, 5e-4, 5e-5, 0.5, 0.5, 0.0],
    'thresar': [3.0, 0.15, 0.10, 0.0, 0.999, 0.0, 0.0, 0.0],
    'elmask': 10.0,
    'maxinno': [10.0, 10.0],
    'maxtdiff': 1.0,
}

# Robust configuration for challenging conditions
ROBUST_STATS = {
    'eratio': [500.0, 500.0, 500.0],
    'err': [1.0, 0.005, 0.005, 0.0, 1.0, 0.0, 0.0, 0.0],
    'std': [50.0, 0.05, 0.5],
    'prn': [2e-4, 2e-3, 2e-4, 2.0, 2.0, 0.0],
    'thresar': [2.5, 0.30, 0.20, 0.0, 0.95, 0.0, 0.0, 0.0],
    'elmask': 20.0,
    'maxinno': [50.0, 50.0],
    'maxtdiff': 5.0,
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_obs_variance(elevation, snr=0.0, baseline=0.0, is_phase=True, freq_idx=0,
                         err=None, eratio=None):
    """
    Compute observation variance using RTKLIB error model
    
    Parameters
    ----------
    elevation : float
        Satellite elevation angle (radians)
    snr : float
        Signal-to-noise ratio (dB-Hz)
    baseline : float
        Baseline length (m)
    is_phase : bool
        True for carrier phase, False for pseudorange
    freq_idx : int
        Frequency index (0=L1, 1=L2, 2=L5)
    err : list
        Observation error terms (default: DEFAULT_ERR)
    eratio : list
        Code/phase error ratios (default: DEFAULT_ERATIO)
        
    Returns
    -------
    float
        Observation variance (m²)
    """
    if err is None:
        err = DEFAULT_ERR
    if eratio is None:
        eratio = DEFAULT_ERATIO
    
    # Base error
    sigma = err[1]  # Constant term
    
    # Elevation-dependent error
    if elevation > 0:
        sigma += err[2] / np.sin(elevation)
    
    # Baseline-dependent error (per 10km)
    if baseline > 0:
        sigma += err[3] * baseline / 10000.0
    
    # SNR-dependent error
    if snr > 0 and err[5] > 0 and snr < err[5]:
        sigma += err[6] * (err[5] - snr) / 10.0
    
    # Apply code/phase error ratio
    if not is_phase:
        sigma *= eratio[min(freq_idx, len(eratio)-1)]
    
    return sigma ** 2

def get_process_noise(dt, prn=None):
    """
    Get process noise values for Kalman filter
    
    Parameters
    ----------
    dt : float
        Time interval (seconds)
    prn : list
        Process noise values (default: DEFAULT_PRN)
        
    Returns
    -------
    dict
        Process noise values for different states
    """
    if prn is None:
        prn = DEFAULT_PRN
    
    return {
        'ambiguity': (prn[0] * np.sqrt(dt)) ** 2,
        'ionosphere': (prn[1] * np.sqrt(dt)) ** 2,
        'troposphere': (prn[2] * np.sqrt(dt)) ** 2,
        'acc_horizontal': (prn[3] * np.sqrt(dt)) ** 2,
        'acc_vertical': (prn[4] * np.sqrt(dt)) ** 2,
        'position': (prn[5] * np.sqrt(dt)) ** 2 if prn[5] > 0 else 0.0
    }

def validate_innovation(innovation, is_phase=True, maxinno=None):
    """
    Check if innovation is within acceptable range
    
    Parameters
    ----------
    innovation : float
        Innovation value (m)
    is_phase : bool
        True for carrier phase, False for pseudorange
    maxinno : list
        Max innovation thresholds (default: DEFAULT_MAXINNO)
        
    Returns
    -------
    bool
        True if innovation is acceptable
    """
    if maxinno is None:
        maxinno = DEFAULT_MAXINNO
    
    threshold = maxinno[0] if is_phase else maxinno[1]
    return abs(innovation) < threshold

def check_ratio_test(ratio, threshold=THRESAR_RATIO):
    """
    Check if ratio test passes for ambiguity resolution
    
    Parameters
    ----------
    ratio : float
        Ratio test value
    threshold : float
        Ratio threshold (default: THRESAR_RATIO)
        
    Returns
    -------
    bool
        True if ratio test passes
    """
    return ratio >= threshold

def check_time_sync(time_diff, max_diff=MAXTDIFF):
    """
    Check if time difference is acceptable
    
    Parameters
    ----------
    time_diff : float
        Time difference between measurements (seconds)
    max_diff : float
        Maximum allowed difference (default: MAXTDIFF)
        
    Returns
    -------
    bool
        True if time sync is acceptable
    """
    return abs(time_diff) <= max_diff