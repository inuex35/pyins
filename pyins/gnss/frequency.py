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

"""GNSS frequency management functions"""

from ..core.constants import *
from ..core.constants import sat2sys


def sat2freq(sat, frq_idx=0, glo_fcn=0):
    """
    Get carrier frequency for a satellite
    
    Parameters
    ----------
    sat : int
        Satellite number
    frq_idx : int
        Frequency index (0=L1/B1I/E1, 1=L2/B2I/E5b, 2=L5/B2a/E5a)
    glo_fcn : int
        GLONASS frequency channel number (-7 to +6)
    
    Returns
    -------
    float
        Carrier frequency in Hz
    """
    sys = sat2sys(sat)
    
    if sys == SYS_GPS:
        if frq_idx == 0:
            return FREQ_L1
        elif frq_idx == 1:
            return FREQ_L2
        elif frq_idx == 2:
            return FREQ_L5
            
    elif sys == SYS_GLO:
        if frq_idx == 0:
            return FREQ_G1 + glo_fcn * DFREQ_G1
        elif frq_idx == 1:
            return FREQ_G2 + glo_fcn * DFREQ_G2
            
    elif sys == SYS_GAL:
        if frq_idx == 0:
            return FREQ_E1
        elif frq_idx == 1:
            return FREQ_E5b
        elif frq_idx == 2:
            return FREQ_E5a
            
    elif sys == SYS_BDS:
        if frq_idx == 0:
            return FREQ_B1I  # B1I for BeiDou MEO/IGSO
        elif frq_idx == 1:
            return FREQ_B3   # B3 for BeiDou  
        elif frq_idx == 2:
            return FREQ_B2a  # B2a
            
    elif sys == SYS_QZS:
        if frq_idx == 0:
            return FREQ_J1
        elif frq_idx == 1:
            return FREQ_J2
        elif frq_idx == 2:
            return FREQ_J5
            
    elif sys == SYS_SBS:
        if frq_idx == 0:
            return FREQ_S1
        elif frq_idx == 2:
            return FREQ_S5
            
    elif sys == SYS_IRN:
        if frq_idx == 0:
            return FREQ_I5
        elif frq_idx == 1:
            return FREQ_IS
            
    return 0.0

