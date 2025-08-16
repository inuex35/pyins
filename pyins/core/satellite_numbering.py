"""Unified satellite numbering system for pyins"""

# Define system IDs (duplicated from constants.py to avoid circular import)
SYS_NONE = 0x00
SYS_GPS = 0x01
SYS_GLO = 0x02
SYS_GAL = 0x04
SYS_BDS = 0x08
SYS_QZS = 0x10
SYS_SBS = 0x20
SYS_IRN = 0x40

# Define satellite number ranges for each system
SATELLITE_RANGES = {
    SYS_GPS: [(1, 32)],                    # GPS: 1-32
    SYS_SBS: [(33, 64), (133, 140)],      # SBAS: 33-64, 133-140
    SYS_GLO: [(65, 88)],                   # GLONASS: 65-88
    SYS_GAL: [(97, 132)],                  # Galileo: 97-132
    SYS_BDS: [(141, 203)],                 # BeiDou: 141-203 (BDS-2: 141-177, BDS-3: 178-203)
    SYS_QZS: [(210, 216)],                 # QZSS: 210-216
    SYS_IRN: [(230, 243)],                 # IRNSS: 230-243
}

# System ID to character mapping
SYS_TO_CHAR = {
    SYS_GPS: 'G',
    SYS_GLO: 'R',
    SYS_GAL: 'E',
    SYS_BDS: 'C',
    SYS_QZS: 'J',
    SYS_SBS: 'S',
    SYS_IRN: 'I',
}

# Character to system ID mapping
CHAR_TO_SYS = {v: k for k, v in SYS_TO_CHAR.items()}

def prn_to_sat(system_char, prn):
    """
    Convert system character and PRN to internal satellite number
    
    Parameters
    ----------
    system_char : str
        Single character system identifier ('G', 'R', 'E', 'C', 'J', etc.)
    prn : int
        PRN number
        
    Returns
    -------
    int
        Internal satellite number, or 0 if invalid
        
    Satellite number ranges:
    - GPS (G): 1-32 -> 1-32
    - SBAS (S): 33-64 -> 33-64  
    - GLONASS (R): 1-24 -> 65-88
    - Reserved: 89-96
    - Galileo (E): 1-36 -> 97-132
    - SBAS (S): 133-140 -> 133-140
    - BeiDou-2 (C): 1-37 -> 141-177
    - BeiDou-3 (C): 38-63 -> 178-203
    - Reserved: 204-209
    - QZSS (J): 1-7 -> 210-216
    - Reserved: 200
    - IRNSS (I): 1-14 -> 201-214
    """
    if system_char == 'G':  # GPS
        if 1 <= prn <= 32:
            return prn
    elif system_char == 'R':  # GLONASS
        if 1 <= prn <= 24:
            return prn + 64
    elif system_char == 'E':  # Galileo
        if 1 <= prn <= 36:
            return prn + 96
    elif system_char == 'C':  # BeiDou
        if 1 <= prn <= 37:
            return prn + 140  # BeiDou-2
        elif 38 <= prn <= 63:
            return prn + 140  # BeiDou-3 (178-203)
    elif system_char == 'J':  # QZSS
        if 1 <= prn <= 7:
            return prn + 209  # J01-J07 -> 210-216
    elif system_char == 'S':  # SBAS
        if 120 <= prn <= 151:
            return prn - 87  # SBAS 120-151 -> 33-64
        elif 152 <= prn <= 159:
            return prn - 19  # SBAS 152-159 -> 133-140
    elif system_char == 'I':  # IRNSS
        if 1 <= prn <= 14:
            return prn + 200
    
    return 0  # Invalid


def sat_to_prn(sat):
    """
    Convert internal satellite number to PRN
    
    Parameters
    ----------
    sat : int
        Internal satellite number
        
    Returns
    -------
    int
        PRN number, or 0 if invalid
    """
    if sat <= 0 or sat > 255:
        return 0
    elif 1 <= sat <= 32:  # GPS
        return sat
    elif 33 <= sat <= 64:  # SBAS
        return sat - 33 + 120
    elif 65 <= sat <= 88:  # GLONASS
        return sat - 64
    elif 97 <= sat <= 132:  # Galileo
        return sat - 96
    elif 133 <= sat <= 140:  # SBAS
        return sat - 133 + 152
    elif 141 <= sat <= 203:  # BeiDou
        return sat - 140
    elif 210 <= sat <= 216:  # QZSS
        return sat - 209
    elif 230 <= sat <= 243:  # IRNSS
        return sat - 229
    else:
        return 0