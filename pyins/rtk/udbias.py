#!/usr/bin/env python3
"""
Update phase biases for DD positioning (based on RTKLIB's udbias)
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

# Constants
CLIGHT = 299792458.0
MAXSAT = 64
MAXFREQ = 3

def IB(sat, freq, na):
    """Get index for phase bias state"""
    return na + MAXSAT * freq + sat - 1

def initx(nav, value, variance, idx):
    """Initialize state and covariance"""
    nav.x[idx] = value
    n = len(nav.x)
    nav.P[idx, :] = 0.0
    nav.P[:, idx] = 0.0
    nav.P[idx, idx] = variance

def sat2freq(sat, freq_idx, nav):
    """Get satellite frequency in Hz"""
    # Simplified - returns L1/L2/L5 frequencies
    from pyins.gnss.constants import FREQ_L1, FREQ_L2, FREQ_L5

    if freq_idx == 0:
        return FREQ_L1
    elif freq_idx == 1:
        return FREQ_L2
    elif freq_idx == 2:
        return FREQ_L5
    return 0.0

def detslp_ll(nav, obs, indices, rcv):
    """Detect cycle slip by LLI flag"""
    for i in indices:
        if hasattr(obs, 'LLI') and obs.LLI[i] & 1:
            sat = obs.sat[i]
            for f in range(nav.nf):
                nav.slip[sat-1, f] |= 1
                logger.debug(f"Cycle slip detected (LLI): sat={sat} rcv={rcv}")

def detslp_gf(nav, obsb, obsr, iu, ir):
    """Detect cycle slip by geometry-free combination"""
    # Simplified implementation
    pass

def detslp_dop(rcv, nav, obs, indices):
    """Detect cycle slip by doppler"""
    # Simplified implementation
    pass

def udbias(nav, obsb, obsr, iu, ir):
    """
    Update phase biases for double difference positioning (RTKLIB udbias equivalent)

    Parameters
    ----------
    nav : Navigation state object
        Contains state vector x, covariance P, and tracking info
    obsb : Base observations
    obsr : Rover observations
    iu : Rover satellite indices
    ir : Base satellite indices

    Notes
    -----
    This function:
    1. Detects cycle slips
    2. Updates phase bias states
    3. Initializes new ambiguities from PR-CP difference
    4. Maintains phase-code coherency
    """

    logger.debug(f'udbias: ns={len(iu)}')

    # Initialize navigation state attributes if not present
    if not hasattr(nav, 'x'):
        nav.x = np.zeros(256)  # State vector
    if not hasattr(nav, 'P'):
        nav.P = np.eye(256) * 1000.0  # Covariance
    if not hasattr(nav, 'slip'):
        nav.slip = np.zeros((MAXSAT, MAXFREQ), dtype=int)
    if not hasattr(nav, 'rejc'):
        nav.rejc = np.zeros((MAXSAT, MAXFREQ), dtype=int)
    if not hasattr(nav, 'outc'):
        nav.outc = np.zeros((MAXSAT, MAXFREQ), dtype=int)
    if not hasattr(nav, 'lock'):
        nav.lock = np.zeros((MAXSAT, MAXFREQ), dtype=int)
    if not hasattr(nav, 'nf'):
        nav.nf = 1  # Number of frequencies
    if not hasattr(nav, 'na'):
        nav.na = 3  # Number of position states
    if not hasattr(nav, 'maxout'):
        nav.maxout = 5
    if not hasattr(nav, 'prnbias'):
        nav.prnbias = 0.0001  # Process noise for phase bias (cycles/sqrt(s))
    if not hasattr(nav, 'sig_n0'):
        nav.sig_n0 = 100.0  # Initial phase bias sigma (cycles)
    if not hasattr(nav, 'tt'):
        nav.tt = 1.0  # Time difference

    # Cycle slip detection from receiver flags
    detslp_ll(nav, obsb, ir, 0)
    detslp_ll(nav, obsr, iu, 1)

    # Cycle slip detection by doppler and geometry-free
    detslp_dop(0, nav, obsb, ir)  # base
    detslp_dop(1, nav, obsr, iu)  # rover
    detslp_gf(nav, obsb, obsr, iu, ir)

    # Get satellite list
    ns = len(iu)
    sat = [obsr.sat[i] for i in iu] if hasattr(obsr, 'sat') else list(range(1, ns+1))

    # Update outage counters
    nav.outc += 1

    # Process each frequency
    for f in range(nav.nf):
        # Reset phase biases for satellites with long outage
        for i in range(MAXSAT):
            ii = IB(i+1, f, nav.na)
            if nav.outc[i, f] > nav.maxout and nav.x[ii] != 0.0:
                logger.debug(f'  Outage counter overflow: sat={i+1} L{f+1} n={nav.outc[i,f]}')
                initx(nav, 0, 0, ii)

        # Update phase bias process noise and check for slips
        for i in range(ns):
            j = IB(sat[i], f, nav.na)
            nav.P[j, j] += nav.prnbias**2 * abs(nav.tt)

            if (nav.slip[sat[i]-1, f] & 1) or nav.rejc[sat[i]-1, f] > 1:
                logger.debug(f'Reset phase bias: sat={sat[i]} f={f} slip={nav.slip[sat[i]-1,f]} rejc={nav.rejc[sat[i]-1,f]}')
                initx(nav, 0, 0, j)

        # Estimate phase bias from PR-CP difference
        bias = np.zeros(ns)
        offset = 0.0
        namb = 0

        for i in range(ns):
            freq = sat2freq(sat[i], f, nav)

            # Check if observations are valid
            L_rover = obsr.L[iu[i], f] if hasattr(obsr, 'L') and iu[i] < len(obsr.L) else 0
            L_base = obsb.L[ir[i], f] if hasattr(obsb, 'L') and ir[i] < len(obsb.L) else 0
            P_rover = obsr.P[iu[i], f] if hasattr(obsr, 'P') and iu[i] < len(obsr.P) else 0
            P_base = obsb.P[ir[i], f] if hasattr(obsb, 'P') and ir[i] < len(obsb.P) else 0

            if L_rover == 0 or L_base == 0 or P_rover == 0 or P_base == 0:
                continue

            # Calculate single differences
            cp = L_rover - L_base  # Carrier phase SD in cycles
            pr = P_rover - P_base  # Pseudorange SD in meters

            if cp == 0 or pr == 0 or freq == 0:
                continue

            # Estimate bias in cycles (RTKLIB formula)
            bias[i] = cp - pr * freq / CLIGHT

            # Accumulate offset for phase-code coherency
            x = nav.x[IB(sat[i], f, nav.na)]
            if x != 0.0:
                offset += bias[i] - x
                namb += 1

        # Correct phase-bias offset to ensure phase-code coherency
        if namb > 0:
            offset = offset / namb
            logger.debug(f'Phase-code coherency adjust={offset:.2f} cycles, n={namb}')

            # Apply offset to all non-zero phase bias states
            ib1 = IB(1, f, nav.na)  # First satellite bias index
            for i in range(MAXSAT):
                idx = ib1 + i
                if idx < len(nav.x) and nav.x[idx] != 0.0:
                    nav.x[idx] += offset

        # Initialize new phase biases
        for i in range(ns):
            j = IB(sat[i], f, nav.na)
            if bias[i] == 0.0 or nav.x[j] != 0.0:
                continue

            # Set initial state of phase bias
            initx(nav, bias[i], nav.sig_n0**2, j)
            nav.outc[sat[i]-1, f] = 1
            nav.rejc[sat[i]-1, f] = 0
            nav.lock[sat[i]-1, f] = 0

            logger.info(f"Initialize phase bias: sat={sat[i]:3d} F={f+1} bias={bias[i]:.3f} cycles")

    return nav


def get_ambiguity_from_pr_cp(pr_dd, cp_dd, wavelength):
    """
    Get integer ambiguity estimate from PR-CP difference

    Parameters
    ----------
    pr_dd : float
        Double differenced pseudorange in meters
    cp_dd : float
        Double differenced carrier phase in cycles
    wavelength : float
        Carrier wavelength in meters

    Returns
    -------
    int
        Estimated integer ambiguity in cycles
    """
    # Convert PR to cycles
    pr_cycles = pr_dd / wavelength

    # Ambiguity = PR - CP (in cycles)
    ambiguity = pr_cycles - cp_dd

    # Round to nearest integer
    return round(ambiguity)