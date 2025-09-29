#!/usr/bin/env python3
"""
Double difference formation following RTKLIB-py approach.
Computes zero-difference residuals first, then forms double differences.
"""

import logging

import numpy as np
from cssrlib.ephemeris import satpos as cssr_satpos
from cssrlib.gnss import (
    gpst2bdt,
    gpst2time,
    gpst2utc,
    timeadd,
    ecef2pos,
    geodist,
    satazel,
    tropmodel,
    tropmapf,
)
from cssrlib.pntpos import ionmodel
from pyins.core.constants import CLIGHT, sat2prn, sat2sys, sys2char
from pyins.gnss.satposs import satposs
from pyins.gnss.ephemeris import seleph
from pyins.gnss.utils import to_cssrlib_sat_num


logger = logging.getLogger(__name__)


def _system_char(sat_num: int) -> str:
    return sys2char(sat2sys(sat_num)).strip()


def compute_zero_diff_residual(obs, sat_pos, sat_clk, receiver_pos, freq_idx=0):
    """
    Compute zero-difference residual for carrier phase like RTKLIB-py.

    RTKLIB: y = L * c/f - r
    where:
    - L: carrier phase observation (cycles)
    - c: speed of light
    - f: frequency
    - r: geometric range with clock corrections

    Returns:
        residual_m: Zero-difference residual in meters
        residual_cycles: Zero-difference residual in cycles
        range_m: Geometric range in meters
    """
    # Get frequency (GPS L1 default)
    if hasattr(obs, 'freq') and obs.freq[freq_idx] > 0:
        freq = obs.freq[freq_idx]
    else:
        freq = 1575.42e6  # GPS L1

    wavelength = CLIGHT / freq

    # Get carrier phase observation (cycles)
    cp_cycles = obs.L[freq_idx]
    if cp_cycles == 0:
        return None, None, None

    # Convert to meters
    cp_meters = cp_cycles * wavelength

    # Compute geometric range
    range_m = np.linalg.norm(sat_pos - receiver_pos)

    # Apply satellite clock correction (remove satellite clock bias)
    range_m -= CLIGHT * sat_clk

    # Zero-difference residual (meters)
    residual_m = cp_meters - range_m
    residual_cycles = residual_m / wavelength

    return residual_m, residual_cycles, range_m


def corrected_range(sat_pos, sat_clk, receiver_pos):
    """Geometric range corrected by satellite clock (meters)."""
    return float(np.linalg.norm(sat_pos - receiver_pos) - CLIGHT * sat_clk)


def cssrlib_measurement_terms(obs, receiver_pos, nav_raw, freq_idx=0):
    """Return cssrlib-style range/troposphere/ionosphere terms (meters)."""

    if nav_raw is None or obs.P[freq_idx] <= 0:
        return None

    gps_time = float(obs.time)
    week = int(gps_time // 604800)
    tow = gps_time - week * 604800.0
    t_obs = gpst2time(week, tow)

    sys_char = _system_char(obs.sat)
    if not sys_char:
        return None
    cssr_sat = to_cssrlib_sat_num(obs.sat)
    if cssr_sat <= 0:
        return None

    if sys_char == 'C':
        t_obs = gpst2bdt(t_obs)
    elif sys_char == 'R':
        t_obs = gpst2utc(t_obs)
        t_obs = timeadd(t_obs, 10800.0)

    t_tx = timeadd(t_obs, -obs.P[freq_idx] / CLIGHT)

    rs, _, dts, svh = cssr_satpos(cssr_sat, t_tx, nav_raw)
    if rs is None or np.isnan(rs).any() or dts is None or np.isnan(dts).any():
        return None

    sat_pos = rs[0]
    sat_clk = dts[0]

    geom_dist, los = geodist(sat_pos, receiver_pos)
    geom = geom_dist - CLIGHT * sat_clk

    pos_llh = ecef2pos(receiver_pos)
    az, el = satazel(pos_llh, los)

    trop_hs, trop_wet, _ = tropmodel(t_obs, pos_llh, model=getattr(nav_raw, 'trpModel', None))
    mapfh, mapfw = tropmapf(t_obs, pos_llh, el, model=getattr(nav_raw, 'trpModel', None))
    trop = mapfh * trop_hs + mapfw * trop_wet

    iono = ionmodel(t_obs, pos_llh, az, el, nav=nav_raw, model=getattr(nav_raw, 'ionoModel', None))

    return {
        'geom': float(geom),
        'trop': float(trop),
        'iono': float(iono),
        'sat_clk': float(sat_clk),
    }


def code_bias_correction(obs, freq_idx, nav_data):
    """Return broadcast code bias (TGD/DCB) in meters for given observation."""

    if nav_data is None:
        return 0.0

    sys_char = _system_char(obs.sat)
    if not sys_char:
        return 0.0

    eph = None
    try:
        eph = seleph(nav_data, obs.time, obs.sat)
    except Exception:
        eph = None

    if eph is None:
        return 0.0

    if sys_char in ('G', 'E', 'J', 'S', 'I'):
        tgd = getattr(eph, 'tgd', 0.0)
        return CLIGHT * float(tgd)

    if sys_char == 'C':  # BeiDou
        if freq_idx == 0 and hasattr(eph, 'tgd1') and eph.tgd1 is not None:
            return CLIGHT * float(eph.tgd1)
        if freq_idx == 1 and hasattr(eph, 'tgd2') and eph.tgd2 is not None:
            return CLIGHT * float(eph.tgd2)
        tgd = getattr(eph, 'tgd', 0.0)
        return CLIGHT * float(tgd)

    # GLONASS/others: handled through inter-channel biases elsewhere
    return 0.0


def form_double_differences(rover_obs, base_obs, nav_data, rover_time,
                           base_position, base_llh_deg,
                           rover_position=None,
                           use_systems=None, use_frequencies=None,
                           cutoff_angle=10.0,
                           base_obs_list=None, base_obs_index=None):
    """
    Form double differences using RTKLIB-py approach.

    Steps:
    1. Compute satellite positions for rover and base times
    2. Compute zero-difference residuals for each satellite
    3. Form single differences between rover and base
    4. Form double differences using reference satellite

    Parameters
    ----------
    rover_obs : list
        Rover observations
    base_obs : list
        Base observations
    nav_data : dict
        Navigation data
    rover_time : float
        Rover observation time (GPS seconds)
    base_position : array
        Base station ECEF position
    base_llh_deg : array
        Base station LLH position in degrees
    rover_position : array, optional
        Rover ECEF position (for satellite selection)
    use_systems : list, optional
        GNSS systems to use (default: ['G'])
    use_frequencies : list, optional
        Frequency indices to use (default: [0])
    cutoff_angle : float
        Elevation cutoff angle in degrees

    Returns
    -------
    list
        List of DD measurements with residuals
    """

    if use_systems is None:
        use_systems = ['G']
    if use_frequencies is None:
        use_frequencies = [0]

    nav_raw = getattr(nav_data, 'raw_nav', None)

    # Get base observation time
    base_time = base_obs[0].time if base_obs else rover_time

    # Compute satellite positions at rover and base times
    rover_sat_pos, rover_sat_clk, _, rover_svh = satposs(rover_obs, nav_data, rover_time)
    base_sat_pos, base_sat_clk, _, base_svh = satposs(base_obs, nav_data, base_time)

    # If no rover position provided, use base position as approximation
    if rover_position is None:
        rover_position = base_position + np.array([100, 100, 10])  # Rough offset

    dd_measurements = []

    # Process each frequency
    for freq_idx in use_frequencies:
        # Compute zero-difference residuals for all satellites
        rover_residuals = {}
        base_residuals = {}

        # Process rover observations
        for i, obs in enumerate(rover_obs):
            if rover_svh[i] != 0:
                continue
            if not np.isfinite(rover_sat_pos[i]).all() or np.linalg.norm(rover_sat_pos[i]) < 1.0:
                continue
            system = _system_char(obs.sat)
            if not system:
                continue
            if system not in use_systems:
                continue

            prn = sat2prn(obs.sat)
            if prn <= 0:
                continue

            sat_id = f"{system}{prn:03d}"

            residual_m, residual_cycles, range_m = compute_zero_diff_residual(
                obs, rover_sat_pos[i], rover_sat_clk[i], rover_position, freq_idx
            )

            terms = cssrlib_measurement_terms(obs, rover_position, nav_raw, freq_idx)
            geom_component = None
            trop_component = 0.0
            iono_component = 0.0
            cssr_range = None
            cssr_clk = None
            if terms is not None:
                geom_component = terms['geom']
                trop_component = terms['trop']
                iono_component = terms['iono']
                cssr_range = geom_component + trop_component + iono_component
                cssr_clk = terms['sat_clk']

            raw_pr = obs.P[freq_idx]
            bias = code_bias_correction(obs, freq_idx, nav_data)

            if residual_cycles is not None:
                rover_residuals[sat_id] = {
                    'residual_m': residual_m,
                    'residual_cycles': residual_cycles,
                    'range_m': range_m,
                    'geom': geom_component,
                    'sat_pos': rover_sat_pos[i],
                    'sat_clk': rover_sat_clk[i],
                    'sat_clk_cssr': cssr_clk,
                    'sat_num': obs.sat,
                    'obs': obs,
                    'raw_pr': raw_pr,
                    'bias': bias,
                    'trop': terms['trop'] if terms else 0.0,
                    'iono': terms['iono'] if terms else 0.0,
                    'cssr_range_m': cssr_range,
                }

        # Process base observations
        for i, obs in enumerate(base_obs):
            if base_svh[i] != 0:
                continue
            if not np.isfinite(base_sat_pos[i]).all() or np.linalg.norm(base_sat_pos[i]) < 1.0:
                continue
            system = _system_char(obs.sat)
            if not system:
                continue
            if system not in use_systems:
                continue

            prn = sat2prn(obs.sat)
            if prn <= 0:
                continue

            sat_id = f"{system}{prn:03d}"

            residual_m, residual_cycles, range_m = compute_zero_diff_residual(
                obs, base_sat_pos[i], base_sat_clk[i], base_position, freq_idx
            )

            terms = cssrlib_measurement_terms(obs, base_position, nav_raw, freq_idx)
            geom_component = None
            trop_component = 0.0
            iono_component = 0.0
            cssr_range = None
            cssr_clk = None
            if terms is not None:
                geom_component = terms['geom']
                trop_component = terms['trop']
                iono_component = terms['iono']
                cssr_range = geom_component + trop_component + iono_component
                cssr_clk = terms['sat_clk']

            raw_pr = obs.P[freq_idx]
            bias = code_bias_correction(obs, freq_idx, nav_data)

            if residual_cycles is not None:
                base_residuals[sat_id] = {
                    'residual_m': residual_m,
                    'residual_cycles': residual_cycles,
                    'range_m': range_m,
                    'geom': geom_component,
                    'sat_pos': base_sat_pos[i],
                    'sat_clk': base_sat_clk[i],
                    'sat_clk_cssr': cssr_clk,
                    'sat_num': obs.sat,
                    'obs': obs,
                    'raw_pr': raw_pr,
                    'bias': bias,
                    'trop': terms['trop'] if terms else 0.0,
                    'iono': terms['iono'] if terms else 0.0,
                    'cssr_range_m': cssr_range,
                }

        # Find common satellites
        common_sats = set(rover_residuals.keys()) & set(base_residuals.keys())

        if len(common_sats) < 2:
            continue  # Need at least 2 satellites for DD

        # Group satellites by GNSS system to avoid inter-system double differences
        system_groups: dict[str, list[str]] = {}
        for sat_id in common_sats:
            system_groups.setdefault(sat_id[0], []).append(sat_id)

        for system, system_sats in system_groups.items():
            if len(system_sats) < 2:
                continue  # Need at least two satellites within the same system

            # Select reference satellite for this system (simple alphabetical order)
            ref_sat_id = sorted(system_sats)[0]
            ref_rover = rover_residuals[ref_sat_id]
            ref_base = base_residuals[ref_sat_id]

            for sat_id in system_sats:
                if sat_id == ref_sat_id:
                    continue

                rover_data = rover_residuals[sat_id]
                base_data = base_residuals[sat_id]

                # Single difference residuals (rover - base)
                sd_residual_ref = ref_rover['residual_cycles'] - ref_base['residual_cycles']
                sd_residual_sat = rover_data['residual_cycles'] - base_data['residual_cycles']

                # Double difference residual
                dd_residual_cycles = sd_residual_sat - sd_residual_ref

                # Get frequency and wavelength
                if hasattr(rover_data['obs'], 'freq') and rover_data['obs'].freq[freq_idx] > 0:
                    freq = rover_data['obs'].freq[freq_idx]
                else:
                    freq = 1575.42e6
                wavelength = CLIGHT / freq

                # Also compute DD pseudorange
                rover_pr = rover_data['obs'].P[freq_idx]
                base_pr = base_data['obs'].P[freq_idx]
                ref_rover_pr = ref_rover['obs'].P[freq_idx]
                ref_base_pr = ref_base['obs'].P[freq_idx]

                rover_pr_corr = None
                ref_rover_pr_corr = None
                base_pr_corr = None
                ref_base_pr_corr = None

                if rover_pr > 0 and base_pr > 0 and ref_rover_pr > 0 and ref_base_pr > 0:
                    rover_bias = code_bias_correction(rover_data['obs'], freq_idx, nav_data)
                    ref_rover_bias = code_bias_correction(ref_rover['obs'], freq_idx, nav_data)
                    base_bias = code_bias_correction(base_data['obs'], freq_idx, nav_data)
                    ref_base_bias = code_bias_correction(ref_base['obs'], freq_idx, nav_data)

                    rover_clk_cssr = rover_data.get('sat_clk_cssr')
                    ref_rover_clk_cssr = ref_rover.get('sat_clk_cssr')
                    base_clk_cssr = base_data.get('sat_clk_cssr')
                    ref_base_clk_cssr = ref_base.get('sat_clk_cssr')

                    rover_clk_corr = rover_clk_cssr if rover_clk_cssr is not None else rover_data['sat_clk']
                    ref_rover_clk_corr = ref_rover_clk_cssr if ref_rover_clk_cssr is not None else ref_rover['sat_clk']
                    base_clk_corr = base_clk_cssr if base_clk_cssr is not None else base_data['sat_clk']
                    ref_base_clk_corr = ref_base_clk_cssr if ref_base_clk_cssr is not None else ref_base['sat_clk']

                    rover_pr_corr = rover_pr + CLIGHT * rover_clk_corr - rover_bias
                    rover_pr_corr -= rover_data.get('trop', 0.0) + rover_data.get('iono', 0.0)
                    ref_rover_pr_corr = ref_rover_pr + CLIGHT * ref_rover_clk_corr - ref_rover_bias
                    ref_rover_pr_corr -= ref_rover.get('trop', 0.0) + ref_rover.get('iono', 0.0)
                    base_pr_corr = base_pr + CLIGHT * base_clk_corr - base_bias
                    base_pr_corr -= base_data.get('trop', 0.0) + base_data.get('iono', 0.0)
                    ref_base_pr_corr = ref_base_pr + CLIGHT * ref_base_clk_corr - ref_base_bias
                    ref_base_pr_corr -= ref_base.get('trop', 0.0) + ref_base.get('iono', 0.0)
                    dd_pr = (rover_pr_corr - ref_rover_pr_corr) - (base_pr_corr - ref_base_pr_corr)
                else:
                    dd_pr = None

                # Get original carrier phase observations for verification
                rover_cp = rover_data['obs'].L[freq_idx]
                base_cp = base_data['obs'].L[freq_idx]
                ref_rover_cp = ref_rover['obs'].L[freq_idx]
                ref_base_cp = ref_base['obs'].L[freq_idx]

                # DD carrier phase (raw observation)
                dd_cp_obs = (rover_cp - ref_rover_cp) - (base_cp - ref_base_cp)

                dd_range_m = None
                dd_range_cycles = None
                dd_residual_m = None
                dd_residual_cycles_geom = None
                try:
                    rover_geom = corrected_range(rover_data['sat_pos'], rover_data['sat_clk'], rover_position)
                    ref_rover_geom = corrected_range(ref_rover['sat_pos'], ref_rover['sat_clk'], rover_position)
                    base_geom = corrected_range(base_data['sat_pos'], base_data['sat_clk'], base_position)
                    ref_base_geom = corrected_range(ref_base['sat_pos'], ref_base['sat_clk'], base_position)
                    dd_range_m = (rover_geom - ref_rover_geom) - (base_geom - ref_base_geom)
                    if wavelength:
                        dd_range_cycles = dd_range_m / wavelength
                except Exception:
                    geom_ranges = (
                        rover_data.get('geom'),
                        ref_rover.get('geom'),
                        base_data.get('geom'),
                        ref_base.get('geom'),
                    )

                    if all(val is not None for val in geom_ranges):
                        rover_range, ref_rover_range, base_range, ref_base_range = geom_ranges
                    else:
                        rover_range = rover_data.get('range_m')
                        ref_rover_range = ref_rover.get('range_m')
                        base_range = base_data.get('range_m')
                        ref_base_range = ref_base.get('range_m')

                    if None not in (rover_range, ref_rover_range, base_range, ref_base_range):
                        dd_range_m = (rover_range - ref_rover_range) - (base_range - ref_base_range)
                        if wavelength:
                            dd_range_cycles = dd_range_m / wavelength

                if dd_range_m is not None and dd_pr is not None:
                    dd_residual_m = dd_pr - dd_range_m

                if dd_range_cycles is not None and dd_cp_obs is not None:
                    dd_residual_cycles_geom = dd_cp_obs - dd_range_cycles

                if dd_residual_cycles_geom is not None:
                    dd_residual_cycles = dd_residual_cycles_geom

                dd_entry = {
                    'sat': rover_data['sat_num'],
                    'ref_sat': ref_rover['sat_num'],
                    'sys': system,
                    'freq_idx': freq_idx,
                    'dd_obs': dd_pr,
                    'dd_carrier': dd_cp_obs,
                    'dd_residual': dd_residual_cycles,
                    'wavelength': wavelength,
                    'dd_range_m': dd_range_m,
                    'dd_range_cycles': dd_range_cycles,
                    'dd_residual_m': dd_residual_m,
                    'sat_pos': rover_data['sat_pos'],
                    'ref_sat_pos': ref_rover['sat_pos'],
                    'sat_clk': rover_data['sat_clk'],
                    'ref_sat_clk': ref_rover['sat_clk'],
                    'rover_residual': rover_data['residual_cycles'],
                    'base_residual': base_data['residual_cycles'],
                    'ref_rover_residual': ref_rover['residual_cycles'],
                    'ref_base_residual': ref_base['residual_cycles'],
                    'base_sat_pos': base_data['sat_pos'],
                    'base_ref_sat_pos': ref_base['sat_pos'],
                    'base_sat_clk': base_data['sat_clk'],
                    'base_ref_sat_clk': ref_base['sat_clk'],
                    'rover_cp_ref': ref_rover_cp,
                    'rover_cp_other': rover_cp,
                    'base_cp_ref': ref_base_cp,
                    'base_cp_other': base_cp
                }

                if logger.isEnabledFor(logging.DEBUG):
                    def _dbg(data, corr):
                        return {
                            'raw_pr': float(data.get('raw_pr', float('nan'))),
                            'sat_clk_m': float(CLIGHT * data['sat_clk']) if data.get('sat_clk') is not None else None,
                            'sat_clk_cssr_m': float(CLIGHT * data['sat_clk_cssr']) if data.get('sat_clk_cssr') is not None else None,
                            'bias_m': float(data.get('bias', 0.0)),
                            'trop_m': float(data.get('trop', 0.0)),
                            'iono_m': float(data.get('iono', 0.0)),
                            'geom_m': float(data.get('geom', data.get('range_m', 0.0))) if data.get('geom') is not None else None,
                            'cssr_range_m': float(data.get('cssr_range_m')) if data.get('cssr_range_m') is not None else None,
                            'range_sum_m': float(data.get('range_m', 0.0)),
                            'corrected_pr': float(corr) if corr is not None else None,
                        }

                    dd_entry['debug'] = {
                        'rover': _dbg(rover_data, locals().get('rover_pr_corr')),
                        'rover_ref': _dbg(ref_rover, locals().get('ref_rover_pr_corr')),
                        'base': _dbg(base_data, locals().get('base_pr_corr')),
                        'base_ref': _dbg(ref_base, locals().get('ref_base_pr_corr')),
                        'dd_obs': float(dd_pr) if dd_pr is not None else None,
                        'dd_range': float(dd_range_m) if dd_range_m is not None else None,
                        'dd_residual_m': float(dd_residual_m) if dd_residual_m is not None else None,
                    }

                    logger.debug(
                        "DD %s %s-%s f=%d dd_obs=%.3f dd_range=%.3f dd_res=%.3f",
                        system,
                        rover_data['sat_num'],
                        ref_rover['sat_num'],
                        freq_idx,
                        float(dd_pr) if dd_pr is not None else float('nan'),
                        float(dd_range_m) if dd_range_m is not None else float('nan'),
                        float(dd_residual_m) if dd_residual_m is not None else float('nan'),
                    )

                dd_measurements.append(dd_entry)

    return dd_measurements
