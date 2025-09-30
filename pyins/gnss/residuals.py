"""Zero-difference residual helpers built on cssrlib."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from cssrlib.gnss import (char2sys, ecef2pos, geodist, rCST, satazel, sat2id,
                          tropmapf, tropmodel, uGNSS, uTYP)
from cssrlib.pntpos import ionmodel

from .geometry import compute_satellite_positions


@dataclass
class ResidualEntry:
    sat: int
    sat_id: str
    system: str
    signal: str | None
    pseudorange: float
    predicted: float
    residual: float
    geom: float
    trop: float
    iono: float
    sat_pos: np.ndarray
    sat_clk: float
    elevation: float


_CODE_PRIORITY = {
    # BeiDou priorities follow RTKLIB's codepris table (B2 first, then B1, B2/B3
    # complementary signals, etc.).
    'C': (
        # B2I/Q/X/D/P/Z/L/S/A/N
        'C2I', 'C2Q', 'C2X', 'C2D', 'C2P', 'C2Z', 'C2S', 'C2L', 'C2A', 'C2N',
        # B1I/Q/X/D/P/Z/L/S/A/N
        'C1I', 'C1Q', 'C1X', 'C1D', 'C1P', 'C1Z', 'C1S', 'C1L', 'C1A', 'C1N',
        # B2b (1207 MHz) family
        'C7I', 'C7Q', 'C7X', 'C7D', 'C7P', 'C7Z', 'C7S', 'C7L', 'C7A', 'C7N',
        # B3 family
        'C6I', 'C6Q', 'C6X', 'C6D', 'C6P', 'C6Z', 'C6S', 'C6L', 'C6A', 'C6N',
        # B2a family
        'C5I', 'C5Q', 'C5X', 'C5D', 'C5P', 'C5Z', 'C5S', 'C5L', 'C5A', 'C5N',
        # B5 family (if present)
        'C8I', 'C8Q', 'C8X', 'C8D', 'C8P', 'C8Z', 'C8S', 'C8L', 'C8A', 'C8N',
    ),
}


def _first_valid_pseudorange(p_values: np.ndarray, codes: list[str]) -> tuple[float, str] | None:
    for idx, value in enumerate(p_values):
        if value > 0:
            code = codes[idx] if idx < len(codes) else ''
            return float(value), code
    return None


def _preferred_pseudorange(obs_epoch, sat_idx: int, sys_char: str) -> tuple[float, str] | None:
    sys_enum = char2sys(sys_char) if sys_char else uGNSS.NONE
    if sys_enum == uGNSS.NONE:
        return None

    sig_map = getattr(obs_epoch, 'sig', None)
    if not sig_map or sys_enum not in sig_map:
        return None

    code_signals = sig_map[sys_enum].get(uTYP.C)
    if not code_signals:
        return None

    preferred_codes = _CODE_PRIORITY.get(sys_char)
    if not preferred_codes:
        return None

    for code in preferred_codes:
        for sig_idx, sig in enumerate(code_signals):
            if sig.str() != code:
                continue
            values = np.asarray(obs_epoch.P[sat_idx], dtype=float)
            if sig_idx < values.size and values[sig_idx] > 0:
                return float(values[sig_idx]), code
    return None


def _code_bias_correction(nav, sat: int, sys_char: str, signal_code: str | None) -> float:
    if not signal_code:
        return 0.0

    eph = next((candidate for candidate in getattr(nav, 'eph', [])
                if candidate.sat == sat), None)
    if eph is None:
        return 0.0

    code_upper = signal_code.upper()
    bias_sec = 0.0

    if sys_char in {'G', 'J', 'R', 'S'}:
        bias_sec = getattr(eph, 'tgd', 0.0) or 0.0
    elif sys_char == 'E':
        if '5' in code_upper or '7' in code_upper:
            bias_sec = getattr(eph, 'tgd_b', None)
            if bias_sec is None:
                bias_sec = getattr(eph, 'tgd', 0.0) or 0.0
        else:
            bias_sec = getattr(eph, 'tgd', 0.0) or 0.0
    elif sys_char == 'C':
        if code_upper.startswith(('C1', 'C2')):
            bias_sec = getattr(eph, 'tgd', 0.0) or 0.0
        elif code_upper.startswith(('C5', 'C6', 'C7', 'C8')):
            bias_sec = getattr(eph, 'tgd_b', None)
            if bias_sec is None:
                bias_sec = getattr(eph, 'tgd', 0.0) or 0.0
    else:
        bias_sec = getattr(eph, 'tgd', 0.0) or 0.0

    if bias_sec is None:
        bias_sec = 0.0

    return float(bias_sec) * rCST.CLIGHT


def _transmit_time(reception_time, pseudorange: float, sys_char: str):
    from cssrlib.gnss import timeadd

    return timeadd(reception_time, -pseudorange / rCST.CLIGHT)


def compute_zero_diff_residuals(obs_epoch, nav, receiver_pos: np.ndarray,
                                min_elevation_deg: float = 10.0) -> Dict[str, ResidualEntry]:
    """Return zero-difference residual information keyed by satellite ID."""

    positions, clocks, health = compute_satellite_positions(obs_epoch, nav)
    pos_llh = ecef2pos(receiver_pos)

    results: Dict[str, ResidualEntry] = {}

    for idx, sat in enumerate(obs_epoch.sat):
        sat_id = sat2id(sat)
        if not sat_id:
            continue
        sys_char = sat_id[0]

        if health[idx] != 0:
            continue

        preferred = _preferred_pseudorange(obs_epoch, idx, sys_char)
        if preferred is not None:
            pr, signal_code = preferred
        else:
            sig_map = getattr(obs_epoch, 'sig', None)
            code_signals = []
            if sig_map and char2sys(sys_char) in sig_map:
                for sig in sig_map[char2sys(sys_char)][uTYP.C]:
                    code_signals.append(sig.str())
            fallback = _first_valid_pseudorange(np.asarray(obs_epoch.P[idx], dtype=float), code_signals)
            if fallback is None:
                continue
            pr, signal_code = fallback

        pr -= _code_bias_correction(nav, sat, sys_char, signal_code)

        if pr is None:
            continue

        sat_pos = positions[idx]
        if not np.isfinite(sat_pos).all() or np.linalg.norm(sat_pos) < 1.0:
            continue

        geom_dist, los = geodist(sat_pos, receiver_pos)
        az, el = satazel(pos_llh, los)
        if np.degrees(el) < min_elevation_deg:
            continue

        trop_hs, trop_wet, _ = tropmodel(obs_epoch.t, pos_llh)
        mapfh, mapfw = tropmapf(obs_epoch.t, pos_llh, el)
        trop = mapfh * trop_hs + mapfw * trop_wet
        iono = ionmodel(obs_epoch.t, pos_llh, az, el, nav=nav)

        geom = geom_dist - rCST.CLIGHT * clocks[idx]
        predicted = geom + trop + iono
        residual = pr - predicted

        results[sat_id] = ResidualEntry(
            sat=sat,
            sat_id=sat_id,
            system=sys_char,
            signal=signal_code,
            pseudorange=pr,
            predicted=predicted,
            residual=residual,
            geom=geom,
            trop=trop,
            iono=iono,
            sat_pos=sat_pos.copy(),
            sat_clk=clocks[idx],
            elevation=el,
        )

    return results
