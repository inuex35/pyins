"""RINEX readers backed by cssrlib.rinex."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from cssrlib.gnss import Nav, time2gpst, gtime_t, uGNSS, uTYP, sat2id
from cssrlib.rinex import rnxdec

from ..core.constants import MAXBAND, sat2sys, prn2sat, char2sys
from ..core.data_structures import NavigationData, Observation


def _gtime_to_gps_seconds(gt: gtime_t) -> float:
    return float(gt.time) + gt.sec


def _gtime_to_unix(gt: gtime_t) -> float:
    return float(gt.time) + gt.sec


def _ensure_array(values, length: int) -> np.ndarray:
    arr = np.zeros(length, dtype=np.float64)
    n = min(len(values), length)
    if n:
        arr[:n] = values[:n]
    return arr


class RinexNavReader:
    """Read broadcast navigation data using cssrlib."""

    def __init__(self, filename: str):
        self.filename = Path(filename)

    def read(self) -> NavigationData:
        if not self.filename.exists():
            raise FileNotFoundError(self.filename)

        decoder = rnxdec()
        nav = Nav()
        decoder.decode_nav(str(self.filename), nav, append=False)

        nav_data = NavigationData()
        nav_data.eph = [self._remap_satellite(eph) for eph in nav.eph]
        nav_data.geph = [self._remap_satellite(geph) for geph in nav.geph]
        nav_data.peph = list(nav.peph)
        nav_data.pclk = list(nav.dclk)
        nav_data.leaps = nav.leaps

        if nav.ion is not None:
            if nav.ion.shape[0] >= 1:
                nav_data.ion_gps = np.array(nav.ion[0]).copy()
            if nav.ion.shape[0] >= 2:
                nav_data.ion_bds = np.array(nav.ion[1]).copy()

        nav_data.sto_prm = nav.sto_prm
        nav_data.eop_prm = nav.eop_prm
        nav_data.ion_prm = nav.ion_prm

        return nav_data

    @staticmethod
    def _remap_satellite(eph):
        """Convert cssrlib satellite numbering to pyins unified numbering."""

        if eph is None or not hasattr(eph, "sat"):
            return eph

        try:
            sat_id = sat2id(eph.sat)
        except Exception:
            return eph

        if not sat_id or len(sat_id) < 2:
            return eph

        sys_char = sat_id[0]
        try:
            prn = int(sat_id[1:])
        except ValueError:
            return eph

        sys = char2sys(sys_char)
        new_sat = prn2sat(prn, sys)
        if new_sat:
            eph.sat = new_sat
        return eph


class RinexObsReader:
    """Read observation epochs using cssrlib."""

    def __init__(self, filename: str):
        self.filename = Path(filename)

    def read(self) -> List[Dict[str, object]]:
        if not self.filename.exists():
            raise FileNotFoundError(self.filename)

        decoder = rnxdec()
        if decoder.decode_obsh(str(self.filename)) < 0:
            raise RuntimeError("Unsupported RINEX version")

        # Build signal table from header to keep all signals
        decoder.setSignals([])
        for sys, sigs in decoder.sig_map.items():
            for sig in sigs.values():
                if sys not in decoder.sig_tab:
                    decoder.sig_tab[sys] = {}
                decoder.sig_tab[sys].setdefault(sig.typ, [])
                if sig not in decoder.sig_tab[sys][sig.typ]:
                    decoder.sig_tab[sys][sig.typ].append(sig)

        # Ensure reshape() dimensions match by normalizing counts per type
        max_counts = {typ: 0 for typ in decoder.nsig}
        for sigs in decoder.sig_tab.values():
            for typ, sig_list in sigs.items():
                max_counts[typ] = max(max_counts[typ], len(sig_list))

        for typ, count in max_counts.items():
            decoder.nsig[typ] = count

        for sys, sigs in decoder.sig_tab.items():
            for typ, sig_list in sigs.items():
                if not sig_list:
                    continue
                while len(sig_list) < max_counts[typ]:
                    sig_list.append(sig_list[-1])

        epochs: List[Dict[str, object]] = []

        while True:
            obs = decoder.decode_obs()
            if obs is None or len(obs.sat) == 0:
                break

            gps_week, gps_tow = time2gpst(obs.t)
            gps_seconds = gps_week * 604800.0 + gps_tow

            epoch_obs: List[Observation] = []
            for idx, sat in enumerate(obs.sat):
                P = _ensure_array(obs.P[idx], MAXBAND)
                L = _ensure_array(obs.L[idx], MAXBAND)
                S = _ensure_array(obs.S[idx], MAXBAND)
                LLI = _ensure_array(obs.lli[idx], MAXBAND).astype(int)
                D = np.zeros(MAXBAND, dtype=np.float64)
                code = np.zeros(MAXBAND, dtype=int)

                epoch_obs.append(
                    Observation(
                        time=gps_seconds,
                        sat=int(sat),
                        system=sat2sys(int(sat)),
                        L=L,
                        P=P,
                        D=D,
                        SNR=S,
                        LLI=LLI,
                        code=code,
                    )
                )

            epochs.append(
                {
                    "gps_time": gps_seconds,
                    "gps_week": int(gps_week),
                    "gps_tow": float(gps_tow),
                    "unix_time": _gtime_to_unix(obs.t),
                    "n_sats": len(epoch_obs),
                    "observations": epoch_obs,
                }
            )

        return epochs
