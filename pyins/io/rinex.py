"""Thin wrappers around cssrlib.rinex."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import List

from cssrlib.gnss import Nav, rSigRnx, sys2char
from cssrlib.rinex import rnxdec


def read_nav(filename: str) -> Nav:
    """Decode a RINEX navigation file into a cssrlib Nav object."""

    nav = Nav()
    decoder = rnxdec()
    decoder.decode_nav(filename, nav, append=False)
    return nav


def _prepare_decoder(decoder: rnxdec) -> None:
    decoder.setSignals([])

    for sys, sigs in decoder.sig_map.items():
        for sig in sigs.values():
            decoder.sig_tab.setdefault(sys, {}).setdefault(sig.typ, [])
            if sig not in decoder.sig_tab[sys][sig.typ]:
                decoder.sig_tab[sys][sig.typ].append(sig)

    max_counts = {typ: 0 for typ in decoder.nsig}
    for sigs in decoder.sig_tab.values():
        for typ, sig_list in sigs.items():
            max_counts[typ] = max(max_counts[typ], len(sig_list))

    for typ, count in max_counts.items():
        decoder.nsig[typ] = count

    for sigs in decoder.sig_tab.values():
        for typ, sig_list in sigs.items():
            if not sig_list:
                continue
            while len(sig_list) < max_counts[typ]:
                sig_list.append(sig_list[-1])


def read_obs(filename: str, signal_codes: List[str] | None = None) -> List:
    """Decode a RINEX observation file into cssrlib Obs epochs."""

    decoder = rnxdec()
    if signal_codes:
        decoder.setSignals([rSigRnx(code) for code in signal_codes])
    if decoder.decode_obsh(filename) < 0:
        raise RuntimeError("Unsupported RINEX version")

    if not signal_codes:
        _prepare_decoder(decoder)

    epochs: List = []
    while True:
        obs = decoder.decode_obs()
        if obs is None or len(obs.sat) == 0:
            break
        epochs.append(deepcopy(obs))
    return epochs


def _extract_signal_codes(decoder: rnxdec) -> List[str]:
    codes: List[str] = []
    for sys_enum, typ_dict in decoder.sig_map.items():
        sys_char = sys2char(sys_enum)
        for sig in typ_dict.values():
            code = f"{sys_char}{sig.str()}"
            if code not in codes:
                codes.append(code)
    return codes


def read_aligned_obs(reference_filename: str, other_filename: str) -> tuple[List, List]:
    """Decode two RINEX obs files keeping only common signal definitions."""

    ref_dec = rnxdec()
    if ref_dec.decode_obsh(reference_filename) < 0:
        raise RuntimeError("Unsupported RINEX version for reference file")

    other_dec = rnxdec()
    if other_dec.decode_obsh(other_filename) < 0:
        raise RuntimeError("Unsupported RINEX version for target file")

    ref_codes = _extract_signal_codes(ref_dec)
    other_codes = set(_extract_signal_codes(other_dec))
    common_codes = [code for code in ref_codes if code in other_codes]

    if not common_codes:
        raise RuntimeError("No common observation signals between the provided RINEX files")

    return (
        read_obs(reference_filename, common_codes),
        read_obs(other_filename, common_codes),
    )


class RinexNavReader:
    """Compatibility wrapper returning cssrlib Nav objects."""

    def __init__(self, filename: str):
        self.filename = Path(filename)

    def read(self) -> Nav:
        return read_nav(str(self.filename))


class RinexObsReader:
    """Compatibility wrapper returning cssrlib Obs epochs."""

    def __init__(self, filename: str, signal_codes: List[str] | None = None):
        self.filename = Path(filename)
        self.signal_codes = signal_codes

    def read(self) -> List:
        return read_obs(str(self.filename), self.signal_codes)
