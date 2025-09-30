"""Thin wrappers around cssrlib.rinex."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import List

from cssrlib.gnss import Nav
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


def read_obs(filename: str) -> List:
    """Decode a RINEX observation file into cssrlib Obs epochs."""

    decoder = rnxdec()
    if decoder.decode_obsh(filename) < 0:
        raise RuntimeError("Unsupported RINEX version")

    _prepare_decoder(decoder)

    epochs: List = []
    while True:
        obs = decoder.decode_obs()
        if obs is None or len(obs.sat) == 0:
            break
        epochs.append(deepcopy(obs))
    return epochs


class RinexNavReader:
    """Compatibility wrapper returning cssrlib Nav objects."""

    def __init__(self, filename: str):
        self.filename = Path(filename)

    def read(self) -> Nav:
        return read_nav(str(self.filename))


class RinexObsReader:
    """Compatibility wrapper returning cssrlib Obs epochs."""

    def __init__(self, filename: str):
        self.filename = Path(filename)

    def read(self) -> List:
        return read_obs(str(self.filename))
