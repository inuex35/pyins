"""Utilities for working with RTKLIB .pos summary files.

The RTKLIB ``.pos`` format is commonly produced by RTKPOST/RTKNAVI and contains
epoch-wise baseline solutions expressed in the local ENU frame.  This module
provides lightweight helpers to parse these files and convert the reported
ENU baselines back to ECEF coordinates so that they can be used as reference
positions when validating residual computations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

from ..coordinate.transforms import ecef2llh
from ..coordinate.dcm import enu2ecef_dcm

WEEK_SECONDS = 604800.0


@dataclass(frozen=True)
class PosRecord:
    """Single epoch record from an RTKLIB ``.pos`` file."""

    gps_seconds: float
    enu: np.ndarray
    quality: Optional[int] = None
    num_sats: Optional[int] = None
    sd_enu: Optional[np.ndarray] = None
    sd_cov: Optional[np.ndarray] = None
    age: Optional[float] = None
    ratio: Optional[float] = None

    def to_ecef(self, base_ecef: np.ndarray, base_llh: Optional[np.ndarray] = None) -> np.ndarray:
        """Convert the ENU baseline into an absolute ECEF coordinate."""
        if base_llh is None:
            base_llh = ecef2llh(np.asarray(base_ecef, dtype=np.float64))
        rotation = enu2ecef_dcm(base_llh)
        return np.asarray(base_ecef, dtype=np.float64) + rotation @ self.enu


def _parse_floats(values: Iterable[str], count: int) -> Optional[np.ndarray]:
    items: List[float] = []
    for idx, token in enumerate(values):
        if idx >= count:
            break
        try:
            items.append(float(token))
        except ValueError:
            return None
    if len(items) != count:
        return None
    return np.array(items, dtype=np.float64)


def read_rtklib_pos(path: str | Path) -> List[PosRecord]:
    """Parse an RTKLIB ``.pos`` file and return epoch-wise records."""
    pos_path = Path(path)
    if not pos_path.exists():
        raise FileNotFoundError(pos_path)

    records: List[PosRecord] = []

    with pos_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.strip() or line.lstrip().startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                gps_week = int(parts[0])
                tow = float(parts[1])
            except ValueError:
                continue

            enu = _parse_floats(parts[2:], 3)
            if enu is None:
                continue

            quality = int(parts[5]) if len(parts) > 5 else None
            num_sats = int(parts[6]) if len(parts) > 6 else None
            sd_enu = _parse_floats(parts[7:], 3)
            sd_cov = _parse_floats(parts[10:], 3)
            age = float(parts[13]) if len(parts) > 13 else None
            ratio = float(parts[14]) if len(parts) > 14 else None

            gps_seconds = gps_week * WEEK_SECONDS + tow
            records.append(
                PosRecord(
                    gps_seconds=gps_seconds,
                    enu=enu,
                    quality=quality,
                    num_sats=num_sats,
                    sd_enu=sd_enu,
                    sd_cov=sd_cov,
                    age=age,
                    ratio=ratio,
                )
            )

    if not records:
        raise ValueError(f"No position records parsed from {pos_path}")

    return records


def parse_rinex_approx_position(path: str | Path) -> np.ndarray:
    """Extract approximate base-station ECEF coordinates from a RINEX file."""
    rinex_path = Path(path)
    if not rinex_path.exists():
        raise FileNotFoundError(rinex_path)

    with rinex_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if "APPROX POSITION XYZ" in line:
                tokens = line.strip().split()
                if len(tokens) < 3:
                    continue
                try:
                    return np.array([float(tokens[0]), float(tokens[1]), float(tokens[2])], dtype=np.float64)
                except ValueError as exc:
                    raise ValueError(f"Failed to parse base position from {rinex_path}") from exc

    raise ValueError(f"APPROX POSITION XYZ not found in {rinex_path}")


def reference_position_from_pos(
    pos_path: str | Path,
    base_ecef: np.ndarray,
    index: int = 0,
) -> tuple[float, np.ndarray]:
    """Return the ECEF rover position for a given epoch from a ``.pos`` file."""
    records = read_rtklib_pos(pos_path)
    if index < 0 or index >= len(records):
        raise IndexError(f"index {index} out of bounds for {len(records)} POS records")

    base_llh = ecef2llh(np.asarray(base_ecef, dtype=np.float64))
    record = records[index]
    return record.gps_seconds, record.to_ecef(base_ecef, base_llh)


__all__ = [
    "PosRecord",
    "parse_rinex_approx_position",
    "read_rtklib_pos",
    "reference_position_from_pos",
]
