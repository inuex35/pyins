"""Tests for cssrlib-based satellite geometry helpers."""

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CSSRLIB_SRC = REPO_ROOT / "cssrlib" / "src"
if str(CSSRLIB_SRC) not in sys.path:
    sys.path.insert(0, str(CSSRLIB_SRC))

from pyins.io.rinex import read_nav, read_obs
from pyins.gnss.geometry import compute_satellite_positions

DATA_DIR = REPO_ROOT / "test_data"


def test_compute_satellite_positions_returns_cssrlib_geometry():
    nav = read_nav(str(DATA_DIR / "rover.nav"))
    epochs = read_obs(str(DATA_DIR / "main.obs"))
    first_epoch = epochs[0]

    positions, clocks, health = compute_satellite_positions(first_epoch, nav)

    assert positions.shape == (len(first_epoch.sat), 3)
    assert clocks.shape == (len(first_epoch.sat),)
    assert health.shape == (len(first_epoch.sat),)

    # Require at least one valid satellite with non-zero geometry
    valid = np.where(health == 0)[0]
    assert valid.size > 0

    idx = valid[0]
    pos_norm = np.linalg.norm(positions[idx])
    assert 2e7 < pos_norm < 5e7
    assert np.isfinite(clocks[idx])
