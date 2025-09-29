"""Tests for cssrlib-backed RINEX readers."""

from pathlib import Path
import importlib.machinery
import importlib.util
import sys
import types

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "test_data"
PYINS_ROOT = REPO_ROOT / "pyins"

cssrlib_path = REPO_ROOT / "cssrlib" / "src"
if str(cssrlib_path) not in sys.path:
    sys.path.insert(0, str(cssrlib_path))


# Provide lightweight package stubs to avoid executing pyins.__init__ side effects.
if "pyins" not in sys.modules:
    pyins_pkg = types.ModuleType("pyins")
    pyins_pkg.__path__ = [str(PYINS_ROOT)]
    pyins_pkg.__spec__ = importlib.machinery.ModuleSpec("pyins", loader=None, is_package=True)
    sys.modules["pyins"] = pyins_pkg

if "pyins.io" not in sys.modules:
    io_pkg = types.ModuleType("pyins.io")
    io_pkg.__path__ = [str(PYINS_ROOT / "io")]
    io_pkg.__spec__ = importlib.machinery.ModuleSpec("pyins.io", loader=None, is_package=True)
    sys.modules["pyins.io"] = io_pkg

core_spec = importlib.util.spec_from_file_location(
    "pyins.core",
    PYINS_ROOT / "core" / "__init__.py",
    submodule_search_locations=[str(PYINS_ROOT / "core")],
)
core_module = importlib.util.module_from_spec(core_spec)
sys.modules.setdefault("pyins.core", core_module)
core_spec.loader.exec_module(core_module)

rinex_spec = importlib.util.spec_from_file_location(
    "pyins.io.rinex", PYINS_ROOT / "io" / "rinex.py"
)
rinex_module = importlib.util.module_from_spec(rinex_spec)
sys.modules.setdefault("pyins.io.rinex", rinex_module)
rinex_spec.loader.exec_module(rinex_module)

RinexNavReader = rinex_module.RinexNavReader
RinexObsReader = rinex_module.RinexObsReader

from pyins.core.constants import MAXBAND
from pyins.core.data_structures import Observation


def test_rinex_nav_reader_reads_broadcast_ephemeris():
    reader = RinexNavReader(str(DATA_DIR / "rover.nav"))
    nav_data = reader.read()

    assert nav_data.eph, "Expected GPS/Galileo ephemerides"
    assert nav_data.leaps == 18

    sat_ids = {eph.sat for eph in nav_data.eph}
    assert sat_ids, "Ephemeris should contain satellite IDs"
    assert all(sat_id > 0 for sat_id in sat_ids)


def test_rinex_obs_reader_parses_epochs():
    reader = RinexObsReader(str(DATA_DIR / "base.obs"))
    epochs = reader.read()

    assert epochs, "Observation file should produce epochs"

    first_epoch = epochs[0]
    assert first_epoch["n_sats"] == len(first_epoch["observations"])
    assert first_epoch["gps_time"] > 0
    assert first_epoch["unix_time"] > 0

    observation = first_epoch["observations"][0]
    assert isinstance(observation, Observation)
    assert observation.L.shape == (MAXBAND,)
    assert observation.P.shape == (MAXBAND,)
    assert observation.SNR.shape == (MAXBAND,)
    assert observation.LLI.dtype == np.int_

    systems = {obs.system for epoch in epochs for obs in epoch["observations"]}
    assert systems, "Expected at least one GNSS constellation"
