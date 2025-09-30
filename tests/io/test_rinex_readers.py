"""Tests for cssrlib-backed RINEX readers."""

from pathlib import Path
import importlib.machinery
import importlib.util
import sys
import types


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

read_nav = rinex_module.read_nav
read_obs = rinex_module.read_obs


def test_read_nav_returns_cssrlib_nav():
    nav = read_nav(str(DATA_DIR / "rover.nav"))

    assert hasattr(nav, 'eph') and nav.eph
    assert hasattr(nav, 'geph')
    assert nav.leaps == 18


def test_read_obs_returns_cssrlib_epochs():
    epochs = read_obs(str(DATA_DIR / "base.obs"))

    assert epochs, "Observation file should produce epochs"

    first = epochs[0]
    assert hasattr(first, 'sat') and len(first.sat) > 0
    assert hasattr(first, 'P') and len(first.P) == len(first.sat)
