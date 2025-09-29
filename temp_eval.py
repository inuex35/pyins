import numpy as np
from pathlib import Path

from pyins.core.constants import (
    CLIGHT,
    sat2sys,
    SYS_GPS,
    SYS_GLO,
    SYS_GAL,
    SYS_BDS,
    SYS_QZS,
    SYS_SBS,
    SYS_IRN,
)
from pyins.io import RinexNavReader, RinexObsReader
from pyins.gnss.double_difference_original import form_double_differences
from pyins.coordinate import ecef2llh
from pyins.coordinate.dcm import enu2ecef_dcm

SYS_CHAR = {
    SYS_GPS: "G",
    SYS_GLO: "R",
    SYS_GAL: "E",
    SYS_BDS: "C",
    SYS_QZS: "J",
    SYS_SBS: "S",
    SYS_IRN: "I",
}

base_path = Path("test_data/base.obs")
pos_path = Path("test_data/main.pos")


def parse_base_ecef(path: Path) -> np.ndarray:
    with path.open() as f:
        for line in f:
            if "APPROX POSITION XYZ" in line:
                parts = line.strip().split()
                return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=float)
    raise RuntimeError("Base ECEF not found")


def parse_pos_baseline(path: Path):
    with path.open() as f:
        for line in f:
            if line.startswith("%") or not line.strip():
                continue
            parts = line.split()
            week = int(parts[0])
            tow = float(parts[1])
            enu = np.array(list(map(float, parts[2:5])))
            return week * 604800.0 + tow, enu
    raise RuntimeError("POS file empty")


def corrected_range(pos_ecef, clk, receiver_ecef):
    geom = np.linalg.norm(pos_ecef - receiver_ecef)
    return geom - CLIGHT * float(clk)


def dd_residual(dd, rover_ecef, base_ecef):
    sat_pos, ref_pos = dd["sat_pos"], dd["ref_sat_pos"]
    sat_clk, ref_clk = dd["sat_clk"], dd["ref_sat_clk"]
    rover_diff = corrected_range(sat_pos, sat_clk, rover_ecef) - corrected_range(ref_pos, ref_clk, rover_ecef)
    base_diff = corrected_range(sat_pos, sat_clk, base_ecef) - corrected_range(ref_pos, ref_clk, base_ecef)
    return float(dd["dd_obs"]) - (rover_diff - base_diff)


base_ecef = parse_base_ecef(base_path)
base_llh = ecef2llh(base_ecef)
_, pos_enu = parse_pos_baseline(pos_path)

R_enu2ecef = enu2ecef_dcm(base_llh)
rover_ecef = base_ecef + R_enu2ecef @ pos_enu
rover_llh = ecef2llh(rover_ecef)

nav = RinexNavReader("test_data/rover.nav").read()
rover_epoch = RinexObsReader("test_data/main.obs").read()[0]
base_epoch = RinexObsReader("test_data/base.obs").read()[0]

entries = form_double_differences(
    rover_epoch["observations"],
    base_epoch["observations"],
    nav,
    rover_epoch["gps_time"],
    rover_ecef,
    rover_llh,
    use_systems=["G", "E", "C", "R", "S"],
    cutoff_angle=10.0,
)

rows = []
for dd in entries:
    sys_char = SYS_CHAR.get(sat2sys(dd["sat"]), "?")
    resid = dd_residual(dd, rover_ecef, base_ecef)
    rows.append((sys_char, int(dd["sat"]), int(dd["ref_sat"]), dd["freq_idx"], resid))

rows.sort()
print(f"{'Sys':>3} {'Sat':>4} {'Ref':>4} {'Freq':>4} | {'PR Resid [m]':>14}")
print("-" * 40)
for sys_char, sat, ref_sat, freq, pr_res in rows:
    print(f"{sys_char:>3} {sat:>4} {ref_sat:>4} {freq:>4} | {pr_res:>14.4f}")

from collections import defaultdict
acc = defaultdict(list)
for sys_char, _, _, _, pr_res in rows:
    acc[sys_char].append(pr_res)

print("\nPer-system stats (POS baseline):")
for sys_char in sorted(acc.keys()):
    vals = np.array(acc[sys_char])
    print(f"  {sys_char}: mean {vals.mean():.4f} m, std {vals.std():.4f} m, n={len(vals)}")
