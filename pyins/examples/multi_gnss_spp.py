# Copyright 2024 inuex35
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3
"""
Multi-GNSS Single Point Positioning Example

This example demonstrates how to perform SPP using multiple GNSS constellations
including GPS, GLONASS, Galileo, BeiDou, and QZSS.
"""

from pathlib import Path
from typing import Optional

import numpy as np

from pyins.coordinate import ecef2llh
from pyins.core.constants import SYS_GLO, sat2sys, sys2char
from pyins.core.unified_time import TimeCore
from pyins.gnss.spp_robust import robust_spp_solve
from pyins.io.rinex import RinexNavReader, RinexObsReader


def run_multi_gnss_spp(obs_file: str, nav_file: str,
                       exclude_glonass: bool = True,
                       init_pos: Optional[np.ndarray] = None) -> None:
    """
    Run multi-GNSS SPP on RINEX observation and navigation files

    Parameters
    ----------
    obs_file : str
        Path to RINEX observation file
    nav_file : str
        Path to RINEX navigation file
    exclude_glonass : bool
        Whether to exclude GLONASS satellites (default: True)
    init_pos : np.ndarray, optional
        Initial position guess in ECEF (m)
    """

    print("Multi-GNSS Single Point Positioning")
    print("=" * 50)

    # Check if files exist
    if not Path(obs_file).exists():
        print(f"Error: Observation file not found: {obs_file}")
        return
    if not Path(nav_file).exists():
        print(f"Error: Navigation file not found: {nav_file}")
        return

    print(f"Observation file: {obs_file}")
    print(f"Navigation file: {nav_file}")
    print()

    # Read RINEX files
    print("Reading RINEX files...")
    obs_reader = RinexObsReader(obs_file)
    nav_reader = RinexNavReader(nav_file)

    obs_data = obs_reader.read()
    nav_data = nav_reader.read()

    if not obs_data:
        print("Error: No observation data found")
        return

    print(f"Read {len(obs_data)} epochs of observation data")
    print(f"Read {len(nav_data.eph)} ephemerides")
    print()

    # Process first epoch (you can modify to process all epochs)
    epoch = obs_data[0]
    observations = epoch['observations']

    # Convert time to TimeCore
    tc = TimeCore.from_auto(epoch['gps_time'])
    print(f"Processing epoch at: {tc}")
    print(f"  GPS time: {tc.get_gps_seconds():.1f}")
    print(f"  GPS week/TOW: {tc.get_gps_week_tow()}")
    print(f"Total observations: {len(observations)}")

    # Filter observations if needed
    if exclude_glonass:
        filtered_obs = [obs for obs in observations if sat2sys(obs.sat) != SYS_GLO]
        print(f"Observations after filtering GLONASS: {len(filtered_obs)}")
    else:
        filtered_obs = observations

    # Count satellites by system
    sys_counts = {}
    for obs in filtered_obs:
        pr = obs.P[0] if obs.P[0] > 0 else obs.P[1]
        if pr > 0:
            sys = sat2sys(obs.sat)
            sys_char = sys2char(sys)
            sys_counts[sys_char] = sys_counts.get(sys_char, 0) + 1

    print("\nAvailable satellites by system:")
    for sys_char, count in sorted(sys_counts.items()):
        print(f"  {sys_char}: {count}")
    print()

    # Run SPP
    print("Running SPP...")
    solution, used_sats = robust_spp_solve(filtered_obs, nav_data, init_pos)

    if solution:
        # Convert to geodetic coordinates
        llh = ecef2llh(solution.rr)

        print("\nSPP Solution:")
        print(f"  ECEF X: {solution.rr[0]:14.3f} m")
        print(f"  ECEF Y: {solution.rr[1]:14.3f} m")
        print(f"  ECEF Z: {solution.rr[2]:14.3f} m")
        print()
        print(f"  Latitude:  {np.rad2deg(llh[0]):11.7f}°")
        print(f"  Longitude: {np.rad2deg(llh[1]):11.7f}°")
        print(f"  Height:    {llh[2]:11.3f} m")
        print()
        print(f"  Used satellites: {len(used_sats)}")

        # Clock biases
        print("\nReceiver clock biases:")
        if solution.dtr[0] != 0:
            print(f"  GPS:     {solution.dtr[0]*1e9:8.1f} ns ({solution.dtr[0]*299792458:8.1f} m)")
        if solution.dtr[2] != 0:
            print(f"  Galileo: {solution.dtr[2]*1e9:8.1f} ns ({solution.dtr[2]*299792458:8.1f} m)")
        if solution.dtr[3] != 0:
            print(f"  BeiDou:  {solution.dtr[3]*1e9:8.1f} ns ({solution.dtr[3]*299792458:8.1f} m)")

        # Show which satellites were used
        sys_used = {}
        for sat in used_sats:
            sys_char = sys2char(sat2sys(sat))
            sys_used[sys_char] = sys_used.get(sys_char, 0) + 1

        print("\nSatellites used by system:")
        for sys_char, count in sorted(sys_used.items()):
            print(f"  {sys_char}: {count}")

        print("\nSolution successful!")
    else:
        print("\nSPP failed to converge")
        print("Possible reasons:")
        print("  - Insufficient satellites")
        print("  - Poor satellite geometry")
        print("  - Large measurement errors")
        print("  - No ephemeris data for visible satellites")


def main():
    """Example usage with sample data"""
    import sys

    # Default files (modify these paths for your data)
    obs_file = "data/sample.obs"
    nav_file = "data/sample.nav"

    # Check command line arguments
    if len(sys.argv) > 2:
        obs_file = sys.argv[1]
        nav_file = sys.argv[2]

    # Run SPP
    # You can provide an initial position guess for faster convergence
    # init_pos = np.array([-3961900.0, 3349000.0, 3698200.0])  # Example for Tokyo

    run_multi_gnss_spp(obs_file, nav_file, exclude_glonass=True)


if __name__ == "__main__":
    main()
