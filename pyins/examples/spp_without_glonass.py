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
SPP example excluding GLONASS satellites

Note: GLONASS is now excluded by default in the SPP implementation.
"""

import numpy as np
from pyins.io.rinex import RinexObsReader, RinexNavReader
from pyins.gnss.spp_robust import robust_spp_solve
from pyins.coordinate import ecef2llh
from pyins.core.constants import sat2sys, SYS_GLO, sys2char
from pyins.core.unified_time import TimeCore

# Read RINEX files
obs_reader = RinexObsReader("opensky_debug/kaiyodai.obs")
nav_reader = RinexNavReader("opensky_debug/kaiyodai.nav")

obs_data = obs_reader.read()
nav_data = nav_reader.read()

# Initial position estimate (can be None)
init_pos = np.array([-3961938.616, 3348998.404, 3698236.116])

# Process first epoch
epoch = obs_data[0]
tc = TimeCore.from_auto(epoch['gps_time'])
print(f"Processing epoch at: {tc}")
print(f"  GPS seconds: {tc.get_gps_seconds():.3f}")
print(f"  Total satellites: {epoch['n_sats']}")

# Filter out GLONASS satellites (now done by default in SPP)
filtered_obs = []
excluded_glo = 0
for obs in epoch['observations']:
    if sat2sys(obs.sat) == SYS_GLO:
        excluded_glo += 1
    else:
        filtered_obs.append(obs)

print(f"Excluded {excluded_glo} GLONASS satellites (default behavior)")
print(f"Using {len(filtered_obs)} satellites for SPP")

# Solve for position
solution, used_sats = robust_spp_solve(filtered_obs, nav_data, init_pos)

if solution:
    # Convert to lat/lon/height
    llh = ecef2llh(solution.rr)
    
    print(f"\nSolution:")
    print(f"  Position: lat={np.rad2deg(llh[0]):.6f}°, lon={np.rad2deg(llh[1]):.6f}°, h={llh[2]:.1f}m")
    print(f"  ECEF: X={solution.rr[0]:.1f}m, Y={solution.rr[1]:.1f}m, Z={solution.rr[2]:.1f}m")
    print(f"  Clock bias: {solution.dtr[0]*1e9:.1f} ns")
    print(f"  Used satellites: {len(used_sats)}")
    
    # Show which systems were used
    systems = {}
    for sat in used_sats:
        sys = sat2sys(sat)
        sys_char = sys2char(sys)
        if sys_char not in systems:
            systems[sys_char] = 0
        systems[sys_char] += 1
    
    print(f"  Systems: {', '.join([f'{s}:{n}' for s, n in systems.items()])}")
else:
    print("SPP failed to converge")