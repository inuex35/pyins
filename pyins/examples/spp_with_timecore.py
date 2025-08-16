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
SPP Example with TimeCore

This example demonstrates how to use TimeCore for unified time management
in Single Point Positioning calculations.
"""

import numpy as np
from datetime import datetime, timezone
from pyins.core.unified_time import TimeCore, TimeSystem, current_time
from pyins.core.data_structures import Observation, NavigationData
from pyins.core.constants import SYS_GPS, SYS_BDS, SYS_GAL, CLIGHT, sys2char
from pyins.gnss.spp import single_point_positioning
from pyins.coordinate import ecef2llh, llh2ecef


def demonstrate_timecore():
    """Demonstrate TimeCore features for GNSS processing"""
    
    print("TimeCore Features for GNSS")
    print("=" * 50)
    
    # 1. Current time
    tc = current_time()
    print(f"\n1. Current time:")
    print(f"   {tc}")
    
    # 2. Different time formats
    print(f"\n2. Multiple time formats:")
    print(f"   GPS seconds:  {tc.get_gps_seconds():.3f}")
    print(f"   GPS week/TOW: {tc.get_gps_week_tow()}")
    print(f"   Unix:         {tc.get_unix():.3f}")
    print(f"   UTC:          {tc.get_datetime()}")
    print(f"   MJD:          {tc.get_mjd():.6f}")
    
    # 3. System-specific TOW
    print(f"\n3. System-specific Time of Week:")
    print(f"   GPS TOW: {tc.get_tow(TimeSystem.GPS):.3f} s")
    print(f"   BDS TOW: {tc.get_tow(TimeSystem.BDS):.3f} s")
    print(f"   GAL TOW: {tc.get_tow(TimeSystem.GAL):.3f} s")
    
    # 4. Time arithmetic
    print(f"\n4. Time arithmetic:")
    tc_future = tc + 3600  # Add 1 hour
    print(f"   Current:     {tc.get_datetime()}")
    print(f"   +1 hour:     {tc_future.get_datetime()}")
    print(f"   Difference:  {tc_future - tc:.1f} seconds")
    
    # 5. Week rollover handling
    print(f"\n5. Week rollover handling:")
    tc_near_end = TimeCore.from_gps(2200, 604700)  # Near week end
    tc_after = tc_near_end + 200  # Cross week boundary
    week1, tow1 = tc_near_end.get_gps_week_tow()
    week2, tow2 = tc_after.get_gps_week_tow()
    print(f"   Before: Week {week1}, TOW {tow1:.1f}")
    print(f"   After:  Week {week2}, TOW {tow2:.1f}")


def spp_with_timecore_example():
    """Example SPP calculation using TimeCore"""
    
    print("\n\nSPP Example with TimeCore")
    print("=" * 50)
    
    # Create observation time
    obs_time = current_time()
    print(f"\nObservation time: {obs_time}")
    
    # Create synthetic navigation data
    nav_data = NavigationData()
    
    # Note: In real applications, ephemerides would be loaded from RINEX
    # Here we just demonstrate the time handling
    
    # Create observations at different systems
    observations = []
    
    # Simulated satellite data
    sat_data = [
        # (PRN, system, approx_range_km)
        (1, SYS_GPS, 20000),      # GPS
        (5, SYS_GPS, 21000),
        (10, SYS_GPS, 22000),
        (15, SYS_GPS, 23000),
        (161, SYS_BDS, 21500),    # BeiDou
        (163, SYS_BDS, 22500),
        (165, SYS_BDS, 23500),
        (120, SYS_GAL, 21000),    # Galileo
        (122, SYS_GAL, 22000),
    ]
    
    # True receiver position (example: Tokyo)
    true_llh = np.array([35.68 * np.pi/180, 139.77 * np.pi/180, 100.0])
    true_pos = llh2ecef(true_llh)
    
    print(f"\nCreating observations for {len(sat_data)} satellites:")
    
    for prn, system, range_km in sat_data:
        # Create observation with TimeCore-derived GPS seconds
        obs = Observation(
            time=obs_time.get_gps_seconds(),  # SPP expects float
            sat=prn,
            system=system,
            P=np.array([range_km * 1000 + np.random.normal(0, 10), 0, 0]),
            L=np.array([0, 0, 0]),
            D=np.array([0, 0, 0]),
            SNR=np.array([45 + np.random.normal(0, 5), 0, 0]),
            LLI=np.array([0, 0, 0], dtype=int),
            code=np.array([0, 0, 0], dtype=int)
        )
        observations.append(obs)
        
        # Show how TimeCore would handle system-specific time
        sys_char = sys2char(system)
        if system == SYS_BDS:
            tow = obs_time.get_tow(TimeSystem.BDS)
            print(f"  {sys_char}{prn:02d}: BDS TOW = {tow:.3f} s")
        else:
            tow = obs_time.get_tow(TimeSystem.GPS)
            print(f"  {sys_char}{prn:02d}: GPS TOW = {tow:.3f} s")
    
    # Run SPP (without GLONASS by default)
    print(f"\nRunning SPP with systems: GPS, Galileo, BeiDou")
    print("(GLONASS excluded by default)")
    
    # Note: This will fail without proper ephemerides, but demonstrates time handling
    solution, used_sats = single_point_positioning(
        observations,
        nav_data,
        initial_pos=true_pos,
        systems_to_use=['G', 'E', 'C']  # No 'R' for GLONASS
    )
    
    if solution:
        print(f"\nSPP solution found (unexpected without ephemerides!)")
        llh = ecef2llh(solution.rr)
        print(f"  Position: {llh[0]*180/np.pi:.6f}°, {llh[1]*180/np.pi:.6f}°, {llh[2]:.1f}m")
    else:
        print(f"\nSPP failed (expected without ephemerides)")
        print("In real usage, load ephemerides from RINEX navigation files")


def time_conversion_examples():
    """Examples of time conversions for GNSS data"""
    
    print("\n\nTime Conversion Examples")
    print("=" * 50)
    
    # Example 1: RINEX observation timestamp
    print("\n1. RINEX observation timestamp:")
    rinex_gps_seconds = 1438000000.0
    tc = TimeCore(rinex_gps_seconds)
    print(f"   GPS seconds: {rinex_gps_seconds}")
    print(f"   TimeCore:    {tc}")
    print(f"   Date/Time:   {tc.get_datetime()}")
    
    # Example 2: Converting between systems
    print("\n2. Multi-GNSS time conversion:")
    tc_gps = TimeCore.from_gps(2200, 345600)
    print(f"   GPS: Week {2200}, TOW {345600.0}")
    
    bds_week, bds_tow = tc_gps.get_bds_week_tow()
    print(f"   BDS: Week {bds_week}, TOW {bds_tow:.3f}")
    print(f"   Difference: {345600.0 - bds_tow:.3f} seconds (GPS-BDS offset)")
    
    # Example 3: Handling ambiguous values
    print("\n3. Automatic format detection:")
    values = [
        1700000000.0,  # Unix timestamp
        (2200, 345600.0),  # GPS week/TOW
        datetime.now(timezone.utc),  # Datetime
    ]
    
    for val in values:
        try:
            tc = TimeCore.from_auto(val)
            print(f"   {str(val)[:30]:30} -> {tc.get_gps_week_tow()}")
        except Exception as e:
            print(f"   {str(val)[:30]:30} -> Error: {e}")


def main():
    """Run all examples"""
    
    print("TimeCore Integration Examples for SPP")
    print("=" * 60)
    print("Note: GLONASS is excluded by default in SPP calculations")
    
    # Demonstrate TimeCore features
    demonstrate_timecore()
    
    # SPP example with TimeCore
    spp_with_timecore_example()
    
    # Time conversion examples
    time_conversion_examples()
    
    print("\n" + "=" * 60)
    print("Examples complete!")


if __name__ == "__main__":
    main()