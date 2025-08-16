#!/usr/bin/env python3
"""
Process All Epochs Example

This example shows how to process all epochs in a RINEX file and
output results in various formats.
"""

import numpy as np
from datetime import datetime
from pathlib import Path

from pyins.io.rinex import RinexObsReader, RinexNavReader
from pyins.gnss.spp_robust import robust_spp_solve
from pyins.coordinate import ecef2llh
from pyins.core.constants import sat2sys, sys2char, SYS_GLO


def process_all_epochs(obs_file: str, nav_file: str, 
                      output_file: str = "spp_results.txt",
                      exclude_glonass: bool = True):
    """
    Process all epochs in RINEX files and save results
    
    Parameters
    ----------
    obs_file : str
        Path to RINEX observation file
    nav_file : str
        Path to RINEX navigation file
    output_file : str
        Path to output file for results
    exclude_glonass : bool
        Whether to exclude GLONASS satellites
    """
    
    print(f"Processing all epochs from {obs_file}")
    print(f"Navigation data from {nav_file}")
    print()
    
    # Read RINEX files
    obs_reader = RinexObsReader(obs_file)
    nav_reader = RinexNavReader(nav_file)
    
    obs_data = obs_reader.read()
    nav_data = nav_reader.read()
    
    if not obs_data:
        print("No observation data found")
        return
    
    print(f"Total epochs to process: {len(obs_data)}")
    print()
    
    # Open output file
    with open(output_file, 'w') as f:
        # Write header
        f.write("# Multi-GNSS SPP Results\n")
        f.write(f"# Observation file: {obs_file}\n")
        f.write(f"# Navigation file: {nav_file}\n")
        f.write(f"# Generated: {datetime.now()}\n")
        f.write("#\n")
        f.write("# Columns:\n")
        f.write("# 1: Epoch number\n")
        f.write("# 2: GPS time (seconds)\n")
        f.write("# 3-5: ECEF X, Y, Z (m)\n")
        f.write("# 6-8: Latitude (deg), Longitude (deg), Height (m)\n")
        f.write("# 9: Number of satellites used\n")
        f.write("# 10-14: Satellites used by system (G,R,E,C,J)\n")
        f.write("#\n")
        f.write("# Epoch GPSTime         X              Y              Z           Lat       Lon      Height NSat G R E C J\n")
        
        # Process each epoch
        successful = 0
        failed = 0
        
        for i, epoch in enumerate(obs_data):
            observations = epoch['observations']
            gps_time = epoch['gps_time']
            
            # Filter GLONASS if requested
            if exclude_glonass:
                filtered_obs = [obs for obs in observations if sat2sys(obs.sat) != SYS_GLO]
            else:
                filtered_obs = observations
            
            # Skip if too few observations
            if len(filtered_obs) < 4:
                failed += 1
                continue
            
            # Compute position
            solution, used_sats = robust_spp_solve(filtered_obs, nav_data)
            
            if solution:
                # Convert to LLH
                llh = ecef2llh(solution.rr)
                
                # Count satellites by system
                sys_count = {'G': 0, 'R': 0, 'E': 0, 'C': 0, 'J': 0}
                for sat in used_sats:
                    sys_char = sys2char(sat2sys(sat))
                    if sys_char in sys_count:
                        sys_count[sys_char] += 1
                
                # Write results
                f.write(f"{i+1:6d} {gps_time:13.3f} "
                       f"{solution.rr[0]:14.3f} {solution.rr[1]:14.3f} {solution.rr[2]:14.3f} "
                       f"{np.rad2deg(llh[0]):9.5f} {np.rad2deg(llh[1]):10.5f} {llh[2]:8.2f} "
                       f"{len(used_sats):4d} "
                       f"{sys_count['G']:2d} {sys_count['R']:2d} {sys_count['E']:2d} "
                       f"{sys_count['C']:2d} {sys_count['J']:2d}\n")
                
                successful += 1
                
                # Progress indicator
                if (i + 1) % 10 == 0:
                    print(f"Processed {i+1}/{len(obs_data)} epochs...")
            else:
                failed += 1
    
    print()
    print(f"Processing complete!")
    print(f"  Successful: {successful} epochs")
    print(f"  Failed: {failed} epochs")
    print(f"  Results saved to: {output_file}")
    
    # Create a simple plot if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        # Read results back for plotting
        data = np.loadtxt(output_file, skiprows=11)
        if len(data) > 0:
            plt.figure(figsize=(12, 4))
            
            # Height plot
            plt.subplot(131)
            plt.plot(data[:, 1] - data[0, 1], data[:, 7])
            plt.xlabel('Time (s)')
            plt.ylabel('Height (m)')
            plt.title('Height vs Time')
            plt.grid(True)
            
            # Horizontal position
            plt.subplot(132)
            plt.plot(data[:, 5], data[:, 6], '.')
            plt.xlabel('Longitude (deg)')
            plt.ylabel('Latitude (deg)')
            plt.title('Horizontal Position')
            plt.grid(True)
            plt.axis('equal')
            
            # Number of satellites
            plt.subplot(133)
            plt.plot(data[:, 1] - data[0, 1], data[:, 8])
            plt.xlabel('Time (s)')
            plt.ylabel('Number of Satellites')
            plt.title('Satellites Used')
            plt.grid(True)
            
            plt.tight_layout()
            plot_file = output_file.replace('.txt', '_plot.png')
            plt.savefig(plot_file)
            print(f"  Plot saved to: {plot_file}")
            
    except ImportError:
        print("  (Install matplotlib to generate plots)")


def main():
    """Example usage"""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python process_all_epochs.py <obs_file> <nav_file> [output_file]")
        print("\nExample:")
        print("  python process_all_epochs.py data/rover.obs data/rover.nav results.txt")
        return
    
    obs_file = sys.argv[1]
    nav_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else "spp_results.txt"
    
    process_all_epochs(obs_file, nav_file, output_file)


if __name__ == "__main__":
    main()