#!/usr/bin/env python3
"""
Basic Single Point Positioning (SPP) Example using PyINS

This example demonstrates:
1. Reading RINEX observation and navigation files
2. Computing satellite positions using ephemeris
3. Solving for receiver position using least squares
4. Handling multiple GNSS systems (GPS, Galileo, BeiDou, etc.)
"""

import numpy as np
import logging
from pathlib import Path
from pyins.io.rinex import RinexObsReader, RinexNavReader
from pyins.gnss.spp import spp_solve
from pyins.gnss.satellite_positions import compute_satellite_info
from pyins.coordinate.transforms import ecef2llh, llh2ecef
from pyins.core.constants import sat2sys, sys2char, SYS_GPS, SYS_GAL, SYS_BDS, SYS_GLO, SYS_QZS
from pyins.core.unified_time import TimeCore, TimeSystem
import matplotlib.pyplot as plt


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def process_spp_basic(obs_file, nav_file, output_file=None):
    """
    Process SPP for all epochs in the observation file
    
    Parameters
    ----------
    obs_file : str
        Path to RINEX observation file
    nav_file : str
        Path to RINEX navigation file
    output_file : str, optional
        Path to output results file
    
    Returns
    -------
    results : list
        List of dictionaries containing SPP results for each epoch
    """
    logger = setup_logging()
    
    # Read RINEX files
    logger.info(f"Reading observation file: {obs_file}")
    obs_reader = RinexObsReader(obs_file)
    obs_data = obs_reader.read()
    
    logger.info(f"Reading navigation file: {nav_file}")
    nav_reader = RinexNavReader(nav_file)
    nav_data = nav_reader.read()
    
    n_eph = len(nav_data.eph) if hasattr(nav_data, 'eph') else len(nav_data) if isinstance(nav_data, list) else 0
    n_geph = len(nav_data.geph) if hasattr(nav_data, 'geph') else 0
    logger.info(f"Loaded {len(obs_data)} epochs, {n_eph} ephemerides, {n_geph} GLONASS ephemerides")
    
    # Process each epoch
    results = []
    positions_ecef = []
    positions_llh = []
    times = []
    satellite_counts = []
    pdops = []
    
    for i, obs_epoch in enumerate(obs_data):
        if i % 100 == 0:
            logger.info(f"Processing epoch {i+1}/{len(obs_data)}")
        
        # Extract observations from epoch dictionary
        observations = obs_epoch['observations'] if isinstance(obs_epoch, dict) else obs_epoch
        
        # Solve SPP for this epoch
        solution, used_satellites = spp_solve(
            observations,
            nav_data,
            max_iter=10,
            converge_threshold=1e-4,
            systems_to_use=['G', 'R', 'E', 'C', 'J'],  # GPS, GLONASS, Galileo, BeiDou, QZSS
            use_glonass_ifb=True  # Enable GLONASS IFB handling
        )
        
        if solution and solution.rr is not None:
            # Convert to geodetic coordinates
            llh = ecef2llh(solution.rr)
            
            # Store results
            result = {
                'time': obs_epoch.get('time', obs_epoch.get('gps_time')),
                'ecef': solution.rr,
                'llh': llh,
                'clock_bias': solution.dtr[0] * 299792458.0 if solution.dtr is not None else 0.0,  # Convert to meters
                'pdop': solution.pdop if hasattr(solution, 'pdop') else None,
                'n_satellites': solution.ns if hasattr(solution, 'ns') else len(used_satellites),
                'satellites': used_satellites,
                'residuals': solution.residuals if hasattr(solution, 'residuals') else None
            }
            
            results.append(result)
            positions_ecef.append(solution.rr)
            positions_llh.append(llh)
            times.append(result['time'])
            satellite_counts.append(len(used_satellites))
            if solution.pdop:
                pdops.append(solution.pdop)
            
            # Log solution
            if i % 100 == 0:
                logger.info(f"  Position: [{solution.rr[0]:.3f}, {solution.rr[1]:.3f}, {solution.rr[2]:.3f}] m")
                logger.info(f"  LLH: [{np.rad2deg(llh[0]):.8f}°, {np.rad2deg(llh[1]):.8f}°, {llh[2]:.3f} m]")
                logger.info(f"  Clock bias: {solution.dtr[0] * 299792458.0:.3f} m")
                logger.info(f"  Satellites used: {len(used_satellites)}")
                
                # Show satellite breakdown by system
                sat_systems = {}
                for sat in used_satellites:
                    sys = sat2sys(sat)
                    sys_char = sys2char(sys)
                    sat_systems[sys_char] = sat_systems.get(sys_char, 0) + 1
                logger.info(f"  Systems: {sat_systems}")
        else:
            logger.warning(f"SPP failed for epoch {i+1}")
    
    logger.info(f"Successfully processed {len(results)} out of {len(obs_data)} epochs")
    
    # Save results if output file specified
    if output_file:
        save_results(results, output_file)
    
    # Plot results
    if len(results) > 0:
        plot_spp_results(positions_llh, satellite_counts, pdops, times)
    
    return results


def save_results(results, output_file):
    """Save SPP results to file"""
    logger = logging.getLogger(__name__)
    
    with open(output_file, 'w') as f:
        # Write header
        f.write("# SPP Results\n")
        f.write("# Time, Lat(deg), Lon(deg), Height(m), Clock_bias(m), N_sats, PDOP\n")
        
        # Write data
        for result in results:
            llh = result['llh']
            f.write(f"{result['time']:.3f}, ")
            f.write(f"{np.rad2deg(llh[0]):.8f}, ")
            f.write(f"{np.rad2deg(llh[1]):.8f}, ")
            f.write(f"{llh[2]:.3f}, ")
            f.write(f"{result['clock_bias']:.3f}, ")
            f.write(f"{result['n_satellites']}, ")
            if result['pdop']:
                f.write(f"{result['pdop']:.2f}\n")
            else:
                f.write("0.00\n")
    
    logger.info(f"Results saved to {output_file}")


def plot_spp_results(positions_llh, satellite_counts, pdops, times):
    """Plot SPP results"""
    if len(positions_llh) == 0:
        return
    
    positions_llh = np.array(positions_llh)
    satellite_counts = np.array(satellite_counts)
    
    # Convert to degrees
    lats = np.rad2deg(positions_llh[:, 0])
    lons = np.rad2deg(positions_llh[:, 1])
    heights = positions_llh[:, 2]
    
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Horizontal position
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(lons, lats, '.', markersize=2)
    ax1.set_xlabel('Longitude (deg)')
    ax1.set_ylabel('Latitude (deg)')
    ax1.set_title('Horizontal Position')
    ax1.grid(True)
    
    # 2. Height time series
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(heights, '.', markersize=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Height (m)')
    ax2.set_title('Height Time Series')
    ax2.grid(True)
    
    # 3. Number of satellites
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(satellite_counts, '.', markersize=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Number of Satellites')
    ax3.set_title('Satellite Count')
    ax3.grid(True)
    
    # 4. Position scatter (centered)
    ax4 = fig.add_subplot(2, 3, 4)
    mean_lat = np.mean(lats)
    mean_lon = np.mean(lons)
    # Convert to meters (approximate)
    x_meters = (lons - mean_lon) * 111000 * np.cos(np.deg2rad(mean_lat))
    y_meters = (lats - mean_lat) * 111000
    ax4.scatter(x_meters, y_meters, c=range(len(x_meters)), cmap='viridis', s=2)
    ax4.set_xlabel('East (m)')
    ax4.set_ylabel('North (m)')
    ax4.set_title('Position Scatter (meters from mean)')
    ax4.axis('equal')
    ax4.grid(True)
    
    # 5. Height histogram
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.hist(heights, bins=50, edgecolor='black')
    ax5.set_xlabel('Height (m)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Height Distribution')
    ax5.grid(True, alpha=0.3)
    
    # 6. PDOP if available
    if len(pdops) > 0:
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.plot(pdops, '.', markersize=2)
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('PDOP')
        ax6.set_title('Position Dilution of Precision')
        ax6.grid(True)
    
    plt.suptitle('SPP Results Analysis')
    plt.tight_layout()
    plt.savefig('spp_results.png', dpi=150)
    print("Results plot saved to spp_results.png")
    plt.show()


def demo_mode():
    """Demonstrate SPP algorithm without actual data"""
    print("\n" + "="*60)
    print("SINGLE POINT POSITIONING (SPP) ALGORITHM")
    print("="*60)
    
    print("\n1. OBSERVATION EQUATION:")
    print("   ρ = |r_sat - r_rcv| + c·dt_rcv - c·dt_sat + T + I + ε")
    print("   where:")
    print("   - ρ: Pseudorange measurement")
    print("   - r_sat: Satellite position (from ephemeris)")
    print("   - r_rcv: Receiver position (unknown)")
    print("   - dt_rcv: Receiver clock bias (unknown)")
    print("   - dt_sat: Satellite clock bias (from ephemeris)")
    print("   - T: Tropospheric delay (modeled)")
    print("   - I: Ionospheric delay (modeled or dual-frequency)")
    print("   - ε: Measurement noise")
    
    print("\n2. LINEARIZED EQUATION:")
    print("   δρ = -e·δr + c·δt + v")
    print("   where:")
    print("   - δρ: Observed minus computed range")
    print("   - e: Unit vector from receiver to satellite")
    print("   - δr: Position correction")
    print("   - δt: Clock bias correction")
    print("   - v: Residual")
    
    print("\n3. LEAST SQUARES SOLUTION:")
    print("   [δr, δt] = (H^T·W·H)^(-1)·H^T·W·δρ")
    print("   where:")
    print("   - H: Design matrix [e1, 1; e2, 1; ...; en, 1]")
    print("   - W: Weight matrix (based on elevation, SNR, etc.)")
    
    print("\n4. SUPPORTED GNSS SYSTEMS:")
    print("   ✓ GPS (G)")
    print("   ✓ GLONASS (R) - with IFB handling")
    print("   ✓ Galileo (E)")
    print("   ✓ BeiDou (C)")
    print("   ✓ QZSS (J)")
    
    print("\n5. ERROR MODELS:")
    print("   - Troposphere: Saastamoinen model")
    print("   - Ionosphere: Klobuchar model or iono-free combination")
    print("   - Satellite clock: Broadcast ephemeris")
    print("   - Relativistic effects: Applied")
    print("   - Earth rotation: Sagnac correction")
    
    print("\n6. TYPICAL ACCURACY:")
    print("   - Horizontal: 5-10 m (single frequency)")
    print("   - Vertical: 10-15 m (single frequency)")
    print("   - With dual-frequency: 2-5 m")
    
    print("\n7. QUALITY INDICATORS:")
    print("   - PDOP: Position Dilution of Precision")
    print("   - Number of satellites (minimum 4)")
    print("   - Residuals RMS")
    print("   - Solution convergence")


def main():
    """Main function"""
    print("PyINS Basic SPP Example")
    print("="*60)
    
    import argparse
    parser = argparse.ArgumentParser(description='Process SPP from RINEX files')
    parser.add_argument('--obs', type=str, help='RINEX observation file')
    parser.add_argument('--nav', type=str, help='RINEX navigation file')
    parser.add_argument('--output', type=str, help='Output results file')
    parser.add_argument('--demo', action='store_true', help='Run in demo mode')
    
    args = parser.parse_args()
    
    if args.demo or (not args.obs or not args.nav):
        print("\n[Demo Mode] Showing SPP algorithm explanation")
        demo_mode()
        print("\n\nTo process actual data, use:")
        print("  python spp_basic.py --obs rover.obs --nav brdc.nav --output results.txt")
        return
    
    # Check files exist
    if not Path(args.obs).exists():
        print(f"Error: Observation file not found: {args.obs}")
        return
    if not Path(args.nav).exists():
        print(f"Error: Navigation file not found: {args.nav}")
        return
    
    # Process SPP
    results = process_spp_basic(args.obs, args.nav, args.output)
    
    # Print summary statistics
    if results:
        positions = np.array([r['ecef'] for r in results])
        mean_pos = np.mean(positions, axis=0)
        std_pos = np.std(positions, axis=0)
        
        mean_llh = ecef2llh(mean_pos)
        
        print("\n" + "="*60)
        print("SPP PROCESSING SUMMARY")
        print("="*60)
        print(f"Total epochs processed: {len(results)}")
        print(f"Mean position (ECEF): [{mean_pos[0]:.3f}, {mean_pos[1]:.3f}, {mean_pos[2]:.3f}] m")
        print(f"Mean position (LLH): [{np.rad2deg(mean_llh[0]):.8f}°, {np.rad2deg(mean_llh[1]):.8f}°, {mean_llh[2]:.3f} m]")
        print(f"Position STD (ECEF): [{std_pos[0]:.3f}, {std_pos[1]:.3f}, {std_pos[2]:.3f}] m")
        
        sat_counts = [r['n_satellites'] for r in results]
        print(f"Average satellites used: {np.mean(sat_counts):.1f}")
        print(f"Min/Max satellites: {np.min(sat_counts)}/{np.max(sat_counts)}")


if __name__ == '__main__':
    main()