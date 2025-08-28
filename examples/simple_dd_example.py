#!/usr/bin/env python3
"""
Simple Double Difference Example for pyins
==========================================
This example demonstrates basic double difference processing.
"""

import sys
import numpy as np
import logging
from pathlib import Path

# Add pyins to path if needed
sys.path.insert(0, './external/pyins')

# Import pyins modules
from pyins.io.rinex import RinexObsReader, RinexNavReader
from pyins.rtk.double_difference import DoubleDifferenceProcessor
from pyins.gnss.ephemeris import satpos
from pyins.coordinate.transforms import ecef2llh, llh2ecef, ecef2enu
from pyins.core.constants import sat2sys, SYS_GPS, SYS_GAL, SYS_BDS, SYS_GLO

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main function"""
    
    # Check if test data exists
    rover_obs_file = "okujo_test/main.obs"
    base_obs_file = "okujo_test/base.obs"
    nav_file = "okujo_test/rover.nav"
    
    # Check files
    for file_path in [rover_obs_file, base_obs_file, nav_file]:
        if not Path(file_path).exists():
            logger.error(f"File not found: {file_path}")
            logger.info("\nThis example requires RINEX observation and navigation files.")
            logger.info("Expected files in okujo_test/ directory:")
            logger.info("  - main.obs (rover observations)")
            logger.info("  - base.obs (base observations)")
            logger.info("  - rover.nav (navigation data)")
            return
    
    logger.info("=" * 60)
    logger.info("Double Difference Processing Example")
    logger.info("=" * 60)
    
    # Read RINEX files
    logger.info("\n1. Reading RINEX files...")
    rover_reader = RinexObsReader(rover_obs_file)
    base_reader = RinexObsReader(base_obs_file)
    nav_reader = RinexNavReader(nav_file)
    
    # Get header information
    rover_header = rover_reader.header
    base_header = base_reader.header
    
    logger.info(f"   Rover station: {rover_header.marker_name if hasattr(rover_header, 'marker_name') else 'Unknown'}")
    logger.info(f"   Base station: {base_header.marker_name if hasattr(base_header, 'marker_name') else 'Unknown'}")
    
    # Get approximate positions
    rover_approx = rover_header.approx_position if hasattr(rover_header, 'approx_position') else None
    base_approx = base_header.approx_position if hasattr(base_header, 'approx_position') else None
    
    if rover_approx is not None:
        logger.info(f"   Rover approx pos (ECEF): [{rover_approx[0]:.2f}, {rover_approx[1]:.2f}, {rover_approx[2]:.2f}]")
    if base_approx is not None:
        logger.info(f"   Base approx pos (ECEF): [{base_approx[0]:.2f}, {base_approx[1]:.2f}, {base_approx[2]:.2f}]")
    
    # Read navigation data
    nav_data = nav_reader.read()
    # nav_data is a NavigationData object, not a list
    logger.info(f"   Navigation data loaded")
    
    # Initialize DD processor
    logger.info("\n2. Initializing Double Difference processor...")
    dd_processor = DoubleDifferenceProcessor()
    
    # Read all observations
    logger.info("\n3. Reading observations...")
    rover_obs_all = rover_reader.read()
    base_obs_all = base_reader.read()
    
    logger.info(f"   Rover observations read")
    logger.info(f"   Base observations read")
    
    # Process some epochs
    logger.info("\n4. Processing epochs...")
    max_epochs = min(5, len(rover_obs_all), len(base_obs_all))
    
    results = []
    
    # Process each epoch
    for epoch_idx in range(max_epochs):
        rover_epoch = rover_obs_all[epoch_idx]
        base_epoch = base_obs_all[epoch_idx]
        
        # Get observations from this epoch
        rover_obs = rover_epoch['observations']
        base_obs = base_epoch['observations']
        
        # Get time
        gps_time = rover_epoch['gps_time']
        
        logger.info(f"\n   Epoch {epoch_idx + 1}:")
        logger.info(f"   - GPS time: {gps_time:.1f}")
        logger.info(f"   - Rover satellites: {len(rover_obs)}")
        logger.info(f"   - Base satellites: {len(base_obs)}")
        
        # Form double differences for L1 frequency (index 0)
        dd_pr, dd_cp, sat_pairs, ref_sats = dd_processor.form_double_differences(
            rover_obs, 
            base_obs, 
            frequency_idx=0
        )
        
        logger.info(f"   - Double differences formed: {len(dd_pr)}")
        
        # Show reference satellites by system
        if ref_sats:
            for sys_id, ref_sat in ref_sats.items():
                sys_name = {
                    SYS_GPS: 'GPS',
                    SYS_GLO: 'GLONASS', 
                    SYS_GAL: 'Galileo',
                    SYS_BDS: 'BeiDou'
                }.get(sys_id, 'Unknown')
                logger.info(f"     {sys_name} reference: SAT {ref_sat}")
        
        # Show some DD values
        if len(dd_pr) > 0:
            logger.info(f"   - DD pseudorange range: [{np.min(dd_pr):.2f}, {np.max(dd_pr):.2f}] m")
            logger.info(f"   - DD pseudorange mean: {np.mean(dd_pr):.2f} m")
            logger.info(f"   - DD pseudorange std: {np.std(dd_pr):.2f} m")
            
            # Show satellite pairs
            logger.info(f"   - Satellite pairs (first 3):")
            for i, (ref_sat, other_sat) in enumerate(sat_pairs[:3]):
                logger.info(f"     Pair {i+1}: SAT{ref_sat} - SAT{other_sat}, DD = {dd_pr[i]:.2f} m")
        
        # Store results
        results.append({
            'epoch': epoch_idx + 1,
            'time': gps_time,
            'n_dd': len(dd_pr),
            'dd_pr': dd_pr,
            'dd_cp': dd_cp,
            'sat_pairs': sat_pairs,
            'ref_sats': ref_sats
        })
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Summary:")
    logger.info(f"  - Epochs processed: {len(results)}")
    
    if results:
        avg_dd = np.mean([r['n_dd'] for r in results])
        logger.info(f"  - Average DD measurements per epoch: {avg_dd:.1f}")
        
        # Count satellites by system
        all_sats = set()
        for r in results:
            for pair in r['sat_pairs']:
                all_sats.add(pair[0])
                all_sats.add(pair[1])
        
        # Group by system
        systems_count = {}
        for sat in all_sats:
            sys_id = sat2sys(sat)
            sys_name = {
                SYS_GPS: 'GPS',
                SYS_GLO: 'GLONASS',
                SYS_GAL: 'Galileo',
                SYS_BDS: 'BeiDou'
            }.get(sys_id, 'Other')
            
            if sys_name not in systems_count:
                systems_count[sys_name] = 0
            systems_count[sys_name] += 1
        
        logger.info(f"  - Satellites used:")
        for sys_name, count in systems_count.items():
            logger.info(f"    {sys_name}: {count}")
    
    logger.info("\nProcessing complete!")
    
    # Save basic results
    if results:
        output_file = "dd_basic_results.txt"
        with open(output_file, 'w') as f:
            f.write("# Basic Double Difference Results\n")
            f.write("# Epoch, Time, N_DD, DD_Mean, DD_Std\n")
            for r in results:
                if r['n_dd'] > 0:
                    f.write(f"{r['epoch']}, {r['time']:.3f}, {r['n_dd']}, ")
                    f.write(f"{np.mean(r['dd_pr']):.3f}, {np.std(r['dd_pr']):.3f}\n")
        logger.info(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()