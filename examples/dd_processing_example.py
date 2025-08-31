#!/usr/bin/env python3
"""
Double Difference (DD) Processing Example
=========================================

This example demonstrates double difference processing using pyins library.
Double differencing is used in RTK (Real-Time Kinematic) positioning to eliminate
common errors between rover and base station observations.

Key concepts:
- Single Difference (SD): Difference between rover and base observations of the same satellite
- Double Difference (DD): Difference between two single differences, eliminating receiver clock errors
- Reference satellite: Selected for each GNSS system to form double differences
"""

import numpy as np
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict

# Import pyins modules
from pyins.io.rinex import RinexObsReader, RinexNavReader
from pyins.rtk.double_difference import DoubleDifferenceProcessor
from pyins.rtk.dd_least_squares import DDLeastSquares
from pyins.gnss.ephemeris import satpos, eph2pos
from pyins.gnss.spp import single_point_positioning
from pyins.coordinate.transforms import ecef2llh, llh2ecef, ecef2enu, enu2ecef
from pyins.core.time import GNSSTime
from pyins.core.constants import CLIGHT, sat2sys, SYS_GPS, SYS_GLO, SYS_GAL, SYS_BDS, SYS_QZS
from pyins.core.data_structures import Observation

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DDProcessor:
    """Enhanced Double Difference processor with advanced features"""
    
    def __init__(self, base_pos_ecef: np.ndarray, 
                 elevation_mask: float = 10.0,
                 snr_threshold: float = 35.0):
        """
        Initialize DD processor
        
        Parameters
        ----------
        base_pos_ecef : np.ndarray
            Base station position in ECEF (m)
        elevation_mask : float
            Minimum elevation angle in degrees
        snr_threshold : float
            Minimum SNR threshold in dB-Hz
        """
        self.base_pos_ecef = base_pos_ecef
        self.base_pos_llh = ecef2llh(base_pos_ecef)
        self.elevation_mask = np.deg2rad(elevation_mask)
        self.snr_threshold = snr_threshold
        
        # Initialize processors
        self.dd_processor = DoubleDifferenceProcessor()
        self.dd_solver = DDLeastSquares(self.dd_processor)
        
        # Statistics
        self.stats = {
            'epochs_processed': 0,
            'dd_measurements': [],
            'position_solutions': [],
            'residuals': []
        }
    
    def check_satellite_quality(self, obs: Observation, sat_pos: np.ndarray, 
                               receiver_pos: np.ndarray) -> bool:
        """
        Check if satellite observation meets quality criteria
        
        Parameters
        ----------
        obs : Observation
            Satellite observation
        sat_pos : np.ndarray
            Satellite position in ECEF
        receiver_pos : np.ndarray
            Receiver position in ECEF
            
        Returns
        -------
        bool
            True if observation passes quality checks
        """
        # Check SNR
        if hasattr(obs, 'S') and len(obs.S) > 0:
            if obs.S[0] < self.snr_threshold:
                return False
        
        # Compute elevation angle
        los = sat_pos - receiver_pos
        los_enu = ecef2enu(receiver_pos + los, ecef2llh(receiver_pos))
        elevation = np.arctan2(los_enu[2], np.linalg.norm(los_enu[:2]))
        
        if elevation < self.elevation_mask:
            return False
        
        # Check for valid measurements
        if obs.P[0] <= 0 or obs.L[0] <= 0:
            return False
        
        return True
    
    def select_reference_satellite(self, common_sats: List[int], 
                                  observations: List[Observation],
                                  sat_positions: Dict[int, np.ndarray],
                                  receiver_pos: np.ndarray) -> int:
        """
        Select best reference satellite based on elevation and SNR
        
        Parameters
        ----------
        common_sats : List[int]
            List of common satellites
        observations : List[Observation]
            Observations for selection
        sat_positions : Dict[int, np.ndarray]
            Satellite positions
        receiver_pos : np.ndarray
            Receiver position
            
        Returns
        -------
        int
            Selected reference satellite ID
        """
        best_sat = None
        best_score = -np.inf
        
        obs_dict = {obs.sat: obs for obs in observations}
        
        for sat in common_sats:
            if sat not in sat_positions or sat not in obs_dict:
                continue
                
            obs = obs_dict[sat]
            sat_pos = sat_positions[sat]
            
            # Compute elevation
            los = sat_pos - receiver_pos
            los_enu = ecef2enu(receiver_pos + los, ecef2llh(receiver_pos))
            elevation = np.arctan2(los_enu[2], np.linalg.norm(los_enu[:2]))
            
            # Compute score (higher elevation and SNR is better)
            score = np.rad2deg(elevation)
            if hasattr(obs, 'S') and len(obs.S) > 0:
                score += obs.S[0] * 0.5  # Weight SNR contribution
            
            if score > best_score:
                best_score = score
                best_sat = sat
        
        return best_sat
    
    def process_epoch(self, rover_obs: List[Observation], 
                     base_obs: List[Observation],
                     nav_data: list,
                     initial_rover_pos: Optional[np.ndarray] = None) -> Optional[Dict]:
        """
        Process single epoch with double differencing
        
        Parameters
        ----------
        rover_obs : List[Observation]
            Rover observations
        base_obs : List[Observation]
            Base observations
        nav_data : list
            Navigation data
        initial_rover_pos : np.ndarray, optional
            Initial rover position estimate
            
        Returns
        -------
        Dict or None
            Processing results or None if failed
        """
        # Get GPS time
        gps_time = rover_obs[0].time if rover_obs else None
        if gps_time is None:
            return None
        
        # Compute satellite positions
        sat_pos_array, sat_clk_array, _, sat_health = satpos(rover_obs, nav_data)
        
        # Build satellite position dictionary
        sat_positions = {}
        for i, obs in enumerate(rover_obs):
            if sat_health[i] == 0 and np.linalg.norm(sat_pos_array[i]) > 0:
                # Sanity check on satellite distance
                radius = np.linalg.norm(sat_pos_array[i])
                if 20000e3 < radius < 30000e3:  # GPS/GNSS orbit range
                    sat_positions[obs.sat] = sat_pos_array[i]
        
        # Get initial position if not provided
        if initial_rover_pos is None:
            # Try SPP first
            spp_solution = single_point_positioning(
                rover_obs, nav_data,
                initial_pos=None,
                max_iter=20
            )
            
            if spp_solution and spp_solution.rr is not None:
                initial_rover_pos = spp_solution.rr
                logger.debug(f"SPP initial position: {ecef2llh(initial_rover_pos)}")
            else:
                logger.warning("SPP failed, need approximate position")
                return None
        
        # Group observations by system
        rover_by_system = self._group_by_system(rover_obs)
        base_by_system = self._group_by_system(base_obs)
        
        # Results containers
        all_dd_pr = []
        all_dd_cp = []
        all_sat_pairs = []
        ref_sats_used = {}
        
        # Process each GNSS system separately
        for sys_id, sys_name in [(SYS_GPS, 'GPS'), (SYS_GAL, 'Galileo'), 
                                  (SYS_BDS, 'BeiDou'), (SYS_QZS, 'QZSS')]:
            if sys_id not in rover_by_system or sys_id not in base_by_system:
                continue
            
            # Find common satellites with valid observations
            rover_sats = set()
            base_sats = set()
            
            for obs in rover_by_system[sys_id]:
                if obs.sat in sat_positions:
                    if self.check_satellite_quality(obs, sat_positions[obs.sat], initial_rover_pos):
                        rover_sats.add(obs.sat)
            
            for obs in base_by_system[sys_id]:
                if obs.sat in sat_positions:
                    if self.check_satellite_quality(obs, sat_positions[obs.sat], self.base_pos_ecef):
                        base_sats.add(obs.sat)
            
            common_sats = rover_sats & base_sats
            
            if len(common_sats) < 2:
                logger.debug(f"{sys_name}: Only {len(common_sats)} common satellites, skipping")
                continue
            
            # Select reference satellite
            ref_sat = self.select_reference_satellite(
                list(common_sats), 
                rover_by_system[sys_id],
                sat_positions,
                initial_rover_pos
            )
            
            if ref_sat is None:
                continue
                
            ref_sats_used[sys_id] = ref_sat
            other_sats = [sat for sat in common_sats if sat != ref_sat]
            
            # Create observation dictionaries
            rover_dict = {obs.sat: obs for obs in rover_by_system[sys_id]}
            base_dict = {obs.sat: obs for obs in base_by_system[sys_id]}
            
            # Form double differences
            for sat in other_sats:
                # Single differences
                sd_pr_rover = rover_dict[sat].P[0] - rover_dict[ref_sat].P[0]
                sd_pr_base = base_dict[sat].P[0] - base_dict[ref_sat].P[0]
                sd_cp_rover = rover_dict[sat].L[0] - rover_dict[ref_sat].L[0]
                sd_cp_base = base_dict[sat].L[0] - base_dict[ref_sat].L[0]
                
                # Double differences
                dd_pr = sd_pr_rover - sd_pr_base
                dd_cp = sd_cp_rover - sd_cp_base
                
                all_dd_pr.append(dd_pr)
                all_dd_cp.append(dd_cp)
                all_sat_pairs.append((ref_sat, sat))
            
            logger.info(f"{sys_name}: {len(other_sats)} DD formed, ref sat: {ref_sat}")
        
        # Check if we have enough DD measurements
        if len(all_dd_pr) < 4:
            logger.warning(f"Only {len(all_dd_pr)} DD measurements, need at least 4")
            return None
        
        # Prepare satellite positions for solver
        ref_positions = []
        other_positions = []
        valid_dd_pr = []
        
        for i, (ref_sat, other_sat) in enumerate(all_sat_pairs):
            if ref_sat in sat_positions and other_sat in sat_positions:
                ref_positions.append(sat_positions[ref_sat])
                other_positions.append(sat_positions[other_sat])
                valid_dd_pr.append(all_dd_pr[i])
        
        if len(valid_dd_pr) < 4:
            return None
        
        # Solve for position
        try:
            rover_position, residuals = self.dd_solver.solve(
                np.array(valid_dd_pr),
                np.array(ref_positions),
                np.array(other_positions),
                self.base_pos_ecef,
                initial_rover_pos
            )
            
            # Compute statistics
            rover_llh = ecef2llh(rover_position)
            baseline_enu = ecef2enu(rover_position, self.base_pos_llh)
            baseline_length = np.linalg.norm(baseline_enu)
            rms_residual = np.sqrt(np.mean(residuals**2))
            
            result = {
                'time': gps_time,
                'position_ecef': rover_position,
                'position_llh': rover_llh,
                'baseline_enu': baseline_enu,
                'baseline_length': baseline_length,
                'n_dd': len(valid_dd_pr),
                'ref_sats': ref_sats_used,
                'residuals': residuals,
                'rms_residual': rms_residual
            }
            
            # Update statistics
            self.stats['epochs_processed'] += 1
            self.stats['dd_measurements'].append(len(valid_dd_pr))
            self.stats['position_solutions'].append(rover_llh)
            self.stats['residuals'].append(rms_residual)
            
            return result
            
        except Exception as e:
            logger.error(f"DD solution failed: {e}")
            return None
    
    def _group_by_system(self, observations: List[Observation]) -> Dict[int, List[Observation]]:
        """Group observations by GNSS system"""
        grouped = {}
        for obs in observations:
            sys_id = sat2sys(obs.sat)
            if sys_id not in grouped:
                grouped[sys_id] = []
            grouped[sys_id].append(obs)
        return grouped
    
    def print_statistics(self):
        """Print processing statistics"""
        if self.stats['epochs_processed'] == 0:
            logger.info("No epochs processed")
            return
        
        logger.info("\n=== Processing Statistics ===")
        logger.info(f"Epochs processed: {self.stats['epochs_processed']}")
        logger.info(f"Average DD measurements: {np.mean(self.stats['dd_measurements']):.1f}")
        logger.info(f"Average RMS residual: {np.mean(self.stats['residuals']):.3f} m")
        
        if len(self.stats['position_solutions']) > 1:
            positions = np.array(self.stats['position_solutions'])
            mean_pos = np.mean(positions, axis=0)
            std_pos = np.std(positions, axis=0)
            
            logger.info(f"Mean position (LLH): [{mean_pos[0]:.8f}, {mean_pos[1]:.8f}, {mean_pos[2]:.3f}]")
            logger.info(f"Position STD (m): E={std_pos[1]*111000*np.cos(np.deg2rad(mean_pos[0])):.3f}, "
                       f"N={std_pos[0]*111000:.3f}, U={std_pos[2]:.3f}")


def main():
    """Main function demonstrating DD processing"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Double Difference (DD) Processing')
    parser.add_argument('--data-dir', type=str, default='../../okujo_test',
                        help='Directory containing observation and navigation files')
    parser.add_argument('--rover-obs', type=str, default='main.obs',
                        help='Rover observation file name')
    parser.add_argument('--base-obs', type=str, default='base.obs',
                        help='Base observation file name')
    parser.add_argument('--nav-file', type=str, default='rover.nav',
                        help='Navigation file name')
    args = parser.parse_args()
    
    # Configure data paths
    data_dir = Path(args.data_dir)
    rover_obs_file = data_dir / args.rover_obs
    base_obs_file = data_dir / args.base_obs
    nav_file = data_dir / args.nav_file
    
    # Check files exist
    for file_path in [rover_obs_file, base_obs_file, nav_file]:
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            logger.info("Please adjust the data paths in the script")
            return
    
    # Base station position (replace with actual coordinates)
    # Example: Tokyo area coordinates
    base_llh = np.array([35.7102, 139.5217, 100.0])  # [lat, lon, height]
    base_ecef = llh2ecef(base_llh)
    
    logger.info("=== Double Difference Processing Example ===")
    logger.info(f"Base position (LLH): {base_llh}")
    logger.info(f"Base position (ECEF): {base_ecef}")
    
    # Read RINEX files
    logger.info("\nReading RINEX files...")
    rover_reader = RinexObsReader(str(rover_obs_file))
    base_reader = RinexObsReader(str(base_obs_file))
    nav_reader = RinexNavReader(str(nav_file))
    
    nav_data = nav_reader.read()
    logger.info(f"Loaded {len(nav_data)} navigation messages")
    
    # Initialize DD processor
    dd_processor = DDProcessor(
        base_pos_ecef=base_ecef,
        elevation_mask=15.0,  # 15 degree elevation mask
        snr_threshold=30.0     # 30 dB-Hz SNR threshold
    )
    
    # Process epochs
    max_epochs = 100  # Process first 100 epochs
    epoch_count = 0
    results = []
    
    logger.info(f"\nProcessing up to {max_epochs} epochs...")
    
    while epoch_count < max_epochs:
        # Read observations
        rover_obs = rover_reader.read_epoch()
        base_obs = base_reader.read_epoch()
        
        if rover_obs is None or base_obs is None:
            break
        
        epoch_count += 1
        
        # Process epoch
        result = dd_processor.process_epoch(
            rover_obs, 
            base_obs, 
            nav_data,
            initial_rover_pos=results[-1]['position_ecef'] if results else None
        )
        
        if result:
            results.append(result)
            
            # Log every 10th epoch
            if epoch_count % 10 == 0:
                logger.info(f"Epoch {epoch_count}: "
                          f"DD={result['n_dd']}, "
                          f"RMS={result['rms_residual']:.3f}m, "
                          f"Baseline={result['baseline_length']:.3f}m")
    
    # Print statistics
    dd_processor.print_statistics()
    
    # Save results
    if results:
        output_file = "dd_results.csv"
        logger.info(f"\nSaving results to {output_file}...")
        
        with open(output_file, 'w') as f:
            # Write header
            f.write("time,lat,lon,height,baseline_e,baseline_n,baseline_u,n_dd,rms_residual\n")
            
            # Write data
            for r in results:
                f.write(f"{r['time']:.3f},")
                f.write(f"{r['position_llh'][0]:.8f},{r['position_llh'][1]:.8f},{r['position_llh'][2]:.3f},")
                f.write(f"{r['baseline_enu'][0]:.3f},{r['baseline_enu'][1]:.3f},{r['baseline_enu'][2]:.3f},")
                f.write(f"{r['n_dd']},{r['rms_residual']:.3f}\n")
        
        logger.info(f"Results saved successfully")
    
    logger.info("\nProcessing complete!")


if __name__ == "__main__":
    main()