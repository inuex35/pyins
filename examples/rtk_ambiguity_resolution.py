#!/usr/bin/env python3
"""
RTK with Ambiguity Resolution Example
======================================

This example demonstrates RTK (Real-Time Kinematic) positioning with integer
ambiguity resolution using the LAMBDA method. It shows how to:

1. Form double differences for carrier phase measurements
2. Estimate float ambiguities
3. Resolve integer ambiguities using LAMBDA
4. Achieve centimeter-level positioning accuracy
"""

import sys
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

# Import pyins modules
from pyins.io.rinex import RinexObsReader, RinexNavReader
from pyins.rtk.double_difference import DoubleDifferenceProcessor
from pyins.rtk.ambiguity_resolution import RTKAmbiguityManager
from pyins.gnss.ephemeris import satpos
from pyins.coordinate.transforms import ecef2llh, llh2ecef, ecef2enu
from pyins.core.constants import (
    CLIGHT, FREQ_L1, FREQ_L2, 
    sat2sys, SYS_GPS, SYS_GAL, SYS_BDS, SYS_GLO
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RTKProcessor:
    """RTK processor with ambiguity resolution"""
    
    def __init__(self, base_pos_ecef: np.ndarray):
        """
        Initialize RTK processor
        
        Parameters
        ----------
        base_pos_ecef : np.ndarray
            Base station position in ECEF
        """
        self.base_pos_ecef = base_pos_ecef
        self.base_pos_llh = ecef2llh(base_pos_ecef)
        self.dd_processor = DoubleDifferenceProcessor()
        self.ambiguity_resolver = RTKAmbiguityManager(ratio_threshold=3.0)
        
        # State
        self.float_ambiguities = {}  # Float ambiguity estimates
        self.fixed_ambiguities = {}  # Fixed integer ambiguities
        self.ambiguity_covariance = None
        self.is_fixed = False
        
    def estimate_float_ambiguities(self, dd_phase: np.ndarray, dd_code: np.ndarray,
                                  sat_pairs: List[Tuple[int, int]], 
                                  sat_positions: Dict[int, np.ndarray],
                                  rover_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate float ambiguities from DD measurements
        
        Parameters
        ----------
        dd_phase : np.ndarray
            Double difference carrier phase measurements (cycles)
        dd_code : np.ndarray
            Double difference code measurements (meters)
        sat_pairs : List[Tuple[int, int]]
            Satellite pairs for each DD
        sat_positions : Dict[int, np.ndarray]
            Satellite positions
        rover_pos : np.ndarray
            Rover position estimate
            
        Returns
        -------
        a_float : np.ndarray
            Float ambiguity estimates (cycles)
        Q_a : np.ndarray
            Ambiguity covariance matrix
        """
        n_dd = len(dd_phase)
        
        # Compute geometric ranges for DD
        dd_ranges = []
        wavelengths = []
        
        for ref_sat, other_sat in sat_pairs[:n_dd]:
            if ref_sat not in sat_positions or other_sat not in sat_positions:
                continue
                
            # Compute ranges
            r_rover_ref = np.linalg.norm(sat_positions[ref_sat] - rover_pos)
            r_rover_other = np.linalg.norm(sat_positions[other_sat] - rover_pos)
            r_base_ref = np.linalg.norm(sat_positions[ref_sat] - self.base_pos_ecef)
            r_base_other = np.linalg.norm(sat_positions[other_sat] - self.base_pos_ecef)
            
            # DD range
            dd_range = (r_rover_other - r_rover_ref) - (r_base_other - r_base_ref)
            dd_ranges.append(dd_range)
            
            # Wavelength (simplified - use L1 for all)
            wavelength = CLIGHT / FREQ_L1
            wavelengths.append(wavelength)
        
        dd_ranges = np.array(dd_ranges)
        wavelengths = np.array(wavelengths)
        
        # Estimate float ambiguities
        # N_float = (DD_phase * wavelength - DD_range) / wavelength
        #         = DD_phase - DD_range / wavelength
        
        # Use combination of phase and code for better estimate
        # Code helps to resolve the integer part
        a_float = dd_phase[:len(dd_ranges)] - dd_ranges / wavelengths
        
        # Estimate covariance (simplified)
        # In practice, use proper error propagation
        phase_std = 0.003  # 3mm phase noise
        code_std = 0.3     # 30cm code noise
        
        # Covariance matrix (diagonal for simplicity)
        Q_a = np.eye(len(a_float)) * (phase_std / wavelengths[0])**2
        
        # Add correlation for same reference satellite
        for i in range(len(a_float)):
            for j in range(i+1, len(a_float)):
                if sat_pairs[i][0] == sat_pairs[j][0]:  # Same reference
                    Q_a[i, j] = Q_a[j, i] = 0.5 * np.sqrt(Q_a[i, i] * Q_a[j, j])
        
        return a_float, Q_a
    
    def apply_fixed_ambiguities(self, dd_phase: np.ndarray, a_fixed: np.ndarray,
                               wavelengths: np.ndarray) -> np.ndarray:
        """
        Apply fixed ambiguities to get precise DD measurements
        
        Parameters
        ----------
        dd_phase : np.ndarray
            DD carrier phase measurements (cycles)
        a_fixed : np.ndarray
            Fixed integer ambiguities
        wavelengths : np.ndarray
            Wavelengths for each DD
            
        Returns
        -------
        dd_precise : np.ndarray
            Precise DD measurements in meters
        """
        # Convert phase to meters and subtract ambiguity term
        dd_precise = (dd_phase[:len(a_fixed)] - a_fixed) * wavelengths
        
        return dd_precise
    
    def process_epoch(self, rover_obs: list, base_obs: list, nav_data,
                     initial_pos: np.ndarray) -> dict:
        """
        Process single epoch with ambiguity resolution
        
        Parameters
        ----------
        rover_obs : list
            Rover observations
        base_obs : list
            Base observations
        nav_data
            Navigation data
        initial_pos : np.ndarray
            Initial position estimate
            
        Returns
        -------
        result : dict
            Processing result
        """
        # Get GPS time
        gps_time = rover_obs[0].time if rover_obs else 0
        
        # Compute satellite positions
        sat_pos_array, _, _, sat_health = satpos(rover_obs, nav_data)
        
        sat_positions = {}
        for i, obs in enumerate(rover_obs):
            if sat_health[i] == 0 and np.linalg.norm(sat_pos_array[i]) > 0:
                radius = np.linalg.norm(sat_pos_array[i])
                if 20000e3 < radius < 30000e3:
                    sat_positions[obs.sat] = sat_pos_array[i]
        
        # Form double differences
        dd_pr, dd_cp, sat_pairs, ref_sats = self.dd_processor.form_double_differences(
            rover_obs, base_obs, frequency_idx=0  # L1
        )
        
        if len(dd_pr) < 4:
            return {'success': False, 'reason': 'Insufficient DD measurements'}
        
        # Get wavelengths
        wavelengths = np.array([CLIGHT / FREQ_L1] * len(dd_cp))
        
        # First solve with float solution
        rover_pos_float = self.solve_position_float(
            dd_pr, dd_cp, sat_pairs, sat_positions, initial_pos, wavelengths
        )
        
        if rover_pos_float is None:
            return {'success': False, 'reason': 'Float solution failed'}
        
        # Estimate float ambiguities
        a_float, Q_a = self.estimate_float_ambiguities(
            dd_cp, dd_pr, sat_pairs, sat_positions, rover_pos_float
        )
        
        # Try to resolve ambiguities
        a_fixed, ar_info = self.ambiguity_resolver.resolve(a_float, Q_a, wavelengths)
        
        result = {
            'success': True,
            'time': gps_time,
            'position_float': rover_pos_float,
            'n_dd': len(dd_pr),
            'ar_attempted': True,
            'ar_info': ar_info
        }
        
        if a_fixed is not None:
            # Apply fixed ambiguities
            dd_precise = self.apply_fixed_ambiguities(dd_cp, a_fixed, wavelengths[:len(a_fixed)])
            
            # Solve with fixed ambiguities
            rover_pos_fixed = self.solve_position_fixed(
                dd_precise, sat_pairs[:len(dd_precise)], sat_positions, rover_pos_float
            )
            
            if rover_pos_fixed is not None:
                result['position_fixed'] = rover_pos_fixed
                result['ambiguities_fixed'] = a_fixed
                result['ar_success'] = True
                self.is_fixed = True
                
                # Store for next epoch
                self.fixed_ambiguities = {
                    tuple(sat_pairs[i]): a_fixed[i] 
                    for i in range(len(a_fixed))
                }
            else:
                result['ar_success'] = False
        else:
            result['ar_success'] = False
        
        return result
    
    def solve_position_float(self, dd_pr: np.ndarray, dd_cp: np.ndarray,
                            sat_pairs: list, sat_positions: dict, 
                            initial_pos: np.ndarray, wavelengths: np.ndarray) -> Optional[np.ndarray]:
        """
        Solve position with float ambiguities (code + phase)
        """
        rover_pos = initial_pos.copy()
        
        # Use weighted combination of code and phase
        weight_code = 1.0
        weight_phase = 100.0  # Phase is more precise
        
        for iteration in range(10):
            H = []
            residuals_code = []
            residuals_phase = []
            
            for i, (ref_sat, other_sat) in enumerate(sat_pairs[:len(dd_pr)]):
                if ref_sat not in sat_positions or other_sat not in sat_positions:
                    continue
                
                # Geometry
                r_ref = sat_positions[ref_sat] - rover_pos
                r_other = sat_positions[other_sat] - rover_pos
                d_ref = np.linalg.norm(r_ref)
                d_other = np.linalg.norm(r_other)
                
                # DD geometry row
                h_row = -r_other/d_other + r_ref/d_ref
                H.append(h_row)
                
                # Predicted DD range
                d_base_ref = np.linalg.norm(sat_positions[ref_sat] - self.base_pos_ecef)
                d_base_other = np.linalg.norm(sat_positions[other_sat] - self.base_pos_ecef)
                dd_range_pred = (d_other - d_ref) - (d_base_other - d_base_ref)
                
                # Residuals
                residuals_code.append(dd_pr[i] - dd_range_pred)
                
                # For phase, include ambiguity as unknown
                # This is simplified - proper implementation would estimate ambiguities
                phase_residual = dd_cp[i] * wavelengths[i] - dd_range_pred
                residuals_phase.append(phase_residual)
            
            if len(H) < 4:
                return None
            
            H = np.array(H)
            
            # Combine code and phase residuals with weights
            residuals = (weight_code * np.array(residuals_code) + 
                        weight_phase * np.array(residuals_phase)) / (weight_code + weight_phase)
            
            # Weighted least squares
            try:
                dx = np.linalg.lstsq(H, residuals, rcond=None)[0]
            except:
                return None
            
            rover_pos += dx
            
            if np.linalg.norm(dx) < 1e-4:
                break
        
        return rover_pos
    
    def solve_position_fixed(self, dd_precise: np.ndarray, sat_pairs: list,
                            sat_positions: dict, initial_pos: np.ndarray) -> Optional[np.ndarray]:
        """
        Solve position with fixed ambiguities (precise)
        """
        rover_pos = initial_pos.copy()
        
        for iteration in range(10):
            H = []
            residuals = []
            
            for i, (ref_sat, other_sat) in enumerate(sat_pairs):
                if ref_sat not in sat_positions or other_sat not in sat_positions:
                    continue
                
                # Geometry
                r_ref = sat_positions[ref_sat] - rover_pos
                r_other = sat_positions[other_sat] - rover_pos
                d_ref = np.linalg.norm(r_ref)
                d_other = np.linalg.norm(r_other)
                
                # DD geometry row
                h_row = -r_other/d_other + r_ref/d_ref
                H.append(h_row)
                
                # Predicted DD range
                d_base_ref = np.linalg.norm(sat_positions[ref_sat] - self.base_pos_ecef)
                d_base_other = np.linalg.norm(sat_positions[other_sat] - self.base_pos_ecef)
                dd_range_pred = (d_other - d_ref) - (d_base_other - d_base_ref)
                
                # Residual (precise with fixed ambiguity)
                residuals.append(dd_precise[i] - dd_range_pred)
            
            if len(H) < 4:
                return None
            
            H = np.array(H)
            residuals = np.array(residuals)
            
            # Least squares
            try:
                dx = np.linalg.lstsq(H, residuals, rcond=None)[0]
            except:
                return None
            
            rover_pos += dx
            
            if np.linalg.norm(dx) < 1e-5:  # Tighter convergence for fixed solution
                break
        
        return rover_pos


def main():
    """Main function demonstrating RTK with ambiguity resolution"""
    
    # File paths
    rover_obs_file = "../../okujo_test/main.obs"
    base_obs_file = "../../okujo_test/base.obs"
    nav_file = "../../okujo_test/rover.nav"
    
    # Check files
    for file_path in [rover_obs_file, base_obs_file, nav_file]:
        if not Path(file_path).exists():
            logger.error(f"File not found: {file_path}")
            logger.info("Please adjust paths to your RINEX files")
            return
    
    logger.info("=" * 60)
    logger.info("RTK with Ambiguity Resolution Example")
    logger.info("=" * 60)
    
    # Read RINEX files
    logger.info("\nReading RINEX files...")
    rover_reader = RinexObsReader(rover_obs_file)
    base_reader = RinexObsReader(base_obs_file)
    nav_reader = RinexNavReader(nav_file)
    
    rover_obs_all = rover_reader.read()
    base_obs_all = base_reader.read()
    nav_data = nav_reader.read()
    
    logger.info(f"Read {len(rover_obs_all)} rover epochs, {len(base_obs_all)} base epochs")
    
    # Base station position
    base_pos_ecef = np.array([-3961905.0321, 3348994.2448, 3698212.3456])
    base_pos_llh = ecef2llh(base_pos_ecef)
    logger.info(f"Base position (LLH): [{base_pos_llh[0]:.8f}, {base_pos_llh[1]:.8f}, {base_pos_llh[2]:.3f}]")
    
    # Initial rover position
    rover_init_ecef = np.array([-3963730.2015, 3355699.6439, 3690265.5217])
    
    # Initialize RTK processor
    rtk = RTKProcessor(base_pos_ecef)
    
    # Process epochs
    n_epochs = min(50, len(rover_obs_all), len(base_obs_all))  # Process first 50 epochs
    logger.info(f"\nProcessing {n_epochs} epochs...")
    
    results = []
    n_fixed = 0
    n_float = 0
    
    for epoch_idx in range(n_epochs):
        rover_epoch = rover_obs_all[epoch_idx]
        base_epoch = base_obs_all[epoch_idx]
        
        # Get observations
        rover_obs = rover_epoch['observations']
        base_obs = base_epoch['observations']
        
        # Add time to observations
        gps_time = rover_epoch['gps_time']
        for obs in rover_obs:
            obs.time = gps_time
        for obs in base_obs:
            obs.time = gps_time
        
        # Use previous solution or initial position
        if results and 'position_fixed' in results[-1]:
            initial_pos = results[-1]['position_fixed']
        elif results and 'position_float' in results[-1]:
            initial_pos = results[-1]['position_float']
        else:
            initial_pos = rover_init_ecef
        
        # Process epoch
        result = rtk.process_epoch(rover_obs, base_obs, nav_data, initial_pos)
        
        if result['success']:
            results.append(result)
            
            if result.get('ar_success', False):
                n_fixed += 1
                status = "FIX"
                pos = result['position_fixed']
            else:
                n_float += 1
                status = "FLOAT"
                pos = result['position_float']
            
            pos_llh = ecef2llh(pos)
            baseline_enu = ecef2enu(pos, base_pos_llh)
            
            if epoch_idx % 10 == 0:
                logger.info(f"Epoch {epoch_idx}: {status} - "
                          f"Baseline ENU: [{baseline_enu[0]:.3f}, {baseline_enu[1]:.3f}, {baseline_enu[2]:.3f}] m")
                
                if result.get('ar_attempted', False):
                    ar_info = result['ar_info']
                    logger.info(f"  AR: ratio={ar_info['ratio']:.2f}, "
                              f"success_rate={ar_info['success_rate']:.3f}, "
                              f"fixed={ar_info['fixed']}")
    
    # Statistics
    logger.info("\n" + "=" * 60)
    logger.info("Processing Summary:")
    logger.info(f"  Total epochs: {n_epochs}")
    logger.info(f"  Fixed solutions: {n_fixed} ({100*n_fixed/max(1,n_fixed+n_float):.1f}%)")
    logger.info(f"  Float solutions: {n_float} ({100*n_float/max(1,n_fixed+n_float):.1f}%)")
    
    # Plot results
    if results:
        plot_rtk_results(results, base_pos_llh)


def plot_rtk_results(results: list, base_pos_llh: np.ndarray):
    """Plot RTK results comparing float and fixed solutions"""
    
    # Separate float and fixed solutions
    float_results = []
    fixed_results = []
    
    for r in results:
        if r.get('ar_success', False) and 'position_fixed' in r:
            fixed_results.append(r)
        if 'position_float' in r:
            float_results.append(r)
    
    if not float_results and not fixed_results:
        logger.warning("No results to plot")
        return
    
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Horizontal positions
    ax1 = plt.subplot(2, 3, 1)
    
    if float_results:
        float_baselines = np.array([ecef2enu(r['position_float'], base_pos_llh) 
                                   for r in float_results])
        ax1.scatter(float_baselines[:, 0], float_baselines[:, 1], 
                   c='blue', s=20, alpha=0.5, label='Float')
    
    if fixed_results:
        fixed_baselines = np.array([ecef2enu(r['position_fixed'], base_pos_llh) 
                                   for r in fixed_results])
        ax1.scatter(fixed_baselines[:, 0], fixed_baselines[:, 1], 
                   c='red', s=20, alpha=0.8, label='Fixed')
    
    ax1.set_xlabel('East (m)')
    ax1.set_ylabel('North (m)')
    ax1.set_title('Horizontal Position (Float vs Fixed)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 2. Height time series
    ax2 = plt.subplot(2, 3, 2)
    
    if float_results:
        times_float = np.array([r['time'] for r in float_results])
        times_float -= times_float[0] if len(times_float) > 0 else 0
        heights_float = float_baselines[:, 2] if 'float_baselines' in locals() else []
        if len(heights_float) > 0:
            ax2.plot(times_float, heights_float, 'b-', alpha=0.5, label='Float', linewidth=0.5)
    
    if fixed_results:
        times_fixed = np.array([r['time'] for r in fixed_results])
        times_fixed -= times_fixed[0] if len(times_fixed) > 0 else 0
        heights_fixed = fixed_baselines[:, 2] if 'fixed_baselines' in locals() else []
        if len(heights_fixed) > 0:
            ax2.plot(times_fixed, heights_fixed, 'r-', label='Fixed', linewidth=1)
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Height (m)')
    ax2.set_title('Height Time Series')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Ambiguity resolution ratio
    ax3 = plt.subplot(2, 3, 3)
    
    ar_attempted = [r for r in results if r.get('ar_attempted', False)]
    if ar_attempted:
        times_ar = np.array([r['time'] for r in ar_attempted])
        times_ar -= times_ar[0]
        ratios = np.array([r['ar_info']['ratio'] for r in ar_attempted])
        fixed_mask = np.array([r['ar_info']['fixed'] for r in ar_attempted])
        
        ax3.scatter(times_ar[~fixed_mask], ratios[~fixed_mask], c='blue', s=10, alpha=0.5, label='Failed')
        ax3.scatter(times_ar[fixed_mask], ratios[fixed_mask], c='green', s=10, alpha=0.8, label='Fixed')
        ax3.axhline(y=3.0, color='red', linestyle='--', alpha=0.5, label='Threshold')
        
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Ratio')
        ax3.set_title('Ambiguity Resolution Ratio Test')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
    
    # 4. Number of DD measurements
    ax4 = plt.subplot(2, 3, 4)
    
    times_all = np.array([r['time'] for r in results if 'time' in r])
    times_all -= times_all[0] if len(times_all) > 0 else 0
    n_dd = np.array([r['n_dd'] for r in results if 'n_dd' in r])
    
    ax4.plot(times_all, n_dd, 'k-', linewidth=1)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('# DD Measurements')
    ax4.set_title('Double Differences Used')
    ax4.grid(True, alpha=0.3)
    
    # 5. Solution type over time
    ax5 = plt.subplot(2, 3, 5)
    
    solution_type = []
    for r in results:
        if r.get('ar_success', False):
            solution_type.append(2)  # Fixed
        elif r.get('success', False):
            solution_type.append(1)  # Float
        else:
            solution_type.append(0)  # Failed
    
    ax5.plot(times_all, solution_type, 'ko-', markersize=3, linewidth=0.5)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Solution Type')
    ax5.set_title('Solution Status')
    ax5.set_yticks([0, 1, 2])
    ax5.set_yticklabels(['Failed', 'Float', 'Fixed'])
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([-0.5, 2.5])
    
    # 6. Position precision (STD)
    ax6 = plt.subplot(2, 3, 6)
    
    if fixed_results and len(fixed_baselines) > 10:
        # Compute moving STD
        window = 10
        std_e = []
        std_n = []
        std_u = []
        
        for i in range(window, len(fixed_baselines)):
            window_data = fixed_baselines[i-window:i]
            std_e.append(np.std(window_data[:, 0]))
            std_n.append(np.std(window_data[:, 1]))
            std_u.append(np.std(window_data[:, 2]))
        
        times_std = times_fixed[window:]
        ax6.plot(times_std, np.array(std_e)*100, 'r-', label='East', linewidth=1)
        ax6.plot(times_std, np.array(std_n)*100, 'g-', label='North', linewidth=1)
        ax6.plot(times_std, np.array(std_u)*100, 'b-', label='Up', linewidth=1)
        
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('STD (cm)')
        ax6.set_title(f'Position Precision (window={window})')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    plt.suptitle('RTK with Ambiguity Resolution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    output_file = 'rtk_ambiguity_resolution.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"\nPlot saved to {output_file}")
    
    # Print statistics
    if fixed_results and len(fixed_baselines) > 0:
        logger.info("\nFixed Solution Statistics:")
        logger.info(f"  Mean baseline (ENU): [{np.mean(fixed_baselines[:, 0]):.3f}, "
                   f"{np.mean(fixed_baselines[:, 1]):.3f}, {np.mean(fixed_baselines[:, 2]):.3f}] m")
        logger.info(f"  STD baseline (ENU): [{np.std(fixed_baselines[:, 0])*100:.1f}, "
                   f"{np.std(fixed_baselines[:, 1])*100:.1f}, {np.std(fixed_baselines[:, 2])*100:.1f}] cm")
    
    plt.show()


if __name__ == "__main__":
    main()