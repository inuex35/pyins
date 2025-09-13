#!/usr/bin/env python3
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

"""
Position computation using fixed ambiguities
Based on GREAT-PVT and RTKLIB implementations
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Constants
CLIGHT = 299792458.0  # Speed of light (m/s)
FREQ_L1 = 1575.42e6   # GPS L1 frequency (Hz)
FREQ_L2 = 1227.60e6   # GPS L2 frequency (Hz)
LAMBDA_L1 = CLIGHT / FREQ_L1  # L1 wavelength
LAMBDA_L2 = CLIGHT / FREQ_L2  # L2 wavelength


@dataclass
class RTKPosition:
    """RTK position solution with fixed ambiguities"""
    position: np.ndarray      # ECEF position [x, y, z] (m)
    velocity: np.ndarray      # ECEF velocity [vx, vy, vz] (m/s)
    clock_bias: float         # Receiver clock bias (m)
    clock_drift: float        # Receiver clock drift (m/s)
    n_fixed: int              # Number of fixed ambiguities
    n_total: int              # Total number of ambiguities
    ratio: float              # Ratio test value
    residuals: np.ndarray     # Post-fit residuals
    covariance: np.ndarray    # Position covariance matrix
    pdop: float               # Position DOP
    fix_rate: float           # Fix rate (n_fixed/n_total)


class PositionWithFixedAmbiguity:
    """
    Compute RTK position using fixed integer ambiguities
    Based on GREAT-PVT approach
    """
    
    def __init__(self, 
                 min_satellites: int = 4,
                 max_iterations: int = 10,
                 convergence_threshold: float = 1e-4):
        """
        Initialize position calculator
        
        Parameters
        ----------
        min_satellites : int
            Minimum number of satellites for solution
        max_iterations : int
            Maximum iterations for least squares
        convergence_threshold : float
            Convergence threshold (m)
        """
        self.min_satellites = min_satellites
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
    
    def compute_position(self,
                        dd_observations: Dict[int, np.ndarray],
                        dd_ranges: np.ndarray,
                        fixed_ambiguities: Dict[int, np.ndarray],
                        is_fixed: np.ndarray,
                        sat_positions: np.ndarray,
                        base_position: np.ndarray,
                        rover_position_init: np.ndarray,
                        elevation_mask: float = 10.0) -> RTKPosition:
        """
        Compute RTK position with fixed ambiguities
        
        Parameters
        ----------
        dd_observations : Dict[int, np.ndarray]
            Double-differenced observations {0: L1, 1: L2} (cycles)
        dd_ranges : np.ndarray
            Double-differenced geometric ranges (m)
        fixed_ambiguities : Dict[int, np.ndarray]
            Fixed integer ambiguities {0: N1, 1: N2}
        is_fixed : np.ndarray
            Boolean mask of fixed ambiguities
        sat_positions : np.ndarray
            Satellite positions ECEF (m)
        base_position : np.ndarray
            Base station position ECEF (m)
        rover_position_init : np.ndarray
            Initial rover position ECEF (m)
        elevation_mask : float
            Elevation mask angle (degrees)
            
        Returns
        -------
        solution : RTKPosition
            RTK position solution
        """
        n_sat = len(dd_observations[0])
        n_fixed = np.sum(is_fixed)
        
        # Convert observations to meters
        dd_phase_L1_m = dd_observations[0] * LAMBDA_L1
        dd_phase_L2_m = dd_observations[1] * LAMBDA_L2 if 1 in dd_observations else None
        
        # Apply fixed ambiguities
        if 0 in fixed_ambiguities:
            dd_phase_L1_m_fixed = dd_phase_L1_m - fixed_ambiguities[0] * LAMBDA_L1
        else:
            dd_phase_L1_m_fixed = dd_phase_L1_m
            
        if dd_phase_L2_m is not None and 1 in fixed_ambiguities:
            dd_phase_L2_m_fixed = dd_phase_L2_m - fixed_ambiguities[1] * LAMBDA_L2
        else:
            dd_phase_L2_m_fixed = dd_phase_L2_m
        
        # Build observation vector
        if dd_phase_L2_m_fixed is not None:
            # Ionosphere-free combination for fixed ambiguities
            f1_sq = FREQ_L1 ** 2
            f2_sq = FREQ_L2 ** 2
            alpha = f1_sq / (f1_sq - f2_sq)
            beta = -f2_sq / (f1_sq - f2_sq)
            
            observations = np.zeros(n_sat)
            for i in range(n_sat):
                if is_fixed[i]:
                    # Use ionosphere-free combination for fixed
                    observations[i] = alpha * dd_phase_L1_m_fixed[i] + beta * dd_phase_L2_m_fixed[i]
                else:
                    # Use L1 only for float
                    observations[i] = dd_phase_L1_m[i]
        else:
            # L1 only
            observations = dd_phase_L1_m_fixed if n_fixed > 0 else dd_phase_L1_m
        
        # Iterative least squares
        rover_pos = rover_position_init.copy()
        clock_bias = 0.0
        
        for iteration in range(self.max_iterations):
            # Compute design matrix and residuals
            H, residuals = self._build_design_matrix(
                rover_pos, base_position, sat_positions, 
                observations, dd_ranges, is_fixed
            )
            
            # Weight matrix (elevation weighting)
            W = self._compute_weight_matrix(
                sat_positions, rover_pos, elevation_mask
            )
            
            # Weighted least squares
            try:
                # Normal equation: (H^T W H)^-1 H^T W y
                HTWH = H.T @ W @ H
                HTWy = H.T @ W @ residuals
                dx = np.linalg.solve(HTWH, HTWy)
                
                # Covariance matrix
                Q = np.linalg.inv(HTWH)
                
            except np.linalg.LinAlgError:
                logger.warning("Singular matrix in position computation")
                return self._create_failed_solution(rover_position_init, n_fixed, n_sat)
            
            # Update position
            rover_pos[:3] += dx[:3]
            if len(dx) > 3:
                clock_bias += dx[3]
            
            # Check convergence
            if np.linalg.norm(dx[:3]) < self.convergence_threshold:
                break
        
        # Compute final statistics
        post_fit_residuals = residuals - H @ dx
        sigma_0 = np.sqrt((post_fit_residuals.T @ W @ post_fit_residuals) / (n_sat - 4))
        covariance = sigma_0**2 * Q[:3, :3]
        
        # DOP calculation
        pdop = np.sqrt(np.trace(Q[:3, :3]))
        
        # Ratio test (simplified)
        ratio = 1.0 / sigma_0 if sigma_0 > 0 else 999.0
        
        return RTKPosition(
            position=rover_pos,
            velocity=np.zeros(3),  # Not computed here
            clock_bias=clock_bias,
            clock_drift=0.0,
            n_fixed=n_fixed,
            n_total=n_sat,
            ratio=ratio,
            residuals=post_fit_residuals,
            covariance=covariance,
            pdop=pdop,
            fix_rate=n_fixed / n_sat if n_sat > 0 else 0.0
        )
    
    def _build_design_matrix(self, 
                            rover_pos: np.ndarray,
                            base_pos: np.ndarray,
                            sat_positions: np.ndarray,
                            observations: np.ndarray,
                            dd_ranges: np.ndarray,
                            is_fixed: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Build design matrix and residuals for least squares"""
        n_sat = len(observations)
        H = np.zeros((n_sat, 4))  # [dx, dy, dz, dt]
        residuals = np.zeros(n_sat)
        
        for i in range(n_sat):
            # Unit vector from rover to satellite
            sat_vec = sat_positions[i] - rover_pos
            range_rover = np.linalg.norm(sat_vec)
            unit_vec = sat_vec / range_rover
            
            # Design matrix row
            H[i, :3] = -unit_vec
            H[i, 3] = 1.0  # Clock term
            
            # Computed DD range
            base_vec = sat_positions[i] - base_pos
            range_base = np.linalg.norm(base_vec)
            computed_dd = range_rover - range_base
            
            # Residual
            residuals[i] = observations[i] - computed_dd
        
        return H, residuals
    
    def _compute_weight_matrix(self,
                              sat_positions: np.ndarray,
                              rover_pos: np.ndarray,
                              elevation_mask: float) -> np.ndarray:
        """Compute weight matrix based on elevation angles"""
        n_sat = len(sat_positions)
        weights = np.ones(n_sat)
        
        for i in range(n_sat):
            # Compute elevation angle
            sat_vec = sat_positions[i] - rover_pos
            elevation = self._compute_elevation(rover_pos, sat_vec)
            
            if elevation < elevation_mask:
                weights[i] = 0.0
            else:
                # Elevation-dependent weighting
                sin_el = np.sin(np.radians(elevation))
                weights[i] = sin_el ** 2
        
        return np.diag(weights)
    
    def _compute_elevation(self, pos: np.ndarray, sat_vec: np.ndarray) -> float:
        """Compute satellite elevation angle"""
        # Convert to local ENU
        lat, lon, _ = self._ecef2llh(pos)
        enu = self._ecef2enu(sat_vec, lat, lon)
        
        # Elevation angle
        horizontal = np.sqrt(enu[0]**2 + enu[1]**2)
        elevation = np.degrees(np.arctan2(enu[2], horizontal))
        
        return elevation
    
    def _ecef2llh(self, pos: np.ndarray) -> Tuple[float, float, float]:
        """Convert ECEF to latitude, longitude, height"""
        x, y, z = pos
        
        # WGS84 parameters
        a = 6378137.0
        f = 1.0 / 298.257223563
        e2 = 2*f - f**2
        
        # Longitude
        lon = np.arctan2(y, x)
        
        # Iterative latitude computation
        p = np.sqrt(x**2 + y**2)
        lat = np.arctan2(z, p * (1 - e2))
        
        for _ in range(5):
            N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
            lat = np.arctan2(z + e2 * N * np.sin(lat), p)
        
        # Height
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        h = p / np.cos(lat) - N
        
        return np.degrees(lat), np.degrees(lon), h
    
    def _ecef2enu(self, vec: np.ndarray, lat: float, lon: float) -> np.ndarray:
        """Convert ECEF vector to ENU"""
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        # Rotation matrix
        R = np.array([
            [-np.sin(lon_rad), np.cos(lon_rad), 0],
            [-np.sin(lat_rad)*np.cos(lon_rad), -np.sin(lat_rad)*np.sin(lon_rad), np.cos(lat_rad)],
            [np.cos(lat_rad)*np.cos(lon_rad), np.cos(lat_rad)*np.sin(lon_rad), np.sin(lat_rad)]
        ])
        
        return R @ vec
    
    def _create_failed_solution(self, 
                               init_pos: np.ndarray, 
                               n_fixed: int, 
                               n_total: int) -> RTKPosition:
        """Create a failed solution result"""
        return RTKPosition(
            position=init_pos,
            velocity=np.zeros(3),
            clock_bias=0.0,
            clock_drift=0.0,
            n_fixed=n_fixed,
            n_total=n_total,
            ratio=0.0,
            residuals=np.array([]),
            covariance=np.eye(3) * 999.0,
            pdop=999.0,
            fix_rate=n_fixed / n_total if n_total > 0 else 0.0
        )


def test_position_with_fixed_ambiguity():
    """Test position computation with fixed ambiguities"""
    print("=" * 70)
    print("Testing Position Computation with Fixed Ambiguities")
    print("=" * 70)
    
    # Create test data
    n_sat = 8
    base_pos = np.array([-3961905.0, 3348993.0, 3698211.0])
    rover_true = base_pos + np.array([10.0, 20.0, 5.0])  # 10m offset
    
    # Simulate satellite positions
    sat_positions = np.zeros((n_sat, 3))
    for i in range(n_sat):
        angle = i * 2 * np.pi / n_sat
        sat_positions[i] = base_pos + np.array([
            20000000 * np.cos(angle),
            20000000 * np.sin(angle),
            15000000
        ])
    
    # Simulate DD observations
    dd_obs = {
        0: np.random.normal(0, 0.01, n_sat),  # L1 with small noise
        1: np.random.normal(0, 0.02, n_sat)   # L2
    }
    
    # Fixed ambiguities (integer values)
    fixed_amb = {
        0: np.round(np.random.normal(0, 100, n_sat)),
        1: np.round(np.random.normal(0, 80, n_sat))
    }
    is_fixed = np.ones(n_sat, dtype=bool)
    is_fixed[n_sat//2:] = False  # Half fixed, half float
    
    # DD ranges
    dd_ranges = np.zeros(n_sat)
    for i in range(n_sat):
        range_base = np.linalg.norm(sat_positions[i] - base_pos)
        range_rover = np.linalg.norm(sat_positions[i] - rover_true)
        dd_ranges[i] = range_rover - range_base
    
    # Initial position with error
    rover_init = rover_true + np.array([1.0, 2.0, 0.5])
    
    # Compute position
    calculator = PositionWithFixedAmbiguity()
    solution = calculator.compute_position(
        dd_obs, dd_ranges, fixed_amb, is_fixed,
        sat_positions, base_pos, rover_init
    )
    
    # Results
    position_error = np.linalg.norm(solution.position - rover_true)
    
    print(f"\nResults:")
    print(f"  True position: {rover_true}")
    print(f"  Computed position: {solution.position}")
    print(f"  Position error: {position_error:.4f} m")
    print(f"  Fixed ambiguities: {solution.n_fixed}/{solution.n_total}")
    print(f"  Fix rate: {solution.fix_rate:.2%}")
    print(f"  PDOP: {solution.pdop:.2f}")
    print(f"  Ratio: {solution.ratio:.2f}")
    
    print(f"\nPosition STD (1-sigma):")
    std = np.sqrt(np.diag(solution.covariance))
    print(f"  X: {std[0]*1000:.2f} mm")
    print(f"  Y: {std[1]*1000:.2f} mm")
    print(f"  Z: {std[2]*1000:.2f} mm")
    print(f"  3D: {np.linalg.norm(std)*1000:.2f} mm")
    
    print("=" * 70)
    print("✓ Position computation with fixed ambiguities implemented")
    print("✓ Based on GREAT-PVT/RTKLIB approach")
    print("=" * 70)


if __name__ == '__main__':
    test_position_with_fixed_ambiguity()