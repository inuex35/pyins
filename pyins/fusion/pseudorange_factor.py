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

"""Weighted pseudorange factor for GNSS/INS integration

This module implements pseudorange factors with dynamic weighting based on
elevation angle, SNR, and system-specific error characteristics, following
RTKLIB's approach.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..coordinate.transforms import ecef2llh
from ..core.constants import CLIGHT, SYS_BDS, SYS_GAL, SYS_GLO, SYS_GPS, SYS_QZS
from ..core.data_structures import Ephemeris, Observation
from ..observation.pseudorange import compute_pseudorange_residual
from ..satellite.satellite_position import compute_satellite_position
from .graph_optimizer import Factor
from .state import NavigationState


@dataclass
class MeasurementWeightConfig:
    """Configuration for measurement weighting"""
    # Base error terms [constant, elevation-dependent] (m)
    err_base: float = 0.3
    err_el: float = 0.3

    # SNR-related terms
    snr_max: float = 50.0  # Maximum/reference SNR (dB-Hz)
    err_snr: float = 0.3   # SNR error coefficient

    # Receiver std deviation scaling
    err_rcv_std: float = 0.0  # Receiver std coefficient (0 to disable)

    # Code/phase error ratio
    eratio_code: float = 100.0  # Code is ~100x noisier than phase

    # System-specific error factors
    sys_error_factors: dict[int, float] = field(default_factory=lambda: {
        SYS_GPS: 1.0,
        SYS_GLO: 1.5,  # GLONASS typically noisier
        SYS_GAL: 1.0,
        SYS_BDS: 1.2,
        SYS_QZS: 1.0
    })

    # Minimum elevation angle (rad)
    min_elevation: float = np.radians(5.0)


@dataclass
class WeightedPseudorangeFactor(Factor):
    """
    Pseudorange factor with dynamic weighting

    Implements RTKLIB-style variance calculation based on:
    - Elevation angle
    - Signal-to-noise ratio (SNR)
    - System-specific error characteristics
    - Receiver measurement quality
    """
    observation: Observation
    ephemeris: Ephemeris
    weight_config: MeasurementWeightConfig = field(default_factory=MeasurementWeightConfig)
    frequency_band: int = 0  # Frequency band index (0=L1, 1=L2, etc.)

    def compute_variance(self, state: NavigationState) -> float:
        """
        Compute measurement variance based on elevation and SNR

        Parameters
        ----------
        state : NavigationState
            Current navigation state for computing elevation angle

        Returns
        -------
        variance : float
            Measurement variance (m²)
        """
        # Compute satellite position
        sat_pos, _, _ = compute_satellite_position(self.ephemeris, self.observation.time)

        # Compute elevation angle
        llh = ecef2llh(state.position)
        lat, lon, _h = llh[0], llh[1], llh[2]

        # Vector from receiver to satellite
        los = sat_pos - state.position
        los_norm = np.linalg.norm(los)
        los_unit = los / los_norm

        # Convert to local ENU for elevation
        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)
        sin_lon = np.sin(lon)
        cos_lon = np.cos(lon)

        # ENU transformation matrix
        R_enu = np.array([
            [-sin_lon, cos_lon, 0],
            [-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat],
            [cos_lat*cos_lon, cos_lat*sin_lon, sin_lat]
        ])

        los_enu = R_enu @ los_unit
        elevation = np.arcsin(los_enu[2])  # Up component

        # Ensure minimum elevation
        elevation = max(elevation, self.weight_config.min_elevation)

        # Base variance calculation (RTKLIB formula)
        var = (self.weight_config.err_base**2 +
               self.weight_config.err_el**2 / np.sin(elevation))

        # SNR-dependent term
        if self.weight_config.err_snr > 0:
            snr = self.observation.SNR[self.frequency_band]
            if snr <= 0:
                snr = 30.0  # Default SNR if not available

            snr_diff = max(self.weight_config.snr_max - snr, 0)
            var += self.weight_config.err_snr**2 * (10 ** (0.1 * snr_diff))

        # Code/phase error ratio
        var *= self.weight_config.eratio_code**2

        # System-specific error factor
        sys_factor = self.weight_config.sys_error_factors.get(
            self.observation.system, 1.0
        )
        var *= sys_factor**2

        # Receiver std deviation (if available)
        if self.weight_config.err_rcv_std > 0 and hasattr(self.observation, 'Pstd'):
            if self.observation.Pstd[self.frequency_band] > 0:
                var += (self.weight_config.err_rcv_std *
                       self.observation.Pstd[self.frequency_band])**2

        return var

    def compute_residual(self, state: NavigationState) -> float:
        """
        Compute pseudorange residual

        Parameters
        ----------
        state : NavigationState
            Current navigation state

        Returns
        -------
        residual : float
            Pseudorange residual (m)
        """
        # Use existing pseudorange residual computation
        residual, _, _ = compute_pseudorange_residual(
            self.observation,
            self.ephemeris,
            state.position,
            state.clock_bias[0] / CLIGHT,
            freq_idx=self.frequency_band
        )
        return residual

    def compute_jacobian(self, state: NavigationState) -> dict[str, np.ndarray]:
        """
        Compute Jacobian of pseudorange with respect to state

        Returns
        -------
        jacobians : dict
            Dictionary containing Jacobians for position and clock
        """
        # Compute satellite position
        sat_pos, _, _ = compute_satellite_position(self.ephemeris, self.observation.time)

        # Line-of-sight unit vector
        los = sat_pos - state.position
        los_norm = np.linalg.norm(los)
        los_unit = los / los_norm

        # Jacobian w.r.t position
        H_pos = -los_unit

        # Jacobian w.r.t clock bias (converted from meters to seconds)
        H_clk = np.array([1.0])  # Already in meters

        return {
            'position': H_pos,
            'clock_bias': H_clk
        }

    def update_information_matrix(self, state: NavigationState):
        """
        Update the information matrix based on current measurement variance

        Parameters
        ----------
        state : NavigationState
            Current navigation state
        """
        variance = self.compute_variance(state)
        self.information = np.array([[1.0 / variance]])


class PseudorangeMeasurementModel:
    """
    Pseudorange measurement model with dynamic weighting for multiple satellites
    """

    def __init__(self, weight_config: Optional[MeasurementWeightConfig] = None):
        """
        Initialize measurement model

        Parameters
        ----------
        weight_config : MeasurementWeightConfig, optional
            Weight configuration (uses defaults if not provided)
        """
        self.weight_config = weight_config or MeasurementWeightConfig()

    def create_factors(self,
                      observations: list,
                      ephemerides: dict,
                      state: NavigationState,
                      frequency_band: int = 0) -> list:
        """
        Create weighted pseudorange factors for multiple satellites

        Parameters
        ----------
        observations : list
            List of Observation objects
        ephemerides : dict
            Dictionary mapping satellite ID to Ephemeris
        state : NavigationState
            Current navigation state (for computing weights)
        frequency_band : int
            Frequency band index

        Returns
        -------
        factors : list
            List of WeightedPseudorangeFactor objects
        """
        factors = []

        for obs in observations:
            if obs.sat not in ephemerides:
                continue

            # Skip if no valid pseudorange
            if obs.P[frequency_band] == 0.0:
                continue

            # Create factor
            factor = WeightedPseudorangeFactor(
                node_ids=[0],  # Single state node
                residual=np.zeros(1),
                jacobian={},
                information=np.eye(1),  # Will be updated
                observation=obs,
                ephemeris=ephemerides[obs.sat],
                weight_config=self.weight_config,
                frequency_band=frequency_band
            )

            # Update information matrix based on current state
            factor.update_information_matrix(state)

            factors.append(factor)

        return factors

    def compute_weighted_residuals(self,
                                  observations: list,
                                  ephemerides: dict,
                                  state: NavigationState,
                                  frequency_band: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute weighted pseudorange residuals for multiple satellites

        Parameters
        ----------
        observations : list
            List of Observation objects
        ephemerides : dict
            Dictionary mapping satellite ID to Ephemeris
        state : NavigationState
            Current navigation state
        frequency_band : int
            Frequency band index

        Returns
        -------
        residuals : np.ndarray
            Weighted residuals
        H : np.ndarray
            Measurement Jacobian matrix
        W : np.ndarray
            Weight matrix (inverse of covariance)
        """
        residuals = []
        jacobians = []
        weights = []

        # Create factors
        factors = self.create_factors(observations, ephemerides, state, frequency_band)

        for factor in factors:
            # Compute residual
            res = factor.compute_residual(state)
            residuals.append(res)

            # Compute Jacobian
            jac = factor.compute_jacobian(state)
            H_row = np.hstack([
                jac['position'],
                np.zeros(3),  # velocity
                np.zeros(3),  # attitude
                jac['clock_bias']
            ])
            jacobians.append(H_row)

            # Get weight (information matrix element)
            weights.append(factor.information[0, 0])

        if not residuals:
            return np.array([]), np.array([[]]), np.array([[]])

        # Stack results
        residuals = np.array(residuals)
        H = np.vstack(jacobians)
        W = np.diag(weights)

        return residuals, H, W
