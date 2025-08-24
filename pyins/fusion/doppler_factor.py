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

"""Doppler factor for GNSS/INS integration with GTSAM

This module implements Doppler shift measurements as factors in a factor graph.
The Doppler measurement provides direct velocity information which is particularly
useful for constraining velocity states in GNSS/INS integration.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..coordinate.transforms import ecef2enu, ecef2llh
from ..core.constants import CLIGHT
from ..core.data_structures import Ephemeris, Observation
from ..satellite.satellite_position import compute_satellite_position, compute_satellite_velocity
from ..sensors.lever_arm import skew_symmetric
from .graph_optimizer import Factor
from .state import NavigationState


@dataclass
class DopplerFactor(Factor):
    """
    Doppler measurement factor for factor graph optimization

    The Doppler shift is related to the relative velocity between satellite and receiver:
    doppler = -f/c * (v_sat - v_rcv) · e_sr

    where:
    - f is the carrier frequency
    - c is speed of light
    - v_sat is satellite velocity
    - v_rcv is receiver velocity (including lever arm effects)
    - e_sr is unit vector from satellite to receiver
    """
    observation: Observation
    ephemeris: Ephemeris
    frequency: float  # Carrier frequency (Hz)
    lever_arm: Optional[np.ndarray] = None  # Lever arm from IMU to antenna (body frame)

    def compute_residual(self, state: NavigationState) -> float:
        """
        Compute Doppler measurement residual

        Parameters
        ----------
        state : NavigationState
            Current navigation state containing position, velocity, attitude

        Returns
        -------
        residual : float
            Doppler residual (Hz)
        """
        # Compute satellite position and velocity
        sat_pos, sat_clk, _ = compute_satellite_position(
            self.ephemeris, self.observation.time
        )
        sat_vel, sat_clk_drift = compute_satellite_velocity(
            self.ephemeris, self.observation.time
        )

        # Get receiver antenna position and velocity
        if self.lever_arm is not None:
            # Apply lever arm compensation
            antenna_pos = state.position + state.dcm @ self.lever_arm
            omega_body = state.angular_velocity if hasattr(state, 'angular_velocity') else np.zeros(3)
            antenna_vel = state.velocity + state.dcm @ np.cross(omega_body, self.lever_arm)
        else:
            antenna_pos = state.position
            antenna_vel = state.velocity

        # Compute range and line-of-sight vector
        range_vec = sat_pos - antenna_pos
        range_norm = np.linalg.norm(range_vec)
        los_unit = range_vec / range_norm  # Unit vector from receiver to satellite

        # Relative velocity along line-of-sight
        rel_velocity = sat_vel - antenna_vel
        radial_velocity = np.dot(rel_velocity, los_unit)

        # Predicted Doppler (negative sign convention)
        predicted_doppler = -self.frequency / CLIGHT * radial_velocity

        # Add satellite clock drift effect
        predicted_doppler += self.frequency * sat_clk_drift

        # Residual = measured - predicted
        measured_doppler = self.observation.D[0]  # Assuming first frequency
        residual = measured_doppler - predicted_doppler

        return residual

    def compute_jacobian(self, state: NavigationState) -> dict[str, np.ndarray]:
        """
        Compute Jacobian of Doppler measurement with respect to state

        Returns
        -------
        jacobians : dict
            Dictionary containing Jacobians for position, velocity, and attitude
        """
        # Compute satellite position and velocity
        sat_pos, _, _ = compute_satellite_position(
            self.ephemeris, self.observation.time
        )
        sat_vel, _ = compute_satellite_velocity(
            self.ephemeris, self.observation.time
        )

        # Get receiver antenna position and velocity
        if self.lever_arm is not None:
            antenna_pos = state.position + state.dcm @ self.lever_arm
            omega_body = state.angular_velocity if hasattr(state, 'angular_velocity') else np.zeros(3)
            antenna_vel = state.velocity + state.dcm @ np.cross(omega_body, self.lever_arm)
        else:
            antenna_pos = state.position
            antenna_vel = state.velocity

        # Compute range and line-of-sight
        range_vec = sat_pos - antenna_pos
        range_norm = np.linalg.norm(range_vec)
        los_unit = range_vec / range_norm

        # Relative velocity
        rel_velocity = sat_vel - antenna_vel

        # Jacobian w.r.t position (through line-of-sight change)
        I = np.eye(3)
        los_outer = np.outer(los_unit, los_unit)
        d_los_d_pos = (I - los_outer) / range_norm

        H_pos = self.frequency / CLIGHT * rel_velocity @ d_los_d_pos

        # Jacobian w.r.t velocity (direct contribution)
        H_vel = self.frequency / CLIGHT * los_unit

        # Jacobian w.r.t attitude (if lever arm exists)
        H_att = np.zeros(3)
        if self.lever_arm is not None:
            # Attitude affects antenna velocity through lever arm
            omega_body = state.angular_velocity if hasattr(state, 'angular_velocity') else np.zeros(3)
            H_att = self.frequency / CLIGHT * los_unit @ state.dcm @ skew_symmetric(
                np.cross(omega_body, self.lever_arm)
            )

            # Also affects antenna position
            H_att += self.frequency / CLIGHT * rel_velocity @ d_los_d_pos @ state.dcm @ skew_symmetric(self.lever_arm)

        return {
            'position': H_pos,
            'velocity': H_vel,
            'attitude': H_att
        }


class DopplerMeasurementModel:
    """
    Doppler measurement model for multiple satellites

    This class handles the computation of Doppler measurements for
    multiple satellites simultaneously, including proper weighting.
    """

    def __init__(self, lever_arm: Optional[np.ndarray] = None):
        """
        Initialize Doppler measurement model

        Parameters
        ----------
        lever_arm : np.ndarray, optional
            Lever arm from IMU to antenna in body frame [x, y, z] (m)
        """
        self.lever_arm = lever_arm

    def compute_doppler_residuals(self,
                                 observations: list,
                                 ephemerides: dict,
                                 state: NavigationState,
                                 frequencies: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Doppler residuals for multiple satellites

        Parameters
        ----------
        observations : list
            List of Observation objects with Doppler measurements
        ephemerides : dict
            Dictionary mapping satellite ID to Ephemeris
        state : NavigationState
            Current navigation state
        frequencies : dict
            Dictionary mapping (sat, band) to frequency in Hz

        Returns
        -------
        residuals : np.ndarray
            Doppler residuals for all satellites (Hz)
        H : np.ndarray
            Stacked measurement Jacobian matrix
        R : np.ndarray
            Measurement noise covariance matrix
        """
        residuals = []
        jacobians = []
        variances = []

        for obs in observations:
            if obs.sat not in ephemerides:
                continue

            # Get frequency for this satellite and band
            freq_key = (obs.sat, 0)  # Assuming first band
            if freq_key not in frequencies:
                continue

            frequency = frequencies[freq_key]

            # Create Doppler factor
            factor = DopplerFactor(
                node_ids=[0],  # Single state node
                residual=np.zeros(1),
                jacobian={},
                information=np.eye(1),
                observation=obs,
                ephemeris=ephemerides[obs.sat],
                frequency=frequency,
                lever_arm=self.lever_arm
            )

            # Compute residual
            res = factor.compute_residual(state)
            residuals.append(res)

            # Compute Jacobian
            jac = factor.compute_jacobian(state)

            # Stack Jacobian components [position, velocity, attitude]
            H_row = np.hstack([jac['position'], jac['velocity'], jac['attitude']])
            jacobians.append(H_row)

            # Estimate measurement variance based on SNR and elevation
            variance = self._estimate_doppler_variance(obs, state.position,
                                                      ephemerides[obs.sat])
            variances.append(variance)

        if not residuals:
            return np.array([]), np.array([[]]), np.array([[]])

        # Stack results
        residuals = np.array(residuals)
        H = np.vstack(jacobians)
        R = np.diag(variances)

        return residuals, H, R

    def _estimate_doppler_variance(self,
                                  obs: Observation,
                                  rcv_pos: np.ndarray,
                                  eph: Ephemeris) -> float:
        """
        Estimate Doppler measurement variance

        Parameters
        ----------
        obs : Observation
            GNSS observation with SNR information
        rcv_pos : np.ndarray
            Receiver position (ECEF)
        eph : Ephemeris
            Satellite ephemeris

        Returns
        -------
        variance : float
            Estimated Doppler variance (Hz²)
        """
        # Base standard deviation (Hz)
        sigma_base = 0.1  # 0.1 Hz base uncertainty

        # SNR-based scaling
        snr = obs.SNR[0] if obs.SNR[0] > 0 else 30.0  # Default 30 dB-Hz
        snr_factor = 10 ** ((45 - snr) / 20)  # Better SNR -> lower noise

        # Elevation-based scaling
        sat_pos, _, _ = compute_satellite_position(eph, obs.time)
        sat_pos - rcv_pos

        # Convert to local coordinates for elevation
        llh = ecef2llh(rcv_pos)
        range_enu = ecef2enu(sat_pos, llh)
        elevation = np.arctan2(range_enu[2], np.linalg.norm(range_enu[:2]))

        # Low elevation -> higher noise
        elev_factor = 1.0 / np.sin(max(elevation, np.radians(5)))

        # Combined variance
        sigma = sigma_base * snr_factor * elev_factor

        return sigma ** 2
