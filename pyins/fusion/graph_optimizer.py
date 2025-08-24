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

"""Factor graph optimization for GNSS/IMU fusion"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..core.data_structures import Ephemeris, Observation
from ..sensors.imu import IMUBias, IMUPreintegration
from .state import NavigationState


@dataclass
class Factor:
    """Base factor class"""
    node_ids: list[int]
    residual: np.ndarray
    jacobian: dict[int, np.ndarray]
    information: np.ndarray

    def error(self, states: dict[int, NavigationState]) -> float:
        """Compute factor error"""
        return 0.5 * self.residual.T @ self.information @ self.residual


@dataclass
class IMUFactor(Factor):
    """IMU preintegration factor"""
    preintegration: IMUPreintegration

    def compute_residual(self, state_i: NavigationState,
                        state_j: NavigationState) -> np.ndarray:
        """Compute IMU factor residual"""
        # Predict state j from state i
        bias_i = IMUBias(state_i.acc_bias, state_i.gyro_bias)
        gravity = np.array([0, 0, -9.81])  # Simplified

        pos_j_pred, vel_j_pred, rot_j_pred = self.preintegration.predict(
            state_i.position, state_i.velocity, state_i.dcm,
            bias_i, gravity
        )

        # Residual
        r_p = state_j.position - pos_j_pred
        r_v = state_j.velocity - vel_j_pred
        r_R = so3_log(state_j.dcm @ rot_j_pred.T)

        return np.concatenate([r_p, r_v, r_R])


@dataclass
class GNSSFactor(Factor):
    """GNSS pseudorange/carrier phase factor"""
    observation: Observation
    ephemeris: Ephemeris
    measurement_type: str  # 'pseudorange' or 'carrier'

    def compute_residual(self, state: NavigationState) -> float:
        """Compute GNSS measurement residual"""
        from ..observation.carrier_phase import compute_carrier_residual
        from ..observation.pseudorange import compute_pseudorange_residual

        if self.measurement_type == 'pseudorange':
            res, _, _ = compute_pseudorange_residual(
                self.observation, self.ephemeris,
                state.position, state.clock_bias[0] / CLIGHT
            )
        else:  # carrier phase
            ambiguity = state.ambiguities.get(self.observation.sat, 0.0)
            res, _, _ = compute_carrier_residual(
                self.observation, self.ephemeris,
                state.position, state.clock_bias[0] / CLIGHT,
                ambiguity
            )
        return res


@dataclass
class PriorFactor(Factor):
    """Prior factor on state"""
    prior_state: NavigationState

    def compute_residual(self, state: NavigationState) -> np.ndarray:
        """Compute prior residual"""
        r_p = state.position - self.prior_state.position
        r_v = state.velocity - self.prior_state.velocity
        r_R = so3_log(state.dcm @ self.prior_state.dcm.T)
        r_ba = state.acc_bias - self.prior_state.acc_bias
        r_bg = state.gyro_bias - self.prior_state.gyro_bias

        return np.concatenate([r_p, r_v, r_R, r_ba, r_bg])


class FactorGraph:
    """Factor graph for GNSS/IMU optimization"""

    def __init__(self):
        self.nodes = {}  # node_id -> NavigationState
        self.factors = []  # List of factors
        self.next_node_id = 0

    def add_node(self, state: NavigationState) -> int:
        """Add state node to graph"""
        node_id = self.next_node_id
        self.nodes[node_id] = state.copy()
        self.next_node_id += 1
        return node_id

    def add_factor(self, factor: Factor):
        """Add factor to graph"""
        self.factors.append(factor)

    def add_imu_factor(self, node_i: int, node_j: int,
                      preintegration: IMUPreintegration,
                      information: Optional[np.ndarray] = None):
        """Add IMU preintegration factor"""
        if information is None:
            # Default information matrix
            information = np.eye(9) * 100  # Position, velocity, rotation

        factor = IMUFactor(
            node_ids=[node_i, node_j],
            residual=np.zeros(9),
            jacobian={},
            information=information,
            preintegration=preintegration
        )
        self.add_factor(factor)

    def add_gnss_factor(self, node_id: int,
                       observation: Observation,
                       ephemeris: Ephemeris,
                       measurement_type: str = 'pseudorange',
                       sigma: float = 1.0):
        """Add GNSS measurement factor"""
        information = np.array([[1.0 / sigma**2]])

        factor = GNSSFactor(
            node_ids=[node_id],
            residual=np.zeros(1),
            jacobian={},
            information=information,
            observation=observation,
            ephemeris=ephemeris,
            measurement_type=measurement_type
        )
        self.add_factor(factor)

    def add_prior_factor(self, node_id: int,
                        prior_state: NavigationState,
                        position_sigma: float = 10.0,
                        velocity_sigma: float = 1.0,
                        attitude_sigma: float = 0.1,
                        bias_sigma: float = 0.01):
        """Add prior factor"""
        sigmas = np.array([
            position_sigma, position_sigma, position_sigma,
            velocity_sigma, velocity_sigma, velocity_sigma,
            attitude_sigma, attitude_sigma, attitude_sigma,
            bias_sigma, bias_sigma, bias_sigma,
            bias_sigma, bias_sigma, bias_sigma
        ])
        information = np.diag(1.0 / sigmas**2)

        factor = PriorFactor(
            node_ids=[node_id],
            residual=np.zeros(15),
            jacobian={},
            information=information,
            prior_state=prior_state
        )
        self.add_factor(factor)

    def optimize(self, max_iterations: int = 10,
                tolerance: float = 1e-6) -> dict[int, NavigationState]:
        """
        Optimize factor graph using Gauss-Newton

        Parameters:
        -----------
        max_iterations : int
            Maximum optimization iterations
        tolerance : float
            Convergence tolerance

        Returns:
        --------
        optimized_nodes : Dict[int, NavigationState]
            Optimized states
        """
        for iteration in range(max_iterations):
            # Linearize factors
            H_total = {}  # Hessian blocks
            b_total = {}  # Right-hand side

            total_error = 0.0

            for factor in self.factors:
                # Compute residual and Jacobians
                if isinstance(factor, IMUFactor):
                    # IMU factor between two nodes
                    i, j = factor.node_ids
                    residual = factor.compute_residual(
                        self.nodes[i], self.nodes[j])

                    # Compute Jacobians (simplified)
                    J_i = -np.eye(9)  # w.r.t state i
                    J_j = np.eye(9)   # w.r.t state j

                    # Add to system
                    self._add_to_system(H_total, b_total,
                                      {i: J_i, j: J_j},
                                      residual, factor.information)

                elif isinstance(factor, GNSSFactor):
                    # GNSS factor on single node
                    node_id = factor.node_ids[0]
                    residual = np.array([factor.compute_residual(
                        self.nodes[node_id])])

                    # Numerical Jacobian (simplified)
                    J = self._compute_gnss_jacobian(
                        factor, self.nodes[node_id])

                    self._add_to_system(H_total, b_total,
                                      {node_id: J},
                                      residual, factor.information)

                elif isinstance(factor, PriorFactor):
                    # Prior factor
                    node_id = factor.node_ids[0]
                    residual = factor.compute_residual(
                        self.nodes[node_id])

                    # Jacobian is identity for prior
                    J = np.eye(len(residual))

                    self._add_to_system(H_total, b_total,
                                      {node_id: J},
                                      residual, factor.information)

                # Accumulate error
                total_error += factor.error(self.nodes)

            # Solve linear system
            dx = self._solve_system(H_total, b_total)

            # Update states
            max_update = 0.0
            for node_id, state in self.nodes.items():
                if node_id in dx:
                    update = dx[node_id]

                    # Update position and velocity
                    state.position += update[0:3]
                    state.velocity += update[3:6]

                    # Update rotation
                    dR = so3_exp(update[6:9])
                    state.dcm = dR @ state.dcm

                    # Update biases
                    if len(update) > 9:
                        state.acc_bias += update[9:12]
                        state.gyro_bias += update[12:15]

                    max_update = max(max_update, np.linalg.norm(update))

            print(f"Iteration {iteration}: error = {total_error:.6f}, "
                  f"max update = {max_update:.6f}")

            if max_update < tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break

        return self.nodes

    def _add_to_system(self, H_total: dict, b_total: dict,
                      jacobians: dict[int, np.ndarray],
                      residual: np.ndarray,
                      information: np.ndarray):
        """Add factor contribution to system"""
        # Weighted residual
        weighted_res = information @ residual

        for i, J_i in jacobians.items():
            # Right-hand side
            if i not in b_total:
                b_total[i] = np.zeros(J_i.shape[1])
            b_total[i] += J_i.T @ weighted_res

            # Hessian blocks
            for j, J_j in jacobians.items():
                key = (min(i, j), max(i, j))
                if key not in H_total:
                    H_total[key] = np.zeros((J_i.shape[1], J_j.shape[1]))

                if i <= j:
                    H_total[key] += J_i.T @ information @ J_j
                else:
                    H_total[key] += J_j.T @ information @ J_i

    def _solve_system(self, H_total: dict, b_total: dict) -> dict[int, np.ndarray]:
        """Solve sparse linear system"""
        # Simple dense solver (for small problems)
        # In practice, would use sparse solver

        # Get node ordering
        node_ids = sorted(self.nodes.keys())
        n_nodes = len(node_ids)
        state_dim = 15  # Simplified

        # Build dense system
        H = np.zeros((n_nodes * state_dim, n_nodes * state_dim))
        b = np.zeros(n_nodes * state_dim)

        for i, node_i in enumerate(node_ids):
            if node_i in b_total:
                b[i*state_dim:(i+1)*state_dim] = -b_total[node_i][:state_dim]

            for j, node_j in enumerate(node_ids):
                key = (min(node_i, node_j), max(node_i, node_j))
                if key in H_total:
                    if node_i <= node_j:
                        H[i*state_dim:(i+1)*state_dim,
                          j*state_dim:(j+1)*state_dim] = H_total[key][:state_dim, :state_dim]
                    else:
                        H[i*state_dim:(i+1)*state_dim,
                          j*state_dim:(j+1)*state_dim] = H_total[key][:state_dim, :state_dim].T

        # Make symmetric
        H = 0.5 * (H + H.T)

        # Solve
        try:
            dx_vec = np.linalg.solve(H, b)
        except:
            # Regularize if singular
            dx_vec = np.linalg.solve(H + 1e-6 * np.eye(H.shape[0]), b)

        # Extract updates
        dx = {}
        for i, node_id in enumerate(node_ids):
            dx[node_id] = dx_vec[i*state_dim:(i+1)*state_dim]

        return dx

    def _compute_gnss_jacobian(self, factor: GNSSFactor,
                             state: NavigationState) -> np.ndarray:
        """Compute GNSS Jacobian numerically"""
        # Simplified - in practice would compute analytically
        eps = 1e-6
        J = np.zeros((1, 15))

        # Position derivatives
        for i in range(3):
            state_plus = state.copy()
            state_plus.position[i] += eps

            res_plus = factor.compute_residual(state_plus)
            res_minus = factor.compute_residual(state)

            J[0, i] = (res_plus - res_minus) / eps

        return J


def so3_exp(v: np.ndarray) -> np.ndarray:
    """SO(3) exponential map"""
    theta = np.linalg.norm(v)
    if theta < 1e-8:
        return np.eye(3) + skew_symmetric(v)

    axis = v / theta
    K = skew_symmetric(axis)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def so3_log(R: np.ndarray) -> np.ndarray:
    """SO(3) logarithm map"""
    theta = np.arccos((np.trace(R) - 1) / 2)

    if theta < 1e-8:
        return np.zeros(3)

    return theta / (2 * np.sin(theta)) * np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ])


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Create skew-symmetric matrix"""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
