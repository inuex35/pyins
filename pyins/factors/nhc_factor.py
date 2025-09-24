"""Non-holonomic constraint factor for ground vehicle INS/GNSS graphs."""

from __future__ import annotations

from typing import Iterable, Sequence

import gtsam
import numpy as np


def _skew(vec: np.ndarray) -> np.ndarray:
    """Return the skew-symmetric matrix of a 3-vector."""

    x, y, z = vec
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


class NonHolonomicConstraintFactor(gtsam.CustomFactor):
    """Enforce the vehicle non-holonomic constraint in body frame.

    The factor constrains the body-frame lateral and vertical velocity components
    (defaults to Y- and Z-axes) to stay close to the supplied measurement, which
    is typically zero for wheeled ground vehicles.
    """

    def __init__(
        self,
        pose_key: int,
        velocity_key: int,
        noise_model: gtsam.noiseModel.Base,
        measured_velocity_body: Sequence[float] | None = None,
        axes: Iterable[int] = (1, 2),
    ) -> None:
        self.pose_key = pose_key
        self.velocity_key = velocity_key

        self.axes: tuple[int, ...] = tuple(axes)
        if not self.axes:
            raise ValueError("axes must contain at least one index")
        if any(idx not in (0, 1, 2) for idx in self.axes):
            raise ValueError("axes indices must be in {0, 1, 2}")

        self.dim = len(self.axes)
        if noise_model.dim() != self.dim:
            raise ValueError(
                f"noise model dimension {noise_model.dim()} does not match number of axes {self.dim}"
            )

        if measured_velocity_body is None:
            measurement = np.zeros(3, dtype=float)
        else:
            arr = np.asarray(measured_velocity_body, dtype=float)
            if arr.shape == (3,):
                measurement = arr
            elif arr.shape == (self.dim,):
                measurement = np.zeros(3, dtype=float)
                measurement[list(self.axes)] = arr
            else:
                raise ValueError(
                    "measured_velocity_body must be length 3 or match the number of constrained axes"
                )

        self.measurement = measurement
        self.selector = np.eye(3, dtype=float)[list(self.axes)]
        self.target = self.selector @ self.measurement

        super().__init__(
            noise_model,
            [pose_key, velocity_key],
            lambda factor, values, jacobians: self._error_func(values, jacobians),
        )

    def _error_func(self, values: gtsam.Values, jacobians: Sequence[np.ndarray] | None) -> np.ndarray:
        pose = values.atPose3(self.pose_key)
        velocity = np.asarray(values.atVector(self.velocity_key), dtype=float)
        if velocity.shape != (3,):
            raise ValueError("velocity state must be a 3-vector")

        R_body_to_world = pose.rotation().matrix()
        velocity_body = R_body_to_world.T @ velocity

        residual = (self.selector @ velocity_body) - self.target

        if jacobians is not None:
            if len(jacobians) > 0:
                H_pose = np.zeros((self.dim, 6), dtype=float)
                rotation_part = -self.selector @ _skew(velocity_body)
                H_pose[:, :3] = rotation_part
                jacobians[0] = H_pose
            if len(jacobians) > 1:
                jacobians[1] = self.selector @ R_body_to_world.T

        return residual


__all__ = ['NonHolonomicConstraintFactor']
