"""Single-receiver phase range factor implementation."""

from __future__ import annotations

from typing import Optional, Sequence

import gtsam
import numpy as np

from .phase_range_utils import normalize_bias_terms, point3_to_array


class PhaseRangeFactor:
    """Single-receiver phase range factor using GICI-Open residual convention."""

    def __init__(
        self,
        position_key: int,
        ambiguity_key: int,
        pseudorange: float,
        phaserange: float,
        sat_pos: np.ndarray,
        wavelength: float,
        noise_model_pr: gtsam.noiseModel.Base,
        noise_model_cp: gtsam.noiseModel.Base,
        reference_ecef: Optional[np.ndarray] = None,
        use_enu: bool = False,
        reference_llh: Optional[np.ndarray] = None,
        bias_keys: Optional[Sequence[int]] = None,
        bias_coeffs: Optional[Sequence[float]] = None,
    ) -> None:
        self.position_key = position_key
        self.ambiguity_key = ambiguity_key

        self.pseudorange = float(pseudorange)
        self.phaserange = float(phaserange)
        self.sat_pos = np.asarray(sat_pos, dtype=float)

        self.sigma_pr = float(noise_model_pr.sigmas()[0])
        self.sigma_cp = float(noise_model_cp.sigmas()[0])
        self.combined_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([self.sigma_pr, self.sigma_cp])
        )

        self.bias_terms = normalize_bias_terms(bias_keys, bias_coeffs)

        self.reference_ecef = (
            np.asarray(reference_ecef, dtype=float)
            if reference_ecef is not None
            else np.zeros(3, dtype=float)
        )
        self.use_enu = use_enu

        if use_enu and reference_llh is not None:
            from pyins.coordinate.dcm import enu2ecef_dcm

            self.R_enu2ecef = enu2ecef_dcm(reference_llh)
        else:
            self.R_enu2ecef = None

        self.wavelength = float(wavelength)

    def create_factor(self) -> gtsam.CustomFactor:
        """Create the GTSAM custom factor instance."""

        def error_func(this: gtsam.CustomFactor, values: gtsam.Values, H: list) -> np.ndarray:
            try:
                position_point = values.atPoint3(self.position_key)
                position_vec = point3_to_array(position_point)
            except Exception:
                position_vec = np.asarray(values.atVector(self.position_key), dtype=float)

            if self.use_enu and self.R_enu2ecef is not None:
                rover_ecef = self.reference_ecef + self.R_enu2ecef @ position_vec
                jac_transform = self.R_enu2ecef
            else:
                rover_ecef = position_vec
                jac_transform = np.eye(3)

            ambiguity_meters = values.atDouble(self.ambiguity_key)
            bias_sum = 0.0
            for key, coeff in self.bias_terms:
                bias_sum += coeff * values.atDouble(key)

            diff = self.sat_pos - rover_ecef
            geometric_range = np.linalg.norm(diff)

            if geometric_range < 1e-6:
                raise ValueError("Geometric range is degenerate (near zero).")

            error_pr = self.pseudorange - geometric_range
            error_cp = self.phaserange - (geometric_range + ambiguity_meters + bias_sum)

            if H is not None and len(H) > 0:
                grad_range_ecef = (rover_ecef - self.sat_pos) / geometric_range
                grad_range_state = grad_range_ecef @ jac_transform

                jac_pos = -grad_range_state
                H[0] = np.vstack([jac_pos, jac_pos])

                if len(H) > 1:
                    H[1] = np.array([[0.0], [-1.0]])
                for idx, (_, coeff) in enumerate(self.bias_terms):
                    if len(H) > 2 + idx:
                        H[2 + idx] = np.array([[0.0], [-coeff]])

            return np.array([error_pr, error_cp])

        factor_keys = [self.position_key, self.ambiguity_key]
        factor_keys.extend(key for key, _ in self.bias_terms)
        return gtsam.CustomFactor(
            self.combined_noise,
            factor_keys,
            error_func,
        )


__all__ = ['PhaseRangeFactor']
