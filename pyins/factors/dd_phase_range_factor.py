"""Double-difference phase range factor implementation."""

from __future__ import annotations

from typing import Optional, Sequence

import gtsam
import numpy as np

from .phase_range_utils import normalize_bias_terms, point3_to_array


class DDPhaseRangeFactor:
    """Double-difference phase range factor following GICI-Open formulation."""

    def __init__(
        self,
        position_key: int,
        dd_ambiguity_key: int,
        dd_measurement: float,
        sat_pos_rover: np.ndarray,
        ref_sat_pos_rover: np.ndarray,
        sat_pos_base: np.ndarray,
        ref_sat_pos_base: np.ndarray,
        base_position_ecef: np.ndarray,
        wavelength: float,
        noise_model: Optional[gtsam.noiseModel.Base] = None,
        reference_llh: Optional[np.ndarray] = None,
        use_enu: bool = True,
        bias_keys: Optional[Sequence[int]] = None,
        bias_coeffs: Optional[Sequence[float]] = None,
        measurement_in_meters: bool = False,
        ambiguity_in_cycles: bool = True,
    ) -> None:
        self.position_key = position_key
        self.dd_ambiguity_key = dd_ambiguity_key
        self.dd_measurement = float(dd_measurement)

        self.sat_pos_rover = np.asarray(sat_pos_rover, dtype=float)
        self.ref_sat_pos_rover = np.asarray(ref_sat_pos_rover, dtype=float)
        self.sat_pos_base = np.asarray(sat_pos_base, dtype=float)
        self.ref_sat_pos_base = np.asarray(ref_sat_pos_base, dtype=float)
        self.base_position = np.asarray(base_position_ecef, dtype=float)

        self.wavelength = float(wavelength)
        if noise_model is None:
            from pyins.core.stats import ERR_CONSTANT

            sigma = ERR_CONSTANT if self.measurement_in_meters else ERR_CONSTANT / self.wavelength
            self.noise_model = gtsam.noiseModel.Isotropic.Sigma(1, sigma)
        else:
            self.noise_model = noise_model
        self.use_enu = use_enu
        self.bias_terms = normalize_bias_terms(bias_keys, bias_coeffs)
        self.measurement_in_meters = measurement_in_meters
        self.ambiguity_in_cycles = ambiguity_in_cycles

        from pyins.coordinate.dcm import enu2ecef_dcm

        self.reference_llh = (
            np.asarray(reference_llh, dtype=float)
            if reference_llh is not None
            else np.zeros(3, dtype=float)
        )
        self.R_enu2ecef = enu2ecef_dcm(self.reference_llh)

    def create_factor(self) -> gtsam.CustomFactor:
        """Create the GTSAM custom factor."""

        def error_func(this: gtsam.CustomFactor, values: gtsam.Values, H: list) -> np.ndarray:
            try:
                rover_point = values.atPoint3(self.position_key)
                rover_vec = point3_to_array(rover_point)
            except Exception:
                rover_vec = np.asarray(values.atVector(self.position_key), dtype=float)

            if self.use_enu:
                rover_ecef = self.base_position + self.R_enu2ecef @ rover_vec
                jac_transform = self.R_enu2ecef
            else:
                rover_ecef = rover_vec
                jac_transform = np.eye(3)

            ambiguity_value = values.atDouble(self.dd_ambiguity_key)
            if self.ambiguity_in_cycles:
                dd_ambiguity_meters = ambiguity_value * self.wavelength
            else:
                dd_ambiguity_meters = ambiguity_value
            bias_sum = 0.0
            for key, coeff in self.bias_terms:
                bias_sum += coeff * values.atDouble(key)

            rho_rover_sat = np.linalg.norm(self.sat_pos_rover - rover_ecef)
            rho_rover_ref = np.linalg.norm(self.ref_sat_pos_rover - rover_ecef)
            rho_base_sat = np.linalg.norm(self.sat_pos_base - self.base_position)
            rho_base_ref = np.linalg.norm(self.ref_sat_pos_base - self.base_position)

            if min(rho_rover_sat, rho_rover_ref) < 1e-6:
                raise ValueError("Rover-satellite geometry is degenerate.")

            dd_range = (rho_rover_sat - rho_base_sat) - (rho_rover_ref - rho_base_ref)
            dd_measurement_m = (
                self.dd_measurement
                if self.measurement_in_meters
                else self.dd_measurement * self.wavelength
            )

            residual = dd_measurement_m - (dd_range + dd_ambiguity_meters + bias_sum)

            if H is not None and len(H) > 0:
                e_sat = (rover_ecef - self.sat_pos_rover) / rho_rover_sat
                e_ref = (rover_ecef - self.ref_sat_pos_rover) / rho_rover_ref
                jac_pos_ecef = -(e_sat - e_ref)
                jac_pos_state = jac_pos_ecef @ jac_transform

                H[0] = jac_pos_state.reshape(1, 3)

                if len(H) > 1:
                    scale = -self.wavelength if self.ambiguity_in_cycles else -1.0
                    H[1] = np.array([[scale]])
                for idx, (_, coeff) in enumerate(self.bias_terms):
                    if len(H) > 2 + idx:
                        H[2 + idx] = np.array([[-coeff]])

            return np.array([residual])

        factor_keys = [self.position_key, self.dd_ambiguity_key]
        factor_keys.extend(key for key, _ in self.bias_terms)
        return gtsam.CustomFactor(
            self.noise_model,
            factor_keys,
            error_func,
        )


__all__ = ['DDPhaseRangeFactor']
