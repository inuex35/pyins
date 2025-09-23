"""Phase range factor implementations aligned with GICI-Open."""

from __future__ import annotations

from typing import Optional

import gtsam
import numpy as np


def _point3_to_array(point: gtsam.Point3) -> np.ndarray:
    """Convert a GTSAM `Point3`/Vector to a NumPy array."""

    try:
        return np.asarray(point)
    except TypeError:
        return np.array([point.x(), point.y(), point.z()])


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
    ) -> None:
        self.position_key = position_key
        self.ambiguity_key = ambiguity_key

        self.pseudorange = float(pseudorange)
        self.phaserange = float(phaserange)
        self.sat_pos = np.asarray(sat_pos, dtype=float)

        # Extract 1-sigma values from supplied noise models
        self.sigma_pr = float(noise_model_pr.sigmas()[0])
        self.sigma_cp = float(noise_model_cp.sigmas()[0])
        self.combined_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([self.sigma_pr, self.sigma_cp])
        )

        self.reference_ecef = (
            np.asarray(reference_ecef, dtype=float)
            if reference_ecef is not None
            else np.zeros(3)
        )
        self.use_enu = use_enu

        if use_enu and reference_llh is not None:
            from pyins.coordinate.dcm import enu2ecef_dcm

            self.R_enu2ecef = enu2ecef_dcm(reference_llh)
        else:
            self.R_enu2ecef = None

    def create_factor(self) -> gtsam.CustomFactor:
        """Create the GTSAM custom factor."""

        def error_func(this: gtsam.CustomFactor, values: gtsam.Values, H: list) -> np.ndarray:
            position_point = values.atPoint3(self.position_key)
            position_vec = _point3_to_array(position_point)

            if self.use_enu and self.R_enu2ecef is not None:
                rover_ecef = self.reference_ecef + self.R_enu2ecef @ position_vec
                jac_transform = self.R_enu2ecef
            else:
                rover_ecef = position_vec
                jac_transform = np.eye(3)

            ambiguity_meters = values.atDouble(self.ambiguity_key)

            diff = self.sat_pos - rover_ecef
            geometric_range = np.linalg.norm(diff)

            if geometric_range < 1e-6:
                raise ValueError("Geometric range is degenerate (near zero).")

            error_pr = self.pseudorange - geometric_range
            error_cp = self.phaserange - (geometric_range + ambiguity_meters)

            if H is not None and len(H) > 0:
                grad_range_ecef = (rover_ecef - self.sat_pos) / geometric_range
                grad_range_state = grad_range_ecef @ jac_transform

                jac_pos = -grad_range_state
                H[0] = np.vstack([jac_pos, jac_pos])

                if len(H) > 1:
                    H[1] = np.array([[0.0], [-1.0]])

            return np.array([error_pr, error_cp])

        return gtsam.CustomFactor(
            self.combined_noise,
            [self.position_key, self.ambiguity_key],
            error_func,
        )


class DDPhaseRangeFactor:
    """Double-difference phase range factor following GICI-Open formulation."""

    def __init__(
        self,
        position_key: int,
        dd_ambiguity_key: int,
        dd_measurement_m: float,
        sat_pos_rover: np.ndarray,
        ref_sat_pos_rover: np.ndarray,
        sat_pos_base: np.ndarray,
        ref_sat_pos_base: np.ndarray,
        base_position_ecef: np.ndarray,
        wavelength: float,
        noise_model: gtsam.noiseModel.Base,
        reference_llh: Optional[np.ndarray] = None,
        use_enu: bool = True,
    ) -> None:
        self.position_key = position_key
        self.dd_ambiguity_key = dd_ambiguity_key
        self.dd_measurement = float(dd_measurement_m)

        self.sat_pos_rover = np.asarray(sat_pos_rover, dtype=float)
        self.ref_sat_pos_rover = np.asarray(ref_sat_pos_rover, dtype=float)
        self.sat_pos_base = np.asarray(sat_pos_base, dtype=float)
        self.ref_sat_pos_base = np.asarray(ref_sat_pos_base, dtype=float)
        self.base_position = np.asarray(base_position_ecef, dtype=float)

        self.wavelength = float(wavelength)
        self.noise_model = noise_model
        self.use_enu = use_enu

        from pyins.coordinate.dcm import enu2ecef_dcm

        self.reference_llh = (
            np.asarray(reference_llh, dtype=float)
            if reference_llh is not None
            else np.zeros(3)
        )
        self.R_enu2ecef = enu2ecef_dcm(self.reference_llh)

    def create_factor(self) -> gtsam.CustomFactor:
        """Create the GTSAM custom factor."""

        def error_func(this: gtsam.CustomFactor, values: gtsam.Values, H: list) -> np.ndarray:
            rover_point = values.atPoint3(self.position_key)
            rover_vec = _point3_to_array(rover_point)

            if self.use_enu:
                rover_ecef = self.base_position + self.R_enu2ecef @ rover_vec
                jac_transform = self.R_enu2ecef
            else:
                rover_ecef = rover_vec
                jac_transform = np.eye(3)

            dd_ambiguity_meters = values.atDouble(self.dd_ambiguity_key)

            rho_rover_sat = np.linalg.norm(self.sat_pos_rover - rover_ecef)
            rho_rover_ref = np.linalg.norm(self.ref_sat_pos_rover - rover_ecef)
            rho_base_sat = np.linalg.norm(self.sat_pos_base - self.base_position)
            rho_base_ref = np.linalg.norm(self.ref_sat_pos_base - self.base_position)

            if min(rho_rover_sat, rho_rover_ref) < 1e-6:
                raise ValueError("Rover-satellite geometry is degenerate.")

            dd_range = (rho_rover_sat - rho_base_sat) - (rho_rover_ref - rho_base_ref)
            residual = self.dd_measurement - (dd_range + dd_ambiguity_meters)

            if H is not None and len(H) > 0:
                e_sat = (rover_ecef - self.sat_pos_rover) / rho_rover_sat
                e_ref = (rover_ecef - self.ref_sat_pos_rover) / rho_rover_ref
                jac_pos_ecef = -(e_sat - e_ref)
                jac_pos_state = jac_pos_ecef @ jac_transform

                H[0] = jac_pos_state.reshape(1, 3)

                if len(H) > 1:
                    H[1] = np.array([[-1.0]])

            return np.array([residual])

        return gtsam.CustomFactor(
            self.noise_model,
            [self.position_key, self.dd_ambiguity_key],
            error_func,
        )


class DDPhaseRangeFactorFixed:
    """Double-difference phase range factor with fixed ambiguity (cycles)."""

    def __init__(
        self,
        position_key: int,
        dd_measurement_m: float,
        sat_pos_rover: np.ndarray,
        ref_sat_pos_rover: np.ndarray,
        sat_pos_base: np.ndarray,
        ref_sat_pos_base: np.ndarray,
        base_position_ecef: np.ndarray,
        wavelength: float,
        fixed_ambiguity_cycles: float,
        noise_model: gtsam.noiseModel.Base,
        reference_llh: np.ndarray,
        use_enu: bool = True,
    ) -> None:
        self.position_key = position_key
        self.noise_model = noise_model
        self.use_enu = use_enu

        self.sat_pos_rover = np.asarray(sat_pos_rover, dtype=float)
        self.ref_sat_pos_rover = np.asarray(ref_sat_pos_rover, dtype=float)
        self.sat_pos_base = np.asarray(sat_pos_base, dtype=float)
        self.ref_sat_pos_base = np.asarray(ref_sat_pos_base, dtype=float)
        self.base_position = np.asarray(base_position_ecef, dtype=float)

        self.wavelength = float(wavelength)
        self.fixed_ambiguity_meters = float(fixed_ambiguity_cycles) * self.wavelength

        self.dd_measurement = float(dd_measurement_m)

        from pyins.coordinate.dcm import enu2ecef_dcm

        self.reference_llh = np.asarray(reference_llh, dtype=float)
        self.R_enu2ecef = enu2ecef_dcm(self.reference_llh)

    def create_factor(self) -> gtsam.CustomFactor:
        def error_func(this: gtsam.CustomFactor, values: gtsam.Values, H: list) -> np.ndarray:
            rover_point = values.atPoint3(self.position_key)
            rover_vec = _point3_to_array(rover_point)

            if self.use_enu:
                rover_ecef = self.base_position + self.R_enu2ecef @ rover_vec
                jac_transform = self.R_enu2ecef
            else:
                rover_ecef = rover_vec
                jac_transform = np.eye(3)

            rho_rover_sat = np.linalg.norm(self.sat_pos_rover - rover_ecef)
            rho_rover_ref = np.linalg.norm(self.ref_sat_pos_rover - rover_ecef)
            rho_base_sat = np.linalg.norm(self.sat_pos_base - self.base_position)
            rho_base_ref = np.linalg.norm(self.ref_sat_pos_base - self.base_position)

            dd_range = (rho_rover_sat - rho_base_sat) - (rho_rover_ref - rho_base_ref)
            residual = self.dd_measurement - (dd_range + self.fixed_ambiguity_meters)

            if H is not None and len(H) > 0:
                e_sat = (rover_ecef - self.sat_pos_rover) / rho_rover_sat
                e_ref = (rover_ecef - self.ref_sat_pos_rover) / rho_rover_ref
                jac_pos_ecef = -(e_sat - e_ref)
                jac_pos_state = jac_pos_ecef @ jac_transform

                H[0] = jac_pos_state.reshape(1, 3)

            return np.array([residual])

        return gtsam.CustomFactor(
            self.noise_model,
            [self.position_key],
            error_func,
        )


__all__ = ['PhaseRangeFactor', 'DDPhaseRangeFactor', 'DDPhaseRangeFactorFixed']
