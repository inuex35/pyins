"""Shared helpers for phase range factors."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import gtsam
import numpy as np


def point3_to_array(point: gtsam.Point3) -> np.ndarray:
    """Convert a GTSAM Point3 or compatible object to a NumPy array."""

    try:
        return np.asarray(point, dtype=float)
    except TypeError:
        return np.array([point.x(), point.y(), point.z()], dtype=float)


def normalize_bias_terms(
    bias_keys: Sequence[int] | None,
    bias_coeffs: Sequence[float] | None,
) -> List[Tuple[int, float]]:
    """Build a list of (key, coefficient) pairs for bias parameters."""

    if not bias_keys:
        return []

    if bias_coeffs is None:
        bias_coeffs = [1.0] * len(bias_keys)

    if len(bias_keys) != len(bias_coeffs):
        raise ValueError("bias_keys and bias_coeffs must have same length")

    return list(zip(bias_keys, bias_coeffs))


__all__ = ['point3_to_array', 'normalize_bias_terms']
