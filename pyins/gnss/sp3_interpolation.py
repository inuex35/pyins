#!/usr/bin/env python3
"""
SP3 interpolation methods compatible with RTKLIB and gnss_lib_py

This module provides polynomial interpolation methods for SP3 precise ephemeris,
including Neville's algorithm (used by RTKLIB) and standard polynomial fitting
(used by GNSSpy).
"""

import warnings

import numpy as np


def neville_interpolation(x: np.ndarray, y: np.ndarray, x0: float) -> float:
    """
    Neville's algorithm for polynomial interpolation (RTKLIB compatible)

    This implements the same algorithm used in RTKLIB's interppol function.

    Parameters
    ----------
    x : np.ndarray
        Array of x values (time points)
    y : np.ndarray
        Array of y values (position/clock values)
    x0 : float
        Point at which to interpolate

    Returns
    -------
    float
        Interpolated value at x0
    """
    n = len(x)
    if n == 0:
        return np.nan
    if n == 1:
        return y[0]

    # Create a copy of y to avoid modifying the original
    p = y.copy()

    # Neville's recursive algorithm
    for j in range(1, n):
        for i in range(n - j):
            p[i] = ((x0 - x[i]) * p[i + 1] - (x0 - x[i + j]) * p[i]) / (x[i + j] - x[i])

    return p[0]


def lagrange_interpolation(x: np.ndarray, y: np.ndarray, degree: int,
                          x0: float, centered: bool = True) -> float:
    """
    Lagrange polynomial interpolation with specified degree

    Parameters
    ----------
    x : np.ndarray
        Array of x values (time points)
    y : np.ndarray
        Array of y values (position/clock values)
    degree : int
        Degree of polynomial (number of points - 1)
    x0 : float
        Point at which to interpolate
    centered : bool
        If True, center the interpolation window around x0

    Returns
    -------
    float
        Interpolated value at x0
    """
    n = len(x)

    # Select points for interpolation
    if centered and n > degree + 1:
        # Find the closest point to x0
        idx = np.searchsorted(x, x0)

        # Center the interpolation window
        n_points = degree + 1
        start_idx = max(0, idx - n_points // 2)
        end_idx = min(n, start_idx + n_points)
        if end_idx - start_idx < n_points:
            start_idx = max(0, end_idx - n_points)

        x_subset = x[start_idx:end_idx]
        y_subset = y[start_idx:end_idx]
    else:
        # Use all points or up to degree+1 points
        n_use = min(n, degree + 1)
        x_subset = x[:n_use]
        y_subset = y[:n_use]

    # Lagrange interpolation
    result = 0.0
    n_subset = len(x_subset)

    for i in range(n_subset):
        term = y_subset[i]
        for j in range(n_subset):
            if i != j:
                term *= (x0 - x_subset[j]) / (x_subset[i] - x_subset[j])
        result += term

    return result


def polyfit_interpolation(x: np.ndarray, y: np.ndarray, degree: int,
                         x0: float, centered: bool = True) -> float:
    """
    Polynomial interpolation using numpy polyfit (GNSSpy compatible)

    Parameters
    ----------
    x : np.ndarray
        Array of x values (time points)
    y : np.ndarray
        Array of y values (position/clock values)
    degree : int
        Degree of polynomial
    x0 : float
        Point at which to interpolate
    centered : bool
        If True, center the interpolation window around x0

    Returns
    -------
    float
        Interpolated value at x0
    """
    n = len(x)

    # Select points for interpolation
    if centered and n > degree + 1:
        # Find the closest point to x0
        idx = np.searchsorted(x, x0)

        # Center the interpolation window
        n_points = degree + 1
        start_idx = max(0, idx - n_points // 2)
        end_idx = min(n, start_idx + n_points)
        if end_idx - start_idx < n_points:
            start_idx = max(0, end_idx - n_points)

        x_subset = x[start_idx:end_idx]
        y_subset = y[start_idx:end_idx]
    else:
        # Use all points or up to degree+1 points
        n_use = min(n, degree + 1)
        x_subset = x[:n_use]
        y_subset = y[:n_use]

    # Adjust degree if not enough points
    actual_degree = min(degree, len(x_subset) - 1)

    # Fit polynomial with warning suppression
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', np.polynomial.polyutils.RankWarning)
        coeffs = np.polyfit(x_subset, y_subset, actual_degree)

    # Evaluate at x0
    return np.polyval(coeffs, x0)


def interpolate_sp3_position(times: np.ndarray, positions: np.ndarray,
                            target_time: float, method: str = 'neville',
                            degree: int = 10) -> tuple[np.ndarray, bool]:
    """
    Interpolate SP3 satellite position at target time

    Parameters
    ----------
    times : np.ndarray
        Array of epoch times (in seconds)
    positions : np.ndarray
        Array of positions (n_epochs x 3) in meters
    target_time : float
        Target time for interpolation (in seconds)
    method : str
        Interpolation method: 'neville', 'lagrange', 'polyfit'
    degree : int
        Polynomial degree (default 10 for RTKLIB compatibility)

    Returns
    -------
    position : np.ndarray
        Interpolated position [x, y, z] in meters
    success : bool
        True if interpolation succeeded
    """
    if len(times) < 2:
        return np.zeros(3), False

    # Check if target time is within range
    if target_time < times[0] or target_time > times[-1]:
        return np.zeros(3), False

    # Select interpolation points
    n_points = min(degree + 1, len(times))
    idx = np.searchsorted(times, target_time)

    # Center the interpolation window
    start_idx = max(0, idx - n_points // 2)
    end_idx = min(len(times), start_idx + n_points)
    if end_idx - start_idx < n_points:
        start_idx = max(0, end_idx - n_points)

    # Extract subset for interpolation
    time_subset = times[start_idx:end_idx]
    pos_subset = positions[start_idx:end_idx]

    # Interpolate each coordinate
    interpolated_pos = np.zeros(3)

    for i in range(3):
        if method == 'neville':
            # RTKLIB-compatible Neville's algorithm
            interpolated_pos[i] = neville_interpolation(
                time_subset, pos_subset[:, i], target_time)
        elif method == 'lagrange':
            # Lagrange interpolation
            interpolated_pos[i] = lagrange_interpolation(
                time_subset, pos_subset[:, i], degree, target_time, centered=False)
        elif method == 'polyfit':
            # GNSSpy-compatible polynomial fitting
            interpolated_pos[i] = polyfit_interpolation(
                time_subset, pos_subset[:, i], degree, target_time, centered=False)
        else:
            raise ValueError(f"Unknown interpolation method: {method}")

    return interpolated_pos, True


def interpolate_sp3_clock(times: np.ndarray, clocks: np.ndarray,
                         target_time: float, method: str = 'linear') -> tuple[float, bool]:
    """
    Interpolate SP3 satellite clock at target time

    For clock interpolation, linear interpolation is typically used
    as it provides sufficient accuracy for clock corrections.

    Parameters
    ----------
    times : np.ndarray
        Array of epoch times (in seconds)
    clocks : np.ndarray
        Array of clock corrections (in seconds)
    target_time : float
        Target time for interpolation (in seconds)
    method : str
        Interpolation method: 'linear' or 'polynomial'

    Returns
    -------
    clock : float
        Interpolated clock correction in seconds
    success : bool
        True if interpolation succeeded
    """
    if len(times) < 2:
        return 0.0, False

    # Check if target time is within range
    if target_time < times[0] or target_time > times[-1]:
        return 0.0, False

    if method == 'linear':
        # Linear interpolation (RTKLIB default for clocks)
        clock = np.interp(target_time, times, clocks)
    elif method == 'polynomial':
        # Polynomial interpolation (degree 2 for clocks)
        clock = polyfit_interpolation(times, clocks, 2, target_time)
    else:
        raise ValueError(f"Unknown clock interpolation method: {method}")

    return clock, True


def get_interpolation_accuracy(method: str, degree: int, interval_minutes: int) -> float:
    """
    Get expected interpolation accuracy based on method and parameters

    Parameters
    ----------
    method : str
        Interpolation method
    degree : int
        Polynomial degree
    interval_minutes : int
        SP3 data interval in minutes (typically 15 or 30)

    Returns
    -------
    float
        Expected accuracy in meters
    """
    # Based on empirical studies from literature
    if interval_minutes == 15:
        if degree >= 10:
            return 0.01  # 1 cm for degree >= 10
        elif degree >= 7:
            return 0.05  # 5 cm for degree 7-9
        else:
            return 0.10  # 10 cm for lower degrees
    elif interval_minutes == 30:
        if degree >= 16:
            return 0.025  # 2.5 cm for degree >= 16
        elif degree >= 10:
            return 0.05  # 5 cm for degree 10-15
        else:
            return 0.15  # 15 cm for lower degrees
    else:
        # Conservative estimate for other intervals
        return 0.10  # 10 cm
