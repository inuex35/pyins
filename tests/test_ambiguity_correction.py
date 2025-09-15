"""
Tests for ambiguity offset correction module.
"""

import numpy as np
import pytest
from pyins.rtk.ambiguity_correction import (
    calculate_ambiguity_offset,
    apply_ambiguity_offset,
    compute_ambiguity_from_measurements,
    correct_ambiguity_offsets
)


class TestAmbiguityCorrection:
    """Test cases for ambiguity correction functions."""

    def test_calculate_offset_no_existing(self):
        """Test offset calculation with no existing estimates."""
        new_estimates = {(10, 25, 0): 7715.7, (12, 25, 0): 6500.3}
        existing_estimates = {}

        offset = calculate_ambiguity_offset(new_estimates, existing_estimates)
        assert offset is None

    def test_calculate_offset_with_common_satellites(self):
        """Test offset calculation with common satellite pairs."""
        new_estimates = {
            (10, 25, 0): 7720.0,
            (12, 25, 0): 6505.0,
            (15, 25, 0): 8100.0
        }

        existing_estimates = {
            (10, 25, 0): 7715.0,  # diff = 5.0
            (12, 25, 0): 6500.0,  # diff = 5.0
            (20, 25, 0): 9000.0   # no match in new
        }

        offset = calculate_ambiguity_offset(new_estimates, existing_estimates)
        assert offset is not None
        assert abs(offset - 5.0) < 1e-6

    def test_calculate_offset_below_threshold(self):
        """Test that small offsets are ignored."""
        new_estimates = {(10, 25, 0): 7715.05}
        existing_estimates = {(10, 25, 0): 7715.0}

        offset = calculate_ambiguity_offset(new_estimates, existing_estimates, offset_threshold=1.0)
        assert offset is None  # 0.05 < threshold of 1.0

    def test_apply_offset_correction(self):
        """Test applying offset to ambiguities."""
        ambiguities = {
            (10, 25, 0): 7715.0,
            (12, 25, 0): 6500.0,
            (15, 25, 0): 8100.0
        }

        offset = 10.5
        corrected = apply_ambiguity_offset(ambiguities, offset)

        assert len(corrected) == 3
        assert abs(corrected[(10, 25, 0)] - 7725.5) < 1e-6
        assert abs(corrected[(12, 25, 0)] - 6510.5) < 1e-6
        assert abs(corrected[(15, 25, 0)] - 8110.5) < 1e-6

    def test_compute_ambiguity_from_measurements(self):
        """Test ambiguity computation from DD measurements."""
        # Example from actual data
        dd_carrier_cycles = 13413.4
        dd_pseudorange_m = 1084.245
        wavelength = 0.190  # L1

        ambiguity = compute_ambiguity_from_measurements(
            dd_carrier_cycles, dd_pseudorange_m, wavelength
        )

        expected = dd_carrier_cycles - (dd_pseudorange_m / wavelength)
        assert abs(ambiguity - expected) < 1e-6
        assert abs(ambiguity - 7706.8) < 0.1  # Close to expected value


def test_correct_ambiguity_offsets():
    """Test the convenience function."""
    dd_measurements = [
        {
            'sat': 10, 'ref_sat': 25, 'freq_idx': 0,
            'dd_carrier': 13420.0,  # Changed from existing
            'dd_obs': 1084.245,
            'wavelength': 0.190
        },
        {
            'sat': 12, 'ref_sat': 25, 'freq_idx': 0,
            'dd_carrier': 10515.0,  # Changed from existing
            'dd_obs': 1084.159,
            'wavelength': 0.244
        }
    ]

    existing_ambiguities = {
        (10, 25, 0): 7715.0,
        (12, 25, 0): 6068.0
    }

    wavelengths = {0: 0.190, 1: 0.244}

    corrected, offset = correct_ambiguity_offsets(
        dd_measurements, existing_ambiguities, wavelengths
    )

    # Should have applied an offset
    assert offset != 0.0
    assert len(corrected) == 2

    # Check that offset was applied to all ambiguities
    for key in existing_ambiguities:
        assert abs(corrected[key] - (existing_ambiguities[key] + offset)) < 1e-6


if __name__ == '__main__':
    pytest.main([__file__, '-v'])