import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from typing import List, Dict, Tuple

from pyins.rtk.ambiguity_resolution import (
    AmbiguityResolver,
    WidelaneAmbiguityResolver,
    RTKAmbiguityManager
)


class TestAmbiguityResolver(unittest.TestCase):
    
    def setUp(self):
        self.resolver = AmbiguityResolver(ratio_threshold=3.0, success_rate_threshold=0.999)
        
    def test_init(self):
        resolver = AmbiguityResolver(ratio_threshold=2.5, success_rate_threshold=0.95)
        self.assertEqual(resolver.ratio_threshold, 2.5)
        self.assertEqual(resolver.success_rate_threshold, 0.95)
    
    def test_resolve_ambiguities_empty(self):
        float_amb = np.array([])
        cov = np.array([[]])
        
        fixed, ratio, success = self.resolver.resolve_ambiguities(float_amb, cov)
        
        self.assertIsNone(fixed)
        self.assertEqual(ratio, 0.0)
        self.assertFalse(success)
    
    @patch('pyins.rtk.ambiguity_resolution.lambda_reduction')
    @patch('pyins.rtk.ambiguity_resolution.lambda_search')
    def test_resolve_ambiguities_success(self, mock_search, mock_reduction):
        # Setup mock returns
        n = 3
        Z = np.eye(n)
        L = np.tril(np.ones((n, n)))
        D = np.diag([1, 2, 3])
        mock_reduction.return_value = (Z, L, D)
        
        candidates = [np.array([1, 2, 3]), np.array([1, 2, 4])]
        residuals = [0.1, 0.4]  # Ratio = 4.0
        mock_search.return_value = (candidates, residuals)
        
        # Test data
        float_amb = np.array([1.1, 2.2, 3.3])
        cov = np.eye(3) * 0.01
        
        fixed, ratio, success = self.resolver.resolve_ambiguities(float_amb, cov)
        
        self.assertIsNotNone(fixed)
        np.testing.assert_array_equal(fixed, [1, 2, 3])
        self.assertEqual(ratio, 4.0)
        self.assertTrue(success)
    
    @patch('pyins.rtk.ambiguity_resolution.lambda_reduction')
    @patch('pyins.rtk.ambiguity_resolution.lambda_search')
    def test_resolve_ambiguities_ratio_failure(self, mock_search, mock_reduction):
        # Setup mock returns with low ratio
        n = 3
        Z = np.eye(n)
        L = np.tril(np.ones((n, n)))
        D = np.diag([1, 2, 3])
        mock_reduction.return_value = (Z, L, D)
        
        candidates = [np.array([1, 2, 3]), np.array([1, 2, 4])]
        residuals = [0.1, 0.2]  # Ratio = 2.0 (below threshold)
        mock_search.return_value = (candidates, residuals)
        
        float_amb = np.array([1.1, 2.2, 3.3])
        cov = np.eye(3) * 0.01
        
        fixed, ratio, success = self.resolver.resolve_ambiguities(float_amb, cov)
        
        self.assertIsNone(fixed)
        self.assertEqual(ratio, 2.0)
        self.assertFalse(success)
    
    @patch('pyins.rtk.ambiguity_resolution.lambda_reduction')
    @patch('pyins.rtk.ambiguity_resolution.lambda_search')
    def test_resolve_ambiguities_insufficient_candidates(self, mock_search, mock_reduction):
        n = 3
        Z = np.eye(n)
        L = np.tril(np.ones((n, n)))
        D = np.diag([1, 2, 3])
        mock_reduction.return_value = (Z, L, D)
        
        # Only one candidate
        candidates = [np.array([1, 2, 3])]
        residuals = [0.1]
        mock_search.return_value = (candidates, residuals)
        
        float_amb = np.array([1.1, 2.2, 3.3])
        cov = np.eye(3) * 0.01
        
        fixed, ratio, success = self.resolver.resolve_ambiguities(float_amb, cov)
        
        self.assertIsNone(fixed)
        self.assertEqual(ratio, 0.0)
        self.assertFalse(success)
    
    @patch('pyins.rtk.ambiguity_resolution.lambda_reduction')
    def test_resolve_ambiguities_exception_handling(self, mock_reduction):
        # Make lambda_reduction raise an exception
        mock_reduction.side_effect = Exception("Test exception")
        
        float_amb = np.array([1.1, 2.2, 3.3])
        cov = np.eye(3) * 0.01
        
        fixed, ratio, success = self.resolver.resolve_ambiguities(float_amb, cov)
        
        self.assertIsNone(fixed)
        self.assertEqual(ratio, 0.0)
        self.assertFalse(success)
    
    def test_estimate_success_rate(self):
        # Test various ratio cases
        rate = self.resolver._estimate_success_rate(0.1, 0.12, 3)  # ratio = 1.2
        self.assertAlmostEqual(rate, 0.1)
        
        rate = self.resolver._estimate_success_rate(0.1, 0.17, 3)  # ratio = 1.7
        self.assertAlmostEqual(rate, 0.5)
        
        rate = self.resolver._estimate_success_rate(0.1, 0.25, 3)  # ratio = 2.5
        self.assertAlmostEqual(rate, 0.9)
        
        rate = self.resolver._estimate_success_rate(0.1, 0.5, 3)  # ratio = 5.0
        self.assertGreater(rate, 0.9)
        self.assertLessEqual(rate, 0.999)
    
    def test_estimate_success_rate_edge_cases(self):
        # Zero or negative residuals
        rate = self.resolver._estimate_success_rate(0.0, 0.1, 3)
        self.assertEqual(rate, 0.0)
        
        rate = self.resolver._estimate_success_rate(-0.1, 0.1, 3)
        self.assertEqual(rate, 0.0)
        
        rate = self.resolver._estimate_success_rate(0.2, 0.1, 3)  # second < first
        self.assertEqual(rate, 0.0)
    
    @patch('pyins.rtk.ambiguity_resolution.AmbiguityResolver.resolve_ambiguities')
    def test_partial_ambiguity_resolution_success(self, mock_resolve):
        # Mock successful resolution for subset
        mock_resolve.return_value = (np.array([1, 2]), 4.0, True)
        
        float_amb = np.array([1.1, 2.2, 3.3, 4.4])
        cov = np.diag([0.01, 0.02, 0.03, 0.04])  # Varying precisions
        
        fixed_subset, indices, ratio = self.resolver.partial_ambiguity_resolution(
            float_amb, cov, min_subset_size=2
        )
        
        self.assertIsNotNone(fixed_subset)
        self.assertEqual(len(fixed_subset), 2)
        self.assertEqual(ratio, 4.0)
        # Should select most precise (smallest covariance)
        np.testing.assert_array_equal(indices, [0, 1])
    
    def test_partial_ambiguity_resolution_too_small(self):
        float_amb = np.array([1.1])
        cov = np.array([[0.01]])
        
        fixed_subset, indices, ratio = self.resolver.partial_ambiguity_resolution(
            float_amb, cov, min_subset_size=2
        )
        
        self.assertIsNone(fixed_subset)
        self.assertEqual(len(indices), 0)
        self.assertEqual(ratio, 0.0)
    
    @patch('pyins.rtk.ambiguity_resolution.AmbiguityResolver.resolve_ambiguities')
    def test_partial_ambiguity_resolution_all_fail(self, mock_resolve):
        # Mock all resolutions failing
        mock_resolve.return_value = (None, 0.0, False)
        
        float_amb = np.array([1.1, 2.2, 3.3])
        cov = np.eye(3) * 0.01
        
        fixed_subset, indices, ratio = self.resolver.partial_ambiguity_resolution(
            float_amb, cov, min_subset_size=2
        )
        
        self.assertIsNone(fixed_subset)
        self.assertEqual(len(indices), 0)
        self.assertEqual(ratio, 0.0)


class TestWidelaneAmbiguityResolver(unittest.TestCase):
    
    def setUp(self):
        self.wl_resolver = WidelaneAmbiguityResolver()
    
    def test_init(self):
        self.assertAlmostEqual(self.wl_resolver.l1_freq, 1575.42e6)
        self.assertAlmostEqual(self.wl_resolver.l2_freq, 1227.60e6)
        self.assertAlmostEqual(self.wl_resolver.wl_wavelength, 0.862, places=2)
    
    def test_resolve_widelane(self):
        # Create test data with known widelane ambiguities
        n_sats = 5
        wl_true = np.array([10, -5, 3, 0, -2])
        
        # Generate observations
        l1_obs = wl_true * 10.0 + np.random.randn(n_sats) * 0.1
        l2_obs = np.random.randn(n_sats) * 0.1
        
        # Generate pseudoranges (simplified)
        l1_pr = np.random.randn(n_sats) * 10 + 20000000
        l2_pr = np.random.randn(n_sats) * 10 + 20000000
        
        # For testing, create observations that will round to expected values
        l1_freq = self.wl_resolver.l1_freq
        l2_freq = self.wl_resolver.l2_freq
        c = 299792458.0
        
        # Reverse engineer observations to get desired widelane
        mw_term = (l1_freq * l1_pr - l2_freq * l2_pr) / (c * (l1_freq - l2_freq))
        l1_obs = wl_true + mw_term + np.random.randn(n_sats) * 0.01
        l2_obs = mw_term + np.random.randn(n_sats) * 0.01
        
        wl_resolved = self.wl_resolver.resolve_widelane(l1_obs, l2_obs, l1_pr, l2_pr)
        
        # Check that resolved values are close to true values
        np.testing.assert_array_almost_equal(wl_resolved, wl_true, decimal=0)
    
    def test_constrain_l1_ambiguities(self):
        l1_float = np.array([10.3, 5.7, -2.2, 0.1])
        l2_float = np.array([5.1, 2.9, -4.8, -1.2])
        wl_fixed = np.array([5, 3, 2, 1])
        
        l2_constrained = self.wl_resolver.constrain_l1_ambiguities(
            l1_float, l2_float, wl_fixed
        )
        
        # L2 = L1 - widelane
        expected = l1_float - wl_fixed
        np.testing.assert_array_almost_equal(l2_constrained, expected)


class TestRTKAmbiguityManager(unittest.TestCase):
    
    def setUp(self):
        self.manager = RTKAmbiguityManager()
    
    def test_init(self):
        self.assertIsInstance(self.manager.resolver, AmbiguityResolver)
        self.assertIsInstance(self.manager.wl_resolver, WidelaneAmbiguityResolver)
        self.assertEqual(len(self.manager.fixed_ambiguities), 0)
        self.assertEqual(len(self.manager.ambiguity_history), 0)
        self.assertEqual(len(self.manager.fix_status), 0)
    
    @patch.object(AmbiguityResolver, 'resolve_ambiguities')
    def test_update_ambiguities_full_success(self, mock_resolve):
        # Mock successful full resolution
        fixed = np.array([10, -5, 3])
        mock_resolve.return_value = (fixed, 4.0, True)
        
        satellites = [1, 2, 3]
        float_estimates = np.array([10.1, -4.9, 3.2])
        covariance = np.eye(3) * 0.01
        
        ambiguities, fix_status = self.manager.update_ambiguities(
            satellites, float_estimates, covariance
        )
        
        self.assertEqual(ambiguities[1], 10)
        self.assertEqual(ambiguities[2], -5)
        self.assertEqual(ambiguities[3], 3)
        self.assertTrue(fix_status[1])
        self.assertTrue(fix_status[2])
        self.assertTrue(fix_status[3])
    
    @patch.object(AmbiguityResolver, 'resolve_ambiguities')
    @patch.object(AmbiguityResolver, 'partial_ambiguity_resolution')
    def test_update_ambiguities_partial_success(self, mock_partial, mock_resolve):
        # Mock full resolution failure
        mock_resolve.return_value = (None, 2.0, False)
        
        # Mock partial resolution success
        fixed_subset = np.array([10, -5])
        indices = np.array([0, 1])
        mock_partial.return_value = (fixed_subset, indices, 3.5)
        
        satellites = [1, 2, 3]
        float_estimates = np.array([10.1, -4.9, 3.2])
        covariance = np.eye(3) * 0.01
        
        ambiguities, fix_status = self.manager.update_ambiguities(
            satellites, float_estimates, covariance
        )
        
        self.assertEqual(ambiguities[1], 10)
        self.assertEqual(ambiguities[2], -5)
        self.assertTrue(fix_status[1])
        self.assertTrue(fix_status[2])
        self.assertFalse(fix_status[3])
    
    @patch.object(AmbiguityResolver, 'resolve_ambiguities')
    @patch.object(AmbiguityResolver, 'partial_ambiguity_resolution')
    def test_update_ambiguities_all_fail(self, mock_partial, mock_resolve):
        # Mock all resolutions failing
        mock_resolve.return_value = (None, 1.5, False)
        mock_partial.return_value = (None, np.array([]), 0.0)
        
        satellites = [1, 2, 3]
        float_estimates = np.array([10.1, -4.9, 3.2])
        covariance = np.eye(3) * 0.01
        
        ambiguities, fix_status = self.manager.update_ambiguities(
            satellites, float_estimates, covariance
        )
        
        self.assertFalse(fix_status[1])
        self.assertFalse(fix_status[2])
        self.assertFalse(fix_status[3])
    
    def test_validate_ambiguities(self):
        satellites = [1, 2, 3, 4]
        residuals = np.array([0.05, 0.08, 0.15, 0.03])
        threshold = 0.1
        
        validation_status = self.manager.validate_ambiguities(
            satellites, residuals, threshold
        )
        
        self.assertTrue(validation_status[1])   # 0.05 < 0.1
        self.assertTrue(validation_status[2])   # 0.08 < 0.1
        self.assertFalse(validation_status[3])  # 0.15 > 0.1
        self.assertTrue(validation_status[4])   # 0.03 < 0.1
    
    def test_validate_ambiguities_missing_residuals(self):
        satellites = [1, 2, 3, 4]
        residuals = np.array([0.05, 0.08])  # Only 2 residuals for 4 satellites
        threshold = 0.1
        
        validation_status = self.manager.validate_ambiguities(
            satellites, residuals, threshold
        )
        
        self.assertTrue(validation_status[1])
        self.assertTrue(validation_status[2])
        self.assertFalse(validation_status[3])  # No residual
        self.assertFalse(validation_status[4])  # No residual
    
    def test_validate_ambiguities_empty(self):
        satellites = []
        residuals = np.array([])
        
        validation_status = self.manager.validate_ambiguities(
            satellites, residuals
        )
        
        self.assertEqual(len(validation_status), 0)
    
    def test_ambiguity_persistence(self):
        # Test that ambiguities persist across updates
        self.manager.fixed_ambiguities[1] = 10
        self.manager.fixed_ambiguities[2] = -5
        self.manager.fix_status[1] = True
        self.manager.fix_status[2] = True
        
        # Get copies
        ambiguities, fix_status = self.manager.update_ambiguities(
            [], np.array([]), np.array([[]])
        )
        
        # Check that copies contain previous values
        self.assertEqual(ambiguities[1], 10)
        self.assertEqual(ambiguities[2], -5)
        self.assertTrue(fix_status[1])
        self.assertTrue(fix_status[2])


class TestIntegrationScenarios(unittest.TestCase):
    
    def test_complete_rtk_ambiguity_workflow(self):
        # Initialize manager
        manager = RTKAmbiguityManager()
        
        # First epoch - float solution
        satellites = [1, 2, 3, 4]
        float_estimates = np.array([10.4, -5.3, 3.2, 0.1])
        
        # Create realistic covariance matrix
        variances = np.array([0.01, 0.02, 0.015, 0.025])
        correlations = 0.3
        cov = np.diag(variances)
        for i in range(len(variances)):
            for j in range(i+1, len(variances)):
                cov[i, j] = cov[j, i] = correlations * np.sqrt(variances[i] * variances[j])
        
        # Mock successful resolution for testing
        with patch.object(manager.resolver, 'resolve_ambiguities') as mock_resolve:
            mock_resolve.return_value = (np.array([10, -5, 3, 0]), 4.5, True)
            
            ambiguities, fix_status = manager.update_ambiguities(
                satellites, float_estimates, cov
            )
            
            # Verify all fixed
            for sat in satellites:
                self.assertTrue(fix_status[sat])
            
            # Validate with residuals
            residuals = np.array([0.02, 0.03, 0.01, 0.05])
            validation = manager.validate_ambiguities(satellites, residuals, 0.1)
            
            for sat in satellites:
                self.assertTrue(validation[sat])
    
    def test_widelane_constrained_resolution(self):
        wl_resolver = WidelaneAmbiguityResolver()
        
        # Test data
        n_sats = 6
        l1_float = np.random.randn(n_sats) * 10
        l2_float = np.random.randn(n_sats) * 10
        
        # Generate widelane observations
        wl_fixed = np.round(l1_float - l2_float)
        
        # Constrain L2 using widelane
        l2_constrained = wl_resolver.constrain_l1_ambiguities(
            l1_float, l2_float, wl_fixed
        )
        
        # Verify constraint
        np.testing.assert_array_almost_equal(
            l2_constrained, l1_float - wl_fixed
        )


if __name__ == '__main__':
    unittest.main()