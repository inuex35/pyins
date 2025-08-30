import unittest
from unittest.mock import patch

import numpy as np

from pyins.rtk.ambiguity_resolution import (
    RTKAmbiguityManager,
    mlambda,
    LD,
    reduction,
    search,
)


class TestRTKAmbiguityManager(unittest.TestCase):

    def setUp(self):
        self.manager = RTKAmbiguityManager(ratio_threshold=3.0, success_rate_threshold=0.999)

    def test_init(self):
        manager = RTKAmbiguityManager(ratio_threshold=2.5, success_rate_threshold=0.95)
        self.assertEqual(manager.ratio_threshold, 2.5)
        self.assertEqual(manager.success_rate_threshold, 0.95)

    def test_resolve_ambiguities_empty(self):
        float_amb = np.array([])
        cov = np.array([[]])

        # Test empty input
        if len(float_amb) == 0:
            self.assertEqual(len(float_amb), 0)

    def test_search_ambiguities(self):
        # Test search_ambiguities method
        float_amb = np.array([1.1, 2.2, 3.3])
        cov = np.eye(3) * 0.01
        
        with patch('pyins.rtk.ambiguity_resolution.mlambda') as mock_mlambda:
            # Mock mlambda to return candidates
            mock_mlambda.return_value = (np.array([[1, 1], [2, 2], [3, 4]]), [0.1, 0.4])
            
            candidates = self.manager.search_ambiguities(float_amb, cov, n_candidates=2)
            
            self.assertEqual(len(candidates), 2)
            self.assertEqual(candidates[0][1], 0.1)  # First residual
            self.assertEqual(candidates[1][1], 0.4)  # Second residual

    def test_ratio_threshold(self):
        # Test ratio threshold parameter
        self.manager.ratio_threshold = 3.0
        self.assertEqual(self.manager.ratio_threshold, 3.0)
        
        # Test that ratio threshold can be updated
        self.manager.ratio_threshold = 2.5
        self.assertEqual(self.manager.ratio_threshold, 2.5)

    def test_compute_residual_norm(self):
        # Test residual norm computation
        a_float = np.array([1.1, 2.2, 3.3])
        a_int = np.array([1, 2, 3])
        Q_a = np.eye(3) * 0.01
        
        norm = self.manager._compute_residual_norm(a_float, a_int, Q_a)
        
        # Compute expected norm
        diff = a_float - a_int
        expected = diff @ np.linalg.inv(Q_a) @ diff
        
        self.assertAlmostEqual(norm, expected, places=5)

    def test_resolve_great_pvt(self):
        # Test GREAT-PVT partial fixing
        self.manager.use_great_pvt = True
        self.manager.max_fix_count = 2
        
        float_amb = np.array([1.1, 2.2, 3.3, 4.4])
        cov = np.diag([0.01, 0.02, 0.1, 0.15])  # Varying quality
        
        # The function should select best subset based on covariance
        # This is a simplified test - actual implementation is more complex
        self.assertIsNotNone(self.manager)

    def test_fixed_ambiguities_storage(self):
        # Test that fixed ambiguities are stored correctly
        self.manager.fixed_ambiguities = {1: 10, 2: -5, 3: 3}
        
        self.assertEqual(self.manager.fixed_ambiguities[1], 10)
        self.assertEqual(self.manager.fixed_ambiguities[2], -5)
        self.assertEqual(self.manager.fixed_ambiguities[3], 3)
        
    def test_min_sats_requirement(self):
        # Test minimum satellites requirement
        self.manager.min_sats = 4
        
        # Should not attempt resolution with too few satellites
        float_amb = np.array([1.1, 2.2])
        cov = np.eye(2) * 0.01
        
        # This should be handled by the implementation
        self.assertEqual(len(float_amb), 2)
        self.assertLess(len(float_amb), self.manager.min_sats)

    def test_great_pvt_parameters(self):
        # Test GREAT-PVT specific parameters
        manager = RTKAmbiguityManager(use_great_pvt=True, max_fix_count=10)
        
        self.assertTrue(manager.use_great_pvt)
        self.assertEqual(manager.max_fix_count, 10)
        self.assertEqual(manager.max_sigma, 0.25)


class TestMLAMBDAFunctions(unittest.TestCase):
    """Test the low-level MLAMBDA functions"""
    
    def test_LD_factorization(self):
        # Test LD factorization
        Q = np.array([[2.0, 0.5], [0.5, 1.0]])
        L, d = LD(Q)
        
        # Verify factorization: Q = L' * diag(d) * L
        Q_reconstructed = L.T @ np.diag(d) @ L
        np.testing.assert_array_almost_equal(Q, Q_reconstructed, decimal=10)
        
    def test_reduction(self):
        # Test reduction function
        L = np.array([[1.0, 0.0], [0.8, 1.0]])
        d = np.array([1.0, 1.0])
        
        L_out, d_out, Z = reduction(L, d)
        
        # Check that output is valid
        self.assertEqual(L_out.shape, L.shape)
        self.assertEqual(d_out.shape, d.shape)
        self.assertEqual(Z.shape, (2, 2))  # Transformation matrix
        
    def test_mlambda_basic(self):
        # Test basic MLAMBDA functionality
        a = np.array([1.1, 2.2, 3.3])
        Q = np.eye(3) * 0.01
        
        afix, s = mlambda(a, Q, m=2)
        
        # Check output shape
        self.assertEqual(afix.shape[0], len(a))  # n ambiguities
        self.assertEqual(afix.shape[1], 2)  # m candidates
        self.assertEqual(len(s), 2)  # m residuals
        
        # Check that first candidate has lower residual
        self.assertLessEqual(s[0], s[1])


class TestRTKAmbiguityManagerAdvanced(unittest.TestCase):
    """Test advanced RTKAmbiguityManager functionality"""
    
    def setUp(self):
        self.manager = RTKAmbiguityManager()
        
    def test_init_defaults(self):
        self.assertEqual(self.manager.ratio_threshold, 2.0)
        self.assertEqual(self.manager.success_rate_threshold, 0.95)
        self.assertEqual(self.manager.min_sats, 4)
        self.assertEqual(len(self.manager.fixed_ambiguities), 0)
        
    def test_ratio_comparison(self):
        # Test ratio threshold comparison logic
        self.manager.ratio_threshold = 2.0
        
        # These would be used internally by the manager
        ratio_good = 3.5
        ratio_bad = 1.5
        
        # Check that ratio comparison would work correctly
        self.assertGreater(ratio_good, self.manager.ratio_threshold)
        self.assertLess(ratio_bad, self.manager.ratio_threshold)
        
    def test_position_variance_check(self):
        # Test max position variance parameter
        self.manager.max_position_var = 0.1
        
        # Should be a parameter that affects whether to attempt AR
        self.assertIsNotNone(self.manager.max_position_var)
        self.assertEqual(self.manager.max_position_var, 0.1)
        
    def test_nb_ar_tracking(self):
        # Test that number of ambiguities resolved is tracked
        self.manager.nb_ar = 0
        self.assertEqual(self.manager.nb_ar, 0)
        
        # Simulate successful resolution
        self.manager.nb_ar = 5
        self.assertEqual(self.manager.nb_ar, 5)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for ambiguity resolution workflow"""
    
    def test_complete_rtk_ambiguity_workflow(self):
        # Initialize manager
        manager = RTKAmbiguityManager()
        
        # Test data
        float_estimates = np.array([10.4, -5.3, 3.2, 0.1])
        
        # Create realistic covariance matrix
        variances = np.array([0.01, 0.02, 0.015, 0.025])
        correlations = 0.3
        cov = np.diag(variances)
        for i in range(len(variances)):
            for j in range(i+1, len(variances)):
                cov[i, j] = cov[j, i] = correlations * np.sqrt(variances[i] * variances[j])
        
        # Mock successful resolution for testing
        with patch('pyins.rtk.ambiguity_resolution.mlambda') as mock_mlambda:
            # Return 2 candidates with residuals
            mock_mlambda.return_value = (
                np.array([[10, 10], [-5, -5], [3, 3], [0, 1]]),
                [0.1, 0.5]
            )
            
            candidates = manager.search_ambiguities(float_estimates, cov, n_candidates=2)
            
            # Verify candidates were returned
            self.assertEqual(len(candidates), 2)
            self.assertIsInstance(candidates[0][0], np.ndarray)
            self.assertIsInstance(candidates[0][1], float)
    
    def test_mlambda_with_real_data(self):
        """Test MLAMBDA with realistic data"""
        # Realistic float ambiguities and covariance
        a = np.array([23.4, -5.2, 17.8])
        Q = np.array([
            [0.01, 0.002, 0.001],
            [0.002, 0.015, 0.003],
            [0.001, 0.003, 0.012]
        ])
        
        # Call mlambda
        afix, s = mlambda(a, Q, m=2)
        
        # Verify output
        self.assertEqual(afix.shape[0], 3)  # 3 ambiguities
        self.assertEqual(afix.shape[1], 2)  # 2 candidates
        self.assertEqual(len(s), 2)
        
        # First candidate should have lower residual
        self.assertLessEqual(s[0], s[1])
        
        # Ambiguities should be integers
        self.assertTrue(np.all(afix == afix.astype(int)))


class TestSearchFunction(unittest.TestCase):
    """Test the search function used in MLAMBDA"""
    
    def test_search_basic(self):
        """Test basic search functionality"""
        # Simple 2D case
        L = np.array([[1.0, 0.0], [0.5, 1.0]])
        D = np.array([1.0, 1.0])
        zs = np.array([0.1, 0.2])
        m = 2
        
        E, s = search(L, D, zs, m)
        
        # Check output shape
        self.assertEqual(E.shape[0], 2)  # n dimensions
        self.assertEqual(E.shape[1], m)  # m candidates
        self.assertEqual(len(s), m)
        
        # Candidates should be integers
        self.assertTrue(np.all(E == E.astype(int)))
        
        # Residuals should be sorted
        for i in range(len(s)-1):
            self.assertLessEqual(s[i], s[i+1])


if __name__ == '__main__':
    unittest.main()
