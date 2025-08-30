#!/usr/bin/env python3
"""Test suite for SPP (Single Point Positioning) functions"""

import unittest
import numpy as np


class TestSPPFunctions(unittest.TestCase):
    """Test SPP basic functions"""
    
    def test_imports(self):
        """Test that SPP functions can be imported"""
        try:
            from pyins.gnss.spp import tropmodel_simple
            from pyins.gnss.spp import satazel
            from pyins.gnss.spp import varerr
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import SPP functions: {e}")
    
    def test_tropmodel_simple(self):
        """Test simple troposphere model"""
        from pyins.gnss.spp import tropmodel_simple
        
        # Position at sea level (lat, lon, height in radians/meters)
        pos = np.array([np.deg2rad(30), 0, 100])  # 30 deg latitude, 100m height
        
        # High elevation (less delay)
        el_high = np.deg2rad(90.0)
        delay_high = tropmodel_simple(pos, el_high)
        self.assertGreater(delay_high, 0)
        self.assertLess(delay_high, 10)  # Should be ~2.3m at zenith
        
        # Low elevation (more delay)
        el_low = np.deg2rad(10.0)
        delay_low = tropmodel_simple(pos, el_low)
        self.assertGreater(delay_low, delay_high)
        self.assertLess(delay_low, 50)  # Reasonable upper bound
    
    def test_satazel(self):
        """Test satellite azimuth/elevation calculation"""
        from pyins.gnss.spp import satazel
        
        # Position (lat, lon, height)
        pos = np.array([0, 0, 0])
        
        # Satellite direction in ENU (normalized)
        e = np.array([1.0, 1.0, 1.0])
        e = e / np.linalg.norm(e)  # Normalize to unit vector
        
        az, el = satazel(pos, e)
        
        # Azimuth should be between 0 and 360
        self.assertGreaterEqual(az, 0)
        self.assertLessEqual(az, 360)
        
        # Elevation should be between -90 and 90
        self.assertGreaterEqual(el, -90)
        self.assertLessEqual(el, 90)
    
    def test_varerr(self):
        """Test variance error model"""
        from pyins.gnss.spp import varerr
        
        # Test GPS system
        sys = 0  # GPS system code
        
        # High elevation (low variance)
        el_high = np.deg2rad(90.0)
        var_high = varerr(sys, el_high)
        self.assertGreater(var_high, 0)
        
        # Low elevation (high variance)
        el_low = np.deg2rad(10.0)
        var_low = varerr(sys, el_low)
        self.assertGreater(var_low, var_high)


class TestGeodesic(unittest.TestCase):
    """Test geodesic calculations"""
    
    def test_geodist(self):
        """Test geometric distance calculation"""
        from pyins.gnss.spp import geodist
        
        # Satellite position
        sat_pos = np.array([20000000, 10000000, 5000000])
        
        # Receiver position
        rec_pos = np.array([6378137, 0, 0])
        
        # Calculate distance and unit vector
        dist, e = geodist(sat_pos, rec_pos)
        
        # Should be positive and reasonable
        self.assertGreater(dist, 1e7)  # More than 10,000 km
        self.assertLess(dist, 5e7)     # Less than 50,000 km
        
        # Unit vector should have norm 1
        self.assertAlmostEqual(np.linalg.norm(e), 1.0, places=10)
        
        # Should be close to Euclidean distance
        euclidean = np.linalg.norm(sat_pos - rec_pos)
        self.assertAlmostEqual(dist, euclidean, delta=1)  # Within 1m


if __name__ == '__main__':
    unittest.main()