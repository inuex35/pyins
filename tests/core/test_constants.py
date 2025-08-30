#!/usr/bin/env python3
"""Test suite for physical constants"""

import unittest
import numpy as np
from pyins.core.constants import (
    CLIGHT, FREQ_L1, FREQ_L2, FREQ_L5,
    FREQ_G1, FREQ_G2, DFREQ_G1, DFREQ_G2,
    RE_WGS84, FE_WGS84, OMGE, MU_GPS, MU_GAL
)


class TestPhysicalConstants(unittest.TestCase):
    """Test physical constants values"""
    
    def test_speed_of_light(self):
        """Test speed of light constant"""
        self.assertEqual(CLIGHT, 299792458.0)
        
    def test_gps_frequencies(self):
        """Test GPS frequency constants"""
        # L1 frequency ~1575.42 MHz
        self.assertAlmostEqual(FREQ_L1, 1575.42e6, delta=1e6)
        
        # L2 frequency ~1227.60 MHz
        self.assertAlmostEqual(FREQ_L2, 1227.60e6, delta=1e6)
        
        # L5 frequency ~1176.45 MHz
        self.assertAlmostEqual(FREQ_L5, 1176.45e6, delta=1e6)
        
        # L1 > L2 > L5
        self.assertGreater(FREQ_L1, FREQ_L2)
        self.assertGreater(FREQ_L2, FREQ_L5)
        
    def test_glonass_frequencies(self):
        """Test GLONASS frequency constants"""
        # G1 base frequency ~1602 MHz
        self.assertAlmostEqual(FREQ_G1, 1602.0e6, delta=1e6)
        
        # G2 base frequency ~1246 MHz
        self.assertAlmostEqual(FREQ_G2, 1246.0e6, delta=1e6)
        
        # Channel spacing
        self.assertAlmostEqual(DFREQ_G1, 0.5625e6, delta=1e4)
        self.assertAlmostEqual(DFREQ_G2, 0.4375e6, delta=1e4)
        
    def test_earth_parameters(self):
        """Test Earth parameters"""
        # WGS84 semi-major axis ~6378137 m
        self.assertAlmostEqual(RE_WGS84, 6378137.0, delta=1.0)
        
        # WGS84 flattening ~1/298.257
        self.assertAlmostEqual(FE_WGS84, 1.0/298.257223563, delta=1e-9)
        
        # Earth rotation rate
        self.assertAlmostEqual(OMGE, 7.2921151467e-5, delta=1e-10)
        
    def test_gravitational_parameters(self):
        """Test gravitational parameters"""
        # GPS/WGS84 gravitational parameter
        self.assertAlmostEqual(MU_GPS, 3.9860050e14, delta=1e9)
        
        # Galileo gravitational parameter
        self.assertAlmostEqual(MU_GAL, 3.986004418e14, delta=1e9)
        
        # Should be similar but not identical
        self.assertAlmostEqual(MU_GPS, MU_GAL, delta=1e10)


class TestDerivedConstants(unittest.TestCase):
    """Test derived values from constants"""
    
    def test_wavelengths(self):
        """Test wavelength calculations"""
        # GPS L1 wavelength
        lambda_L1 = CLIGHT / FREQ_L1
        self.assertAlmostEqual(lambda_L1, 0.190, delta=0.01)
        
        # GPS L2 wavelength
        lambda_L2 = CLIGHT / FREQ_L2
        self.assertAlmostEqual(lambda_L2, 0.244, delta=0.01)
        
        # GPS L5 wavelength
        lambda_L5 = CLIGHT / FREQ_L5
        self.assertAlmostEqual(lambda_L5, 0.255, delta=0.01)
        
    def test_widelane_narrowlane(self):
        """Test wide-lane and narrow-lane combinations"""
        # Wide-lane frequency
        f_wl = FREQ_L1 - FREQ_L2
        lambda_wl = CLIGHT / f_wl
        
        # Wide-lane wavelength ~86 cm
        self.assertAlmostEqual(lambda_wl, 0.862, delta=0.01)
        
        # Narrow-lane frequency
        f_nl = FREQ_L1 + FREQ_L2
        lambda_nl = CLIGHT / f_nl
        
        # Narrow-lane wavelength ~10.7 cm
        self.assertAlmostEqual(lambda_nl, 0.107, delta=0.01)
        
    def test_glonass_channels(self):
        """Test GLONASS frequency channels"""
        # Channel numbers from -7 to +6
        for k in range(-7, 7):
            # G1 frequency for channel k
            f_g1_k = FREQ_G1 + k * DFREQ_G1
            
            # Should be in L-band
            self.assertGreater(f_g1_k, 1.5e9)
            self.assertLess(f_g1_k, 1.7e9)
            
            # G2 frequency for channel k
            f_g2_k = FREQ_G2 + k * DFREQ_G2
            
            # Should be in L-band
            self.assertGreater(f_g2_k, 1.2e9)
            self.assertLess(f_g2_k, 1.3e9)


class TestConstantRelationships(unittest.TestCase):
    """Test relationships between constants"""
    
    def test_frequency_ratios(self):
        """Test frequency ratios used in combinations"""
        # L1/L2 ratio
        ratio_12 = FREQ_L1 / FREQ_L2
        self.assertAlmostEqual(ratio_12, 154/120, delta=0.001)
        
        # L1/L5 ratio
        ratio_15 = FREQ_L1 / FREQ_L5
        self.assertAlmostEqual(ratio_15, 154/115, delta=0.001)
        
    def test_earth_ellipsoid(self):
        """Test Earth ellipsoid parameters"""
        # Semi-major axis
        a = RE_WGS84
        
        # Flattening
        f = FE_WGS84
        
        # Semi-minor axis
        b = a * (1 - f)
        
        # Eccentricity squared
        e2 = 2 * f - f * f
        
        # Check eccentricity
        self.assertAlmostEqual(e2, 0.00669437999, delta=1e-9)
        
        # Check semi-minor axis
        self.assertAlmostEqual(b, 6356752.314, delta=0.01)
        
    def test_orbital_parameters(self):
        """Test orbital parameter relationships"""
        # GPS orbital radius ~26,560 km
        r_gps = 26560000  # meters
        
        # Mean motion
        n = np.sqrt(MU_GPS / r_gps**3)
        
        # Orbital period
        T = 2 * np.pi / n
        
        # Should be ~12 hours (11.97 hours)
        T_hours = T / 3600
        self.assertAlmostEqual(T_hours, 11.97, delta=0.1)
        
        # Angular velocity relative to Earth
        omega_rel = n - OMGE
        
        # GPS satellites complete 2 orbits per sidereal day
        self.assertAlmostEqual(n / OMGE, 2.0, delta=0.01)


if __name__ == '__main__':
    unittest.main()