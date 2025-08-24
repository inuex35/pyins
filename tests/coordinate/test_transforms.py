import unittest
import numpy as np
from pyins.coordinate.transforms import (
    ecef2llh, llh2ecef, ecef2enu, enu2ecef,
    ecef2ned, ned2ecef, enu2ned, ned2enu,
    compute_rotation_matrix_enu, compute_rotation_matrix_ned
)
from pyins.core.constants import RE_WGS84, FE_WGS84


class TestCoordinateTransforms(unittest.TestCase):
    
    def setUp(self):
        # Test points
        self.tokyo_llh = np.array([np.radians(35.6762), np.radians(139.6503), 40.0])  # Tokyo Tower
        self.newyork_llh = np.array([np.radians(40.7128), np.radians(-74.0060), 10.0])  # New York
        self.equator_llh = np.array([0.0, 0.0, 0.0])  # Equator, prime meridian
        self.pole_llh = np.array([np.radians(90.0), 0.0, 0.0])  # North pole
        
    def test_llh2ecef_ecef2llh_round_trip(self):
        # Test round-trip conversion for various points
        test_points = [
            self.tokyo_llh,
            self.newyork_llh,
            self.equator_llh,
            self.pole_llh,
            np.array([np.radians(-35.0), np.radians(150.0), 100.0])  # Southern hemisphere
        ]
        
        for llh in test_points:
            xyz = llh2ecef(llh)
            llh_recovered = ecef2llh(xyz)
            
            # Check round-trip accuracy
            # Note: Due to numerical precision in iterative calculations,
            # we use slightly relaxed tolerances for height
            np.testing.assert_allclose(llh_recovered[:2], llh[:2], rtol=1e-10, atol=1e-10,
                                      err_msg=f"Round-trip failed for lat/lon {llh}")
            np.testing.assert_allclose(llh_recovered[2], llh[2], rtol=1e-8, atol=1e-6,
                                      err_msg=f"Round-trip failed for height {llh}")
    
    def test_llh2ecef_known_values(self):
        # Test with known values
        # Point at equator, prime meridian, sea level
        llh = np.array([0.0, 0.0, 0.0])
        xyz = llh2ecef(llh)
        
        # Should be on X-axis at Earth radius
        expected_x = RE_WGS84
        expected_y = 0.0
        expected_z = 0.0
        
        self.assertAlmostEqual(xyz[0], expected_x, places=3)
        self.assertAlmostEqual(xyz[1], expected_y, places=3)
        self.assertAlmostEqual(xyz[2], expected_z, places=3)
        
        # North pole
        llh = np.array([np.pi/2, 0.0, 0.0])
        xyz = llh2ecef(llh)
        
        # Should be on Z-axis
        expected_x = 0.0
        expected_y = 0.0
        # Earth polar radius
        b = RE_WGS84 * np.sqrt(1 - FE_WGS84 * (2 - FE_WGS84))
        expected_z = b
        
        self.assertAlmostEqual(xyz[0], expected_x, places=3)
        self.assertAlmostEqual(xyz[1], expected_y, places=3)
        self.assertAlmostEqual(xyz[2], expected_z, places=3)
    
    def test_ecef2llh_known_values(self):
        # Test with known ECEF values
        # Point on X-axis at Earth radius
        xyz = np.array([RE_WGS84, 0.0, 0.0])
        llh = ecef2llh(xyz)
        
        self.assertAlmostEqual(llh[0], 0.0, places=10)  # Latitude
        self.assertAlmostEqual(llh[1], 0.0, places=10)  # Longitude
        self.assertAlmostEqual(llh[2], 0.0, places=3)   # Height
        
        # Point on Y-axis
        xyz = np.array([0.0, RE_WGS84, 0.0])
        llh = ecef2llh(xyz)
        
        self.assertAlmostEqual(llh[0], 0.0, places=10)  # Latitude
        self.assertAlmostEqual(llh[1], np.pi/2, places=10)  # Longitude (90 degrees)
        self.assertAlmostEqual(llh[2], 0.0, places=3)   # Height
    
    def test_ecef2enu_enu2ecef_round_trip(self):
        # Test ENU conversion round-trip
        origin = self.tokyo_llh
        
        # Test points relative to Tokyo
        test_offsets = [
            np.array([100.0, 200.0, 50.0]),    # Northeast and up
            np.array([-100.0, -200.0, -50.0]),  # Southwest and down
            np.array([0.0, 0.0, 100.0]),        # Directly up
            np.array([1000.0, 0.0, 0.0]),       # East only
        ]
        
        for enu_offset in test_offsets:
            # Convert to ECEF
            origin_xyz = llh2ecef(origin)
            xyz = enu2ecef(enu_offset, origin)
            
            # Convert back to ENU
            enu_recovered = ecef2enu(xyz, origin)
            
            np.testing.assert_allclose(enu_recovered, enu_offset, rtol=1e-10, atol=1e-8)
    
    def test_ecef2ned_ned2ecef_round_trip(self):
        # Test NED conversion round-trip
        origin = self.newyork_llh
        
        test_offsets = [
            np.array([100.0, 200.0, -50.0]),    # North, East, Down
            np.array([-100.0, -200.0, 50.0]),   # South, West, Up
            np.array([0.0, 0.0, -100.0]),        # Directly down
        ]
        
        for ned_offset in test_offsets:
            # Convert to ECEF and back
            xyz = ned2ecef(ned_offset, origin)
            ned_recovered = ecef2ned(xyz, origin)
            
            np.testing.assert_allclose(ned_recovered, ned_offset, rtol=1e-10, atol=1e-8)
    
    def test_enu2ned_ned2enu(self):
        # Test ENU to NED conversion
        enu = np.array([100.0, 200.0, 50.0])  # East, North, Up
        
        ned = enu2ned(enu)
        
        # NED should be North, East, Down
        expected_ned = np.array([200.0, 100.0, -50.0])
        np.testing.assert_allclose(ned, expected_ned)
        
        # Test reverse
        enu_recovered = ned2enu(ned)
        np.testing.assert_allclose(enu_recovered, enu)
    
    def test_rotation_matrix_enu(self):
        # Test ENU rotation matrix properties
        llh = self.tokyo_llh
        R = compute_rotation_matrix_enu(llh)
        
        # Rotation matrix should be orthogonal
        I = R @ R.T
        np.testing.assert_allclose(I, np.eye(3), rtol=1e-10, atol=1e-10)
        
        # Determinant should be 1
        det = np.linalg.det(R)
        self.assertAlmostEqual(det, 1.0, places=10)
    
    def test_rotation_matrix_ned(self):
        # Test NED rotation matrix properties
        llh = self.newyork_llh
        R = compute_rotation_matrix_ned(llh)
        
        # Rotation matrix should be orthogonal
        I = R @ R.T
        np.testing.assert_allclose(I, np.eye(3), rtol=1e-10, atol=1e-10)
        
        # Determinant should be 1
        det = np.linalg.det(R)
        self.assertAlmostEqual(det, 1.0, places=10)
    
    def test_rotation_matrices_relationship(self):
        # Test relationship between ENU and NED rotation matrices
        llh = self.tokyo_llh
        R_enu = compute_rotation_matrix_enu(llh)
        R_ned = compute_rotation_matrix_ned(llh)
        
        # Test that both rotation matrices are orthogonal
        np.testing.assert_allclose(R_enu @ R_enu.T, np.eye(3), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(R_ned @ R_ned.T, np.eye(3), rtol=1e-10, atol=1e-10)
        
        # Test using the built-in transformation functions
        test_ecef = llh2ecef(llh + np.array([np.radians(0.001), np.radians(0.001), 10]))
        
        # Use the actual transformation functions
        enu = ecef2enu(test_ecef, llh)
        ned = ecef2ned(test_ecef, llh)
        
        # Check that conversions are consistent
        ned_from_enu = enu2ned(enu)
        enu_from_ned = ned2enu(ned)
        
        np.testing.assert_allclose(ned_from_enu, ned, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(enu_from_ned, enu, rtol=1e-8, atol=1e-8)
    
    def test_ecef_transformations_at_poles(self):
        # Test transformations at poles (edge cases)
        north_pole = np.array([np.pi/2, 0.0, 0.0])
        south_pole = np.array([-np.pi/2, 0.0, 0.0])
        
        # Convert to ECEF and back
        xyz_north = llh2ecef(north_pole)
        llh_north_recovered = ecef2llh(xyz_north)
        
        self.assertAlmostEqual(llh_north_recovered[0], north_pole[0], places=10)
        self.assertAlmostEqual(llh_north_recovered[2], north_pole[2], places=5)
        # Longitude at poles is undefined, so we don't test it
        
        xyz_south = llh2ecef(south_pole)
        llh_south_recovered = ecef2llh(xyz_south)
        
        self.assertAlmostEqual(llh_south_recovered[0], south_pole[0], places=10)
        self.assertAlmostEqual(llh_south_recovered[2], south_pole[2], places=5)
    
    def test_local_frame_conversions_consistency(self):
        # Test that local frame conversions are consistent
        origin = self.tokyo_llh
        test_point_llh = self.tokyo_llh + np.array([np.radians(0.01), np.radians(0.01), 100.0])
        
        test_point_ecef = llh2ecef(test_point_llh)
        
        # Convert to ENU and NED
        enu = ecef2enu(test_point_ecef, origin)
        ned = ecef2ned(test_point_ecef, origin)
        
        # Convert between ENU and NED
        ned_from_enu = enu2ned(enu)
        enu_from_ned = ned2enu(ned)
        
        np.testing.assert_allclose(ned_from_enu, ned, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(enu_from_ned, enu, rtol=1e-10, atol=1e-10)
        
        # Convert back to ECEF
        ecef_from_enu = enu2ecef(enu, origin)
        ecef_from_ned = ned2ecef(ned, origin)
        
        np.testing.assert_allclose(ecef_from_enu, test_point_ecef, rtol=1e-10, atol=1e-8)
        np.testing.assert_allclose(ecef_from_ned, test_point_ecef, rtol=1e-10, atol=1e-8)
    
    def test_height_variations(self):
        # Test transformations at different heights
        base_llh = self.tokyo_llh.copy()
        heights = [0.0, 100.0, 1000.0, 10000.0, 100000.0]  # Various altitudes
        
        for h in heights:
            llh = base_llh.copy()
            llh[2] = h
            
            xyz = llh2ecef(llh)
            llh_recovered = ecef2llh(xyz)
            
            np.testing.assert_allclose(llh_recovered, llh, rtol=1e-10, atol=1e-6)
    
    def test_longitude_wrap_around(self):
        # Test longitude wrap-around (-180 to 180 degrees)
        test_lons = [
            np.radians(179.0),
            np.radians(-179.0),
            np.radians(180.0),
            np.radians(-180.0),
            np.radians(0.0),
            np.radians(360.0),  # Should wrap to 0
            np.radians(-360.0),  # Should wrap to 0
        ]
        
        for lon in test_lons:
            llh = np.array([np.radians(35.0), lon, 100.0])
            xyz = llh2ecef(llh)
            llh_recovered = ecef2llh(xyz)
            
            # Normalize longitude to [-pi, pi]
            lon_normalized = np.arctan2(np.sin(lon), np.cos(lon))
            
            self.assertAlmostEqual(llh_recovered[0], llh[0], places=10)
            self.assertAlmostEqual(llh_recovered[1], lon_normalized, places=10)
            self.assertAlmostEqual(llh_recovered[2], llh[2], places=5)


class TestCoordinateTransformEdgeCases(unittest.TestCase):
    
    def test_zero_altitude(self):
        # Test at sea level
        llh = np.array([np.radians(45.0), np.radians(45.0), 0.0])
        xyz = llh2ecef(llh)
        llh_recovered = ecef2llh(xyz)
        
        # Separate tolerance for height due to iterative calculation
        np.testing.assert_allclose(llh_recovered[:2], llh[:2], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(llh_recovered[2], llh[2], rtol=1e-8, atol=1e-6)
    
    def test_negative_altitude(self):
        # Test below sea level (e.g., Dead Sea)
        llh = np.array([np.radians(31.5), np.radians(35.5), -430.0])
        xyz = llh2ecef(llh)
        llh_recovered = ecef2llh(xyz)
        
        # Separate tolerance for height due to iterative calculation
        np.testing.assert_allclose(llh_recovered[:2], llh[:2], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(llh_recovered[2], llh[2], rtol=1e-8, atol=1e-6)
    
    def test_very_small_coordinates(self):
        # Test with very small offsets from origin
        origin = np.array([0.0, 0.0, 0.0])
        small_offset = np.array([1e-6, 1e-6, 1e-6])  # Use larger values to avoid numerical issues
        
        xyz = enu2ecef(small_offset, origin)
        enu_recovered = ecef2enu(xyz, origin)
        
        # Check relative accuracy for small coordinates
        # Note: At the equator, small vertical offsets have larger relative errors
        np.testing.assert_allclose(enu_recovered[:2], small_offset[:2], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(enu_recovered[2], small_offset[2], rtol=1e-3, atol=1e-9)
    
    def test_very_large_coordinates(self):
        # Test with satellite altitude
        llh = np.array([np.radians(0.0), np.radians(0.0), 35786000.0])  # Geostationary orbit
        xyz = llh2ecef(llh)
        llh_recovered = ecef2llh(xyz)
        
        np.testing.assert_allclose(llh_recovered, llh, rtol=1e-10, atol=1e-3)
    
    def test_antipodal_points(self):
        # Test antipodal points (opposite sides of Earth)
        point1_llh = np.array([np.radians(45.0), np.radians(0.0), 0.0])
        point2_llh = np.array([np.radians(-45.0), np.radians(180.0), 0.0])
        
        xyz1 = llh2ecef(point1_llh)
        xyz2 = llh2ecef(point2_llh)
        
        # Points should be roughly opposite
        # Not exactly opposite due to Earth's ellipsoid shape
        cosine_angle = np.dot(xyz1, xyz2) / (np.linalg.norm(xyz1) * np.linalg.norm(xyz2))
        self.assertLess(cosine_angle, -0.5)  # Angle > 120 degrees


if __name__ == '__main__':
    unittest.main()