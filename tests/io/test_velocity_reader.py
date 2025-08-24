import unittest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import logging

from pyins.io.velocity_reader import VelocityReader


class TestVelocityReader(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.csv_file = os.path.join(self.test_dir, "test_velocity.csv")
        
        # Create test velocity data in ENU coordinates
        self.test_data = pd.DataFrame({
            'time': [1000.0, 1000.1, 1000.2, 1000.3, 1000.4],
            'vx': [1.0, 1.1, 1.2, 1.3, 1.4],
            'vy': [2.0, 2.1, 2.2, 2.3, 2.4],
            'vz': [0.1, 0.11, 0.12, 0.13, 0.14]
        })
        self.test_data.to_csv(self.csv_file, index=False)
        
        # Create NED velocity file
        self.ned_file = os.path.join(self.test_dir, "test_ned_velocity.csv")
        self.ned_data = pd.DataFrame({
            'time': [1000.0, 1000.1, 1000.2],
            'vel_n': [2.0, 2.1, 2.2],  # North
            'vel_e': [1.0, 1.1, 1.2],  # East
            'vel_d': [-0.1, -0.11, -0.12]  # Down
        })
        self.ned_data.to_csv(self.ned_file, index=False)
        
        # Create ENU velocity file
        self.enu_file = os.path.join(self.test_dir, "test_enu_velocity.csv")
        self.enu_data = pd.DataFrame({
            'time': [1000.0, 1000.1, 1000.2],
            'vel_e': [1.0, 1.1, 1.2],  # East
            'vel_n': [2.0, 2.1, 2.2],  # North
            'vel_u': [0.1, 0.11, 0.12]  # Up
        })
        self.enu_data.to_csv(self.enu_file, index=False)
    
    def tearDown(self):
        # Clean up test files
        for file in [self.csv_file, self.ned_file, self.enu_file]:
            if os.path.exists(file):
                os.remove(file)
        os.rmdir(self.test_dir)
    
    def test_init_valid_file(self):
        reader = VelocityReader(self.csv_file, format='csv', time_format='gps')
        self.assertEqual(reader.file_path, Path(self.csv_file))
        self.assertEqual(reader.format, 'csv')
        self.assertEqual(reader.time_format, 'gps')
    
    def test_init_invalid_file(self):
        with self.assertRaises(FileNotFoundError):
            VelocityReader("nonexistent_file.csv", format='csv')
    
    def test_read_csv_all_data(self):
        reader = VelocityReader(self.csv_file, format='csv')
        df = reader.read()
        
        self.assertEqual(len(df), 5)
        self.assertListEqual(list(df.columns), ['time', 'vx', 'vy', 'vz'])
        np.testing.assert_array_almost_equal(df['time'].values, self.test_data['time'].values)
        np.testing.assert_array_almost_equal(df['vx'].values, self.test_data['vx'].values)
    
    def test_read_csv_with_time_filter(self):
        reader = VelocityReader(self.csv_file, format='csv')
        df = reader.read(start_time=1000.1, duration=0.2)
        
        self.assertEqual(len(df), 3)  # Should get times 1000.1, 1000.2, 1000.3
        self.assertAlmostEqual(df['time'].iloc[0], 1000.1)
        self.assertAlmostEqual(df['time'].iloc[-1], 1000.3)
    
    def test_read_ned_format(self):
        reader = VelocityReader(self.ned_file, format='csv')
        df = reader.read()
        
        self.assertEqual(len(df), 3)
        self.assertIn('vx', df.columns)
        self.assertIn('vy', df.columns)
        self.assertIn('vz', df.columns)
        
        # Check NED to ENU conversion
        np.testing.assert_array_almost_equal(df['vx'].values, [1.0, 1.1, 1.2])  # East
        np.testing.assert_array_almost_equal(df['vy'].values, [2.0, 2.1, 2.2])  # North
        np.testing.assert_array_almost_equal(df['vz'].values, [0.1, 0.11, 0.12])  # Up (negative of down)
    
    def test_read_enu_format(self):
        reader = VelocityReader(self.enu_file, format='csv')
        df = reader.read()
        
        self.assertEqual(len(df), 3)
        np.testing.assert_array_almost_equal(df['vx'].values, [1.0, 1.1, 1.2])  # East
        np.testing.assert_array_almost_equal(df['vy'].values, [2.0, 2.1, 2.2])  # North
        np.testing.assert_array_almost_equal(df['vz'].values, [0.1, 0.11, 0.12])  # Up
    
    def test_alternative_column_names(self):
        # Test with alternative column names
        alt_data = pd.DataFrame({
            'timestamp': [1000.0, 1000.1],
            'velocity_x': [1.0, 1.1],
            'velocity_y': [2.0, 2.1],
            'velocity_z': [0.1, 0.11]
        })
        alt_file = os.path.join(self.test_dir, "alt_velocity.csv")
        alt_data.to_csv(alt_file, index=False)
        
        reader = VelocityReader(alt_file, format='csv')
        df = reader.read()
        
        self.assertIn('time', df.columns)
        self.assertIn('vx', df.columns)
        self.assertIn('vy', df.columns)
        self.assertIn('vz', df.columns)
        
        os.remove(alt_file)
    
    def test_unix_time_conversion(self):
        # Create file with UNIX time
        unix_data = pd.DataFrame({
            'time': [1577836800.0, 1577836801.0],  # Jan 1, 2020 in UNIX time
            'vx': [1.0, 1.1],
            'vy': [2.0, 2.1],
            'vz': [0.1, 0.11]
        })
        unix_file = os.path.join(self.test_dir, "unix_velocity.csv")
        unix_data.to_csv(unix_file, index=False)
        
        reader = VelocityReader(unix_file, format='csv', time_format='unix')
        df = reader.read()
        
        # Check that time was converted to GPS
        GPS_UNIX_OFFSET = 315964800 - 18
        expected_gps_time = 1577836800.0 - GPS_UNIX_OFFSET
        self.assertAlmostEqual(df['time'].iloc[0], expected_gps_time, places=1)
        
        os.remove(unix_file)
    
    def test_missing_columns(self):
        bad_data = pd.DataFrame({
            'time': [1000.0, 1000.1],
            'vx': [1.0, 1.1]
            # Missing vy and vz
        })
        bad_file = os.path.join(self.test_dir, "bad_velocity.csv")
        bad_data.to_csv(bad_file, index=False)
        
        reader = VelocityReader(bad_file, format='csv')
        with self.assertRaises(RuntimeError) as context:
            reader.read()
        self.assertIn("Missing required columns", str(context.exception))
        
        os.remove(bad_file)
    
    def test_unsupported_format(self):
        reader = VelocityReader(self.csv_file, format='binary')
        with self.assertRaises(ValueError) as context:
            reader.read()
        self.assertIn("Unsupported format", str(context.exception))
    
    def test_get_velocity_at_time_linear(self):
        reader = VelocityReader(self.csv_file, format='csv')
        
        # Test exact time
        vel = reader.get_velocity_at_time(1000.1, method='linear')
        np.testing.assert_array_almost_equal(vel, [1.1, 2.1, 0.11])
        
        # Test interpolated time
        vel = reader.get_velocity_at_time(1000.05, method='linear')
        np.testing.assert_array_almost_equal(vel, [1.05, 2.05, 0.105])
        
        # Test boundary
        vel = reader.get_velocity_at_time(1000.0, method='linear')
        np.testing.assert_array_almost_equal(vel, [1.0, 2.0, 0.1])
    
    def test_get_velocity_at_time_nearest(self):
        reader = VelocityReader(self.csv_file, format='csv')
        
        # Test nearest neighbor interpolation
        vel = reader.get_velocity_at_time(1000.04, method='nearest')
        np.testing.assert_array_almost_equal(vel, [1.0, 2.0, 0.1])  # Nearest to 1000.0
        
        vel = reader.get_velocity_at_time(1000.06, method='nearest')
        np.testing.assert_array_almost_equal(vel, [1.1, 2.1, 0.11])  # Nearest to 1000.1
    
    def test_get_velocity_at_time_out_of_range(self):
        reader = VelocityReader(self.csv_file, format='csv')
        
        # Test time before data
        vel = reader.get_velocity_at_time(999.0, method='linear')
        self.assertIsNone(vel)
        
        # Test time after data
        vel = reader.get_velocity_at_time(1001.0, method='linear')
        self.assertIsNone(vel)
    
    def test_get_velocity_at_time_invalid_method(self):
        reader = VelocityReader(self.csv_file, format='csv')
        
        with self.assertRaises(ValueError) as context:
            reader.get_velocity_at_time(1000.1, method='cubic')
        self.assertIn("Unknown interpolation method", str(context.exception))
    
    def test_data_sorting(self):
        # Create unsorted data
        unsorted_data = pd.DataFrame({
            'time': [1000.2, 1000.0, 1000.3, 1000.1],
            'vx': [1.2, 1.0, 1.3, 1.1],
            'vy': [2.2, 2.0, 2.3, 2.1],
            'vz': [0.12, 0.1, 0.13, 0.11]
        })
        unsorted_file = os.path.join(self.test_dir, "unsorted_velocity.csv")
        unsorted_data.to_csv(unsorted_file, index=False)
        
        reader = VelocityReader(unsorted_file, format='csv')
        df = reader.read()
        
        # Check that data is sorted by time
        self.assertTrue(df['time'].is_monotonic_increasing)
        self.assertAlmostEqual(df['time'].iloc[0], 1000.0)
        self.assertAlmostEqual(df['time'].iloc[-1], 1000.3)
        
        os.remove(unsorted_file)
    
    @patch('pyins.io.velocity_reader.logger')
    def test_logging_output(self, mock_logger):
        reader = VelocityReader(self.csv_file, format='csv')
        df = reader.read()
        
        # Check that appropriate logging was done
        info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        self.assertTrue(any("Loaded 5 velocity samples" in str(call) for call in info_calls))
        self.assertTrue(any("Time range:" in str(call) for call in info_calls))
        self.assertTrue(any("Velocity stats" in str(call) for call in info_calls))
    
    def test_utc_time_format_not_implemented(self):
        reader = VelocityReader(self.csv_file, format='csv', time_format='utc')
        
        with self.assertRaises(RuntimeError) as context:
            reader.read()
        self.assertIn("UTC time conversion not yet implemented", str(context.exception))


class TestVelocityReaderEdgeCases(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        # Clean up any created files
        for file in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, file))
        os.rmdir(self.test_dir)
    
    def test_empty_file(self):
        empty_file = os.path.join(self.test_dir, "empty.csv")
        pd.DataFrame(columns=['time', 'vx', 'vy', 'vz']).to_csv(empty_file, index=False)
        
        reader = VelocityReader(empty_file, format='csv')
        df = reader.read()
        self.assertEqual(len(df), 0)
    
    def test_single_row(self):
        single_file = os.path.join(self.test_dir, "single.csv")
        pd.DataFrame({
            'time': [1000.0],
            'vx': [1.0],
            'vy': [2.0],
            'vz': [0.1]
        }).to_csv(single_file, index=False)
        
        reader = VelocityReader(single_file, format='csv')
        df = reader.read()
        self.assertEqual(len(df), 1)
        
        # Test interpolation with single point
        vel = reader.get_velocity_at_time(1000.0, method='linear')
        np.testing.assert_array_almost_equal(vel, [1.0, 2.0, 0.1])
    
    def test_duplicate_timestamps(self):
        dup_file = os.path.join(self.test_dir, "duplicate.csv")
        pd.DataFrame({
            'time': [1000.0, 1000.0, 1000.1],
            'vx': [1.0, 1.5, 2.0],
            'vy': [2.0, 2.5, 3.0],
            'vz': [0.1, 0.15, 0.2]
        }).to_csv(dup_file, index=False)
        
        reader = VelocityReader(dup_file, format='csv')
        df = reader.read()
        self.assertEqual(len(df), 3)  # Should keep all rows including duplicates
    
    def test_large_time_gaps(self):
        gap_file = os.path.join(self.test_dir, "gaps.csv")
        pd.DataFrame({
            'time': [1000.0, 1000.1, 2000.0, 2000.1],  # Large gap between 1000.1 and 2000.0
            'vx': [1.0, 1.1, 2.0, 2.1],
            'vy': [2.0, 2.1, 3.0, 3.1],
            'vz': [0.1, 0.11, 0.2, 0.21]
        }).to_csv(gap_file, index=False)
        
        reader = VelocityReader(gap_file, format='csv')
        df = reader.read()
        self.assertEqual(len(df), 4)
        
        # Test interpolation across gap
        vel = reader.get_velocity_at_time(1500.0, method='linear')
        # Should interpolate across the gap
        expected_vx = np.interp(1500.0, [1000.1, 2000.0], [1.1, 2.0])
        self.assertAlmostEqual(vel[0], expected_vx)


if __name__ == '__main__':
    unittest.main()