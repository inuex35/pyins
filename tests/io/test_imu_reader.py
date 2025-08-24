import unittest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import logging

from pyins.io.imu_reader import (
    IMUReader, IMUPreprocessor, 
    load_imu_data, preprocess_imu_data, sync_gnss_imu
)
from pyins.sensors.imu import IMUData
from pyins.core.data_structures import SensorType


class TestIMUReader(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.csv_file = os.path.join(self.test_dir, "test_imu.csv")
        self.txt_file = os.path.join(self.test_dir, "test_imu.txt")
        
        # Create test CSV data
        self.test_data = pd.DataFrame({
            'time': [1000.0, 1000.01, 1000.02, 1000.03, 1000.04],
            'accel_x': [0.1, 0.2, 0.3, 0.4, 0.5],
            'accel_y': [0.11, 0.21, 0.31, 0.41, 0.51],
            'accel_z': [9.8, 9.81, 9.82, 9.83, 9.84],
            'gyro_x': [0.001, 0.002, 0.003, 0.004, 0.005],
            'gyro_y': [0.0011, 0.0021, 0.0031, 0.0041, 0.0051],
            'gyro_z': [0.0012, 0.0022, 0.0032, 0.0042, 0.0052]
        })
        self.test_data.to_csv(self.csv_file, index=False)
        
        # Create test TXT data
        with open(self.txt_file, 'w') as f:
            f.write("# Test IMU data\n")
            for _, row in self.test_data.iterrows():
                f.write(f"{row['time']} {row['accel_x']} {row['accel_y']} {row['accel_z']} ")
                f.write(f"{row['gyro_x']} {row['gyro_y']} {row['gyro_z']}\n")
    
    def tearDown(self):
        # Clean up test files
        if os.path.exists(self.csv_file):
            os.remove(self.csv_file)
        if os.path.exists(self.txt_file):
            os.remove(self.txt_file)
        os.rmdir(self.test_dir)
    
    def test_init_valid_file(self):
        reader = IMUReader(self.csv_file, format='csv')
        self.assertEqual(reader.file_path, Path(self.csv_file))
        self.assertEqual(reader.format, 'csv')
        self.assertEqual(reader.input_time_system, 'gps')
    
    def test_init_invalid_file(self):
        with self.assertRaises(FileNotFoundError):
            IMUReader("nonexistent_file.csv", format='csv')
    
    def test_read_csv_all_data(self):
        reader = IMUReader(self.csv_file, format='csv')
        df = reader.read()
        self.assertEqual(len(df), 5)
        self.assertListEqual(list(df.columns), 
                           ['time', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z'])
        np.testing.assert_array_almost_equal(df['time'].values, self.test_data['time'].values)
    
    def test_read_csv_with_time_filter(self):
        reader = IMUReader(self.csv_file, format='csv')
        df = reader.read(start_time=1000.02, duration=0.02)
        self.assertEqual(len(df), 3)  # Should get times 1000.02, 1000.03, 1000.04
        self.assertAlmostEqual(df['time'].iloc[0], 1000.02)
        self.assertAlmostEqual(df['time'].iloc[-1], 1000.04)
    
    def test_read_txt_format(self):
        reader = IMUReader(self.txt_file, format='txt')
        df = reader.read()
        self.assertEqual(len(df), 5)
        np.testing.assert_array_almost_equal(df['accel_x'].values, self.test_data['accel_x'].values)
    
    def test_time_system_conversion_gps_to_unix(self):
        reader = IMUReader(self.csv_file, format='csv', time_system='gps')
        df = reader.read(output_time_system='unix')
        
        expected_offset = reader.GPS_TO_UNIX_OFFSET - reader.LEAP_SECONDS_2015
        expected_times = self.test_data['time'].values + expected_offset
        np.testing.assert_array_almost_equal(df['time'].values, expected_times)
    
    def test_time_system_conversion_unix_to_gps(self):
        reader = IMUReader(self.csv_file, format='csv', time_system='unix')
        df = reader.read(output_time_system='gps')
        
        expected_offset = -reader.GPS_TO_UNIX_OFFSET + reader.LEAP_SECONDS_2015
        expected_times = self.test_data['time'].values + expected_offset
        np.testing.assert_array_almost_equal(df['time'].values, expected_times)
    
    def test_read_as_imu_data_list(self):
        reader = IMUReader(self.csv_file, format='csv')
        imu_list = reader.read_as_imu_data_list()
        
        self.assertEqual(len(imu_list), 5)
        self.assertIsInstance(imu_list[0], IMUData)
        self.assertEqual(imu_list[0].sensor_id, "imu_0")
        self.assertAlmostEqual(imu_list[0].timestamp, 1000.0)
        
        # Check measurement data
        expected_measurement = np.array([0.1, 0.11, 9.8, 0.001, 0.0011, 0.0012])
        np.testing.assert_array_almost_equal(imu_list[0].data, expected_measurement)
    
    def test_alternative_column_names(self):
        # Test with alternative column names
        alt_data = pd.DataFrame({
            'timestamp': [1000.0, 1000.01],
            'ax': [0.1, 0.2],
            'ay': [0.11, 0.21],
            'az': [9.8, 9.81],
            'wx': [0.001, 0.002],
            'wy': [0.0011, 0.0021],
            'wz': [0.0012, 0.0022]
        })
        alt_file = os.path.join(self.test_dir, "alt_imu.csv")
        alt_data.to_csv(alt_file, index=False)
        
        reader = IMUReader(alt_file, format='csv')
        df = reader.read()
        
        self.assertIn('time', df.columns)
        self.assertIn('accel_x', df.columns)
        self.assertIn('gyro_x', df.columns)
        
        os.remove(alt_file)
    
    def test_missing_columns(self):
        bad_data = pd.DataFrame({
            'time': [1000.0, 1000.01],
            'accel_x': [0.1, 0.2],
            # Missing other required columns
        })
        bad_file = os.path.join(self.test_dir, "bad_imu.csv")
        bad_data.to_csv(bad_file, index=False)
        
        reader = IMUReader(bad_file, format='csv')
        with self.assertRaises(ValueError) as context:
            reader.read()
        self.assertIn("Missing required IMU columns", str(context.exception))
        
        os.remove(bad_file)


class TestIMUPreprocessor(unittest.TestCase):
    
    def setUp(self):
        # Create test IMU data with bias
        self.imu_data = pd.DataFrame({
            'time': np.arange(0, 2, 0.01),
            'accel_x': np.ones(200) * 0.1,  # Constant bias
            'accel_y': np.ones(200) * 0.2,
            'accel_z': np.ones(200) * 9.90665,  # Gravity + bias
            'gyro_x': np.ones(200) * 0.01,
            'gyro_y': np.ones(200) * 0.02,
            'gyro_z': np.ones(200) * 0.03
        })
    
    def test_remove_bias_with_known_values(self):
        accel_bias = np.array([0.1, 0.2, 0.1])
        gyro_bias = np.array([0.01, 0.02, 0.03])
        
        processed = IMUPreprocessor.remove_bias(
            self.imu_data, 
            accel_bias=accel_bias,
            gyro_bias=gyro_bias,
            estimate_from_stationary=False
        )
        
        np.testing.assert_array_almost_equal(processed['accel_x'].values, np.zeros(200))
        np.testing.assert_array_almost_equal(processed['accel_y'].values, np.zeros(200))
        np.testing.assert_array_almost_equal(processed['accel_z'].values, np.ones(200) * 9.80665)
        np.testing.assert_array_almost_equal(processed['gyro_x'].values, np.zeros(200))
        np.testing.assert_array_almost_equal(processed['gyro_y'].values, np.zeros(200))
        np.testing.assert_array_almost_equal(processed['gyro_z'].values, np.zeros(200))
    
    def test_remove_bias_estimate_from_stationary(self):
        processed = IMUPreprocessor.remove_bias(
            self.imu_data,
            estimate_from_stationary=True
        )
        
        # Check that bias was estimated and removed
        # The first second should be used for estimation
        self.assertAlmostEqual(processed['accel_x'].mean(), 0.0, places=5)
        self.assertAlmostEqual(processed['accel_y'].mean(), 0.0, places=5)
        # Z-axis: After bias removal, should be close to gravity (9.80665)
        # Original data has 9.90665, estimated bias is 0.1, result is 9.80665
        self.assertAlmostEqual(processed['accel_z'].mean(), 9.80665, places=4)
        self.assertAlmostEqual(processed['gyro_x'].mean(), 0.0, places=5)
        self.assertAlmostEqual(processed['gyro_y'].mean(), 0.0, places=5)
        self.assertAlmostEqual(processed['gyro_z'].mean(), 0.0, places=5)
    
    def test_sync_with_gnss(self):
        # Create test GNSS epochs
        gnss_epochs = [
            {'time': 0.0, 'gps_time': 1000.0, 'observations': {'sat1': 'obs1'}},
            {'time': 0.5, 'gps_time': 1000.5, 'observations': {'sat2': 'obs2'}},
            {'time': 1.0, 'gps_time': 1001.0, 'observations': {'sat3': 'obs3'}},
            {'time': 1.5, 'gps_time': 1001.5, 'observations': {'sat4': 'obs4'}}
        ]
        
        synced = IMUPreprocessor.sync_with_gnss(gnss_epochs, self.imu_data)
        
        self.assertEqual(len(synced), 3)  # One less than GNSS epochs
        
        # Check first synchronized epoch
        self.assertEqual(synced[0]['time'], 0.0)
        self.assertEqual(synced[0]['gps_time'], 1000.0)
        self.assertEqual(synced[0]['gnss_obs'], {'sat1': 'obs1'})
        
        # Check IMU data between epochs
        self.assertEqual(len(synced[0]['imu_data']), 50)  # 50 IMU samples between 0.0 and 0.5
        self.assertAlmostEqual(synced[0]['imu_data'][0]['time'], 0.0)
        np.testing.assert_array_equal(
            synced[0]['imu_data'][0]['accel'],
            np.array([0.1, 0.2, 9.90665])
        )
    
    def test_sync_with_gnss_no_imu_data(self):
        # Create GNSS epochs outside IMU time range
        gnss_epochs = [
            {'time': 10.0, 'observations': {'sat1': 'obs1'}},
            {'time': 11.0, 'observations': {'sat2': 'obs2'}}
        ]
        
        # Just check that empty list is returned when no data matches
        synced = IMUPreprocessor.sync_with_gnss(gnss_epochs, self.imu_data)
        self.assertEqual(len(synced), 0)
    
    def test_sync_with_gnss_time_difference_warning(self):
        # Create GNSS epochs with slight time offset
        gnss_epochs = [
            {'time': 0.02, 'observations': {'sat1': 'obs1'}},  # 20ms offset
            {'time': 0.5, 'observations': {'sat2': 'obs2'}}
        ]
        
        # Test that synchronization still works with time offset
        synced = IMUPreprocessor.sync_with_gnss(
            gnss_epochs, self.imu_data, max_time_diff=0.01
        )
        # Should have one synchronized epoch (between 0.02 and 0.5)
        self.assertEqual(len(synced), 1)


class TestConvenienceFunctions(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.csv_file = os.path.join(self.test_dir, "test_imu.csv")
        
        test_data = pd.DataFrame({
            'time': [1000.0, 1000.01, 1000.02],
            'accel_x': [0.1, 0.2, 0.3],
            'accel_y': [0.11, 0.21, 0.31],
            'accel_z': [9.8, 9.81, 9.82],
            'gyro_x': [0.001, 0.002, 0.003],
            'gyro_y': [0.0011, 0.0021, 0.0031],
            'gyro_z': [0.0012, 0.0022, 0.0032]
        })
        test_data.to_csv(self.csv_file, index=False)
    
    def tearDown(self):
        if os.path.exists(self.csv_file):
            os.remove(self.csv_file)
        os.rmdir(self.test_dir)
    
    def test_load_imu_data(self):
        df = load_imu_data(self.csv_file, format='csv')
        self.assertEqual(len(df), 3)
        self.assertIn('time', df.columns)
        self.assertIn('accel_x', df.columns)
    
    def test_load_imu_data_with_filters(self):
        df = load_imu_data(self.csv_file, start_time=1000.01, duration=0.01)
        self.assertEqual(len(df), 2)
        self.assertAlmostEqual(df['time'].iloc[0], 1000.01)
    
    def test_preprocess_imu_data_no_bias_removal(self):
        df = pd.DataFrame({
            'time': [0, 0.01],
            'accel_x': [0.1, 0.2],
            'accel_y': [0.11, 0.21],
            'accel_z': [9.8, 9.81],
            'gyro_x': [0.001, 0.002],
            'gyro_y': [0.0011, 0.0021],
            'gyro_z': [0.0012, 0.0022]
        })
        
        processed = preprocess_imu_data(df, remove_bias=False)
        pd.testing.assert_frame_equal(processed, df)
    
    def test_preprocess_imu_data_with_bias_removal(self):
        df = pd.DataFrame({
            'time': np.arange(0, 2, 0.01),
            'accel_x': np.ones(200) * 0.1,
            'accel_y': np.ones(200) * 0.2,
            'accel_z': np.ones(200) * 9.90665,
            'gyro_x': np.ones(200) * 0.01,
            'gyro_y': np.ones(200) * 0.02,
            'gyro_z': np.ones(200) * 0.03
        })
        
        accel_bias = np.array([0.1, 0.2, 0.1])
        gyro_bias = np.array([0.01, 0.02, 0.03])
        
        processed = preprocess_imu_data(
            df, 
            remove_bias=True,
            accel_bias=accel_bias,
            gyro_bias=gyro_bias
        )
        
        self.assertAlmostEqual(processed['accel_x'].mean(), 0.0, places=5)
        self.assertAlmostEqual(processed['gyro_x'].mean(), 0.0, places=5)
    
    def test_sync_gnss_imu(self):
        gnss_epochs = [
            {'time': 1000.0, 'observations': {'sat1': 'obs1'}},
            {'time': 1000.01, 'observations': {'sat2': 'obs2'}},
            {'time': 1000.02, 'observations': {'sat3': 'obs3'}}
        ]
        
        imu_data = pd.DataFrame({
            'time': [1000.0, 1000.005, 1000.01, 1000.015, 1000.02],
            'accel_x': [0.1, 0.15, 0.2, 0.25, 0.3],
            'accel_y': [0.11, 0.16, 0.21, 0.26, 0.31],
            'accel_z': [9.8, 9.805, 9.81, 9.815, 9.82],
            'gyro_x': [0.001, 0.0015, 0.002, 0.0025, 0.003],
            'gyro_y': [0.0011, 0.0016, 0.0021, 0.0026, 0.0031],
            'gyro_z': [0.0012, 0.0017, 0.0022, 0.0027, 0.0032]
        })
        
        synced = sync_gnss_imu(gnss_epochs, imu_data)
        
        self.assertEqual(len(synced), 2)
        self.assertEqual(synced[0]['time'], 1000.0)
        self.assertEqual(len(synced[0]['imu_data']), 2)  # IMU data at 1000.0 and 1000.005


if __name__ == '__main__':
    unittest.main()