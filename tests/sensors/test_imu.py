import unittest
import numpy as np
from dataclasses import dataclass
from pyins.sensors.imu import (
    IMUData, IMUConfig, IMUBias, IMUPreintegration
)
from pyins.sensors.sensor_base import SensorType


class TestIMUData(unittest.TestCase):
    
    def test_init_valid(self):
        # Valid 6D IMU data
        data = np.array([0.1, 0.2, 9.8, 0.01, 0.02, 0.03])
        imu_data = IMUData(
            timestamp=1000.0,
            sensor_id="imu_0",
            data=data
        )
        
        self.assertEqual(imu_data.sensor_type, SensorType.IMU)
        self.assertEqual(imu_data.timestamp, 1000.0)
        self.assertEqual(imu_data.sensor_id, "imu_0")
        np.testing.assert_array_equal(imu_data.data, data)
    
    def test_init_invalid_dimensions(self):
        # Invalid data dimensions
        with self.assertRaises(ValueError) as context:
            IMUData(
                timestamp=1000.0,
                sensor_id="imu_0",
                data=np.array([0.1, 0.2, 9.8])  # Only 3D
            )
        self.assertIn("IMU data must be 6D", str(context.exception))
    
    def test_acceleration_property(self):
        data = np.array([0.1, 0.2, 9.8, 0.01, 0.02, 0.03])
        imu_data = IMUData(
            timestamp=1000.0,
            sensor_id="imu_0",
            data=data
        )
        
        acc = imu_data.acceleration
        np.testing.assert_array_equal(acc, np.array([0.1, 0.2, 9.8]))
    
    def test_angular_velocity_property(self):
        data = np.array([0.1, 0.2, 9.8, 0.01, 0.02, 0.03])
        imu_data = IMUData(
            timestamp=1000.0,
            sensor_id="imu_0",
            data=data
        )
        
        gyro = imu_data.angular_velocity
        np.testing.assert_array_equal(gyro, np.array([0.01, 0.02, 0.03]))
    
    def test_remove_gravity(self):
        data = np.array([0.1, 0.2, 9.9, 0.01, 0.02, 0.03])
        imu_data = IMUData(
            timestamp=1000.0,
            sensor_id="imu_0",
            data=data
        )
        
        gravity = np.array([0.0, 0.0, 9.80665])
        acc_no_gravity = imu_data.remove_gravity(gravity)
        
        expected = np.array([0.1, 0.2, 9.9 - 9.80665])
        np.testing.assert_array_almost_equal(acc_no_gravity, expected)
    
    def test_with_covariance(self):
        data = np.array([0.1, 0.2, 9.8, 0.01, 0.02, 0.03])
        cov = np.eye(6) * 0.01
        
        imu_data = IMUData(
            timestamp=1000.0,
            sensor_id="imu_0",
            data=data,
            covariance=cov
        )
        
        np.testing.assert_array_equal(imu_data.covariance, cov)
    
    def test_with_metadata(self):
        data = np.array([0.1, 0.2, 9.8, 0.01, 0.02, 0.03])
        metadata = {"temperature": 25.0, "calibrated": True}
        
        imu_data = IMUData(
            timestamp=1000.0,
            sensor_id="imu_0",
            data=data,
            metadata=metadata
        )
        
        self.assertEqual(imu_data.metadata["temperature"], 25.0)
        self.assertTrue(imu_data.metadata["calibrated"])


class TestIMUConfig(unittest.TestCase):
    
    def test_init_with_defaults(self):
        config = IMUConfig(
            sensor_id="imu_0",
            sensor_type=SensorType.IMU,
            sampling_rate=100.0
        )
        
        self.assertEqual(config.sensor_type, SensorType.IMU)
        self.assertEqual(config.sensor_id, "imu_0")
        self.assertEqual(config.sampling_rate, 100.0)
        
        # Check default noise parameters
        self.assertEqual(config.noise_params["acc_noise"], 0.01)
        self.assertEqual(config.noise_params["gyro_noise"], 0.001)
        self.assertEqual(config.noise_params["acc_bias_walk"], 0.0001)
        self.assertEqual(config.noise_params["gyro_bias_walk"], 0.00001)
    
    def test_init_with_custom_noise(self):
        custom_noise = {
            "acc_noise": 0.05,
            "gyro_noise": 0.005,
            "acc_bias_walk": 0.001,
            "gyro_bias_walk": 0.0001
        }
        
        config = IMUConfig(
            sensor_id="imu_0",
            sensor_type=SensorType.IMU,
            sampling_rate=200.0,
            noise_params=custom_noise
        )
        
        self.assertEqual(config.noise_params["acc_noise"], 0.05)
        self.assertEqual(config.noise_params["gyro_noise"], 0.005)
        self.assertEqual(config.noise_params["acc_bias_walk"], 0.001)
        self.assertEqual(config.noise_params["gyro_bias_walk"], 0.0001)
    
    def test_partial_custom_noise(self):
        # Only specify some noise parameters
        partial_noise = {
            "acc_noise": 0.02,
            "gyro_noise": 0.002
        }
        
        config = IMUConfig(
            sensor_id="imu_0",
            sensor_type=SensorType.IMU,
            sampling_rate=100.0,
            noise_params=partial_noise
        )
        
        # Custom values
        self.assertEqual(config.noise_params["acc_noise"], 0.02)
        self.assertEqual(config.noise_params["gyro_noise"], 0.002)
        # Default values
        self.assertEqual(config.noise_params["acc_bias_walk"], 0.0001)
        self.assertEqual(config.noise_params["gyro_bias_walk"], 0.00001)


class TestIMUBias(unittest.TestCase):
    
    def test_init_default(self):
        bias = IMUBias()
        
        np.testing.assert_array_equal(bias.acc_bias, np.zeros(3))
        np.testing.assert_array_equal(bias.gyro_bias, np.zeros(3))
    
    def test_init_with_values(self):
        acc_bias = np.array([0.01, 0.02, 0.03])
        gyro_bias = np.array([0.001, 0.002, 0.003])
        
        bias = IMUBias(acc_bias=acc_bias, gyro_bias=gyro_bias)
        
        np.testing.assert_array_equal(bias.acc_bias, acc_bias)
        np.testing.assert_array_equal(bias.gyro_bias, gyro_bias)
    
    def test_correct_measurement(self):
        # Create IMU data with bias
        data = np.array([0.11, 0.22, 9.83, 0.011, 0.022, 0.033])
        imu_data = IMUData(
            timestamp=1000.0,
            sensor_id="imu_0",
            data=data
        )
        
        # Create bias
        bias = IMUBias(
            acc_bias=np.array([0.01, 0.02, 0.03]),
            gyro_bias=np.array([0.001, 0.002, 0.003])
        )
        
        # Correct measurement
        corrected = bias.correct_measurement(imu_data)
        
        # Check corrected values
        expected_acc = np.array([0.1, 0.2, 9.8])
        expected_gyro = np.array([0.01, 0.02, 0.03])
        
        np.testing.assert_array_almost_equal(corrected.acceleration, expected_acc)
        np.testing.assert_array_almost_equal(corrected.angular_velocity, expected_gyro)
        
        # Check that other properties are preserved
        self.assertEqual(corrected.timestamp, imu_data.timestamp)
        self.assertEqual(corrected.sensor_id, imu_data.sensor_id)
        self.assertEqual(corrected.sensor_type, imu_data.sensor_type)
    
    def test_correct_measurement_with_metadata(self):
        # Test that metadata and covariance are preserved
        data = np.array([0.1, 0.2, 9.8, 0.01, 0.02, 0.03])
        cov = np.eye(6) * 0.01
        metadata = {"temperature": 25.0}
        
        imu_data = IMUData(
            timestamp=1000.0,
            sensor_id="imu_0",
            data=data,
            covariance=cov,
            metadata=metadata
        )
        
        bias = IMUBias()  # Zero bias
        corrected = bias.correct_measurement(imu_data)
        
        np.testing.assert_array_equal(corrected.covariance, cov)
        self.assertEqual(corrected.metadata, metadata)


class TestIMUPreintegration(unittest.TestCase):
    
    def test_init_default(self):
        preint = IMUPreintegration()
        
        self.assertEqual(preint.dt, 0.0)
        np.testing.assert_array_equal(preint.dP, np.zeros(3))
        np.testing.assert_array_equal(preint.dV, np.zeros(3))
        np.testing.assert_array_equal(preint.dR, np.eye(3))
        
        # Check Jacobians
        np.testing.assert_array_equal(preint.dP_dba, np.zeros((3, 3)))
        np.testing.assert_array_equal(preint.dP_dbg, np.zeros((3, 3)))
        np.testing.assert_array_equal(preint.dV_dba, np.zeros((3, 3)))
        np.testing.assert_array_equal(preint.dV_dbg, np.zeros((3, 3)))
        np.testing.assert_array_equal(preint.dR_dbg, np.zeros((3, 3)))
        
        # Check covariance
        np.testing.assert_array_equal(preint.cov, np.zeros((9, 9)))
        
        # Check noise parameters
        self.assertEqual(preint.acc_noise, 0.01)
        self.assertEqual(preint.gyro_noise, 0.001)
    
    def test_integrate_zero_motion(self):
        preint = IMUPreintegration()
        
        # Zero acceleration and rotation
        acc = np.zeros(3)
        gyro = np.zeros(3)
        dt = 0.01
        
        preint.integrate(acc, gyro, dt)
        
        # Should remain at zero
        np.testing.assert_array_almost_equal(preint.dP, np.zeros(3))
        np.testing.assert_array_almost_equal(preint.dV, np.zeros(3))
        np.testing.assert_array_almost_equal(preint.dR, np.eye(3))
        self.assertAlmostEqual(preint.dt, dt)
    
    def test_integrate_constant_acceleration(self):
        preint = IMUPreintegration()
        
        # Constant acceleration in x direction
        acc = np.array([1.0, 0.0, 0.0])
        gyro = np.zeros(3)
        dt = 0.1
        
        # Integrate for 1 second (10 steps)
        for _ in range(10):
            preint.integrate(acc, gyro, dt)
        
        # After 1 second with 1 m/s^2 acceleration:
        # Velocity should be 1 m/s
        # Position should be 0.5 m
        np.testing.assert_array_almost_equal(preint.dV, np.array([1.0, 0.0, 0.0]), decimal=5)
        np.testing.assert_array_almost_equal(preint.dP, np.array([0.5, 0.0, 0.0]), decimal=5)
        np.testing.assert_array_almost_equal(preint.dR, np.eye(3))
        self.assertAlmostEqual(preint.dt, 1.0)
    
    def test_integrate_constant_rotation(self):
        preint = IMUPreintegration()
        
        # Constant rotation around z-axis
        acc = np.zeros(3)
        gyro = np.array([0.0, 0.0, np.pi/2])  # 90 deg/s
        dt = 0.01
        
        # Integrate for 1 second
        for _ in range(100):
            preint.integrate(acc, gyro, dt)
        
        # After 1 second at 90 deg/s, should have rotated 90 degrees
        expected_R = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        
        np.testing.assert_array_almost_equal(preint.dR, expected_R, decimal=2)
        np.testing.assert_array_almost_equal(preint.dP, np.zeros(3))
        np.testing.assert_array_almost_equal(preint.dV, np.zeros(3))
    
    def test_integrate_small_angle_approximation(self):
        preint = IMUPreintegration()
        
        # Very small rotation (should use small angle approximation)
        acc = np.zeros(3)
        gyro = np.array([1e-10, 1e-10, 1e-10])
        dt = 0.01
        
        preint.integrate(acc, gyro, dt)
        
        # Should be very close to identity
        np.testing.assert_array_almost_equal(preint.dR, np.eye(3), decimal=10)
    
    def test_predict_basic(self):
        preint = IMUPreintegration()
        
        # Setup some preintegrated values
        preint.dP = np.array([1.0, 2.0, 3.0])
        preint.dV = np.array([0.1, 0.2, 0.3])
        preint.dR = np.eye(3)
        preint.dt = 1.0
        
        # Initial state
        pos_i = np.array([10.0, 20.0, 30.0])
        vel_i = np.array([1.0, 2.0, 3.0])
        rot_i = np.eye(3)
        
        # Bias and gravity
        bias = IMUBias()
        gravity = np.array([0.0, 0.0, -9.81])
        
        # Predict next state
        pos_j, vel_j, rot_j = preint.predict(pos_i, vel_i, rot_i, bias, gravity)
        
        # Basic checks (full implementation would need more complex calculations)
        self.assertEqual(len(pos_j), 3)
        self.assertEqual(len(vel_j), 3)
        self.assertEqual(rot_j.shape, (3, 3))
    
    def test_jacobians_update(self):
        preint = IMUPreintegration()
        
        # Initial non-zero Jacobians
        preint.dV_dba = np.ones((3, 3)) * 0.1
        preint.dV_dbg = np.ones((3, 3)) * 0.05
        
        # Integrate
        acc = np.array([0.1, 0.2, 9.8])
        gyro = np.array([0.01, 0.02, 0.03])
        dt = 0.01
        
        initial_dP_dba = preint.dP_dba.copy()
        initial_dP_dbg = preint.dP_dbg.copy()
        
        preint.integrate(acc, gyro, dt)
        
        # Jacobians should have been updated
        self.assertFalse(np.array_equal(preint.dP_dba, initial_dP_dba))
        self.assertFalse(np.array_equal(preint.dP_dbg, initial_dP_dbg))


class TestIMUIntegration(unittest.TestCase):
    
    def test_full_imu_processing_pipeline(self):
        # Create IMU configuration
        config = IMUConfig(
            sensor_id="imu_0",
            sensor_type=SensorType.IMU,
            sampling_rate=100.0,
            noise_params={
                "acc_noise": 0.02,
                "gyro_noise": 0.002
            }
        )
        
        # Create IMU measurements with bias
        true_bias = IMUBias(
            acc_bias=np.array([0.05, -0.03, 0.02]),
            gyro_bias=np.array([0.001, -0.002, 0.0005])
        )
        
        # Generate biased measurements
        measurements = []
        for i in range(10):
            true_acc = np.array([0.1 * np.sin(i * 0.1), 0.0, 9.81])
            true_gyro = np.array([0.0, 0.0, 0.01])
            
            biased_data = np.concatenate([
                true_acc + true_bias.acc_bias,
                true_gyro + true_bias.gyro_bias
            ])
            
            imu_data = IMUData(
                timestamp=1000.0 + i * 0.01,
                sensor_id=config.sensor_id,
                data=biased_data
            )
            measurements.append(imu_data)
        
        # Process measurements
        preint = IMUPreintegration(
            acc_noise=config.noise_params["acc_noise"],
            gyro_noise=config.noise_params["gyro_noise"]
        )
        
        for i in range(len(measurements) - 1):
            # Correct for bias
            corrected = true_bias.correct_measurement(measurements[i])
            
            # Integrate
            dt = measurements[i+1].timestamp - measurements[i].timestamp
            preint.integrate(corrected.acceleration, corrected.angular_velocity, dt)
        
        # Check that preintegration ran
        self.assertGreater(preint.dt, 0)
        self.assertFalse(np.array_equal(preint.dV, np.zeros(3)))


if __name__ == '__main__':
    unittest.main()