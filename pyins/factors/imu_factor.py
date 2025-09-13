# Copyright 2024 inuex35
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""IMU factors for GTSAM optimization"""

import numpy as np
import gtsam


class IMUFactorBuilder:
    """Builder class for creating IMU-related factors"""
    
    def __init__(self, imu_config):
        """
        Initialize IMU factor builder
        
        Parameters
        ----------
        imu_config : dict
            IMU configuration parameters including:
            - gravity: gravity magnitude (m/s²)
            - accelerometer_noise_density: accelerometer noise (m/s²/√Hz)
            - gyroscope_noise_density: gyroscope noise (rad/s/√Hz)
            - accelerometer_bias_stability: accelerometer bias stability (m/s²)
            - gyroscope_bias_stability: gyroscope bias stability (rad/s)
            - integration_uncertainty: integration uncertainty
        """
        self.imu_config = imu_config
        self.preint_params = self._create_preintegration_params()
        
    def _create_preintegration_params(self):
        """Create IMU preintegration parameters"""
        # Create parameters with gravity pointing down in navigation frame
        params = gtsam.PreintegrationParams.MakeSharedU(-self.imu_config['gravity'])
        
        # Set accelerometer and gyroscope noise models
        acc_noise_density = self.imu_config.get('accelerometer_noise_density', 0.01)
        gyro_noise_density = self.imu_config.get('gyroscope_noise_density', 0.001)
        integration_uncertainty = self.imu_config.get('integration_uncertainty', 1e-7)
        
        # Convert noise density to discrete-time noise (assuming 100 Hz IMU)
        imu_rate = self.imu_config.get('imu_rate', 100.0)
        dt = 1.0 / imu_rate
        
        acc_noise_sigma = acc_noise_density * np.sqrt(dt)
        gyro_noise_sigma = gyro_noise_density * np.sqrt(dt)
        
        params.setAccelerometerCovariance(acc_noise_sigma**2 * np.eye(3))
        params.setGyroscopeCovariance(gyro_noise_sigma**2 * np.eye(3))
        params.setIntegrationCovariance(integration_uncertainty**2 * np.eye(3))
        
        return params
        
    def create_preintegrated_measurements(self, bias):
        """
        Create new preintegrated IMU measurements
        
        Parameters
        ----------
        bias : gtsam.imuBias.ConstantBias
            Initial IMU bias estimate
            
        Returns
        -------
        gtsam.PreintegratedImuMeasurements
            Preintegrated measurements object
        """
        return gtsam.PreintegratedImuMeasurements(self.preint_params, bias)
        
    def create_imu_factor(self, pose_i, vel_i, pose_j, vel_j, bias_i, preintegrated):
        """
        Create IMU factor between two poses
        
        Parameters
        ----------
        pose_i : gtsam.Key
            Key for pose at time i
        vel_i : gtsam.Key
            Key for velocity at time i
        pose_j : gtsam.Key
            Key for pose at time j
        vel_j : gtsam.Key
            Key for velocity at time j
        bias_i : gtsam.Key
            Key for IMU bias at time i
        preintegrated : gtsam.PreintegratedImuMeasurements
            Preintegrated measurements between times i and j
            
        Returns
        -------
        gtsam.ImuFactor
            IMU factor connecting the states
        """
        return gtsam.ImuFactor(pose_i, vel_i, pose_j, vel_j, bias_i, preintegrated)
        
    def create_bias_evolution_factor(self, bias_i, bias_j, dt):
        """
        Create bias evolution factor (random walk model)
        
        Parameters
        ----------
        bias_i : gtsam.Key
            Key for bias at time i
        bias_j : gtsam.Key
            Key for bias at time j
        dt : float
            Time difference between i and j
            
        Returns
        -------
        gtsam.BetweenFactorConstantBias
            Bias evolution factor
        """
        # Bias random walk parameters
        acc_bias_stability = self.imu_config.get('accelerometer_bias_stability', 1e-4)
        gyro_bias_stability = self.imu_config.get('gyroscope_bias_stability', 1e-5)
        
        # Create noise model for bias evolution
        bias_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([
            acc_bias_stability * np.sqrt(dt),
            acc_bias_stability * np.sqrt(dt),
            acc_bias_stability * np.sqrt(dt),
            gyro_bias_stability * np.sqrt(dt),
            gyro_bias_stability * np.sqrt(dt),
            gyro_bias_stability * np.sqrt(dt)
        ]))
        
        # Zero change expected (random walk)
        zero_bias = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))
        
        return gtsam.BetweenFactorConstantBias(bias_i, bias_j, zero_bias, bias_noise)
        
    def create_pose_prior(self, pose_key, pose, noise_sigmas):
        """
        Create pose prior factor
        
        Parameters
        ----------
        pose_key : gtsam.Key
            Key for the pose
        pose : gtsam.Pose3
            Prior pose estimate
        noise_sigmas : np.ndarray
            6D noise standard deviations [rotation (rad), position (m)]
            
        Returns
        -------
        gtsam.PriorFactorPose3
            Pose prior factor
        """
        noise = gtsam.noiseModel.Diagonal.Sigmas(noise_sigmas)
        return gtsam.PriorFactorPose3(pose_key, pose, noise)
        
    def create_velocity_prior(self, vel_key, velocity, noise_sigmas):
        """
        Create velocity prior factor
        
        Parameters
        ----------
        vel_key : gtsam.Key
            Key for the velocity
        velocity : np.ndarray
            Prior velocity estimate (3D)
        noise_sigmas : np.ndarray
            3D noise standard deviations (m/s)
            
        Returns
        -------
        gtsam.PriorFactorVector
            Velocity prior factor
        """
        noise = gtsam.noiseModel.Diagonal.Sigmas(noise_sigmas)
        return gtsam.PriorFactorVector(vel_key, velocity, noise)
        
    def create_bias_prior(self, bias_key, bias, noise_sigmas):
        """
        Create bias prior factor
        
        Parameters
        ----------
        bias_key : gtsam.Key
            Key for the bias
        bias : gtsam.imuBias.ConstantBias
            Prior bias estimate
        noise_sigmas : np.ndarray
            6D noise standard deviations [acc bias (m/s²), gyro bias (rad/s)]
            
        Returns
        -------
        gtsam.PriorFactorConstantBias
            Bias prior factor
        """
        noise = gtsam.noiseModel.Diagonal.Sigmas(noise_sigmas)
        return gtsam.PriorFactorConstantBias(bias_key, bias, noise)