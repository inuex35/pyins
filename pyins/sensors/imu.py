"""IMU sensor data structures and processing"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from .sensor_base import SensorConfig, SensorData, SensorInterface, SensorType

@dataclass
class IMUData(SensorData):
    """IMU measurement data"""
    def __post_init__(self):
        self.sensor_type = SensorType.IMU
        if self.data.shape != (6,):
            raise ValueError("IMU data must be 6D [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]")
    
    @property
    def acceleration(self) -> np.ndarray:
        """Get acceleration measurements (m/s^2)"""
        return self.data[:3]
    
    @property
    def angular_velocity(self) -> np.ndarray:
        """Get angular velocity measurements (rad/s)"""
        return self.data[3:6]
    
    def remove_gravity(self, gravity_vector: np.ndarray) -> np.ndarray:
        """Remove gravity from acceleration measurements"""
        return self.acceleration - gravity_vector


@dataclass
class IMUConfig(SensorConfig):
    """IMU configuration parameters"""
    def __post_init__(self):
        self.sensor_type = SensorType.IMU
        
        # Default IMU noise parameters if not provided
        default_noise = {
            'acc_noise': 0.01,      # m/s^2
            'gyro_noise': 0.001,    # rad/s
            'acc_bias_walk': 0.0001,  # m/s^3
            'gyro_bias_walk': 0.00001 # rad/s^2
        }
        
        for key, value in default_noise.items():
            if key not in self.noise_params:
                self.noise_params[key] = value


@dataclass
class IMUBias:
    """IMU bias state"""
    acc_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))   # accelerometer bias (m/s^2)
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))  # gyroscope bias (rad/s)
    
    def correct_measurement(self, imu_data: IMUData) -> IMUData:
        """Apply bias correction to IMU measurements"""
        corrected_data = imu_data.data.copy()
        corrected_data[:3] -= self.acc_bias
        corrected_data[3:6] -= self.gyro_bias
        
        return IMUData(
            timestamp=imu_data.timestamp,
            sensor_id=imu_data.sensor_id,
            sensor_type=imu_data.sensor_type,
            data=corrected_data,
            covariance=imu_data.covariance,
            metadata=imu_data.metadata
        )


@dataclass
class IMUPreintegration:
    """IMU preintegration for factor graph optimization"""
    dt: float = 0.0                                           # integration time interval
    dP: np.ndarray = field(default_factory=lambda: np.zeros(3))     # preintegrated position
    dV: np.ndarray = field(default_factory=lambda: np.zeros(3))     # preintegrated velocity  
    dR: np.ndarray = field(default_factory=lambda: np.eye(3))       # preintegrated rotation
    
    # Jacobians with respect to bias
    dP_dba: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    dP_dbg: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    dV_dba: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    dV_dbg: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    dR_dbg: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    
    # Covariance
    cov: np.ndarray = field(default_factory=lambda: np.zeros((9, 9)))
    
    # Noise parameters
    acc_noise: float = 0.01
    gyro_noise: float = 0.001
    
    def integrate(self, acc: np.ndarray, gyro: np.ndarray, dt: float):
        """Integrate a single IMU measurement"""
        # Store previous values
        dP_prev = self.dP.copy()
        dV_prev = self.dV.copy()
        dR_prev = self.dR.copy()
        
        # Compute rotation increment
        theta = gyro * dt
        theta_norm = np.linalg.norm(theta)
        
        if theta_norm > 1e-8:
            axis = theta / theta_norm
            K = np.array([[0, -axis[2], axis[1]],
                         [axis[2], 0, -axis[0]],
                         [-axis[1], axis[0], 0]])
            dR_inc = np.eye(3) + np.sin(theta_norm) * K + (1 - np.cos(theta_norm)) * K @ K
        else:
            dR_inc = np.eye(3) + skew(theta)
        
        # Update preintegrated values
        self.dR = dR_prev @ dR_inc
        self.dV = dV_prev + dR_prev @ acc * dt
        self.dP = dP_prev + dV_prev * dt + 0.5 * dR_prev @ acc * dt * dt
        
        # Update Jacobians (simplified - full implementation would be more complex)
        I = np.eye(3)
        self.dP_dba += self.dV_dba * dt - 0.5 * dR_prev * dt * dt
        self.dP_dbg += self.dV_dbg * dt
        self.dV_dba -= dR_prev * dt
        
        # Update time
        self.dt += dt
        
        # Update covariance (simplified)
        Q = np.diag([self.acc_noise**2] * 3 + [self.gyro_noise**2] * 3) * dt
        # Full covariance propagation would go here
        
    def predict(self, pos_i: np.ndarray, vel_i: np.ndarray, rot_i: np.ndarray, 
                bias: IMUBias, gravity: np.ndarray) -> tuple:
        """Predict state at j given state at i"""
        # Apply bias correction to preintegrated values
        dP_corrected = self.dP + self.dP_dba @ bias.acc_bias + self.dP_dbg @ bias.gyro_bias
        dV_corrected = self.dV + self.dV_dba @ bias.acc_bias + self.dV_dbg @ bias.gyro_bias
        dR_corrected = self.dR @ exp_so3(self.dR_dbg @ bias.gyro_bias)
        
        # Predict state at j
        pos_j = pos_i + vel_i * self.dt + 0.5 * gravity * self.dt**2 + rot_i @ dP_corrected
        vel_j = vel_i + gravity * self.dt + rot_i @ dV_corrected
        rot_j = rot_i @ dR_corrected
        
        return pos_j, vel_j, rot_j


def skew(v: np.ndarray) -> np.ndarray:
    """Create skew-symmetric matrix from vector"""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def exp_so3(v: np.ndarray) -> np.ndarray:
    """Exponential map from so(3) to SO(3)"""
    theta = np.linalg.norm(v)
    if theta < 1e-8:
        return np.eye(3) + skew(v)
    
    axis = v / theta
    K = skew(axis)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


class IMUSensor(SensorInterface):
    """IMU sensor interface implementation"""
    
    def __init__(self, config: IMUConfig):
        super().__init__(config)
        self.bias = IMUBias()
        self.preintegration = None
        
    def initialize(self) -> bool:
        """Initialize the IMU sensor"""
        # Initialization logic here
        self._is_initialized = True
        return True
    
    def read(self) -> Optional[IMUData]:
        """Read IMU data"""
        # Implementation would interface with actual IMU hardware/driver
        # This is a placeholder
        return None
    
    def calibrate(self, calibration_data: Dict[str, Any]) -> bool:
        """Apply IMU calibration"""
        if 'acc_bias' in calibration_data:
            self.bias.acc_bias = np.array(calibration_data['acc_bias'])
        if 'gyro_bias' in calibration_data:
            self.bias.gyro_bias = np.array(calibration_data['gyro_bias'])
        
        # Update configuration with calibration data
        self.config.calibration.update(calibration_data)
        return True
    
    def start_preintegration(self):
        """Start a new preintegration period"""
        self.preintegration = IMUPreintegration(
            acc_noise=self.config.get_noise_std('acc_noise'),
            gyro_noise=self.config.get_noise_std('gyro_noise')
        )
    
    def add_measurement(self, imu_data: IMUData, dt: float):
        """Add IMU measurement to preintegration"""
        if self.preintegration is None:
            self.start_preintegration()
        
        # Apply bias correction
        corrected = self.bias.correct_measurement(imu_data)
        
        # Integrate
        self.preintegration.integrate(
            corrected.acceleration,
            corrected.angular_velocity,
            dt
        )