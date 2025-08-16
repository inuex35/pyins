"""Base sensor classes and interfaces"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from ..core.data_structures import SensorType

@dataclass
class SensorConfig:
    """Sensor configuration parameters"""
    sensor_id: str
    sensor_type: SensorType
    sampling_rate: float  # Hz
    noise_params: Dict[str, float] = field(default_factory=dict)
    calibration: Dict[str, Any] = field(default_factory=dict)
    lever_arm: np.ndarray = field(default_factory=lambda: np.zeros(3))  # lever arm in body frame (m)
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))     # rotation from sensor to body frame
    
    def get_noise_std(self, param_name: str, default: float = 0.0) -> float:
        """Get noise standard deviation for a parameter"""
        return self.noise_params.get(param_name, default)


@dataclass
class SensorData:
    """Base sensor data container"""
    timestamp: float  # Unix timestamp or GNSS time
    sensor_id: str
    sensor_type: SensorType
    data: np.ndarray
    covariance: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_valid(self) -> bool:
        """Check if sensor data is valid"""
        return self.data is not None and np.all(np.isfinite(self.data))


class SensorInterface(ABC):
    """Abstract base class for sensor interfaces"""
    
    def __init__(self, config: SensorConfig):
        self.config = config
        self._is_initialized = False
        
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the sensor"""
        pass
    
    @abstractmethod
    def read(self) -> Optional[SensorData]:
        """Read data from the sensor"""
        pass
    
    @abstractmethod
    def calibrate(self, calibration_data: Dict[str, Any]) -> bool:
        """Apply calibration to the sensor"""
        pass
    
    def get_lever_arm(self) -> np.ndarray:
        """Get lever arm vector"""
        return self.config.lever_arm
    
    def get_rotation(self) -> np.ndarray:
        """Get rotation matrix from sensor to body frame"""
        return self.config.rotation
    
    def transform_to_body(self, sensor_data: np.ndarray) -> np.ndarray:
        """Transform sensor data to body frame"""
        return self.config.rotation @ sensor_data