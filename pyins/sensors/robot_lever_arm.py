"""Robot lever arm management system with arbitrary component support"""

import numpy as np
from typing import Dict, Optional, List, Tuple, Union
from dataclasses import dataclass, field
import json
import yaml
from pathlib import Path

@dataclass
class ComponentFrame:
    """Represents a component's coordinate frame relative to robot base"""
    position: np.ndarray  # 3D position vector [x, y, z] in meters
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))  # Rotation matrix from component to base frame
    description: str = ""  # Optional description of the component
    
    def __post_init__(self):
        """Validate inputs"""
        if self.position.shape != (3,):
            self.position = np.array(self.position).reshape(3)
        if self.rotation.shape != (3, 3):
            self.rotation = np.array(self.rotation).reshape(3, 3)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'position': self.position.tolist(),
            'rotation': self.rotation.tolist(),
            'description': self.description
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ComponentFrame':
        """Create from dictionary"""
        return cls(
            position=np.array(data['position']),
            rotation=np.array(data.get('rotation', np.eye(3))),
            description=data.get('description', '')
        )


class RobotLeverArm:
    """Manages lever arms for arbitrary robot components"""
    
    def __init__(self):
        """Initialize the robot lever arm manager"""
        self.components: Dict[str, ComponentFrame] = {}
        self._base_frame_id = "base"  # Default base frame identifier
        
    def add_component(self, 
                     component_id: str, 
                     position: Union[np.ndarray, List[float]], 
                     rotation: Optional[Union[np.ndarray, List[List[float]]]] = None,
                     description: str = "") -> None:
        """
        Add a component with its position and orientation
        
        Parameters:
        -----------
        component_id : str
            Unique identifier for the component (e.g., "imu", "gnss_main", "wheel_front_left")
        position : np.ndarray or list
            3D position vector [x, y, z] relative to base frame (meters)
        rotation : np.ndarray or list, optional
            3x3 rotation matrix from component to base frame
        description : str, optional
            Description of the component
        """
        if rotation is None:
            rotation = np.eye(3)
        else:
            rotation = np.array(rotation)
            
        self.components[component_id] = ComponentFrame(
            position=np.array(position),
            rotation=rotation,
            description=description
        )
    
    def get_component(self, component_id: str) -> Optional[ComponentFrame]:
        """Get component frame information"""
        return self.components.get(component_id)
    
    def get_position(self, component_id: str) -> Optional[np.ndarray]:
        """Get component position"""
        component = self.get_component(component_id)
        return component.position if component else None
    
    def get_rotation(self, component_id: str) -> Optional[np.ndarray]:
        """Get component rotation matrix"""
        component = self.get_component(component_id)
        return component.rotation if component else None
    
    def get_lever_arm(self, from_id: str, to_id: str) -> Optional[np.ndarray]:
        """
        Get lever arm vector from one component to another
        
        Parameters:
        -----------
        from_id : str
            Source component ID
        to_id : str
            Target component ID
            
        Returns:
        --------
        lever_arm : np.ndarray
            3D lever arm vector from source to target in base frame
        """
        from_component = self.get_component(from_id)
        to_component = self.get_component(to_id)
        
        if from_component is None or to_component is None:
            return None
            
        return to_component.position - from_component.position
    
    def compensate_position(self,
                          component_id: str,
                          base_position: np.ndarray,
                          base_rotation: np.ndarray) -> Optional[np.ndarray]:
        """
        Calculate component position in navigation frame
        
        Parameters:
        -----------
        component_id : str
            Component identifier
        base_position : np.ndarray
            Base position in navigation frame (3x1)
        base_rotation : np.ndarray
            Rotation matrix from base to navigation frame (3x3)
            
        Returns:
        --------
        component_position : np.ndarray
            Component position in navigation frame
        """
        component = self.get_component(component_id)
        if component is None:
            return None
            
        return base_position + base_rotation @ component.position
    
    def compensate_velocity(self,
                          component_id: str,
                          base_velocity: np.ndarray,
                          base_angular_velocity: np.ndarray,
                          base_rotation: np.ndarray) -> Optional[np.ndarray]:
        """
        Calculate component velocity in navigation frame
        
        Parameters:
        -----------
        component_id : str
            Component identifier
        base_velocity : np.ndarray
            Base velocity in navigation frame (3x1)
        base_angular_velocity : np.ndarray
            Angular velocity in base frame (3x1)
        base_rotation : np.ndarray
            Rotation matrix from base to navigation frame (3x3)
            
        Returns:
        --------
        component_velocity : np.ndarray
            Component velocity in navigation frame
        """
        component = self.get_component(component_id)
        if component is None:
            return None
            
        # v_component = v_base + R_base * (omega_base x lever_arm)
        return base_velocity + base_rotation @ np.cross(base_angular_velocity, component.position)
    
    def compensate_acceleration(self,
                              component_id: str,
                              base_acceleration: np.ndarray,
                              base_angular_velocity: np.ndarray,
                              base_angular_acceleration: np.ndarray,
                              base_rotation: np.ndarray) -> Optional[np.ndarray]:
        """
        Calculate component acceleration in navigation frame
        
        Parameters:
        -----------
        component_id : str
            Component identifier
        base_acceleration : np.ndarray
            Base acceleration in navigation frame (3x1)
        base_angular_velocity : np.ndarray
            Angular velocity in base frame (3x1)
        base_angular_acceleration : np.ndarray
            Angular acceleration in base frame (3x1)
        base_rotation : np.ndarray
            Rotation matrix from base to navigation frame (3x3)
            
        Returns:
        --------
        component_acceleration : np.ndarray
            Component acceleration in navigation frame
        """
        component = self.get_component(component_id)
        if component is None:
            return None
            
        lever_arm = component.position
        
        # Centripetal acceleration: omega x (omega x lever_arm)
        centripetal = np.cross(base_angular_velocity, 
                              np.cross(base_angular_velocity, lever_arm))
        
        # Tangential acceleration: alpha x lever_arm
        tangential = np.cross(base_angular_acceleration, lever_arm)
        
        # Total acceleration
        return base_acceleration + base_rotation @ (centripetal + tangential)
    
    def transform_measurement(self,
                            from_component_id: str,
                            to_component_id: str,
                            measurement: np.ndarray,
                            measurement_type: str = 'vector') -> Optional[np.ndarray]:
        """
        Transform measurement from one component frame to another
        
        Parameters:
        -----------
        from_component_id : str
            Source component ID
        to_component_id : str
            Target component ID
        measurement : np.ndarray
            Measurement in source component frame
        measurement_type : str
            Type of measurement: 'vector' or 'point'
            
        Returns:
        --------
        transformed : np.ndarray
            Measurement transformed to target component frame
        """
        from_comp = self.get_component(from_component_id)
        to_comp = self.get_component(to_component_id)
        
        if from_comp is None or to_comp is None:
            return None
        
        # Transform to base frame
        in_base = from_comp.rotation @ measurement
        
        # Transform to target frame
        transformed = to_comp.rotation.T @ in_base
        
        # If it's a point, also account for position offset
        if measurement_type == 'point':
            lever_arm = self.get_lever_arm(from_component_id, to_component_id)
            if lever_arm is not None:
                transformed += to_comp.rotation.T @ lever_arm
                
        return transformed
    
    def list_components(self) -> List[str]:
        """List all registered component IDs"""
        return list(self.components.keys())
    
    def get_all_positions(self) -> Dict[str, np.ndarray]:
        """Get positions of all components"""
        return {comp_id: comp.position 
                for comp_id, comp in self.components.items()}
    
    def save_to_file(self, filepath: Union[str, Path], format: str = 'yaml') -> None:
        """
        Save lever arm configuration to file
        
        Parameters:
        -----------
        filepath : str or Path
            Path to save file
        format : str
            File format: 'yaml' or 'json'
        """
        data = {
            'base_frame_id': self._base_frame_id,
            'components': {
                comp_id: comp.to_dict() 
                for comp_id, comp in self.components.items()
            }
        }
        
        filepath = Path(filepath)
        if format == 'yaml':
            with open(filepath, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        elif format == 'json':
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def load_from_file(self, filepath: Union[str, Path]) -> None:
        """
        Load lever arm configuration from file
        
        Parameters:
        -----------
        filepath : str or Path
            Path to configuration file
        """
        filepath = Path(filepath)
        
        if filepath.suffix in ['.yaml', '.yml']:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
        elif filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        self._base_frame_id = data.get('base_frame_id', 'base')
        self.components = {
            comp_id: ComponentFrame.from_dict(comp_data)
            for comp_id, comp_data in data.get('components', {}).items()
        }
    
    def visualize_components(self, ax=None, show_labels: bool = True) -> None:
        """
        Visualize component positions in 3D
        
        Parameters:
        -----------
        ax : matplotlib axis, optional
            3D axis to plot on
        show_labels : bool
            Whether to show component labels
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            raise ImportError("Matplotlib required for visualization")
        
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        
        # Plot base frame
        ax.scatter([0], [0], [0], c='red', s=100, marker='o', label='Base')
        
        # Plot components
        for comp_id, comp in self.components.items():
            pos = comp.position
            ax.scatter([pos[0]], [pos[1]], [pos[2]], s=50, marker='o')
            
            if show_labels:
                ax.text(pos[0], pos[1], pos[2], f'  {comp_id}', fontsize=8)
            
            # Draw lever arm from base
            ax.plot([0, pos[0]], [0, pos[1]], [0, pos[2]], 'k--', alpha=0.3)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Robot Component Configuration')
        ax.legend()
        
        # Equal aspect ratio
        max_range = np.max(np.abs([comp.position for comp in self.components.values()]))
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        
        return ax


# Convenience functions for common robot configurations
def create_standard_robot_config() -> RobotLeverArm:
    """Create a standard robot configuration with common components"""
    robot = RobotLeverArm()
    
    # Add common components (example configuration)
    robot.add_component("base", [0, 0, 0], description="Robot base/center")
    robot.add_component("imu", [0.1, 0, 0.05], description="IMU sensor")
    robot.add_component("gnss_main", [0.2, 0, 0.3], description="Main GNSS antenna")
    robot.add_component("gnss_aux", [-0.2, 0, 0.3], description="Auxiliary GNSS antenna")
    robot.add_component("wheel_fl", [0.3, 0.2, -0.1], description="Front left wheel")
    robot.add_component("wheel_fr", [0.3, -0.2, -0.1], description="Front right wheel")
    robot.add_component("wheel_rl", [-0.3, 0.2, -0.1], description="Rear left wheel")
    robot.add_component("wheel_rr", [-0.3, -0.2, -0.1], description="Rear right wheel")
    
    return robot