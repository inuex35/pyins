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

"""Example usage of the robot lever arm system"""

import numpy as np
from pyins.sensors.robot_lever_arm import RobotLeverArm, create_standard_robot_config

# Example 1: Basic lever arm setup and usage
def example_basic_usage():
    """Basic lever arm configuration and compensation"""
    print("=== Example 1: Basic Lever Arm Usage ===\n")
    
    # Create robot lever arm manager
    robot = RobotLeverArm()
    
    # Add components with arbitrary string IDs
    robot.add_component("base", [0, 0, 0], description="Robot base center")
    robot.add_component("imu", [0.15, 0.0, 0.05], description="IMU sensor")
    robot.add_component("gnss_main", [0.25, 0.0, 0.35], description="Main GNSS antenna")
    robot.add_component("gnss_aux", [-0.25, 0.0, 0.35], description="Auxiliary GNSS antenna")
    robot.add_component("lidar", [0.0, 0.0, 0.45], description="Lidar sensor")
    
    # Get lever arm between components
    lever_imu_to_gnss = robot.get_lever_arm("imu", "gnss_main")
    print(f"Lever arm from IMU to GNSS: {lever_imu_to_gnss}")
    
    # Calculate GNSS baseline for heading
    gnss_baseline = robot.get_lever_arm("gnss_aux", "gnss_main")
    print(f"GNSS baseline vector: {gnss_baseline}")
    print(f"GNSS baseline length: {np.linalg.norm(gnss_baseline):.3f} m\n")
    
    # Compensate position for lever arm
    base_position = np.array([100.0, 200.0, 50.0])  # Base position in navigation frame
    base_rotation = np.eye(3)  # No rotation for simplicity
    
    gnss_position = robot.compensate_position("gnss_main", base_position, base_rotation)
    print(f"Base position: {base_position}")
    print(f"GNSS position after lever arm: {gnss_position}\n")


# Example 2: Dynamic compensation with rotation
def example_dynamic_compensation():
    """Lever arm compensation with vehicle dynamics"""
    print("=== Example 2: Dynamic Lever Arm Compensation ===\n")
    
    robot = RobotLeverArm()
    robot.add_component("base", [0, 0, 0])
    robot.add_component("gnss", [0.3, 0.0, 0.4])
    
    # Vehicle state
    base_velocity = np.array([10.0, 0.0, 0.0])  # 10 m/s forward
    angular_velocity = np.array([0.0, 0.0, 0.5])  # 0.5 rad/s yaw rate
    base_rotation = np.eye(3)
    
    # Compensate velocity for GNSS antenna
    gnss_velocity = robot.compensate_velocity("gnss", base_velocity, angular_velocity, base_rotation)
    
    print(f"Base velocity: {base_velocity} m/s")
    print(f"Angular velocity: {angular_velocity} rad/s")
    print(f"GNSS velocity after compensation: {gnss_velocity} m/s")
    print(f"Velocity difference: {gnss_velocity - base_velocity} m/s\n")


# Example 3: Configuration file usage
def example_config_file():
    """Load and save lever arm configuration"""
    print("=== Example 3: Configuration File Usage ===\n")
    
    # Create and configure robot
    robot = create_standard_robot_config()
    
    # Save to YAML file
    config_path = "robot_config.yaml"
    robot.save_to_file(config_path, format='yaml')
    print(f"Saved configuration to {config_path}")
    
    # Load from file
    robot2 = RobotLeverArm()
    robot2.load_from_file(config_path)
    print(f"Loaded configuration from {config_path}")
    
    # List all components
    print("\nLoaded components:")
    for comp_id in robot2.list_components():
        pos = robot2.get_position(comp_id)
        comp = robot2.get_component(comp_id)
        print(f"  {comp_id}: position={pos}, description='{comp.description}'")
    
    # Clean up
    import os
    os.remove(config_path)


# Example 4: Multi-sensor fusion application
def example_sensor_fusion():
    """Example for multi-sensor fusion with lever arms"""
    print("\n=== Example 4: Multi-Sensor Fusion Application ===\n")
    
    robot = RobotLeverArm()
    
    # Configure sensors for a typical autonomous vehicle
    robot.add_component("base", [0, 0, 0], description="Vehicle center of gravity")
    robot.add_component("imu", [0.1, 0.0, 0.02], description="IMU near CG")
    robot.add_component("gnss_front", [1.5, 0.0, 0.5], description="Front GNSS antenna")
    robot.add_component("gnss_rear", [-1.5, 0.0, 0.5], description="Rear GNSS antenna")
    robot.add_component("camera_front", [2.0, 0.0, 1.2], 
                       rotation=np.array([[1, 0, 0], [0, 0.866, -0.5], [0, 0.5, 0.866]]),
                       description="Front camera tilted down 30 degrees")
    
    # Simulate vehicle state
    position = np.array([1000.0, 2000.0, 100.0])
    velocity = np.array([15.0, 0.0, 0.0])  # 15 m/s forward
    yaw = np.radians(45)  # 45 degrees heading
    yaw_rate = np.radians(10)  # 10 deg/s turn rate
    
    # Rotation matrix for 45 degree heading
    R_body = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw), np.cos(yaw), 0],
                       [0, 0, 1]])
    
    omega = np.array([0, 0, yaw_rate])
    
    # Calculate positions for all sensors
    print("Sensor positions in navigation frame:")
    for sensor_id in robot.list_components():
        sensor_pos = robot.compensate_position(sensor_id, position, R_body)
        print(f"  {sensor_id}: {sensor_pos}")
    
    # Calculate velocities for all sensors
    print("\nSensor velocities in navigation frame:")
    for sensor_id in robot.list_components():
        sensor_vel = robot.compensate_velocity(sensor_id, velocity, omega, R_body)
        speed = np.linalg.norm(sensor_vel)
        print(f"  {sensor_id}: {sensor_vel} (speed: {speed:.2f} m/s)")
    
    # GNSS baseline for heading computation
    baseline = robot.get_lever_arm("gnss_rear", "gnss_front")
    baseline_nav = R_body @ baseline
    heading_from_baseline = np.arctan2(baseline_nav[1], baseline_nav[0])
    print(f"\nGNSS baseline heading: {np.degrees(heading_from_baseline):.1f} degrees")


# Example 5: Custom robot configuration
def example_custom_robot():
    """Example of custom robot with unique components"""
    print("\n=== Example 5: Custom Robot Configuration ===\n")
    
    robot = RobotLeverArm()
    
    # Configure a custom robot (e.g., agricultural robot)
    robot.add_component("base", [0, 0, 0], description="Robot chassis center")
    robot.add_component("gps_rover", [0.2, 0.0, 2.1], description="RTK rover antenna")
    robot.add_component("spray_nozzle_left", [0.0, 3.0, 0.5], description="Left spray nozzle")
    robot.add_component("spray_nozzle_right", [0.0, -3.0, 0.5], description="Right spray nozzle")
    robot.add_component("soil_sensor", [1.5, 0.0, -0.3], description="Ground penetrating sensor")
    robot.add_component("weather_station", [0.0, 0.0, 2.5], description="Weather sensors")
    
    # Calculate working width
    working_width = np.linalg.norm(
        robot.get_lever_arm("spray_nozzle_left", "spray_nozzle_right")
    )
    print(f"Spray working width: {working_width:.1f} m")
    
    # Position offset from GPS to soil sensor
    gps_to_soil = robot.get_lever_arm("gps_rover", "soil_sensor")
    print(f"GPS to soil sensor offset: {gps_to_soil}")
    
    # Save custom configuration
    robot.save_to_file("agricultural_robot.yaml")
    print("\nSaved agricultural robot configuration")
    
    # Clean up
    import os
    os.remove("agricultural_robot.yaml")


if __name__ == "__main__":
    example_basic_usage()
    example_dynamic_compensation()
    example_config_file()
    example_sensor_fusion()
    example_custom_robot()