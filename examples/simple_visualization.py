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

"""Simple robot lever arm visualization using modern PyINS API

This example demonstrates:
1. Creating a robot configuration with multiple sensors
2. Adding lever arms for different components
3. Visualizing the sensor configuration in 3D
"""

import numpy as np
import matplotlib.pyplot as plt
from pyins.sensors.robot_lever_arm import RobotLeverArm, create_standard_robot_config

def main():
    """Main function to demonstrate robot lever arm visualization"""
    
    print("PyINS Robot Lever Arm Visualization Example")
    print("="*50)
    
    # Method 1: Create robot manually
    print("\nMethod 1: Manual Configuration")
    robot = RobotLeverArm()
    
    # Add components with lever arms (offsets from robot center)
    # Format: name, [x, y, z] offset in meters, description
    robot.add_component("base", [0, 0, 0], description="Robot base/center")
    robot.add_component("imu", [0.1, 0.0, 0.05], description="IMU sensor")
    robot.add_component("gnss_main", [0.2, 0.0, 0.3], description="Main GNSS antenna")
    robot.add_component("gnss_aux", [-0.2, 0.0, 0.3], description="Auxiliary GNSS antenna")
    robot.add_component("lidar", [0.0, 0.0, 0.4], description="Lidar sensor")
    robot.add_component("camera", [0.15, 0.05, 0.25], description="Camera")
    
    # Print component information
    print("\nRobot components:")
    for name, comp in robot.components.items():
        pos = comp.position
        desc = comp.description
        print(f"  {name:12s}: [{pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}] m - {desc}")
    
    # Calculate some useful offsets
    print("\nRelative offsets:")
    imu_to_gnss = robot.get_offset("imu", "gnss_main")
    print(f"  IMU to Main GNSS: [{imu_to_gnss[0]:.3f}, {imu_to_gnss[1]:.3f}, {imu_to_gnss[2]:.3f}] m")
    
    gnss_baseline = robot.get_offset("gnss_main", "gnss_aux")
    print(f"  GNSS baseline: [{gnss_baseline[0]:.3f}, {gnss_baseline[1]:.3f}, {gnss_baseline[2]:.3f}] m")
    print(f"  Baseline length: {np.linalg.norm(gnss_baseline):.3f} m")
    
    # Method 2: Use standard configuration
    print("\n\nMethod 2: Standard Configuration")
    standard_robot = create_standard_robot_config()
    
    print("\nStandard robot components:")
    for name, comp in standard_robot.components.items():
        pos = comp.position
        desc = comp.description
        print(f"  {name:12s}: [{pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}] m - {desc}")
    
    # Visualize both configurations
    fig = plt.figure(figsize=(14, 6))
    
    # Manual configuration
    ax1 = fig.add_subplot(121, projection='3d')
    robot.visualize_components(ax=ax1, show=False)
    ax1.set_title('Manual Configuration')
    
    # Standard configuration
    ax2 = fig.add_subplot(122, projection='3d')
    standard_robot.visualize_components(ax=ax2, show=False)
    ax2.set_title('Standard Configuration')
    
    plt.tight_layout()
    plt.show()
    
    print("\nVisualization complete!")
    print("\nKey concepts demonstrated:")
    print("1. Lever arms define sensor positions relative to robot center")
    print("2. Accurate lever arms are critical for sensor fusion")
    print("3. Offsets between sensors can be computed for calibration")
    print("4. Standard configurations provide typical sensor layouts")


if __name__ == '__main__':
    main()