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

"""Robot lever arm visualization example

This example demonstrates:
1. Visualizing standard robot configurations
2. Creating custom vehicle sensor layouts
3. Displaying lever arms between components
4. Multiple viewing angles for sensor positions
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyins.sensors.robot_lever_arm import RobotLeverArm, create_standard_robot_config

def visualize_standard_robot():
    """Visualize standard robot configuration"""
    # Create standard robot configuration
    robot = create_standard_robot_config()
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Visualize components
    robot.visualize_components(ax=ax, show_labels=True)
    
    # Adjust view angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.show()


def visualize_custom_robot():
    """Visualize custom robot with various sensors"""
    # Create custom robot
    robot = RobotLeverArm()
    
    # Add components for an autonomous vehicle
    robot.add_component("base", [0, 0, 0], description="Vehicle center")
    robot.add_component("imu", [0.15, 0.0, 0.05], description="IMU")
    robot.add_component("gnss_front", [1.8, 0.0, 0.5], description="Front GNSS")
    robot.add_component("gnss_rear", [-1.8, 0.0, 0.5], description="Rear GNSS")
    robot.add_component("lidar_top", [0.0, 0.0, 1.5], description="Top Lidar")
    robot.add_component("camera_front", [2.0, 0.0, 1.2], description="Front camera")
    robot.add_component("radar_front", [2.2, 0.0, 0.3], description="Front radar")
    
    # Add wheel positions
    robot.add_component("wheel_fl", [1.5, 0.8, -0.3], description="Front left wheel")
    robot.add_component("wheel_fr", [1.5, -0.8, -0.3], description="Front right wheel")
    robot.add_component("wheel_rl", [-1.5, 0.8, -0.3], description="Rear left wheel")
    robot.add_component("wheel_rr", [-1.5, -0.8, -0.3], description="Rear right wheel")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 3D view
    ax1 = fig.add_subplot(221, projection='3d')
    robot.visualize_components(ax=ax1, show_labels=True, show=False)
    ax1.set_title('3D View')
    ax1.view_init(elev=20, azim=45)
    
    # Top view (X-Y plane)
    ax2 = fig.add_subplot(222)
    positions = robot.get_all_positions()
    for comp_id, pos in positions.items():
        ax2.scatter(pos[0], pos[1], s=100)
        ax2.text(pos[0]+0.1, pos[1]+0.1, comp_id, fontsize=8)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top View (X-Y Plane)')
    ax2.grid(True)
    ax2.axis('equal')
    
    # Side view (X-Z plane)
    ax3 = fig.add_subplot(223)
    for comp_id, pos in positions.items():
        ax3.scatter(pos[0], pos[2], s=100)
        ax3.text(pos[0]+0.1, pos[2]+0.05, comp_id, fontsize=8)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('Side View (X-Z Plane)')
    ax3.grid(True)
    ax3.axis('equal')
    
    # Front view (Y-Z plane)
    ax4 = fig.add_subplot(224)
    for comp_id, pos in positions.items():
        ax4.scatter(pos[1], pos[2], s=100)
        ax4.text(pos[1]+0.05, pos[2]+0.05, comp_id, fontsize=8)
    ax4.set_xlabel('Y (m)')
    ax4.set_ylabel('Z (m)')
    ax4.set_title('Front View (Y-Z Plane)')
    ax4.grid(True)
    ax4.axis('equal')
    
    plt.tight_layout()
    plt.show()


def visualize_lever_arms():
    """Visualize specific lever arms between components"""
    robot = RobotLeverArm()
    
    # Simple robot configuration
    robot.add_component("base", [0, 0, 0])
    robot.add_component("imu", [0.2, 0.0, 0.1])
    robot.add_component("gnss_main", [0.5, 0.0, 0.8])
    robot.add_component("gnss_aux", [-0.5, 0.0, 0.8])
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot components
    positions = robot.get_all_positions()
    for comp_id, pos in positions.items():
        ax.scatter(pos[0], pos[1], pos[2], s=150, label=comp_id)
        ax.text(pos[0], pos[1], pos[2]+0.1, comp_id, fontsize=10)
    
    # Highlight specific lever arms
    # IMU to GNSS main
    imu_pos = robot.get_position("imu")
    gnss_pos = robot.get_position("gnss_main")
    ax.plot([imu_pos[0], gnss_pos[0]], 
            [imu_pos[1], gnss_pos[1]], 
            [imu_pos[2], gnss_pos[2]], 
            'r-', linewidth=3, label='IMU to GNSS')
    
    # GNSS baseline
    gnss_aux_pos = robot.get_position("gnss_aux")
    ax.plot([gnss_aux_pos[0], gnss_pos[0]], 
            [gnss_aux_pos[1], gnss_pos[1]], 
            [gnss_aux_pos[2], gnss_pos[2]], 
            'g-', linewidth=3, label='GNSS Baseline')
    
    # Base to all
    base_pos = robot.get_position("base")
    for comp_id in ["imu", "gnss_main", "gnss_aux"]:
        comp_pos = robot.get_position(comp_id)
        ax.plot([base_pos[0], comp_pos[0]], 
                [base_pos[1], comp_pos[1]], 
                [base_pos[2], comp_pos[2]], 
                'k--', alpha=0.3)
    
    # Calculate and display lever arm magnitudes
    lever_imu_gnss = robot.get_lever_arm("imu", "gnss_main")
    lever_gnss_baseline = robot.get_lever_arm("gnss_aux", "gnss_main")
    
    ax.text(0, 0, -0.5, f"IMU to GNSS: {np.linalg.norm(lever_imu_gnss):.3f} m", fontsize=12)
    ax.text(0, 0, -0.7, f"GNSS Baseline: {np.linalg.norm(lever_gnss_baseline):.3f} m", fontsize=12)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Robot Lever Arms Visualization')
    ax.legend()
    ax.view_init(elev=20, azim=60)
    
    plt.show()


def main():
    """Run all visualization examples"""
    print("Robot Lever Arm Visualization Examples")
    print("=====================================")
    print("1. Standard robot configuration")
    print("2. Custom autonomous vehicle")
    print("3. Lever arms visualization")
    print("4. All examples")
    
    choice = input("\nSelect example (1-4): ")
    
    if choice == '1':
        visualize_standard_robot()
    elif choice == '2':
        visualize_custom_robot()
    elif choice == '3':
        visualize_lever_arms()
    elif choice == '4':
        visualize_standard_robot()
        visualize_custom_robot()
        visualize_lever_arms()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    # If matplotlib is not installed, provide instructions
    try:
        import matplotlib
        main()
    except ImportError:
        print("Matplotlib is required for visualization.")
        print("Please install it with: pip install matplotlib")
        print("\nAlternatively, you can use the visualization method in your code:")
        print("""
# Example code:
from robot_lever_arm import RobotLeverArm, create_standard_robot_config
import matplotlib.pyplot as plt

robot = create_standard_robot_config()
robot.visualize_components()
plt.show()
""")