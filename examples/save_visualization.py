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

"""Save robot lever arm visualization to file

This example demonstrates:
1. Creating multiple robot configurations
2. Visualizing from different viewpoints
3. Saving visualizations to files
4. Creating 2D projections of sensor layouts
"""

import numpy as np
import matplotlib.pyplot as plt
from pyins.sensors.robot_lever_arm import RobotLeverArm, create_standard_robot_config
import os

def main():
    """Main function to create and save robot visualizations"""
    
    print("PyINS Robot Lever Arm Visualization - Save to File")
    print("="*60)
    
    # Create output directory if it doesn't exist
    output_dir = 'visualizations'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Create figure with multiple views
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Multi-Sensor Robot Configurations', fontsize=16, fontweight='bold')
    
    # 1. Standard robot configuration
    print("\nCreating standard robot configuration...")
    robot1 = create_standard_robot_config()
    ax1 = fig.add_subplot(221, projection='3d')
    robot1.visualize_components(ax=ax1, show_labels=True, show=False)
    ax1.set_title('Standard Robot Configuration')
    ax1.view_init(elev=20, azim=45)

    # 2. Custom autonomous vehicle
    print("Creating autonomous vehicle configuration...")
    robot2 = RobotLeverArm()
    robot2.add_component("base", [0, 0, 0], description="Vehicle center")
    robot2.add_component("imu", [0.15, 0.0, 0.05], description="High-precision IMU")
    robot2.add_component("gnss_front", [1.8, 0.0, 0.5], description="Front RTK GNSS")
    robot2.add_component("gnss_rear", [-1.8, 0.0, 0.5], description="Rear RTK GNSS")
    robot2.add_component("lidar", [0.0, 0.0, 1.5], description="3D Lidar")
    robot2.add_component("camera_front", [2.0, 0.0, 1.2], description="Front camera")
    robot2.add_component("radar", [2.2, 0.0, 0.3], description="mmWave radar")
    
    ax2 = fig.add_subplot(222, projection='3d')
    robot2.visualize_components(ax=ax2, show_labels=True, show=False)
    ax2.set_title('Autonomous Vehicle Configuration')
    ax2.view_init(elev=20, azim=45)
    
    # Print vehicle configuration details
    print("\nAutonomous vehicle sensor configuration:")
    for name, comp in robot2.components.items():
        pos = comp.position
        desc = comp.description
        print(f"  {name:15s}: [{pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f}] m - {desc}")
    
    # Calculate GNSS baseline
    gnss_baseline = robot2.get_offset("gnss_front", "gnss_rear")
    baseline_length = np.linalg.norm(gnss_baseline)
    print(f"\n  GNSS Baseline length: {baseline_length:.2f} m")

    # 3. Top view of robot2
    ax3 = fig.add_subplot(223)
    positions = robot2.get_all_positions()
    
    # Color code by sensor type
    colors = {
        'base': 'black',
        'imu': 'red',
        'gnss': 'blue',
        'lidar': 'green',
        'camera': 'orange',
        'radar': 'purple'
    }
    
    for comp_id, pos in positions.items():
        # Determine color based on component type
        color = 'gray'
        for key, col in colors.items():
            if key in comp_id.lower():
                color = col
                break
        
        ax3.scatter(pos[0], pos[1], s=150, c=color, alpha=0.7, edgecolors='black')
        ax3.text(pos[0]+0.1, pos[1]+0.1, comp_id, fontsize=8, fontweight='bold')
    
    ax3.set_xlabel('X (Forward) [m]', fontsize=10)
    ax3.set_ylabel('Y (Left) [m]', fontsize=10)
    ax3.set_title('Top View (X-Y Plane)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    ax3.set_xlim([-2.5, 2.5])
    ax3.set_ylim([-1, 1])

    # 4. Side view of robot2
    ax4 = fig.add_subplot(224)
    for comp_id, pos in positions.items():
        # Determine color based on component type
        color = 'gray'
        for key, col in colors.items():
            if key in comp_id.lower():
                color = col
                break
        
        ax4.scatter(pos[0], pos[2], s=150, c=color, alpha=0.7, edgecolors='black')
        ax4.text(pos[0]+0.1, pos[2]+0.05, comp_id, fontsize=8, fontweight='bold')
    
    ax4.set_xlabel('X (Forward) [m]', fontsize=10)
    ax4.set_ylabel('Z (Up) [m]', fontsize=10)
    ax4.set_title('Side View (X-Z Plane)', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.axis('equal')
    ax4.set_xlim([-2.5, 2.5])
    ax4.set_ylim([-0.5, 2])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save main figure
    output_file = os.path.join(output_dir, 'robot_lever_arm_visualization.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nMain visualization saved to: {output_file}")
    
    # Create and save individual 3D views
    print("\nCreating individual 3D views...")
    
    # Standard robot - multiple angles
    angles = [(25, 60), (45, 135), (10, 90), (60, 45)]
    angle_names = ['perspective', 'rear_view', 'top_view', 'side_view']
    
    for angle_name, (elev, azim) in zip(angle_names, angles):
        fig2 = plt.figure(figsize=(10, 8))
        ax = fig2.add_subplot(111, projection='3d')
        robot1.visualize_components(ax=ax, show_labels=True, show=False)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f'Standard Robot - {angle_name.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        
        filename = os.path.join(output_dir, f'robot_3d_{angle_name}.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  Saved: {filename}")
        plt.close(fig2)
    
    # Autonomous vehicle - perspective view
    fig3 = plt.figure(figsize=(12, 8))
    ax = fig3.add_subplot(111, projection='3d')
    robot2.visualize_components(ax=ax, show_labels=True, show=False)
    ax.view_init(elev=15, azim=45)
    ax.set_title('Autonomous Vehicle Sensor Suite', fontsize=14, fontweight='bold')
    
    filename = os.path.join(output_dir, 'autonomous_vehicle_3d.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close(fig3)
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print(f"All files saved to: {output_dir}/")
    print("\nFiles created:")
    print("  - robot_lever_arm_visualization.png (main overview)")
    print("  - robot_3d_perspective.png")
    print("  - robot_3d_rear_view.png")
    print("  - robot_3d_top_view.png")
    print("  - robot_3d_side_view.png")
    print("  - autonomous_vehicle_3d.png")
    

if __name__ == '__main__':
    main()