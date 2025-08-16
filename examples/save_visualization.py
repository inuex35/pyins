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

"""Save robot lever arm visualization to file"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pyins', 'sensors'))
from robot_lever_arm import RobotLeverArm, create_standard_robot_config

# Create figure with multiple views
fig = plt.figure(figsize=(16, 12))

# 1. Standard robot configuration
robot1 = create_standard_robot_config()
ax1 = fig.add_subplot(221, projection='3d')
robot1.visualize_components(ax=ax1, show_labels=True)
ax1.set_title('Standard Robot Configuration')
ax1.view_init(elev=20, azim=45)

# 2. Custom autonomous vehicle
robot2 = RobotLeverArm()
robot2.add_component("base", [0, 0, 0], description="Vehicle center")
robot2.add_component("imu", [0.15, 0.0, 0.05], description="IMU")
robot2.add_component("gnss_front", [1.8, 0.0, 0.5], description="Front GNSS")
robot2.add_component("gnss_rear", [-1.8, 0.0, 0.5], description="Rear GNSS")
robot2.add_component("lidar", [0.0, 0.0, 1.5], description="Lidar")
robot2.add_component("camera", [2.0, 0.0, 1.2], description="Camera")

ax2 = fig.add_subplot(222, projection='3d')
robot2.visualize_components(ax=ax2, show_labels=True)
ax2.set_title('Autonomous Vehicle Configuration')
ax2.view_init(elev=20, azim=45)

# 3. Top view of robot2
ax3 = fig.add_subplot(223)
positions = robot2.get_all_positions()
for comp_id, pos in positions.items():
    ax3.scatter(pos[0], pos[1], s=100)
    ax3.text(pos[0]+0.1, pos[1]+0.1, comp_id, fontsize=8)
ax3.set_xlabel('X (m)')
ax3.set_ylabel('Y (m)')
ax3.set_title('Top View')
ax3.grid(True)
ax3.axis('equal')

# 4. Side view of robot2
ax4 = fig.add_subplot(224)
for comp_id, pos in positions.items():
    ax4.scatter(pos[0], pos[2], s=100)
    ax4.text(pos[0]+0.1, pos[2]+0.05, comp_id, fontsize=8)
ax4.set_xlabel('X (m)')
ax4.set_ylabel('Z (m)')
ax4.set_title('Side View')
ax4.grid(True)
ax4.axis('equal')

plt.tight_layout()

# Save to file
output_file = 'robot_lever_arm_visualization.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"Visualization saved to: {output_file}")

# Also save individual figures
fig2 = plt.figure(figsize=(10, 8))
ax = fig2.add_subplot(111, projection='3d')
robot1.visualize_components(ax=ax, show_labels=True)
ax.view_init(elev=25, azim=60)
plt.savefig('robot_3d_view.png', dpi=150, bbox_inches='tight')
print("3D view saved to: robot_3d_view.png")