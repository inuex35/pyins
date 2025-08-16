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

"""Simple robot lever arm visualization"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pyins', 'sensors'))
from robot_lever_arm import RobotLeverArm

# Create robot
robot = RobotLeverArm()

# Add components
robot.add_component("base", [0, 0, 0], description="Robot base")
robot.add_component("imu", [0.1, 0.0, 0.05], description="IMU")
robot.add_component("gnss_main", [0.2, 0.0, 0.3], description="Main GNSS")
robot.add_component("gnss_aux", [-0.2, 0.0, 0.3], description="Aux GNSS")
robot.add_component("lidar", [0.0, 0.0, 0.4], description="Lidar")

# Visualize
robot.visualize_components()
plt.show()