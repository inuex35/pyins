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

"""
PyINS - Comprehensive GNSS/INS Processing Library

A Python library for satellite positioning, pseudorange/carrier phase processing,
IMU mechanization, sensor fusion, and coordinate transformations.
Inspired by RTKLIB and gnss-py.
"""

__version__ = "1.0.0"
__author__ = "PyINS Development Team"
__title__ = "pyins"
__description__ = "Comprehensive GNSS/INS processing library"

from .core import *
from .satellite import *
from .observation import *
from .coordinate import *
from .attitude import *
# from .plot import *  # Temporarily disabled due to missing planar dependency
from .utils import *