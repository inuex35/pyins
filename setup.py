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

"""Setup script for PyINS library"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

requirements = [
    "numpy>=1.19.0",
    "scipy>=1.5.0",
    "pandas>=1.1.0",
    "matplotlib>=3.3.0",
    "numba>=0.51.2",
    "gnsspy @ git+https://github.com/inuex35/gnsspy.git",
]

setup(
    name="pyins",
    version="1.0.0",
    author="PyINS Development Team",
    description="Comprehensive GNSS/INS processing library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/inuex35/pyins",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: GIS",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
    keywords="GNSS GPS INS IMU navigation positioning geodesy",
)