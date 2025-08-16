"""
PyINS - Comprehensive GNSS/INS Processing Library

A Python library for satellite positioning, pseudorange/carrier phase processing,
IMU mechanization, sensor fusion, and coordinate transformations.
Inspired by gnss-py, rtklib-py, and OB-GINS.
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